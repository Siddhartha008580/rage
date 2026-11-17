"""Compare retrieval modes against labeled queries.

Usage:
    python scripts/eval_retrieval.py eval_selection.json \
        --modes hybrid spectrag agentic logical --top-k 10 --out-dir .baseline
    python scripts/eval_retrieval.py eval_selection.json --debug-chunks

The evaluator-local "logical" mode performs deterministic clause, bridge-query,
and optional graph-neighbor retrieval without adding a mode to chat_rag.py.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chat_rag import (
    CANDIDATE_K,
    INDEX_DIR,
    _load_chunk_graph,
    _normalize_chunk,
    diversify_candidates,
    is_comparison_query,
    reciprocal_rank_fusion,
    rerank_candidates,
    retrieve_chunks,
    select_best_per_source,
    select_with_subquery_coverage,
)
from src.rag import get_index_chunks, query_index, query_lexical_index
from src.spectrag import expand_candidate_ids


VALID_MODES = ("hybrid", "spectrag", "agentic", "logical")
TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
CLAUSE_SPLIT_RE = re.compile(
    r"\b(?:versus|vs\.?|compared to|compared with|whereas|while|"
    r"unless|except when|except if|except for|provided that|assuming that|"
    r"subject to|and if|or if|when|if)\b|;",
    re.IGNORECASE,
)
CROSS_REF_RE = re.compile(
    r"\b((?:Note|Table|Section|Item|Appendix|Exhibit|Schedule|Figure|Footnote)"
    r"\s+\d+[A-Za-z]?(?:\.\d+)*)\b",
    re.IGNORECASE,
)
BRIDGE_ENTITY_RE = re.compile(
    r"\b(?:[A-Z]{2,}[0-9A-Z-]*|[A-Z][a-z]+(?:[A-Z][a-z0-9]+)+|"
    r"[A-Za-z]+\d+[A-Za-z0-9]*)\b"
)
STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "before", "both",
        "but", "by", "can", "compared", "could", "did", "do", "does", "for",
        "from", "had", "has", "have", "how", "if", "in", "into", "is", "it",
        "its", "may", "might", "must", "of", "on", "or", "should", "than",
        "that", "the", "their", "then", "there", "these", "this", "those",
        "to", "under", "unless", "was", "were", "what", "when", "where",
        "which", "while", "who", "why", "will", "with", "would",
    }
)
MAX_LOCAL_SUBQUERIES = 4
MAX_LOCAL_BRIDGE_QUERIES = 3


def _tokens(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_RE.findall(text)]


def _content_tokens(text: str) -> list[str]:
    return [
        token
        for token in _tokens(text)
        if token not in STOPWORDS
        and (len(token) > 2 or any(character.isdigit() for character in token))
    ]


def _decompose_query_local(query: str) -> list[str]:
    """Split conditional and comparison queries into deterministic search branches."""
    clauses = [
        part.strip(" ,?.:;-")
        for part in CLAUSE_SPLIT_RE.split(query)
        if len(_content_tokens(part.strip(" ,?.:;-"))) >= 2
    ]
    if len(clauses) <= 1 and " and " in query.casefold():
        and_clauses = [
            part.strip(" ,?.:;-")
            for part in re.split(r"\band\b", query, flags=re.IGNORECASE)
            if len(_content_tokens(part.strip(" ,?.:;-"))) >= 3
        ]
        if len(and_clauses) >= 2:
            clauses = and_clauses
    return (clauses or [query])[:MAX_LOCAL_SUBQUERIES]


def _local_bridge_queries(
    query: str,
    clauses: list[str],
    seed_chunks: list[dict[str, Any]],
) -> list[str]:
    """Generate local follow-up searches from references and uncovered query terms."""
    top_chunks = seed_chunks[:4]
    if not top_chunks:
        return []

    top_text = " ".join(chunk["text"] for chunk in top_chunks)
    top_tokens = set(_tokens(top_text))
    existing_headers = " ".join(
        " ".join(chunk.get("metadata", {}).get("section_path", []))
        + " "
        + chunk.get("metadata", {}).get("section", "")
        for chunk in top_chunks
    ).casefold()
    bridge_queries: list[str] = []

    for chunk in top_chunks:
        metadata = chunk.get("metadata") or {}
        source = metadata.get("source") or chunk.get("source", "")
        source_stem = Path(source).stem.replace("_", " ")
        for match in CROSS_REF_RE.finditer(chunk["text"]):
            reference = match.group(1).strip()
            if reference.casefold() not in existing_headers:
                candidate = f"{source_stem} {reference}".strip()
                if candidate and candidate not in bridge_queries:
                    bridge_queries.append(candidate)
                if len(bridge_queries) >= MAX_LOCAL_BRIDGE_QUERIES:
                    return bridge_queries

    bridge_entities: list[str] = []
    for chunk in top_chunks[:2]:
        for match in BRIDGE_ENTITY_RE.finditer(chunk["text"]):
            entity = match.group(0)
            if (
                len(entity) >= 3
                and entity.casefold() not in STOPWORDS
                and entity not in bridge_entities
            ):
                bridge_entities.append(entity)
            if len(bridge_entities) >= 5:
                break

    query_terms = list(
        dict.fromkeys(
            token
            for text in [query, *clauses]
            for token in _content_tokens(text)
        )
    )
    uncovered_terms = [term for term in query_terms if term not in top_tokens]
    if uncovered_terms and bridge_entities:
        candidate = " ".join(
            [*uncovered_terms[:4], *bridge_entities[:3]]
        )
        if candidate not in bridge_queries:
            bridge_queries.append(candidate)

    for clause in clauses:
        clause_terms = _content_tokens(clause)
        if not clause_terms or len(bridge_queries) >= MAX_LOCAL_BRIDGE_QUERIES:
            continue
        overlap = (
            sum(term in top_tokens for term in clause_terms) / len(clause_terms)
        )
        if overlap < 0.35:
            reinforced = " ".join([clause, *bridge_entities[:2]]).strip()
            if reinforced and reinforced not in bridge_queries:
                bridge_queries.append(reinforced)
    return bridge_queries[:MAX_LOCAL_BRIDGE_QUERIES]


def _retrieve_logical(
    query: str,
    top_k: int,
    *,
    index_dir: Path,
    candidate_k: int,
) -> list[dict[str, Any]]:
    """Evaluator-local deterministic multi-hop retrieval; does not change chat modes."""
    if not _tokens(query):
        return []

    clauses = _decompose_query_local(query)
    dense_hits = query_index(query, index_dir, top_k=candidate_k)
    lexical_hits = query_lexical_index(
        index_dir, _tokens(query), top_k=candidate_k
    )

    sub_k = max(15, candidate_k // 2)
    clause_lists: list[list[dict[str, Any]]] = []
    for clause in clauses:
        if clause.casefold() == query.casefold():
            continue
        clause_lists.extend(
            [
                query_index(clause, index_dir, top_k=sub_k),
                query_lexical_index(index_dir, _tokens(clause), top_k=sub_k),
            ]
        )
    hop1 = reciprocal_rank_fusion(
        dense_hits,
        lexical_hits,
        extra_hit_lists=clause_lists,
        candidate_k=candidate_k,
    )
    if not hop1:
        return []

    bridge_queries = _local_bridge_queries(query, clauses, hop1)
    bridge_lists: list[list[dict[str, Any]]] = []
    for bridge_query in bridge_queries:
        bridge_terms = _tokens(bridge_query)
        if bridge_terms:
            bridge_lists.extend(
                [
                    query_index(
                        bridge_query, index_dir, top_k=max(10, top_k)
                    ),
                    query_lexical_index(
                        index_dir, bridge_terms, top_k=max(10, top_k)
                    ),
                ]
            )

    graph_path = index_dir / "spectrag_graph.npz"
    if graph_path.is_file():
        graph = _load_chunk_graph(index_dir.resolve())
        seed_ids = [chunk["id"] for chunk in hop1[:8]]
        neighbor_ids = expand_candidate_ids(
            graph, seed_ids, max_hops=1, max_candidates=30
        )
        existing_ids = {chunk["id"] for chunk in hop1}
        new_ids = [
            chunk_id for chunk_id in neighbor_ids if chunk_id not in existing_ids
        ]
        if new_ids:
            graph_chunks = get_index_chunks(new_ids, index_dir)
            if graph_chunks:
                bridge_lists.append(graph_chunks)

    candidates = reciprocal_rank_fusion(
        dense_hits,
        lexical_hits,
        extra_hit_lists=[*clause_lists, *bridge_lists],
        candidate_k=min(candidate_k + 25, 80),
    )
    active_subqueries = list(
        dict.fromkeys([*clauses, *bridge_queries[:2]])
    )
    reranked = rerank_candidates(
        query, candidates, sub_queries=active_subqueries
    )
    if is_comparison_query(query):
        selected = select_best_per_source(reranked, top_k)
    elif len(active_subqueries) > 1:
        selected = select_with_subquery_coverage(
            reranked, active_subqueries, top_k
        )
    else:
        selected = diversify_candidates(reranked, top_k)
    return [_normalize_chunk(chunk) for chunk in selected]


def _validate_specs(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("evaluation file must contain a non-empty JSON array")

    specs = []
    for index, spec in enumerate(value):
        if not isinstance(spec, dict):
            raise ValueError(f"query {index + 1} must be a JSON object")
        for field in ("id", "type", "query"):
            if not isinstance(spec.get(field), str) or not spec[field].strip():
                raise ValueError(
                    f"query {index + 1} must have a non-empty string {field!r}"
                )
        for field in (
            "expected_sources",
            "expected_any_sources",
            "expected_terms",
        ):
            items = spec.get(field, [])
            if not isinstance(items, list) or not all(
                isinstance(item, str) for item in items
            ):
                raise ValueError(
                    f"query {spec['id']!r} field {field!r} must be a list of strings"
                )
        expected_pages = spec.get("expected_pages", [])
        if not isinstance(expected_pages, list) or not all(
            type(page) is int and page >= 1 for page in expected_pages
        ):
            raise ValueError(
                f"query {spec['id']!r} field 'expected_pages' "
                "must be a list of positive integers"
            )
        specs.append(spec)
    return specs


def evaluate_one(
    spec: dict[str, Any],
    mode: str,
    top_k: int,
    *,
    index_dir: Path = INDEX_DIR,
    debug_chunks: bool = False,
) -> dict[str, Any]:
    """Retrieve for one query and check all labeled sources and terms."""
    started = time.perf_counter()
    if mode == "logical":
        chunks = _retrieve_logical(
            spec["query"],
            top_k,
            index_dir=index_dir,
            candidate_k=max(CANDIDATE_K, top_k),
        )
    else:
        chunks = retrieve_chunks(
            spec["query"],
            top_k=top_k,
            index_dir=index_dir,
            retrieval_mode=mode,
            candidate_k=max(CANDIDATE_K, top_k),
        )
    retrieval_ms = (time.perf_counter() - started) * 1000

    sources = {chunk["source"] for chunk in chunks}
    pages = {chunk["page"] for chunk in chunks}
    combined_text = "\n".join(chunk["text"] for chunk in chunks).casefold()
    expected_sources = spec.get("expected_sources", [])
    expected_any_sources = spec.get("expected_any_sources", [])
    expected_terms = spec.get("expected_terms", [])
    expected_pages = spec.get("expected_pages", [])
    sources_found = [source for source in expected_sources if source in sources]
    sources_missing = [
        source for source in expected_sources if source not in sources
    ]
    any_sources_found = [
        source for source in expected_any_sources if source in sources
    ]
    terms_found = [
        term for term in expected_terms if term.casefold() in combined_text
    ]
    terms_missing = [
        term for term in expected_terms if term.casefold() not in combined_text
    ]
    pages_found = [page for page in expected_pages if page in pages]
    pages_missing = [page for page in expected_pages if page not in pages]
    sources_ok = not sources_missing and (
        not expected_any_sources or bool(any_sources_found)
    )
    expected_source_count = len(expected_sources) + int(bool(expected_any_sources))
    sources_found_count = len(sources_found) + int(bool(any_sources_found))
    source_recall = (
        sources_found_count / expected_source_count
        if expected_source_count
        else None
    )
    terms_ok = not terms_missing
    pages_ok = not pages_missing

    result = {
        "id": spec["id"],
        "type": spec["type"],
        "hit": sources_ok and terms_ok and pages_ok,
        "sources_ok": sources_ok,
        "terms_ok": terms_ok,
        "pages_ok": pages_ok,
        "sources_retrieved": sorted(sources),
        "expected_sources": expected_sources,
        "sources_found": sources_found,
        "sources_missing": sources_missing,
        "expected_any_sources": expected_any_sources,
        "any_sources_found": any_sources_found,
        "source_recall": source_recall,
        "sources_found_count": sources_found_count,
        "expected_source_count": expected_source_count,
        "terms_found": terms_found,
        "terms_missing": terms_missing,
        "pages_retrieved": sorted(pages),
        "expected_pages": expected_pages,
        "pages_found": pages_found,
        "pages_missing": pages_missing,
        "n_chunks": len(chunks),
        "retrieval_ms": retrieval_ms,
    }
    if debug_chunks and not result["hit"]:
        result["retrieved_chunks"] = [
            {
                "id": chunk["id"],
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk_type": chunk["chunk_type"],
                "matched_expected_terms": _matched_terms(
                    chunk["text"], expected_terms
                ),
            }
            for chunk in chunks
        ]
    return result


def _matched_terms(text: str, expected_terms: list[str]) -> list[str]:
    folded_text = text.casefold()
    return [
        term for term in expected_terms if term.casefold() in folded_text
    ]


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "hits": 0,
            "n": 0,
            "sources_found": 0,
            "expected_sources": 0,
            "retrieval_ms": 0.0,
        }
    )
    for result in results:
        counts = by_type[result["type"]]
        counts["n"] += 1
        counts["hits"] += int(result["hit"])
        counts["sources_found"] += result["sources_found_count"]
        counts["expected_sources"] += result["expected_source_count"]
        counts["retrieval_ms"] += result["retrieval_ms"]

    hits = sum(int(result["hit"]) for result in results)
    count = len(results)
    sources_found = sum(result["sources_found_count"] for result in results)
    expected_sources = sum(result["expected_source_count"] for result in results)
    return {
        "overall": {
            "hits": hits,
            "n": count,
            "rate": hits / count if count else 0.0,
            "source_recall": (
                sources_found / expected_sources if expected_sources else None
            ),
            "sources_found": sources_found,
            "expected_sources": expected_sources,
            "avg_retrieval_ms": (
                sum(result["retrieval_ms"] for result in results) / count
                if count
                else 0.0
            ),
        },
        "by_type": {
            query_type: {
                "hits": int(counts["hits"]),
                "n": int(counts["n"]),
                "rate": counts["hits"] / counts["n"] if counts["n"] else 0.0,
                "source_recall": (
                    counts["sources_found"] / counts["expected_sources"]
                    if counts["expected_sources"]
                    else None
                ),
                "avg_retrieval_ms": (
                    counts["retrieval_ms"] / counts["n"] if counts["n"] else 0.0
                ),
            }
            for query_type, counts in sorted(by_type.items())
        },
    }


def _print_result(result: dict[str, Any]) -> None:
    mark = "✓" if result["hit"] else "✗"
    extra = ""
    if result["sources_missing"]:
        extra += f" missing_sources={result['sources_missing']}"
    if result["expected_any_sources"] and not result["any_sources_found"]:
        extra += f" missing_any_source={result['expected_any_sources']}"
    if result["terms_missing"]:
        extra += f" missing_terms={result['terms_missing']}"
    if result["pages_missing"]:
        extra += f" missing_pages={result['pages_missing']}"
    print(
        f"  {mark} {result['id']:<6} {result['type']:<12}"
        f"{result['retrieval_ms']:.0f}ms{extra}"
    )
    for chunk in result.get("retrieved_chunks", []):
        print(
            f"      {chunk['id']}  source={chunk['source']} "
            f"page={chunk['page']} type={chunk['chunk_type']} "
            f"matched_terms={chunk['matched_expected_terms']}"
        )


def _print_comparison(summaries: dict[str, dict[str, Any]]) -> None:
    modes = list(summaries)
    if len(modes) < 2:
        return

    print("\n" + "=" * 72)
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 72)
    column_width = 34
    header = f"{'type':<12}" + "".join(
        f"{mode:^{column_width}}" for mode in modes
    )
    print(header)
    print(
        f"{'':<12}"
        + "".join(
            f"{'hits/rate':<15}{'src recall':<11}{'avg ms':<8}"
            for _ in modes
        )
    )
    print("-" * len(header))

    all_types = sorted(
        {
            query_type
            for summary in summaries.values()
            for query_type in summary["by_type"]
        }
    )
    for query_type in [*all_types, "overall"]:
        row = f"{query_type:<14}"
        for mode in modes:
            data = (
                summaries[mode]["overall"]
                if query_type == "overall"
                else summaries[mode]["by_type"].get(
                    query_type, {"hits": 0, "n": 0, "rate": 0.0}
                )
            )
            recall = data.get("source_recall")
            recall_text = f"{recall:.0%}" if recall is not None else "n/a"
            latency = data.get("avg_retrieval_ms", 0.0)
            row += (
                f"{data['hits']}/{data['n']} ({data['rate']:.0%})".ljust(15)
                + f"{recall_text:<11}{latency:<8.0f}"
            )
        print(row)

    best_rate = max(summaries[mode]["overall"]["rate"] for mode in modes)
    best_modes = [
        mode
        for mode in modes
        if summaries[mode]["overall"]["rate"] == best_rate
    ]
    print(
        "\nBest retrieval-only hit rate: "
        f"{', '.join(best_modes)} ({best_rate:.0%})"
    )
    fastest_mode = min(
        modes, key=lambda mode: summaries[mode]["overall"]["avg_retrieval_ms"]
    )
    print(
        f"Fastest average retrieval: {fastest_mode} "
        f"({summaries[fastest_mode]['overall']['avg_retrieval_ms']:.0f} ms)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_file", type=Path)
    parser.add_argument(
        "--modes", nargs="+", choices=VALID_MODES, default=list(VALID_MODES)
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    parser.add_argument("--out-dir", type=Path, default=Path(".baseline"))
    parser.add_argument(
        "--debug-chunks",
        action="store_true",
        help="include retrieved chunk metadata and matched expected terms for failures",
    )
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if len(set(args.modes)) != len(args.modes):
        parser.error("--modes must not contain duplicates")

    specs = _validate_specs(json.loads(args.eval_file.read_text(encoding="utf-8")))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict[str, Any]] = {}
    for mode in args.modes:
        print("\n" + "=" * 72)
        print(
            f"MODE: {mode}  (top_k={args.top_k}, "
            f"{len(specs)} queries, "
            + (
                "agentic gap audit may call the planner LLM)"
                if mode == "agentic"
                and os.environ.get("OPENROUTER_API_KEY")
                else "0 LLM calls)"
            )
        )
        print("=" * 72)
        results = [
            evaluate_one(
                spec,
                mode,
                args.top_k,
                index_dir=args.index_dir,
                debug_chunks=args.debug_chunks,
            )
            for spec in specs
        ]
        summary = summarize(results)
        summaries[mode] = summary

        for result in results:
            _print_result(result)

        overall = summary["overall"]
        recall = overall["source_recall"]
        recall_text = f"{recall:.0%}" if recall is not None else "n/a"
        print(
            f"\n  Overall: {overall['hits']}/{overall['n']} "
            f"= {overall['rate']:.2f}; source recall={recall_text}; "
            f"avg retrieval={overall['avg_retrieval_ms']:.0f} ms"
        )
        for query_type, counts in summary["by_type"].items():
            recall = counts["source_recall"]
            recall_text = f"{recall:.0%}" if recall is not None else "n/a"
            print(
                f"    {query_type:<12}: {counts['hits']}/{counts['n']} "
                f"= {counts['rate']:.2f}; source recall={recall_text}; "
                f"avg retrieval={counts['avg_retrieval_ms']:.0f} ms"
            )

        (args.out_dir / f"{mode}.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (args.out_dir / f"{mode}_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    _print_comparison(summaries)
    print(f"\nResults saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
