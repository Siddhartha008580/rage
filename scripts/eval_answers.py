"""Evaluate retrieved evidence and generated answers against a changing JSON set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chat_rag import (  # noqa: E402
    INDEX_DIR,
    ask_openrouter,
    plan_logical_query,
    retrieve_chunks,
)


def _validate_specs(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("evaluation file must contain a non-empty JSON array")
    specs: list[dict[str, Any]] = []
    list_fields = (
        "expected_sources",
        "expected_any_sources",
        "expected_terms",
        "expected_answer_terms",
        "expected_pages",
    )
    for index, spec in enumerate(value, 1):
        if not isinstance(spec, dict):
            raise ValueError(f"query {index} must be a JSON object")
        for field in ("id", "query"):
            if not isinstance(spec.get(field), str) or not spec[field].strip():
                raise ValueError(
                    f"query {index} must have a non-empty string {field!r}"
                )
        for field in list_fields:
            items = spec.get(field, [])
            if not isinstance(items, list):
                raise ValueError(
                    f"query {spec['id']!r} field {field!r} must be a list"
                )
            item_type = int if field == "expected_pages" else str
            if not all(type(item) is item_type for item in items):
                raise ValueError(
                    f"query {spec['id']!r} field {field!r} has invalid items"
                )
            if field == "expected_pages" and any(item < 1 for item in items):
                raise ValueError(
                    f"query {spec['id']!r} field 'expected_pages' "
                    "must contain positive page numbers"
                )
        specs.append(spec)
    return specs


def score_answer(
    spec: dict[str, Any],
    chunks: list[dict[str, Any]],
    answer_result: dict[str, Any],
) -> dict[str, Any]:
    """Score dynamic expected evidence and answer terms without domain assumptions."""
    sources = {str(chunk.get("source") or "") for chunk in chunks}
    pages = {chunk.get("page") for chunk in chunks}
    evidence = "\n".join(str(chunk.get("text") or "") for chunk in chunks).casefold()
    answer = str(answer_result.get("answer") or "")
    answer_folded = answer.casefold()

    expected_sources = spec.get("expected_sources", [])
    expected_any_sources = spec.get("expected_any_sources", [])
    expected_terms = spec.get("expected_terms", [])
    expected_answer_terms = spec.get("expected_answer_terms", [])
    expected_pages = spec.get("expected_pages", [])
    missing_sources = [source for source in expected_sources if source not in sources]
    missing_any_sources = (
        [] if not expected_any_sources or sources.intersection(expected_any_sources)
        else list(expected_any_sources)
    )
    missing_terms = [
        term for term in expected_terms if term.casefold() not in evidence
    ]
    missing_answer_terms = [
        term for term in expected_answer_terms if term.casefold() not in answer_folded
    ]
    missing_pages = [page for page in expected_pages if page not in pages]
    unverified_citations = answer_result.get("unverified_citations") or []
    generation_complete = answer_result.get("generation_complete") is True
    claim_support_verified = answer_result.get("claim_support_verified") is True

    passed = (
        not missing_sources
        and not missing_any_sources
        and not missing_terms
        and not missing_answer_terms
        and not missing_pages
        and not unverified_citations
        and generation_complete
        and claim_support_verified
    )
    return {
        "id": spec["id"],
        "passed": passed,
        "sources_retrieved": sorted(sources),
        "sources_missing": missing_sources,
        "any_sources_missing": missing_any_sources,
        "evidence_terms_missing": missing_terms,
        "answer_terms_missing": missing_answer_terms,
        "pages_retrieved": sorted(page for page in pages if page is not None),
        "pages_missing": missing_pages,
        "unverified_citations": unverified_citations,
        "generation_complete": generation_complete,
        "claim_support_verified": answer_result.get("claim_support_verified"),
        "answer": answer,
        "retrieved_citations": sorted(
            {
                str(chunk.get("citation"))
                for chunk in chunks
                if chunk.get("citation")
            }
        ),
    }


def evaluate_one(
    spec: dict[str, Any],
    *,
    index_dir: Path,
    top_k: int,
    candidate_k: int,
    retrieval_mode: str,
) -> dict[str, Any]:
    plan = plan_logical_query(spec["query"])
    chunks = retrieve_chunks(
        spec["query"],
        top_k=top_k,
        candidate_k=candidate_k,
        index_dir=index_dir,
        retrieval_mode=retrieval_mode,
        query_plan=plan,
    )
    answer_result = ask_openrouter(
        spec["query"],
        chunks,
        query_plan=plan,
        high_stakes=True,
    )
    return score_answer(spec, chunks, answer_result)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate RAG answers against a JSON set of expected evidence and answer terms."
    )
    parser.add_argument("specs", type=Path, help="JSON array of evaluation cases")
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument(
        "--retrieval-mode",
        choices=("hybrid", "spectrag", "agentic"),
        default="hybrid",
    )
    parser.add_argument("--out", type=Path, help="Optional JSON results path")
    args = parser.parse_args()
    if args.top_k < 1 or args.candidate_k < args.top_k:
        parser.error("--top-k must be positive and --candidate-k >= --top-k")

    specs = _validate_specs(json.loads(args.specs.read_text(encoding="utf-8")))
    results = [
        evaluate_one(
            spec,
            index_dir=args.index_dir,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            retrieval_mode=args.retrieval_mode,
        )
        for spec in specs
        if spec.get("status") != "blocked"
    ]
    payload = {
        "passed": sum(result["passed"] for result in results),
        "total": len(results),
        "results": results,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if all(result["passed"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
