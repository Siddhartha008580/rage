"""Trace failed retrieval evals through Markdown, chunks, page labels, and retrieval.

Usage:
    python scripts/trace_failure.py
    python scripts/trace_failure.py --query C-06
    python scripts/trace_failure.py --json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_eval(path: Path) -> list[dict[str, Any]]:
    data = _read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")
    specs = []
    for spec in data:
        if not isinstance(spec, dict) or not isinstance(spec.get("id"), str):
            raise ValueError(f"{path} contains an eval entry without a string id")
        for field in ("expected_sources", "expected_any_sources", "expected_terms"):
            if not isinstance(spec.get(field, []), list):
                raise ValueError(f"{spec['id']}: {field} must be a list")
        if not isinstance(spec.get("expected_pages", []), list):
            raise ValueError(f"{spec['id']}: expected_pages must be a list")
        specs.append(spec)
    return specs


def _load_baselines(paths: list[Path]) -> dict[str, dict[str, dict[str, Any]]]:
    baselines = {}
    for path in paths:
        records = _read_json(path)
        if not isinstance(records, list):
            raise ValueError(f"{path} must contain a JSON array")
        baselines[path.stem] = {
            record["id"]: record
            for record in records
            if isinstance(record, dict) and isinstance(record.get("id"), str)
        }
    return baselines


def _load_chunks(path: Path) -> list[dict[str, Any]]:
    chunks = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if isinstance(record, dict):
                chunks.append(record)
    return chunks


def _markdown_by_source(output_dir: Path) -> dict[str, list[Path]]:
    files: dict[str, list[Path]] = defaultdict(list)
    for tables_path in output_dir.rglob("*.tables.json"):
        if ".progress" in tables_path.parts:
            continue
        try:
            metadata = _read_json(tables_path)
        except (OSError, json.JSONDecodeError):
            continue
        source = metadata.get("source_file") if isinstance(metadata, dict) else None
        markdown_path = tables_path.with_suffix(".md")
        if isinstance(source, str) and markdown_path.is_file():
            files[source].append(markdown_path)

    # Older conversions may not have sidecars; use the conventional source stem.
    for markdown_path in output_dir.rglob("*.md"):
        if ".progress" in markdown_path.parts:
            continue
        source_stem = markdown_path.name.split("--", 1)[0]
        for source in (f"{source_stem}.pdf",):
            if markdown_path not in files[source]:
                files[source].append(markdown_path)
    return {source: sorted(set(paths)) for source, paths in files.items()}


def _terms_in_text(text: str, terms: list[str]) -> list[str]:
    folded = text.casefold()
    return [term for term in terms if term.casefold() in folded]


def trace_one(
    spec: dict[str, Any],
    chunks: list[dict[str, Any]],
    markdown_files: dict[str, list[Path]],
) -> dict[str, Any]:
    expected_sources = spec.get("expected_sources", [])
    expected_any_sources = spec.get("expected_any_sources", [])
    terms = spec.get("expected_terms", [])
    expected_pages = spec.get("expected_pages", [])
    sources = expected_sources or expected_any_sources
    result: dict[str, Any] = {
        "id": spec["id"],
        "verdict": "",
        "detail": "",
        "markdown_files": {},
        "term_chunk_pages": {},
        "baseline_failures": [],
    }

    if not sources:
        result.update(
            verdict="UNSCOPED_EVAL",
            detail="eval has no expected source; this query cannot be traced to a document",
        )
        return result

    missing_markdown = [source for source in sources if source not in markdown_files]
    if missing_markdown:
        result.update(
            verdict="NO_MARKDOWN_FILE",
            detail=f"no converted Markdown found for {', '.join(missing_markdown)}",
        )
        return result

    candidate_paths = [
        path for source in sources for path in markdown_files.get(source, [])
    ]
    result["markdown_files"] = {
        source: [str(path) for path in markdown_files.get(source, [])]
        for source in sources
    }
    markdown_texts = [path.read_text(encoding="utf-8").casefold() for path in candidate_paths]
    missing_markdown_terms = [
        term
        for term in terms
        if not any(term.casefold() in text for text in markdown_texts)
    ]
    if missing_markdown_terms:
        result.update(
            verdict="MARKDOWN_LOSS",
            detail=f"eval terms absent from Markdown: {missing_markdown_terms}",
        )
        return result

    source_chunks = [
        chunk for chunk in chunks if chunk.get("source") in sources
    ]
    missing_chunk_sources = [
        source
        for source in expected_sources
        if not any(chunk.get("source") == source for chunk in source_chunks)
    ]
    missing_chunk_terms = [
        term
        for term in terms
        if not any(
            term.casefold() in str(chunk.get("text", "")).casefold()
            for chunk in source_chunks
        )
    ]
    result["term_chunk_pages"] = {
        term: {
            str(source): sorted(
                {
                    chunk.get("pdf_page")
                    for chunk in source_chunks
                    if chunk.get("source") == source
                    and term.casefold() in str(chunk.get("text", "")).casefold()
                    and isinstance(chunk.get("pdf_page"), int)
                }
            )
            for source in sources
        }
        for term in terms
    }
    if missing_chunk_sources or missing_chunk_terms:
        details = []
        if missing_chunk_sources:
            details.append(f"expected sources have no chunks: {missing_chunk_sources}")
        if missing_chunk_terms:
            details.append(f"Markdown terms absent from chunks: {missing_chunk_terms}")
        result.update(verdict="CHUNKER_LOSS", detail="; ".join(details))
        return result

    available_pages = {
        chunk.get("pdf_page")
        for chunk in source_chunks
        if isinstance(chunk.get("pdf_page"), int)
    }
    missing_pages = [page for page in expected_pages if page not in available_pages]
    if missing_pages:
        result.update(
            verdict="PAGE_LABEL_REVIEW",
            detail=(
                f"expected page(s) {missing_pages} have no chunks for the expected "
                "source(s); verify the eval page labels against the PDF"
            ),
        )
        return result

    result.update(
        verdict="RETRIEVAL_LOSS",
        detail=(
            f"expected source(s), terms, and labeled pages exist in chunks "
            f"({len(source_chunks)} matching-source chunks); inspect retrieval selection"
        ),
    )
    return result


VERDICT_MEANING = {
    "NO_MARKDOWN_FILE": "conversion/output plumbing",
    "MARKDOWN_LOSS": "text absent from converted Markdown",
    "CHUNKER_LOSS": "text or expected source absent from chunks",
    "PAGE_LABEL_REVIEW": "expected page is not represented in chunks; verify the PDF",
    "RETRIEVAL_LOSS": "evidence exists in chunks; inspect retrieval selection",
    "UNSCOPED_EVAL": "eval has no source document to trace",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-file", type=Path, default=ROOT / "eval_selection.json")
    parser.add_argument("--chunks", type=Path, default=ROOT / "chunks.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "markdown")
    parser.add_argument(
        "--baseline",
        type=Path,
        nargs="*",
        help="baseline result JSON files (default: existing hybrid/spectrag results)",
    )
    parser.add_argument("--query", help="trace one eval id, even if it passed")
    parser.add_argument("--json", action="store_true", help="emit JSON results")
    args = parser.parse_args()

    baseline_paths = args.baseline
    if baseline_paths is None:
        baseline_paths = [
            ROOT / ".baseline" / f"{mode}.json"
            for mode in ("hybrid", "spectrag")
            if (ROOT / ".baseline" / f"{mode}.json").is_file()
        ]
    if not baseline_paths:
        parser.error("no baseline result files found; provide --baseline or run eval_retrieval.py")
    if not args.chunks.is_file():
        parser.error(f"chunks file not found: {args.chunks}")
    if not args.eval_file.is_file():
        parser.error(f"eval file not found: {args.eval_file}")
    if not args.output_dir.is_dir():
        parser.error(f"Markdown output directory not found: {args.output_dir}")

    specs = _load_eval(args.eval_file)
    baselines = _load_baselines(baseline_paths)
    failures = {
        query_id
        for results in baselines.values()
        for query_id, record in results.items()
        if not record.get("hit", False)
    }
    if args.query:
        specs = [spec for spec in specs if spec["id"] == args.query]
        if not specs:
            parser.error(f"no eval entry found for {args.query}")
    else:
        specs = [spec for spec in specs if spec["id"] in failures]
    if not specs:
        parser.error("no failed evals found in the baseline files")

    chunks = _load_chunks(args.chunks)
    markdown_files = _markdown_by_source(args.output_dir)
    results = []
    for spec in specs:
        result = trace_one(spec, chunks, markdown_files)
        result["baseline_failures"] = sorted(
            mode
            for mode, records in baselines.items()
            if spec["id"] in records and not records[spec["id"]].get("hit", False)
        )
        if any(len(paths) > 1 for paths in result["markdown_files"].values()):
            result["detail"] += (
                " (multiple Markdown conversion variants found; confirm the active one)"
            )
        results.append(result)

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    print(f"\n{'ID':<7} {'verdict':<20} {'failed baselines':<22} detail")
    print("=" * 120)
    for result in results:
        print(
            f"{result['id']:<7} {result['verdict']:<20} "
            f"{', '.join(result['baseline_failures']):<22} {result['detail']}"
        )
    print("\nSUMMARY")
    counts: dict[str, int] = defaultdict(int)
    for result in results:
        counts[result["verdict"]] += 1
    for verdict, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {verdict:<20} {count:>2}  {VERDICT_MEANING[verdict]}")


if __name__ == "__main__":
    main()
