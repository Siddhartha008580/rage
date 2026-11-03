from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pdfplumber
import pymupdf

try:
    from .convert import _bbox_iou
    from .table_transformer import TableTransformerExtractor
except ImportError:
    from convert import _bbox_iou
    from table_transformer import TableTransformerExtractor


def _page_table_summary(page: Any) -> tuple[float, list[dict[str, Any]]]:
    tables = []
    score = 0.0
    try:
        detected_tables = page.find_tables()
    except Exception:
        detected_tables = []
    for table in detected_tables:
        try:
            rows = table.extract()
        except Exception:
            continue
        columns = max((len(row) for row in rows), default=0)
        if not rows or columns < 2:
            continue
        score += len(rows) * columns
        tables.append(
            {
                "bbox": tuple(float(value) for value in table.bbox),
                "rows": len(rows),
                "columns": columns,
            }
        )
    return score, tables


def select_sample_pages(input_dir: Path, max_documents: int = 10) -> list[dict[str, Any]]:
    samples = []
    for path in sorted(input_dir.rglob("*.pdf"))[:max_documents]:
        best_table_page = None
        best_table_score = 0.0
        best_text_page = 0
        best_text_score = -1
        with pdfplumber.open(path) as pdf:
            for page_index, page in enumerate(pdf.pages):
                score, tables = _page_table_summary(page)
                text_score = len(page.extract_text() or "")
                if text_score > best_text_score:
                    best_text_score = text_score
                    best_text_page = page_index
                if score > best_table_score:
                    best_table_score = score
                    best_table_page = {
                        "source": path,
                        "page_index": page_index,
                        "selection": "highest table row-column density",
                        "baseline_tables": tables,
                    }
        if best_table_page is not None:
            samples.append(best_table_page)
        else:
            samples.append(
                {
                    "source": path,
                    "page_index": best_text_page,
                    "selection": "highest text density control",
                    "baseline_tables": [],
                }
            )
    return samples


def evaluate_sample(
    input_dir: Path,
    max_documents: int = 10,
    dpi: int = 150,
) -> list[dict[str, Any]]:
    extractor = TableTransformerExtractor(dpi=dpi)
    results = []
    for sample in select_sample_pages(input_dir, max_documents):
        source = sample["source"]
        page_index = sample["page_index"]
        with pymupdf.open(source) as document:
            predictions = extractor.extract_page(document[page_index])

        unmatched = set(range(len(predictions)))
        matches = []
        for baseline in sample["baseline_tables"]:
            best_index, best_overlap = None, 0.0
            for index in unmatched:
                overlap = _bbox_iou(baseline["bbox"], predictions[index].bbox)
                if overlap > best_overlap:
                    best_index, best_overlap = index, overlap
            if best_index is not None and best_overlap >= 0.3:
                unmatched.remove(best_index)
                prediction = predictions[best_index]
                matches.append(
                    {
                        "bbox_iou": round(best_overlap, 3),
                        "baseline_shape": [baseline["rows"], baseline["columns"]],
                        "transformer_shape": [
                            len(prediction.rows),
                            max((len(row) for row in prediction.rows), default=0),
                        ],
                        "shape_match": baseline["rows"] == len(prediction.rows)
                        and baseline["columns"]
                        == max((len(row) for row in prediction.rows), default=0),
                        "confidence": round(prediction.confidence, 3),
                        "word_coverage": round(prediction.word_coverage, 3),
                    }
                )
        results.append(
            {
                "source": source.relative_to(input_dir).as_posix(),
                "page": page_index + 1,
                "selection": sample["selection"],
                "baseline_table_count": len(sample["baseline_tables"]),
                "transformer_table_count": len(predictions),
                "matched_tables": len(matches),
                "unmatched_transformer_tables": len(unmatched),
                "matches": matches,
            }
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a small CPU Table Transformer pilot across input PDFs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "input",
    )
    parser.add_argument("--max-documents", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    results = evaluate_sample(args.input_dir.resolve(), args.max_documents, args.dpi)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())