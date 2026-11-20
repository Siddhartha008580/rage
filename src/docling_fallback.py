"""Optional Docling fallback for difficult PDF pages."""
from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path
from typing import Any


DOCLING_AVAILABLE: bool | None = None
DOCLING_HEADER = "<!-- rendered-by: docling -->\n"

SPARSE_TEXT_THRESHOLD = 100
IMAGE_COVERAGE_THRESHOLD = 0.4
FULL_PAGE_IMAGE_COVERAGE = 0.8
FULL_PAGE_TEXT_THRESHOLD = 50
MIN_TEXT_FOR_TABLE_DETECTION = 200


def _enabled(value: str | None) -> bool:
    return (value or "").strip().casefold() in {"1", "true", "yes"}


def docling_configuration() -> dict[str, bool]:
    """Return settings that affect cached PDF conversion output."""
    return {
        "enabled": not _enabled(os.environ.get("DOCLING_DISABLE")),
        "ocr": _enabled(os.environ.get("DOCLING_OCR")),
        "available": importlib.util.find_spec("docling") is not None,
    }


def _check_docling() -> bool:
    """Check lazily whether Docling is importable."""
    global DOCLING_AVAILABLE
    if not docling_configuration()["enabled"]:
        return False
    if DOCLING_AVAILABLE is None:
        try:
            import docling  # noqa: F401
            from docling.document_converter import DocumentConverter  # noqa: F401
        except ImportError:
            DOCLING_AVAILABLE = False
        else:
            DOCLING_AVAILABLE = True
    return DOCLING_AVAILABLE is True


def _has_aligned_numeric_column(page: Any) -> bool:
    """Detect repeated right-aligned numeric values likely belonging to a table."""
    lines: dict[tuple[int, int], list[tuple[Any, ...]]] = {}
    for word in page.get_text("words"):
        if len(word) < 8:
            continue
        y0 = float(word[1])
        if not (page.rect.height * 0.1 <= y0 <= page.rect.height * 0.9):
            continue
        lines.setdefault((int(word[5]), int(word[6])), []).append(word)

    aligned_columns: dict[int, list[str]] = {}
    for words in lines.values():
        words.sort(key=lambda word: float(word[0]))
        if len(words) < 2:
            continue
        value = re.sub(r"[,.;:()\[\]]+$", "", str(words[-1][4]).strip())
        if not re.fullmatch(r"[-+]?\d[\d,]*(?:\.\d+)?%?", value):
            continue
        right_edge_bin = round(float(words[-1][2]) / 12)
        aligned_columns.setdefault(right_edge_bin, []).append(value)
    return any(
        len(values) >= 5 and len(set(values)) >= 3
        for values in aligned_columns.values()
    )


def _has_dot_leader_rows(page: Any) -> bool:
    """Detect pages with multiple table-of-contents dot leaders."""
    page_numbers: list[str] = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            text = "".join(
                span.get("text", "") for span in line.get("spans", [])
            ).strip()
            match = re.search(r"\.{3,}\s*(\d{1,5})$", text)
            if match:
                page_numbers.append(match.group(1))
    return len(page_numbers) >= 4 and len(set(page_numbers)) >= 3


def _has_multiple_text_columns(page: Any) -> bool:
    page_width = float(page.rect.width)
    if page_width <= 0:
        return False
    center = page_width / 2
    minimum_gap = page_width * 0.025
    left_blocks: list[tuple[float, float]] = []
    right_blocks: list[tuple[float, float]] = []
    for block in page.get_text("blocks"):
        if len(block) < 5 or not str(block[4]).strip():
            continue
        x0, y0, x1, y1 = (float(value) for value in block[:4])
        block_width = x1 - x0
        if block_width <= 0 or block_width > page_width * 0.65:
            continue
        if x1 <= center - minimum_gap:
            left_blocks.append((y0, y1))
        elif x0 >= center + minimum_gap:
            right_blocks.append((y0, y1))

    if len(left_blocks) < 3 or len(right_blocks) < 3:
        return False
    left_top = min(y0 for y0, _ in left_blocks)
    left_bottom = max(y1 for _, y1 in left_blocks)
    right_top = min(y0 for y0, _ in right_blocks)
    right_bottom = max(y1 for _, y1 in right_blocks)
    shared_height = max(
        0.0, min(left_bottom, right_bottom) - max(left_top, right_top)
    )
    smaller_column_height = min(
        left_bottom - left_top, right_bottom - right_top
    )
    return (
        smaller_column_height > 0
        and shared_height / smaller_column_height >= 0.5
    )


def page_needs_docling(page: Any) -> tuple[bool, str]:
    """Return whether layout-aware processing may improve this PDF page."""
    native_text_chars = len(page.get_text().strip())
    page_area = page.rect.width * page.rect.height
    image_area = 0.0
    for image in page.get_image_info():
        x0, y0, x1, y1 = image["bbox"]
        x0 = max(x0, page.rect.x0)
        y0 = max(y0, page.rect.y0)
        x1 = min(x1, page.rect.x1)
        y1 = min(y1, page.rect.y1)
        image_area += max(0, x1 - x0) * max(0, y1 - y0)
    image_coverage = min(1.0, image_area / page_area) if page_area else 0.0

    if (
        image_coverage >= FULL_PAGE_IMAGE_COVERAGE
        and FULL_PAGE_TEXT_THRESHOLD > native_text_chars > 10
    ):
        return (
            True,
            f"full-page image with sparse text "
            f"({image_coverage:.0%} coverage, {native_text_chars} chars)",
        )
    if (
        native_text_chars < SPARSE_TEXT_THRESHOLD
        and image_coverage >= IMAGE_COVERAGE_THRESHOLD
    ):
        return (
            True,
            f"sparse page with images "
            f"({native_text_chars} chars, {image_coverage:.0%} coverage)",
        )
    if _has_multiple_text_columns(page):
        return True, "multiple text columns (reading order)"
    if (
        native_text_chars > MIN_TEXT_FOR_TABLE_DETECTION
        and _has_aligned_numeric_column(page)
    ):
        return True, "aligned numeric column (potential table)"
    if _has_dot_leader_rows(page):
        return True, "dot-leader rows (table of contents)"
    return False, ""


def find_pages_needing_docling(document: Any) -> dict[int, str]:
    """Return flagged zero-based page indices unless fallback is disabled."""
    if not docling_configuration()["enabled"]:
        return {}
    pages: dict[int, str] = {}
    for page_index, page in enumerate(document):
        needs_docling, reason = page_needs_docling(page)
        if needs_docling:
            pages[page_index] = reason
    return pages


def _strip_docling_page_artifacts(markdown: str) -> str:
    """Remove Docling page markers; ingestion adds its own page boundaries."""
    return re.sub(
        r"<!--\s*page[\s_-]?(?:break|no|marker|number).*?-->",
        "",
        markdown,
        flags=re.IGNORECASE,
    ).strip()


def convert_pages_with_docling(
    source: Path,
    page_indices: set[int],
    warnings: list[str],
) -> dict[int, str]:
    """Convert flagged pages, retaining standard-pipeline fallback on errors."""
    if not page_indices:
        return {}
    if not _check_docling():
        warnings.append(
            f"Docling fallback requested for {len(page_indices)} pages but "
            "Docling is unavailable; using the standard pipeline"
        )
        return {}

    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
    except ImportError as error:
        warnings.append(
            f"Docling import failed: {error}; using the standard pipeline"
        )
        return {}

    try:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = docling_configuration()["ocr"]
        pipeline_options.do_table_structure = True
        pipeline_options.generate_page_images = False
        pipeline_options.generate_picture_images = False
        pipeline_options.images_scale = 1.0
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                )
            }
        )
    except Exception as error:
        warnings.append(
            f"Docling converter could not be initialized: {error}; "
            "using the standard pipeline"
        )
        return {}

    results: dict[int, str] = {}
    for page_index in sorted(page_indices):
        try:
            conversion = converter.convert(
                source,
                page_range=(page_index + 1, page_index + 1),
            )
            markdown = _strip_docling_page_artifacts(
                conversion.document.export_to_markdown()
            )
            if markdown:
                results[page_index] = markdown
            else:
                warnings.append(
                    f"Docling returned empty Markdown for PDF page "
                    f"{page_index + 1}; using the standard pipeline"
                )
        except Exception as error:
            warnings.append(
                f"Docling failed on PDF page {page_index + 1}: "
                f"{str(error)[:200]}; using the standard pipeline"
            )
    return results


def summarize_usage(pages_with_reasons: dict[int, str]) -> str:
    """Summarize why Docling fallback pages were selected."""
    if not pages_with_reasons:
        return "Docling fallback: not needed (0 pages flagged)"
    reasons_by_type: dict[str, int] = {}
    for reason in pages_with_reasons.values():
        key = reason.split(" (")[0]
        reasons_by_type[key] = reasons_by_type.get(key, 0) + 1
    parts = ", ".join(
        f"{count}x {reason}"
        for reason, count in sorted(reasons_by_type.items())
    )
    return (
        f"Docling fallback: {len(pages_with_reasons)} pages flagged ({parts})"
    )
