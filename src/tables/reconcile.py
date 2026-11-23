from __future__ import annotations

import re
from typing import Any

try:
    from ..contracts import PageArtifact, TableArtifact
except ImportError:
    from contracts import PageArtifact, TableArtifact


def _replace(table: TableArtifact, *, extractor: str, rejection_reason: str | None = None) -> TableArtifact:
    return TableArtifact(
        table_id=table.table_id,
        source_page=table.source_page,
        bbox=table.bbox,
        extractor=extractor,  # type: ignore[arg-type]
        headers=list(table.headers),
        rows=[list(row) for row in table.rows],
        raw_html=table.raw_html,
        quality_score=table.quality_score,
        repair_notes=list(table.repair_notes),
        rejection_reason=rejection_reason,
    )


def _with_warning(table: TableArtifact, warning: str) -> TableArtifact:
    return TableArtifact(
        table_id=table.table_id,
        source_page=table.source_page,
        bbox=table.bbox,
        extractor=table.extractor,
        headers=list(table.headers),
        rows=[list(row) for row in table.rows],
        raw_html=table.raw_html,
        quality_score=table.quality_score,
        repair_notes=list(table.repair_notes) + [warning],
        rejection_reason=table.rejection_reason,
    )


def _populated_count(rows: list[list[str | None]]) -> int:
    return sum(1 for row in rows for cell in row if cell and cell.strip())


def _table_rows_are_plausible(rows: list[list[str | None]]) -> bool:
    if not rows:
        return False
    widths = [len(row) for row in rows]
    return len(set(widths)) == 1 and _populated_count(rows) >= 2


def _table_complexity(rows: list[list[str | None]]) -> dict[str, Any]:
    widths = [len(row) for row in rows]
    blank_ratio = 1.0 - _populated_count(rows) / max(1, sum(widths))
    is_complex = len(set(widths)) > 1 or blank_ratio > 0.25
    return {"is_complex": is_complex}


def reconcile_table(
    native: TableArtifact | None,
    docling: TableArtifact | None,
    page_artifact: PageArtifact,
) -> TableArtifact | None:
    if native is None and docling is None:
        return None
    if native is None:
        if _docling_table_acceptable(docling, page_artifact):
            return _replace(docling, extractor="docling")
        return None
    if docling is None:
        return native

    docling_has_spans = bool(re.search(r"rowspan|colspan", docling.raw_html, re.I))
    docling_populated = _populated_count(docling.rows)
    native_populated = _populated_count(native.rows)
    native_implausible = not _table_rows_are_plausible(native.rows)
    native_complex = _table_complexity(native.rows)["is_complex"]

    if native_complex and docling_has_spans:
        return _replace(native, extractor="native+docling_hint")
    if not native_implausible and not docling_has_spans:
        return native
    if native_implausible and docling_has_spans and docling_populated > native_populated:
        return _replace(docling, extractor="docling+repair")
    chosen = max([native, docling], key=lambda table: _populated_count(table.rows))
    return _with_warning(chosen, "TABLE_RECONCILIATION_AMBIGUOUS")


def _docling_table_acceptable(docling: TableArtifact, page_artifact: PageArtifact) -> bool:
    if "<table" not in docling.raw_html.lower():
        return False
    if re.search(r"<table[^>]*>.*<table", docling.raw_html, re.I | re.S):
        return False
    populated_rows = sum(
        1 for row in docling.rows
        if any(cell and cell.strip() for cell in row)
    )
    if populated_rows < 2:
        return False
    return bool(re.search(r"<(?:th)\b", docling.raw_html, re.I))
