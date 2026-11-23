from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None


DEFAULT_ROUTING = {
    "docling_confidence_threshold": 0.55,
    "force_native_page_types": ["digital_text", "clean_table"],
    "max_docling_pages_per_document": 20,
    "hyperlink_guard": True,
    "coalesce_page_ranges": True,
}


def load_routing_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path or "config/routing.yaml")
    if not config_path.exists():
        return dict(DEFAULT_ROUTING)
    if yaml is None:
        raise RuntimeError("PyYAML is required to load config/routing.yaml")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("routing"), dict):
        raise ValueError("routing.yaml must contain a top-level 'routing:' mapping")
    routing = data["routing"]
    unknown = set(routing) - set(DEFAULT_ROUTING)
    if unknown:
        raise ValueError(f"Unknown routing config keys: {', '.join(sorted(unknown))}")
    threshold = routing.get("docling_confidence_threshold", DEFAULT_ROUTING["docling_confidence_threshold"])
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("docling_confidence_threshold must be a number")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("docling_confidence_threshold must be between 0.0 and 1.0")
    routing["docling_confidence_threshold"] = float(threshold)
    for key in ("force_native_page_types",):
        if key in routing and not isinstance(routing[key], list):
            raise ValueError(f"{key} must be a list")
    for key in ("max_docling_pages_per_document",):
        if key in routing and (isinstance(routing[key], bool) or not isinstance(routing[key], int)):
            raise ValueError(f"{key} must be an integer")
    for key in ("hyperlink_guard", "coalesce_page_ranges"):
        if key in routing and not isinstance(routing[key], bool):
            raise ValueError(f"{key} must be a boolean")
    result = dict(DEFAULT_ROUTING)
    result.update(routing)
    return result


def _image_coverage(page: Any) -> float:
    rect = getattr(page, "rect", None)
    page_area = float(getattr(rect, "width", 0) * getattr(rect, "height", 0))
    if page_area <= 0:
        return 0.0
    image_area = 0.0
    for image in page.get_image_info():
        bbox = image.get("bbox")
        if not bbox:
            continue
        x0, y0, x1, y1 = (float(v) for v in bbox)
        x0 = max(x0, getattr(rect, "x0", 0))
        y0 = max(y0, getattr(rect, "y0", 0))
        x1 = min(x1, getattr(rect, "x1", x1))
        y1 = min(y1, getattr(rect, "y1", y1))
        image_area += max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return min(1.0, image_area / page_area)


def _rect_area(rect: tuple[float, float, float, float]) -> float:
    return max(0.0, rect[2] - rect[0]) * max(0.0, rect[3] - rect[1])


def _intersects(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x_overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    y_overlap = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return x_overlap * y_overlap


def _layout_regularity(page: Any) -> float:
    blocks = page.get_text("blocks")
    rects = [tuple(float(v) for v in block[:4]) for block in blocks if len(block) >= 5]
    if len(rects) < 2:
        return 1.0
    overlap_count = 0
    for index, rect in enumerate(rects):
        area = _rect_area(rect)
        for other in rects[index + 1:]:
            overlap = _intersects(rect, other)
            if area > 0 and overlap / area > 0.05:
                overlap_count += 1
                break
    return max(0.0, 1.0 - overlap_count / len(rects))


def _table_rows_are_plausible(rows: list[list[Any]]) -> bool:
    if not rows:
        return True
    columns = max((len(row) for row in rows), default=0)
    if columns == 0:
        return False
    return len(set(len(row) for row in rows)) == 1


def score_page_confidence(page: Any, native_artifacts: Any) -> tuple[float, list[str]]:
    reasons: list[str] = []
    text_score = min(1.0, len(page.get_text().strip()) / 1200)
    if text_score < 0.25:
        reasons.append("SPARSE_TEXT")
    image_coverage = _image_coverage(page)
    image_score = 1.0 - min(1.0, image_coverage)
    if image_coverage > 0.30:
        reasons.append("IMAGE_HEAVY")
    tables = getattr(native_artifacts, "tables", [])
    if tables:
        table_score = sum(
            1.0 if _table_rows_are_plausible(table.rows) else 0.0
            for table in tables
        ) / len(tables)
        if table_score < 1.0:
            reasons.append("TABLE_IMPLAUSIBLE")
    else:
        table_score = 1.0
    layout_score = _layout_regularity(page)
    if layout_score < 0.6:
        reasons.append("IRREGULAR_LAYOUT")
    confidence = 0.30 * text_score + 0.20 * image_score + 0.30 * table_score + 0.20 * layout_score
    return confidence, reasons


def _pdf_page_needs_ocr(native_text_chars: int, image_coverage: float) -> bool:
    sparse = native_text_chars < 300 and image_coverage >= 0.05
    mixed = image_coverage >= 0.25 and native_text_chars < 1200
    return sparse or mixed
