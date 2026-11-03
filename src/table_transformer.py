from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import html
import json
from pathlib import Path
from statistics import mean
from typing import Any


DETECTION_CHECKPOINT = "microsoft/table-transformer-detection"
STRUCTURE_CHECKPOINT = "microsoft/table-transformer-structure-recognition-v1.1-all"


def _normalize_processor_size(processor: Any) -> None:
    size = getattr(processor, "size", None)
    get_size = getattr(size, "get", None)
    if (
        callable(get_size)
        and get_size("longest_edge")
        and not get_size("shortest_edge")
    ):
        longest_edge = int(get_size("longest_edge"))
        processor.size = {
            "max_height": longest_edge,
            "max_width": longest_edge,
        }


@dataclass
class TablePrediction:
    bbox: tuple[float, float, float, float]
    rows: list[list[str]]
    html: str
    confidence: float
    detection_score: float
    structure_score: float
    word_coverage: float


def _load_model_bundle(model_name: str) -> tuple[Any, Any]:
    from huggingface_hub import hf_hub_download
    from transformers import (
        AutoConfig,
        AutoImageProcessor,
        TableTransformerConfig,
        TableTransformerForObjectDetection,
    )

    try:
        config = AutoConfig.from_pretrained(model_name)
    except Exception:
        config_path = hf_hub_download(model_name, "config.json")
        config_data = json.loads(Path(config_path).read_text(encoding="utf-8"))
        if config_data.get("dilation") is not None:
            raise
        config_data["dilation"] = False
        backbone_config = config_data.get("backbone_config")
        if isinstance(backbone_config, dict) and backbone_config.get("dilation") is None:
            backbone_config["dilation"] = False
        config = TableTransformerConfig.from_dict(config_data)

    processor = AutoImageProcessor.from_pretrained(model_name)
    _normalize_processor_size(processor)
    model = TableTransformerForObjectDetection.from_pretrained(
        model_name,
        config=config,
    )
    model.to("cpu")
    model.eval()
    return processor, model


@lru_cache(maxsize=2)
def _model_bundle(model_name: str) -> tuple[Any, Any]:
    import torch

    torch.set_num_threads(min(2, max(1, torch.get_num_threads())))
    return _load_model_bundle(model_name)


def _predict(
    model_name: str,
    image: Any,
    threshold: float,
) -> list[dict[str, Any]]:
    import torch

    processor, model = _model_bundle(model_name)
    inputs = processor(images=image, return_tensors="pt")
    with torch.inference_mode():
        outputs = model(**inputs)
    result = processor.post_process_object_detection(
        outputs,
        threshold=threshold,
        target_sizes=torch.tensor([[image.height, image.width]]),
    )[0]
    predictions: list[dict[str, Any]] = []
    for box, score, label in zip(
        result["boxes"], result["scores"], result["labels"]
    ):
        label_id = int(label.item())
        predictions.append(
            {
                "bbox": tuple(float(value) for value in box.tolist()),
                "score": float(score.item()),
                "label_id": label_id,
                "label": model.config.id2label.get(
                    label_id,
                    model.config.id2label.get(str(label_id), str(label_id)),
                ),
            }
        )
    return predictions


def _clip_box(
    box: tuple[float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box
    return (
        max(0.0, min(float(width), x0)),
        max(0.0, min(float(height), y0)),
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
    )


def _inside(box: tuple[float, float, float, float], x: float, y: float) -> bool:
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def _choose_interval(boxes: list[tuple[float, float, float, float]], x: float, y: float) -> int | None:
    matches = [
        (index, box)
        for index, box in enumerate(boxes)
        if _inside(box, x, y)
    ]
    if not matches:
        return None
    return min(
        matches,
        key=lambda item: (item[1][2] - item[1][0]) * (item[1][3] - item[1][1]),
    )[0]


def _text_for_words(words: list[tuple[Any, ...]]) -> str:
    by_line: dict[tuple[int, int], list[tuple[int, str]]] = {}
    for word in words:
        block_number = int(word[5]) if len(word) > 5 else 0
        line_number = int(word[6]) if len(word) > 6 else 0
        word_number = int(word[7]) if len(word) > 7 else 0
        by_line.setdefault((block_number, line_number), []).append(
            (word_number, str(word[4]))
        )
    return "\n".join(
        " ".join(text for _, text in sorted(line_words))
        for _, line_words in sorted(by_line.items())
    )


def _span_indices(
    boxes: list[tuple[float, float, float, float]],
    start: float,
    end: float,
    axis: int,
) -> list[int]:
    indices = []
    for index, box in enumerate(boxes):
        low = box[axis]
        high = box[axis + 2]
        overlap = max(0.0, min(high, end) - max(low, start))
        if overlap >= 0.5 * (high - low):
            indices.append(index)
    return indices


def _table_html(
    cells: list[list[str]],
    row_spans: dict[tuple[int, int], tuple[int, int]],
    header_rows: set[int],
) -> str:
    covered: set[tuple[int, int]] = set()
    parts = ["<table>"]
    for row_index, row in enumerate(cells):
        parts.append("<tr>")
        for column_index, value in enumerate(row):
            if (row_index, column_index) in covered:
                continue
            rowspan, colspan = row_spans.get((row_index, column_index), (1, 1))
            for covered_row in range(row_index, min(len(cells), row_index + rowspan)):
                for covered_column in range(
                    column_index, min(len(row), column_index + colspan)
                ):
                    if (covered_row, covered_column) != (row_index, column_index):
                        covered.add((covered_row, covered_column))
            tag = "th" if row_index in header_rows else "td"
            attributes = "".join(
                [
                    f' rowspan="{rowspan}"' if rowspan > 1 else "",
                    f' colspan="{colspan}"' if colspan > 1 else "",
                ]
            )
            rendered = "<br>".join(html.escape(line) for line in value.splitlines())
            parts.append(f"<{tag}{attributes}>{rendered}</{tag}>")
        parts.append("</tr>")
    parts.append("</table>")
    return "\n".join(parts)


class TableTransformerExtractor:
    def __init__(
        self,
        dpi: int = 150,
        detection_threshold: float = 0.5,
        structure_threshold: float = 0.3,
        minimum_confidence: float = 0.45,
    ) -> None:
        self.dpi = dpi
        self.detection_threshold = detection_threshold
        self.structure_threshold = structure_threshold
        self.minimum_confidence = minimum_confidence

    def extract_page(self, page: Any) -> list[TablePrediction]:
        import pymupdf
        from PIL import Image

        pixmap = page.get_pixmap(dpi=self.dpi, alpha=False)
        page_image = Image.frombytes(
            "RGB", (pixmap.width, pixmap.height), pixmap.samples
        )
        detected_tables = [
            prediction
            for prediction in _predict(
                DETECTION_CHECKPOINT,
                page_image,
                self.detection_threshold,
            )
            if prediction["label"].casefold() in {"table", "table rotated"}
        ]
        words = self._page_words(page)
        scale_x = pixmap.width / page.rect.width
        scale_y = pixmap.height / page.rect.height
        output = []

        for detection in detected_tables:
            table_box = _clip_box(detection["bbox"], pixmap.width, pixmap.height)
            x0, y0, x1, y1 = table_box
            if x1 - x0 < 16 or y1 - y0 < 16:
                continue
            crop_box = (int(x0), int(y0), int(x1 + 0.5), int(y1 + 0.5))
            crop = page_image.crop(crop_box)
            structure = _predict(
                STRUCTURE_CHECKPOINT,
                crop,
                self.structure_threshold,
            )
            table = self._reconstruct(
                page,
                words,
                structure,
                table_box,
                crop.size,
                (scale_x, scale_y),
                detection["score"],
            )
            if table is not None:
                output.append(table)
        return output

    @staticmethod
    def _page_words(page: Any) -> list[tuple[Any, ...]]:
        words = page.get_text("words")
        if words:
            return words
        try:
            text_page = page.get_textpage_ocr(language="eng", dpi=300, full=True)
            return page.get_text("words", textpage=text_page)
        except Exception:
            return []

    def _reconstruct(
        self,
        page: Any,
        words: list[tuple[Any, ...]],
        structure: list[dict[str, Any]],
        table_box: tuple[float, float, float, float],
        crop_size: tuple[int, int],
        scale: tuple[float, float],
        detection_score: float,
    ) -> TablePrediction | None:
        row_predictions = [
            item for item in structure if item["label"].casefold() == "table row"
        ]
        column_predictions = [
            item for item in structure if item["label"].casefold() == "table column"
        ]
        if not row_predictions or not column_predictions:
            return None

        rows = sorted(
            (_clip_box(item["bbox"], *crop_size) for item in row_predictions),
            key=lambda box: box[1],
        )
        columns = sorted(
            (_clip_box(item["bbox"], *crop_size) for item in column_predictions),
            key=lambda box: box[0],
        )
        crop_x, crop_y, _, _ = table_box
        rows = [(x0 + crop_x, y0 + crop_y, x1 + crop_x, y1 + crop_y) for x0, y0, x1, y1 in rows]
        columns = [(x0 + crop_x, y0 + crop_y, x1 + crop_x, y1 + crop_y) for x0, y0, x1, y1 in columns]

        cell_words: list[list[list[tuple[Any, ...]]]] = [
            [[] for _ in columns] for _ in rows
        ]
        table_words = 0
        assigned_words = 0
        scale_x, scale_y = scale
        page_x0, page_y0 = page.rect.x0, page.rect.y0
        for word in words:
            word_box = (
                (float(word[0]) - page_x0) * scale_x,
                (float(word[1]) - page_y0) * scale_y,
                (float(word[2]) - page_x0) * scale_x,
                (float(word[3]) - page_y0) * scale_y,
            )
            center_x = (word_box[0] + word_box[2]) / 2
            center_y = (word_box[1] + word_box[3]) / 2
            if not _inside(table_box, center_x, center_y):
                continue
            table_words += 1
            row_index = _choose_interval(rows, center_x, center_y)
            column_index = _choose_interval(columns, center_x, center_y)
            if row_index is None or column_index is None:
                continue
            cell_words[row_index][column_index].append(word)
            assigned_words += 1

        row_spans: dict[tuple[int, int], tuple[int, int]] = {}
        for span in (
            item for item in structure
            if item["label"].casefold() == "table spanning cell"
        ):
            sx0, sy0, sx1, sy1 = span["bbox"]
            global_span = (sx0 + crop_x, sy0 + crop_y, sx1 + crop_x, sy1 + crop_y)
            covered_rows = _span_indices(rows, global_span[1], global_span[3], 1)
            covered_columns = _span_indices(columns, global_span[0], global_span[2], 0)
            if not covered_rows or not covered_columns:
                continue
            first_row, first_column = covered_rows[0], covered_columns[0]
            span_words = []
            for row_index in covered_rows:
                for column_index in covered_columns:
                    kept = []
                    for word in cell_words[row_index][column_index]:
                        cx = (
                            (float(word[0]) - page_x0)
                            + (float(word[2]) - page_x0)
                        ) * scale_x / 2
                        cy = (
                            (float(word[1]) - page_y0)
                            + (float(word[3]) - page_y0)
                        ) * scale_y / 2
                        if _inside(global_span, cx, cy):
                            span_words.append(word)
                        else:
                            kept.append(word)
                    cell_words[row_index][column_index] = kept
            cell_words[first_row][first_column].extend(span_words)
            row_spans[(first_row, first_column)] = (
                len(covered_rows), len(covered_columns)
            )

        cell_text = [
            [_text_for_words(words_in_cell) for words_in_cell in row]
            for row in cell_words
        ]
        if not any(value for row in cell_text for value in row):
            return None
        header_rows: set[int] = set()
        header_predictions = [
            item
            for item in structure
            if item["label"].casefold() == "table column header"
        ]
        for header in header_predictions:
            _, header_y0, _, header_y1 = header["bbox"]
            for row_index, row_box in enumerate(rows):
                row_y0, row_y1 = row_box[1] - crop_y, row_box[3] - crop_y
                overlap = max(0.0, min(row_y1, header_y1) - max(row_y0, header_y0))
                if overlap >= 0.5 * (row_y1 - row_y0):
                    header_rows.add(row_index)

        structure_scores = [
            item["score"] for item in structure
            if item["label"].casefold() in {"table row", "table column"}
        ]
        structure_score = mean(structure_scores) if structure_scores else 0.0
        word_coverage = assigned_words / table_words if table_words else 0.0
        confidence = (
            0.35 * detection_score
            + 0.35 * structure_score
            + 0.30 * word_coverage
        )
        if confidence < self.minimum_confidence:
            return None

        html_table = _table_html(cell_text, row_spans, header_rows)
        pdf_bbox = (
            page.rect.x0 + table_box[0] / scale_x,
            page.rect.y0 + table_box[1] / scale_y,
            page.rect.x0 + table_box[2] / scale_x,
            page.rect.y0 + table_box[3] / scale_y,
        )
        return TablePrediction(
            bbox=pdf_bbox,
            rows=cell_text,
            html=html_table,
            confidence=confidence,
            detection_score=detection_score,
            structure_score=structure_score,
            word_coverage=word_coverage,
        )