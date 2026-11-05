"""Reliability helpers for the document ingestion pipeline.

Standalone module — no dependency on the main script. Imported by ingest.py.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PAGE_BREAK_PLACEHOLDER = "\n\n<!-- DOC_PAGE_BREAK -->\n\n"
PAGE_MARKER_RE = re.compile(
    r">\s*\*\*PDF page (\d+)(?:\s*\(printed page ([^)]*)\))?\*\*"
)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
DATE_ONLY_RE = re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}$")
DATE_FOOTER_START_RE = re.compile(
    r"^\d{1,2}/\d{1,2}/\d{2,4}\s*,?\s+"
    r"\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm)\b",
    re.IGNORECASE,
)
INGEST_RESULT_PREFIX = "__INGEST_RESULT__"


# ---------------------------------------------------------------- page markers
def format_markdown(
    markdown: str,
    first_page: int = 1,
    page_label_prefix: str = "Page",
    printed_page_labels: list[str | None] | None = None,
) -> str:
    """Rewrite PAGE_BREAK_PLACEHOLDER so the marker *precedes* its page content."""
    parts = markdown.split(PAGE_BREAK_PLACEHOLDER)
    if parts and not parts[0].strip():
        parts = parts[1:]
    if parts and not parts[-1].strip():
        parts = parts[:-1]
    if not parts:
        return markdown

    out: list[str] = []
    for offset, content in enumerate(parts):
        page_number = first_page + offset
        printed = None
        if printed_page_labels:
            idx = page_number - first_page
            if 0 <= idx < len(printed_page_labels):
                printed = printed_page_labels[idx]
        marker = f"\n\n> **{page_label_prefix} {page_number}"
        if printed:
            marker += f" (printed page {printed})"
        marker += "**\n\n"
        out.append(marker + content.lstrip("\n"))
    return "".join(out)


def build_page_index(markdown: str) -> list[dict[str, Any]]:
    """Map every PDF page marker to its character range in the combined markdown."""
    matches = list(PAGE_MARKER_RE.finditer(markdown))
    index: list[dict[str, Any]] = []
    for i, match in enumerate(matches):
        start = match.end()
        while start < len(markdown) and markdown[start].isspace():
            start += 1
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        while end > start and markdown[end - 1].isspace():
            end -= 1
        index.append(
            {
                "pdf_page": int(match.group(1)),
                "printed_page": match.group(2) or None,
                "char_start": start,
                "char_end": end,
            }
        )
    return index


# ------------------------------------------------------- printed page numbers
def printed_pdf_page_number(page: Any) -> str | None:
    """Best-effort printed page number from a PyMuPDF page (footer/header digits)."""
    candidates: list[tuple[float, str]] = []
    rect = page.rect
    width = rect.width
    height = rect.height
    if width <= 0 or height <= 0:
        return None
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            x0, y0, x1, y1 = line["bbox"]
            top = y0 - rect.y0
            bottom = y1 - rect.y0
            if top > height * 0.08 and bottom < height * 0.92:
                continue
            text = "".join(
                span.get("text", "") for span in line.get("spans", [])
            ).strip()
            if not re.fullmatch(r"\d{1,4}", text):
                continue
            if 1900 <= int(text) <= 2100:
                continue
            left = x0 - rect.x0
            right = x1 - rect.x0
            if left > width * 0.25 and right < width * 0.75:
                continue
            candidates.append((top, text))
    return min(candidates)[1] if candidates else None


def consistent_printed_labels(
    labels: list[str | None],
    max_back_jump: int = 2,
    max_forward_step: int = 10,
) -> list[str | None]:
    """Blank out labels that break page-number progression."""
    if not labels:
        return labels
    result = list(labels)
    running_max = -1
    for i, value in enumerate(labels):
        if not value or not value.isdigit():
            continue
        numeric = int(value)
        if numeric < running_max - max_back_jump:
            result[i] = None
        elif running_max >= 0 and numeric > running_max + max_forward_step:
            result[i] = None
        else:
            running_max = max(running_max, numeric)
    return result


# ----------------------------------------------- running header/footer strip
def strip_repeated_lines(
    markdown_pages: list[str],
    threshold: float = 0.6,
    edge_fraction: float = 0.15,
    min_pages: int = 3,
    max_line_length: int = 200,
) -> tuple[list[str], list[str]]:
    """Remove lines that recur near page edges on most pages (headers/footers)."""
    if len(markdown_pages) < min_pages:
        return markdown_pages, []

    counts: Counter[str] = Counter()
    for page in markdown_pages:
        nonempty = [line.strip() for line in page.splitlines() if line.strip()]
        if not nonempty:
            continue
        edge = max(1, int(len(nonempty) * edge_fraction))
        edge_lines = set(nonempty[:edge] + nonempty[-edge:])
        for line in edge_lines:
            if not (2 < len(line) < max_line_length):
                continue
            if any(tok in line for tok in ("![", "](", "|", "```", "~~~")):
                continue
            counts[line] += 1

    required = max(3, int(len(markdown_pages) * threshold))
    repeated = {line for line, n in counts.items() if n >= required}
    if not repeated:
        return markdown_pages, []

    cleaned: list[str] = []
    for page in markdown_pages:
        lines = page.splitlines()
        nonempty_idx = [i for i, line in enumerate(lines) if line.strip()]
        edge = max(1, int(len(nonempty_idx) * edge_fraction))
        edge_idx = set(nonempty_idx[:edge] + nonempty_idx[-edge:])
        kept = [
            line
            for i, line in enumerate(lines)
            if i not in edge_idx or line.strip() not in repeated
        ]
        cleaned.append("\n".join(kept))
    return cleaned, sorted(repeated)


def clean_page_markdown(page_md: str) -> str:
    """Drop date footers, clean presentation wrappers, and normalize spacing."""
    lines = [line.rstrip() for line in page_md.splitlines()]
    filtered: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and (
            DATE_ONLY_RE.match(stripped)
            or DATE_FOOTER_START_RE.match(stripped)
        ):
            continue
        line = re.sub(
            r"~~\s*(<u\b[^>]*>.*?</u>)\s*~~",
            r"\1",
            line,
            flags=re.IGNORECASE,
        )
        line = re.sub(
            r"</?(?:mark|u|em|b|strong)\b[^>]*>",
            "",
            line,
            flags=re.IGNORECASE,
        )
        filtered.append(line)

    result: list[str] = []
    blanks = 0
    for line in filtered:
        if not line.strip():
            blanks += 1
            if blanks <= 2:
                result.append("")
        else:
            blanks = 0
            result.append(line)
    return ("\n".join(result).strip() + "\n") if result else ""


# ------------------------------------------------------------- value tables
@dataclass
class ValueTable:
    markdown: str
    records: list[dict[str, Any]]
    warnings: list[str]


def _parse_number(cell: str | None) -> float | None:
    if cell is None:
        return None
    text = str(cell).strip()
    if not text:
        return None
    text = re.sub(r"[, $€£¥₹₩₽¢]", "", text)
    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1]
    if text.endswith("%"):
        return None
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", text):
        return None
    value = float(text)
    return -value if negative else value


def _format_number(value: float | None) -> str:
    if value is None:
        return ""
    if value == int(value):
        return f"{int(value):,}"
    return f"{value:,.2f}"


def _headers_from_context(context: str, count: int) -> list[str]:
    years = [(m.start(), int(m.group(0))) for m in YEAR_RE.finditer(context or "")]
    if len(years) < count:
        return []
    for end in range(len(years), count - 1, -1):
        window = years[end - count : end]
        values = [year for _, year in window]
        if all(0 <= values[i + 1] - values[i] <= 3 for i in range(len(values) - 1)):
            return [str(year) for year in values]
    return []


def value_table_from_rows(
    rows: list[list[str | None]],
    context: str = "",
) -> ValueTable | None:
    """Recognise a plain numeric grid (income statement, roll-forward, etc.).

    Returns None when the table is not confidently a value grid so the caller
    can fall back to the complexity pipeline.
    """
    if not rows or len(rows) < 2:
        return None
    width = max((len(row) for row in rows), default=0)
    if width < 2:
        return None

    header_years: list[str] = []
    header_rows: set[int] = set()
    for row_index, row in enumerate(rows):
        row_years = [
            match.group(0)
            for cell in row
            if cell
            for match in [YEAR_RE.search(str(cell))]
            if match
        ]
        if len(row_years) >= 2:
            header_years = row_years
            header_rows.add(row_index)

    numeric_columns = [
        column
        for column in range(1, width)
        if any(
            _parse_number(row[column] if column < len(row) else None)
            is not None
            for row_index, row in enumerate(rows)
            if row_index not in header_rows
        )
    ]
    if not numeric_columns:
        return None

    currency_only = re.compile(r"[$€£¥₹₩₽¢]")
    currency_columns = {
        column
        for column in range(1, width)
        if any(
            row[column]
            and currency_only.fullmatch(row[column].strip())
            for row_index, row in enumerate(rows)
            if row_index not in header_rows and column < len(row)
        )
    }
    paired_currency_columns: dict[int, int] = {}
    for column in sorted(currency_columns):
        following_column = column + 1
        if following_column not in numeric_columns:
            continue
        has_split_amount = any(
            column < len(row)
            and following_column < len(row)
            and row[column]
            and currency_only.fullmatch(row[column].strip())
            and _parse_number(row[following_column]) is not None
            for row_index, row in enumerate(rows)
            if row_index not in header_rows
        )
        has_two_values = any(
            column < len(row)
            and following_column < len(row)
            and _parse_number(row[column]) is not None
            and _parse_number(row[following_column]) is not None
            for row_index, row in enumerate(rows)
            if row_index not in header_rows
        )
        if has_split_amount and not has_two_values:
            paired_currency_columns[column] = following_column

    paired_value_columns = set(paired_currency_columns.values())
    data_columns: list[tuple[int, ...]] = []
    for column in range(1, width):
        if column in paired_value_columns:
            continue
        if column in paired_currency_columns:
            data_columns.append(
                (column, paired_currency_columns[column])
            )
        elif column in numeric_columns:
            data_columns.append((column,))

    contextual_headers: list[str] = []
    for row in rows:
        labels = [
            " ".join(
                row[column].strip()
                for column in columns
                if column < len(row)
                and row[column]
                and not currency_only.fullmatch(row[column].strip())
                and _parse_number(row[column]) is None
            )
            for columns in data_columns
        ]
        if sum(bool(label) for label in labels) >= 2:
            contextual_headers = labels

    data_rows: list[tuple[str, list[float | None]]] = []
    for row_index, row in enumerate(rows):
        if not row:
            continue
        if row_index in header_rows:
            continue
        label = (row[0] or "").strip()
        if not label:
            continue
        values = [
            next(
                (
                    value
                    for column in reversed(columns)
                    if column < len(row)
                    and (value := _parse_number(row[column])) is not None
                ),
                None,
            )
            for columns in data_columns
        ]
        numeric_count = sum(1 for value in values if value is not None)
        if numeric_count == 0:
            continue
        if numeric_count < max(1, len(values) // 2):
            continue
        data_rows.append((label, values))

    if len(data_rows) < 3:
        return None

    widths = Counter(len(values) for _, values in data_rows)
    common_width, common_count = widths.most_common(1)[0]
    if common_count < len(data_rows) * 0.7:
        return None
    data_rows = [(label, values[:common_width]) for label, values in data_rows]

    warnings: list[str] = []
    if header_years and len(header_years) >= common_width:
        headers = ["Item"] + header_years[:common_width]
    elif (
        contextual_headers
        and len(contextual_headers) == common_width
        and all(contextual_headers)
    ):
        headers = ["Item"] + contextual_headers
    else:
        context_years = _headers_from_context(context, common_width)
        if context_years:
            headers = ["Item"] + context_years
        else:
            warnings.append(
                "numeric table detected but no year headers found in "
                "context or first rows"
            )
            headers = ["Item"] + [f"Column {i + 1}" for i in range(common_width)]

    md_lines = ["| " + " | ".join(headers) + " |"]
    md_lines.append("|" + "|".join([" --- " for _ in headers]) + "|")
    records: list[dict[str, Any]] = []
    for label, values in data_rows:
        formatted = [_format_number(value) for value in values]
        md_lines.append("| " + " | ".join([label] + formatted) + " |")
        records.append({"label": label, "values": values})
    return ValueTable("\n".join(md_lines), records, warnings)


# --------------------------------------------------------- source planning
def plan_source(
    previous_record: dict[str, Any] | None,
    current_hash: str,
    output_ok: bool,
    force: bool = False,
) -> str:
    """Decide what to do with an input file: reuse / convert / reconvert_*."""
    if force:
        return "reconvert_forced"
    if not previous_record:
        return "convert"
    if output_ok:
        return "reuse"
    if previous_record.get("source_sha256") != current_hash:
        return "reconvert_changed"
    return "reconvert_stale"


def remove_stale_outputs(
    previous_record: dict[str, Any], workspace: Path
) -> list[str]:
    """Delete md + sidecars + images for a stale record. Returns removed paths."""
    removed: list[str] = []
    markdown_info = previous_record.get("markdown") or {}
    output = markdown_info.get("output")
    if not isinstance(output, str):
        return removed
    md_path = workspace / output
    for candidate in (
        md_path,
        md_path.with_suffix(".tables.json"),
        md_path.with_suffix(".pages.json"),
    ):
        if candidate.is_file():
            candidate.unlink()
            removed.append(str(candidate))
    images_dir = md_path.parent / "images" / md_path.stem
    if images_dir.is_dir():
        shutil.rmtree(images_dir)
        removed.append(str(images_dir))
    return removed


# ----------------------------------------------------------- subprocess guard
def worker_timeout_seconds() -> float:
    try:
        return float(os.environ.get("INGEST_WORKER_TIMEOUT_SECONDS", "3600"))
    except ValueError:
        return 3600.0


def run_command_with_timeout(
    command: list[str],
    cwd: Path,
    timeout_secs: float,
    label: str = "worker",
) -> tuple[dict[str, Any] | None, int, bool]:
    """Run a subprocess with streaming + hard timeout. Returns (result, code, timed_out)."""
    result_box: list[dict[str, Any] | None] = [None]
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    def reader() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            line = line.rstrip()
            if line.startswith(INGEST_RESULT_PREFIX):
                try:
                    result_box[0] = json.loads(
                        line.removeprefix(INGEST_RESULT_PREFIX)
                    )
                except json.JSONDecodeError:
                    pass
            elif line:
                print(f"[{label}] {line}", flush=True)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()

    timed_out = False
    try:
        return_code = process.wait(timeout=timeout_secs)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        return_code = process.returncode if process.returncode is not None else -1

    thread.join(timeout=15)
    if process.stdout is not None:
        process.stdout.close()
    return result_box[0], return_code, timed_out


# ------------------------------------------------------ environment checks
def validate_environment() -> str | None:
    """Return a human-readable error string when configuration is invalid."""
    errors: list[str] = []
    backend = os.environ.get("PDF_OCR_BACKEND", "tesseract").strip().lower()
    if backend not in {"tesseract", "paddle", "unstructured", "auto", "none"}:
        errors.append(f"Invalid PDF_OCR_BACKEND: {backend}")
    table = os.environ.get("PDF_TABLE_BACKEND", "pdfplumber").strip().lower()
    if table not in {"pdfplumber", "hybrid", "table-transformer"}:
        errors.append(f"Invalid PDF_TABLE_BACKEND: {table}")
    fallback = os.environ.get("PDF_TABLE_FALLBACK", "none").strip().lower()
    if fallback not in {"none", "unstructured"}:
        errors.append(f"Invalid PDF_TABLE_FALLBACK: {fallback}")
    try:
        float(os.environ.get("INGEST_WORKER_TIMEOUT_SECONDS", "3600"))
    except ValueError:
        errors.append("INGEST_WORKER_TIMEOUT_SECONDS is not numeric")

    for module in ("pymupdf", "pymupdf4llm", "pdfplumber"):
        if importlib.util.find_spec(module) is None:
            errors.append(f"Missing required module: {module}")

    if errors:
        return "Environment configuration errors:\n" + "\n".join(
            f"  - {error}" for error in errors
        )
    return None