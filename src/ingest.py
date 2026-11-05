import argparse
from collections import Counter
import hashlib
from html import escape, unescape
from html.parser import HTMLParser
import importlib.util
import json
import os
import re
import shutil
import sys
import subprocess
import tempfile
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote

try:
    from .ingest_fixes import (
        build_page_index,
        clean_page_markdown,
        consistent_printed_labels,
        format_markdown,
        plan_source,
        printed_pdf_page_number,
        remove_stale_outputs,
        run_command_with_timeout,
        strip_repeated_lines,
        validate_environment,
        value_table_from_rows,
        worker_timeout_seconds,
    )
except ImportError:
    from ingest_fixes import (
        build_page_index,
        clean_page_markdown,
        consistent_printed_labels,
        format_markdown,
        plan_source,
        printed_pdf_page_number,
        remove_stale_outputs,
        run_command_with_timeout,
        strip_repeated_lines,
        validate_environment,
        value_table_from_rows,
        worker_timeout_seconds,
    )

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx"}
CONVERSION_FORMAT_VERSION = 34
PAGE_BREAK_PLACEHOLDER = "\n\n<!-- DOC_PAGE_BREAK -->\n\n"
GROQ_VISION_MODEL = "qwen/qwen3.8-27b"
GROQ_IMAGE_BATCH_SIZE = 3
DIAGRAM_CONFIDENCE_THRESHOLD = 0.55
MAX_VISION_IMAGE_BYTES = 20 * 1024 * 1024
MAX_VISION_BATCH_BYTES = 20 * 1024 * 1024
MIN_VISION_IMAGE_DIMENSION = 32
SLIDE_RENDER_DPI = 144
IMAGE_LINK_PATTERN = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
PAGE_FURNITURE_EDGE_FRACTION = 0.1
SPARSE_PAGE_TEXT_CHARS = 300
MIN_IMAGE_COVERAGE_FOR_OCR_WARNING = 0.05
PDF_OCR_BACKEND = "tesseract"
PDF_OCR_LANGUAGES = "eng"
TABLE_PLACEHOLDER_PREFIX = "ZZTABLE"
FIGURE_CAPTION_PATTERN = re.compile(
    r"^(?:figure|fig\.?|chart|exhibit|diagram|graph|illustration|flowchart)"
    r"\s*(?:no\.?\s*)?\d+[a-z]?(?:\s*[:.)\-–—]\s*|\s+).+",
    re.IGNORECASE,
)


# -------------------------------------------------- heading/fence normalization
def _normalized_heading_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[#*_`]", "", text)
    return " ".join(unescape(text).split()).casefold()


def _numbered_section_heading(text: str) -> str | None:
    match = re.fullmatch(r"(\d+(?:\.\d+)*\.?)\s+(.+)", text.strip())
    if not match:
        return None
    title = match.group(2).strip()
    if (
        len(title.split()) > 12
        or re.search(r"[.!?;:]$", title)
        or re.search(r"[{}=<>]", title)
    ):
        return None
    level = min(len(match.group(1).rstrip(".").split(".")), 6)
    return f"{'#' * level} {text.strip()}"


def _has_aligned_columns(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    column_starts = Counter(
        match.start()
        for line in lines
        for match in re.finditer(r" {5,}(?=\S)", line)
    )
    required_lines = max(2, (len(lines) + 3) // 4)
    return max(column_starts.values(), default=0) >= required_lines


def _is_prose_block(text: str) -> bool:
    normalized = textwrap.dedent(text).strip()
    words = re.findall(r"\b[^\W\d_]+(?:['’][^\W\d_]+)?\b", normalized)
    if (
        len(words) < 6
        or re.search(r"[{};=]", normalized)
        or _has_aligned_columns(normalized)
    ):
        return False
    nonspace = [character for character in normalized if not character.isspace()]
    if (
        not nonspace
        or sum(character.isalpha() for character in nonspace) / len(nonspace) < 0.55
    ):
        return False
    lines = normalized.splitlines()
    indents = [
        len(line) - len(line.lstrip()) for line in lines if line.strip()
    ]
    if indents and max(indents) - min(indents) > 2:
        return False
    return len(words) >= 18 or bool(re.search(r"[.!?][\"')\]]?$", normalized))


def _normalize_pdf_fenced_blocks(markdown: str) -> str:
    lines = markdown.splitlines()
    result: list[str] = []
    index = 0
    while index < len(lines):
        opening = re.match(r"^\s*(`{3,}|~{3,})(.*)$", lines[index])
        if not opening:
            result.append(lines[index])
            index += 1
            continue

        fence = opening.group(1)
        info = opening.group(2).strip()
        closing_index = index + 1
        while closing_index < len(lines):
            if re.fullmatch(
                rf"\s*{re.escape(fence[0])}{{{len(fence)},}}\s*",
                lines[closing_index],
            ):
                break
            closing_index += 1
        if closing_index == len(lines) or info:
            result.extend(
                lines[index : closing_index + (closing_index < len(lines))]
            )
            index = closing_index + (closing_index < len(lines))
            continue

        content = "\n".join(lines[index + 1 : closing_index])
        if not content.strip():
            index = closing_index + 1
            continue
        heading = _numbered_section_heading(content)
        if heading:
            result.append(heading)
        elif _is_prose_block(content):
            paragraph = " ".join(
                line.strip() for line in textwrap.dedent(content).splitlines()
            )
            result.append(paragraph)
        else:
            result.extend(lines[index : closing_index + 1])
        index = closing_index + 1
    return "\n".join(result)


def _fence_unmarked_pdf_examples(markdown: str) -> str:
    blocks = re.split(r"(\n[ \t]*\n)", markdown)
    fenced: list[str] = []
    in_fence = False
    for block in blocks:
        fence_markers = re.findall(r"(?m)^\s*(```|~~~)", block)
        if fence_markers:
            for _ in fence_markers:
                in_fence = not in_fence
            fenced.append(block)
            continue
        if in_fence or not block.strip():
            fenced.append(block)
            continue

        content = block.strip()
        lines = content.splitlines()
        if any(re.match(r"^\s*\|.*\|\s*$", line) for line in lines):
            fenced.append(block)
            continue
        if re.match(
            r"^(?:(?:GET|HEAD|POST|PUT|DELETE|CONNECT|OPTIONS|TRACE|PATCH)"
            r"\s+\S+\s+HTTP/\d\.\d|HTTP/\d\.\d\s+\d{3}\b)",
            content,
        ):
            fenced.append(f"```http\n{content}\n```")
            continue

        assignments = list(
            re.finditer(r"(?<![\w-])[A-Za-z][A-Za-z0-9-]*\s*=", content)
        )
        if assignments and assignments[0].start() == 0 and len(assignments) >= 2:
            abnf = re.sub(
                r"\s+(?=[A-Za-z][A-Za-z0-9-]*\s*=)",
                "\n",
                content,
            )
            fenced.append(f"```abnf\n{abnf}\n```")
            continue
        fenced.append(block)
    return "".join(fenced)


def _pdf_heading_font_sizes(document: Any) -> dict[str, float]:
    font_sizes: dict[str, float] = {}
    for page in document:
        for block in page.get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                text = "".join(span.get("text", "") for span in spans)
                key = _normalized_heading_text(text)
                if key and spans:
                    font_sizes[key] = max(
                        font_sizes.get(key, 0),
                        max(span.get("size", 0) for span in spans),
                    )
    return font_sizes


def _normalize_pdf_heading_levels(
    markdown: str,
    heading_font_sizes: dict[str, float],
) -> str:
    lines = markdown.splitlines()
    current_major_font_size: float | None = None
    found_major_heading = False
    for index, line in enumerate(lines):
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if not match:
            continue

        heading = match.group(2).strip()
        plain_heading = unescape(re.sub(r"<[^>]+>", "", heading))
        plain_heading = re.sub(r"[#*_`]", "", plain_heading).strip()
        heading_key = _normalized_heading_text(heading)
        if re.match(r"^table\s+\d+\s*:", plain_heading, re.IGNORECASE):
            caption = heading
            if not (caption.startswith("**") and caption.endswith("**")):
                caption = f"**{caption}**"
            lines[index] = caption
            continue

        letters = [character for character in plain_heading if character.isalpha()]
        font_size = heading_font_sizes.get(heading_key)
        numbered_section = re.match(
            r"^(\d+(?:\.\d+)*)\.\s+\S",
            plain_heading,
        )
        if numbered_section:
            level = min(numbered_section.group(1).count(".") + 1, 6)
            found_major_heading = True
        elif letters and all(character.isupper() for character in letters):
            level = 1
            found_major_heading = True
            current_major_font_size = font_size or current_major_font_size
        elif not found_major_heading:
            continue
        elif (
            font_size is not None
            and current_major_font_size is not None
            and font_size < current_major_font_size - 0.5
        ):
            level = 3
        elif len(match.group(1)) == 1:
            level = 2
        else:
            level = len(match.group(1))

        lines[index] = f"{'#' * level} {heading}"
    return "\n".join(lines)


# ------------------------------------------------------------ PDF OCR helpers
def pdf_ocr_warnings(source: Path) -> list[str]:
    import pymupdf

    warnings: list[str] = []
    with pymupdf.open(source) as document:
        for page_index, page in enumerate(document):
            native_text_chars, image_coverage = _pdf_page_content_metrics(page)
            if _pdf_page_needs_ocr(native_text_chars, image_coverage):
                warnings.append(
                    f"Page {page_index + 1} has {native_text_chars} native text "
                    f"characters and image bounds totaling about "
                    f"{image_coverage:.0%} of the page area; OCR has not yet "
                    "been confirmed, so image text may be missing"
                )
    return warnings


def _pdf_page_content_metrics(page: Any) -> tuple[int, float]:
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
    coverage = min(1.0, image_area / page_area) if page_area else 0
    return native_text_chars, coverage


def _pdf_page_needs_ocr(native_text_chars: int, image_coverage: float) -> bool:
    sparse_page = (
        native_text_chars < SPARSE_PAGE_TEXT_CHARS
        and image_coverage >= MIN_IMAGE_COVERAGE_FOR_OCR_WARNING
    )
    mixed_page = image_coverage >= 0.25 and native_text_chars < 1200
    return sparse_page or mixed_page


def _supplemental_ocr_text(native_text: str, ocr_text: str) -> str:
    remaining_native_lines = Counter(
        " ".join(line.split()).casefold()
        for line in native_text.splitlines()
        if line.strip()
    )
    supplemental_lines: list[str] = []
    for line in ocr_text.splitlines():
        normalized = " ".join(line.split()).casefold()
        if not normalized:
            continue
        if remaining_native_lines[normalized]:
            remaining_native_lines[normalized] -= 1
            continue
        supplemental_lines.append(line.strip())
    return "\n".join(supplemental_lines).strip()


def _configured_pdf_ocr_backend() -> str:
    backend = os.environ.get("PDF_OCR_BACKEND", PDF_OCR_BACKEND).strip().lower()
    if backend not in {"tesseract", "paddle", "unstructured", "auto", "none"}:
        raise ValueError(
            "PDF_OCR_BACKEND must be tesseract, paddle, unstructured, auto, or none"
        )
    if backend == "auto":
        if importlib.util.find_spec("paddleocr") is not None:
            return "paddle"
        if importlib.util.find_spec("unstructured") is not None:
            return "unstructured"
        return "tesseract"
    return backend


_PADDLE_OCR_PIPELINE: Any | None = None


def _paddle_ocr_image(image_path: Path) -> str:
    global _PADDLE_OCR_PIPELINE
    if _PADDLE_OCR_PIPELINE is None:
        paddleocr = importlib.import_module("paddleocr")
        pipeline_type = getattr(paddleocr, "PPStructureV3", None)
        if not callable(pipeline_type):
            raise RuntimeError("PaddleOCR does not provide PPStructureV3")
        _PADDLE_OCR_PIPELINE = pipeline_type()

    results = _PADDLE_OCR_PIPELINE.predict(input=str(image_path))
    markdown_parts: list[str] = []
    for result in results:
        markdown = getattr(result, "markdown", None)
        if isinstance(markdown, str):
            markdown_parts.append(markdown)
        elif isinstance(markdown, dict):
            text = markdown.get("markdown_texts")
            if isinstance(text, str):
                markdown_parts.append(text)
            elif isinstance(text, list):
                markdown_parts.extend(
                    part for part in text if isinstance(part, str)
                )
    text = "\n\n".join(part.strip() for part in markdown_parts if part.strip())
    if not text:
        raise RuntimeError("PaddleOCR returned no Markdown text for the page")
    return IMAGE_LINK_PATTERN.sub(lambda match: match.group(1), text)


def _unstructured_ocr_image(image_path: Path) -> str:
    image_partition = importlib.import_module("unstructured.partition.image")
    partition_image = getattr(image_partition, "partition_image", None)
    if not callable(partition_image):
        raise RuntimeError("Unstructured does not provide partition_image")

    languages = [
        language
        for language in re.split(
            r"[+,]",
            os.environ.get("PDF_OCR_LANGUAGES", PDF_OCR_LANGUAGES),
        )
        if language
    ]
    elements = partition_image(
        filename=str(image_path),
        strategy="hi_res",
        infer_table_structure=True,
        languages=languages,
    )
    parts: list[str] = []
    for element in elements:
        metadata = getattr(element, "metadata", None)
        html = getattr(metadata, "text_as_html", None)
        category = getattr(element, "category", "")
        content = (
            html
            if category == "Table" and isinstance(html, str)
            else str(element)
        )
        if content.strip():
            parts.append(content.strip())
    text = "\n\n".join(parts)
    if not text:
        raise RuntimeError("Unstructured returned no text for the page")
    return text


def _run_page_ocr(
    page: Any,
    native_text: str,
    backend: str,
    page_number: int,
    warnings: list[str],
) -> str:
    if backend == "none":
        return ""
    try:
        if backend == "tesseract":
            ocr_textpage = page.get_textpage_ocr(
                language=os.environ.get(
                    "PDF_OCR_LANGUAGES", PDF_OCR_LANGUAGES
                ),
                dpi=300,
                full=False,
            )
            ocr_text = page.get_text("text", textpage=ocr_textpage)
        else:
            with tempfile.TemporaryDirectory(
                prefix="document-page-ocr-"
            ) as temp_dir:
                image_path = Path(temp_dir) / f"page-{page_number:04}.png"
                page.get_pixmap(dpi=300, alpha=False).save(image_path)
                if backend == "paddle":
                    ocr_text = _paddle_ocr_image(image_path)
                else:
                    ocr_text = _unstructured_ocr_image(image_path)
        if not ocr_text.strip():
            warnings.append(f"Page {page_number} {backend} OCR returned no text")
            return ""
        supplemental_text = _supplemental_ocr_text(native_text, ocr_text)
        if supplemental_text:
            warnings.append(
                f"Page {page_number} added OCR-only text from {backend}; "
                "verify recognized text against the source"
            )
            return supplemental_text
        warnings.append(
            f"Page {page_number} {backend} OCR found no text missing from the "
            "native text layer"
        )
    except Exception as error:
        warnings.append(f"Page {page_number} {backend} OCR failed: {error}")
        if backend != "tesseract":
            try:
                ocr_textpage = page.get_textpage_ocr(
                    language=os.environ.get(
                        "PDF_OCR_LANGUAGES", PDF_OCR_LANGUAGES
                    ),
                    dpi=300,
                    full=False,
                )
                ocr_text = page.get_text("text", textpage=ocr_textpage)
                if not ocr_text.strip():
                    warnings.append(
                        f"Page {page_number} Tesseract fallback returned no text"
                    )
                    return ""
                supplemental_text = _supplemental_ocr_text(native_text, ocr_text)
                if supplemental_text:
                    warnings.append(
                        f"Page {page_number} fell back to Tesseract OCR"
                    )
                    warnings.append(
                        f"Page {page_number} added OCR-only text from Tesseract; "
                        "verify recognized text against the source"
                    )
                    return supplemental_text
            except Exception as fallback_error:
                warnings.append(
                    f"Page {page_number} Tesseract fallback also failed: "
                    f"{fallback_error}"
                )
    return ""


# ---------------------------------------------------------------- token/config
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pdf_ocr_configuration() -> dict[str, str]:
    return {
        "backend": os.environ.get("PDF_OCR_BACKEND", PDF_OCR_BACKEND)
        .strip()
        .lower(),
        "languages": os.environ.get("PDF_OCR_LANGUAGES", PDF_OCR_LANGUAGES)
        .strip(),
    }


def _configured_pdf_table_backend() -> str:
    backend = os.environ.get("PDF_TABLE_BACKEND", "pdfplumber").strip().lower()
    if backend not in {"pdfplumber", "hybrid", "table-transformer"}:
        raise ValueError(
            "PDF_TABLE_BACKEND must be pdfplumber, hybrid, or table-transformer"
        )
    return backend


def _configured_pdf_table_fallback() -> str:
    fallback = os.environ.get("PDF_TABLE_FALLBACK", "none").strip().lower()
    if fallback not in {"none", "unstructured"}:
        raise ValueError("PDF_TABLE_FALLBACK must be none or unstructured")
    return fallback


def output_token(relative_path: Path, source_sha256: str) -> str:
    path_hash = hashlib.sha256(
        relative_path.as_posix().encode("utf-8")
    ).hexdigest()[:10]
    content_hash = source_sha256[:12]
    safe_stem = re.sub(
        r"[^A-Za-z0-9._-]+", "_", relative_path.stem
    ).strip("._")
    ocr_suffix = ""
    table_suffix = ""
    if relative_path.suffix.lower() == ".pdf":
        configuration = {
            "ocr": _pdf_ocr_configuration(),
            "table_backend": _configured_pdf_table_backend(),
            "table_fallback": _configured_pdf_table_fallback(),
        }
        configuration_json = json.dumps(
            configuration, sort_keys=True
        ).encode("utf-8")
        configuration_hash = hashlib.sha256(
            configuration_json
        ).hexdigest()[:8]
        ocr_suffix = f"-ocr-{configuration_hash}"
        table_suffix = (
            f"-table-{configuration['table_backend']}"
            f"-fallback-{configuration['table_fallback']}"
        )
    return (
        f"{safe_stem or 'document'}--{path_hash}-{content_hash}"
        f"{ocr_suffix}{table_suffix}"
        f"-v{CONVERSION_FORMAT_VERSION}"
    )


def atomic_write_text(destination: Path, content: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def relative_output_path(path: Path, workspace: Path) -> str:
    return path.relative_to(workspace).as_posix()


def _markdown_images_exist(
    markdown: str, markdown_dir: Path, image_root: Path
) -> bool:
    for image_uri in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", markdown):
        if image_uri.startswith(("http://", "https://", "data:")):
            continue
        image_path = (markdown_dir / image_uri.split("#", 1)[0]).resolve()
        try:
            image_path.relative_to(image_root.resolve())
        except ValueError:
            return False
        if not image_path.is_file():
            return False
    return True


# ----------------------------------------------------------- Groq vision (unchanged)
def _normalize_groq_image_analysis(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise ValueError("Groq image-analysis result was not a JSON object")

    confidence_value = result.get("confidence", 0)
    confidence = (
        float(confidence_value)
        if isinstance(confidence_value, (int, float))
        and not isinstance(confidence_value, bool)
        else 0.0
    )
    if not 0 <= confidence <= 1:
        confidence = 0.0
    is_diagram = result.get("is_diagram") is True
    diagram_type = result.get("diagram_type")
    if diagram_type not in {
        "flowchart", "architecture", "sequence", "org_chart", "other",
    }:
        diagram_type = None
    mermaid = result.get("mermaid")
    if (
        not is_diagram
        or confidence < DIAGRAM_CONFIDENCE_THRESHOLD
        or not isinstance(mermaid, str)
        or not mermaid.strip()
        or "```" in mermaid
        or re.search(r"(?im)^\s*(?:click\b|%%\{)", mermaid)
        or re.search(r"(?i)<\s*/?\s*script\b", mermaid)
    ):
        mermaid = None
    description = result.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("Groq image-analysis response has no description")
    description = " ".join(description.split())[:1000]
    return {
        "is_diagram": is_diagram and confidence >= DIAGRAM_CONFIDENCE_THRESHOLD,
        "diagram_type": diagram_type,
        "mermaid": mermaid.strip() if mermaid else None,
        "description": description.strip(),
    }


def _groq_images_analysis(
    image_paths: list[Path], client: Any
) -> list[dict[str, Any]]:
    import base64
    import mimetypes

    if not image_paths:
        raise ValueError("At least one image is required for Groq analysis")
    if len(image_paths) > GROQ_IMAGE_BATCH_SIZE:
        raise ValueError(
            f"Groq image batches cannot exceed {GROQ_IMAGE_BATCH_SIZE} images"
        )

    prompt = (
        "Analyze each document image independently. Return only a JSON object "
        'with an "images" array. Include exactly one object for each image, using '
        'its provided integer index, and these fields: index, is_diagram '
        "(boolean), diagram_type (flowchart, architecture, sequence, org_chart, "
        "other, or null), confidence (number 0..1), mermaid (string or null), "
        "and description (one to three concise sentences). Describe only visible "
        "information; do not infer missing details. Set is_diagram true only for "
        "a clearly visible flowchart, architecture, sequence diagram, or org "
        "chart. Provide Mermaid only when a faithful conversion is possible."
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for index, image_path in enumerate(image_paths, start=1):
        mime_type, _ = mimetypes.guess_type(str(image_path))
        mime_type = mime_type or "image/png"
        image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
        content.extend(
            [
                {"type": "text", "text": f"Image index: {index}"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{image_data}"
                    },
                },
            ]
        )
    response = client.chat.completions.create(
        model=GROQ_VISION_MODEL,
        messages=[{"role": "user", "content": content}],
        temperature=0.1,
        max_completion_tokens=1500 * len(image_paths),
        response_format={"type": "json_object"},
    )
    response_content = response.choices[0].message.content
    if not isinstance(response_content, str):
        raise ValueError("Groq returned an empty image-analysis response")
    result = json.loads(response_content)
    if not isinstance(result, dict) or not isinstance(
        result.get("images"), list
    ):
        raise ValueError("Groq image-analysis response has no images array")

    analyses: dict[int, dict[str, Any]] = {}
    for image_result in result["images"]:
        if not isinstance(image_result, dict):
            raise ValueError("Groq returned an invalid image-analysis entry")
        index = image_result.get("index")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 1
            or index > len(image_paths)
            or index in analyses
        ):
            raise ValueError("Groq returned an invalid or duplicate image index")
        analyses[index] = _normalize_groq_image_analysis(image_result)
    if len(analyses) != len(image_paths):
        raise ValueError("Groq did not return an analysis for every image")
    return [analyses[index] for index in range(1, len(image_paths) + 1)]


def _groq_image_analysis(image_path: Path, client: Any) -> dict[str, Any]:
    return _groq_images_analysis([image_path], client)[0]


# ---------------------------------------------------------------- PPTX render
def _render_pptx_slides(source: Path, image_dir: Path) -> list[Path]:
    import pymupdf

    office_binary = shutil.which("libreoffice") or shutil.which("soffice")
    if office_binary is None:
        raise RuntimeError(
            "LibreOffice is not installed; install LibreOffice Impress to render slides"
        )

    image_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="document-slide-render-"
    ) as temp_dir:
        temp_path = Path(temp_dir)
        pdf_dir = temp_path / "pdf"
        profile_dir = temp_path / "profile"
        pdf_dir.mkdir()
        command = [
            office_binary,
            "--headless",
            f"-env:UserInstallation={profile_dir.as_uri()}",
            "--convert-to",
            "pdf",
            "--outdir",
            str(pdf_dir),
            str(source.resolve()),
        ]
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                "LibreOffice timed out while rendering slides"
            ) from error
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(
                "LibreOffice could not render slides"
                + (f": {detail[:300]}" if detail else "")
            )

        rendered_pdf = pdf_dir / f"{source.stem}.pdf"
        if not rendered_pdf.is_file():
            raise RuntimeError(
                "LibreOffice did not produce the expected slide PDF"
            )

        scale = SLIDE_RENDER_DPI / 72
        matrix = pymupdf.Matrix(scale, scale)
        slide_paths: list[Path] = []
        with pymupdf.open(rendered_pdf) as document:
            if not len(document):
                raise RuntimeError("LibreOffice produced an empty slide PDF")
            for page_index, page in enumerate(document):
                image_path = image_dir / f"slide-{page_index + 1:04}.png"
                page.get_pixmap(matrix=matrix, alpha=False).save(image_path)
                slide_paths.append(image_path)
        return slide_paths


def _append_rendered_slides(
    markdown: str, slide_paths: list[Path], markdown_dir: Path
) -> str:
    if not slide_paths:
        return markdown
    image_prefix = (
        slide_paths[0].parent.relative_to(markdown_dir).as_posix()
    )
    slide_section = ["## Rendered slide visuals"]
    for slide_number, slide_path in enumerate(slide_paths, start=1):
        slide_section.extend(
            [
                f"### Slide {slide_number}",
                "",
                f"![Rendered slide {slide_number}]"
                f"({image_prefix}/{slide_path.name})",
                "",
            ]
        )
    return (
        markdown.rstrip()
        + "\n\n"
        + "\n".join(slide_section).rstrip()
        + "\n"
    )


# ------------------------------------------------------- vision enrichment
def _image_dimensions(image_path: Path) -> tuple[int, int]:
    import pymupdf

    pixmap = pymupdf.Pixmap(str(image_path))
    return pixmap.width, pixmap.height


def _image_is_too_small(image_path: Path) -> bool:
    width, height = _image_dimensions(image_path)
    return min(width, height) < MIN_VISION_IMAGE_DIMENSION


def _enrich_markdown_images(
    markdown: str, markdown_dir: Path, warnings: list[str]
) -> str:
    if os.environ.get("ENABLE_DIAGRAM_CONVERSION", "").strip().lower() not in {
        "1", "true", "yes",
    }:
        return markdown

    image_matches = list(IMAGE_LINK_PATTERN.finditer(markdown))
    if not image_matches:
        return markdown

    if not os.environ.get("GROQ_API_KEY"):
        warnings.append(
            "Groq image enrichment is enabled but GROQ_API_KEY is unset; "
            "image links and source captions remain unchanged"
        )
        return markdown

    try:
        from groq import Groq

        client = Groq()
    except Exception as error:
        warnings.append(
            f"Groq image enrichment could not be initialized: {error}; "
            "image links and source captions remain unchanged"
        )
        return markdown

    enriched: list[str] = []
    cursor = 0
    failed_images: list[str] = []
    analysis_cache: dict[Path, dict[str, Any] | Exception] = {}
    pending_paths: list[Path] = []
    pending_set: set[Path] = set()
    skipped_small_images: set[Path] = set()
    for match in image_matches:
        if re.match(
            r"\s*\n\s*\*\*AI-generated image description",
            markdown[match.end() :],
        ):
            continue
        image_uri = unquote(
            match.group(2).split("#", 1)[0].split("?", 1)[0]
        )
        if image_uri.startswith(("http://", "https://", "data:")):
            continue
        image_path = (markdown_dir / image_uri).resolve()
        try:
            image_path.relative_to(markdown_dir.resolve())
        except ValueError:
            continue
        if (
            not image_path.is_file()
            or image_path.stat().st_size > MAX_VISION_IMAGE_BYTES
            or image_path in pending_set
        ):
            continue
        try:
            if _image_is_too_small(image_path):
                skipped_small_images.add(image_path)
                failed_images.append(
                    f"{image_path.name} (smaller than "
                    f"{MIN_VISION_IMAGE_DIMENSION} pixels in one dimension)"
                )
                continue
        except Exception as error:
            warnings.append(
                f"Could not inspect image dimensions for {image_path.name}: "
                f"{error}; attempting image analysis"
            )
        pending_paths.append(image_path)
        pending_set.add(image_path)

    def analyze_batch(batch: list[Path]) -> None:
        try:
            analyses = _groq_images_analysis(batch, client)
            if len(analyses) != len(batch):
                raise ValueError(
                    "Groq returned the wrong number of image analyses"
                )
            for image_path, analysis in zip(batch, analyses):
                analysis_cache[image_path] = analysis
        except Exception as error:
            for image_path in batch:
                analysis_cache[image_path] = error

    batch: list[Path] = []
    batch_bytes = 0
    for image_path in pending_paths:
        image_size = image_path.stat().st_size
        if batch and (
            len(batch) >= GROQ_IMAGE_BATCH_SIZE
            or batch_bytes + image_size > MAX_VISION_BATCH_BYTES
        ):
            analyze_batch(batch)
            batch = []
            batch_bytes = 0
        batch.append(image_path)
        batch_bytes += image_size
    if batch:
        analyze_batch(batch)

    for match in image_matches:
        enriched.append(markdown[cursor : match.start()])
        image_uri = unquote(match.group(2).split("#", 1)[0].split("?", 1)[0])
        if image_uri.startswith(("http://", "https://", "data:")):
            enriched.append(match.group(0))
            cursor = match.end()
            continue
        image_path = (markdown_dir / image_uri).resolve()
        try:
            image_path.relative_to(markdown_dir.resolve())
        except ValueError:
            enriched.append(match.group(0))
            failed_images.append(image_path.name or image_uri)
            cursor = match.end()
            continue
        if not image_path.is_file():
            enriched.append(match.group(0))
            failed_images.append(image_path.name)
            cursor = match.end()
            continue
        if image_path in skipped_small_images:
            enriched.append(match.group(0))
            cursor = match.end()
            continue

        next_text = markdown[match.end() :]
        if re.match(
            r"\s*\n\s*\*\*AI-generated image description", next_text
        ):
            enriched.append(match.group(0))
            cursor = match.end()
            continue
        existing_fallback = re.match(
            r"\n\n\*\*Image description unavailable:\*\*[^\n]*", next_text
        )
        if existing_fallback:
            cursor = match.end() + len(existing_fallback.group(0))
        if image_path.stat().st_size > MAX_VISION_IMAGE_BYTES:
            enriched.append(match.group(0))
            failed_images.append(f"{image_path.name} (exceeds 20 MiB)")
            cursor = match.end()
            continue

        try:
            cached_analysis = analysis_cache.get(image_path)
            if cached_analysis is None:
                cached_analysis = _groq_image_analysis(image_path, client)
                analysis_cache[image_path] = cached_analysis
            if isinstance(cached_analysis, Exception):
                raise cached_analysis
            analysis = cached_analysis
        except Exception as error:
            enriched.append(match.group(0))
            analysis_cache[image_path] = error
            failed_images.append(
                f"{image_path.name} ({str(error)[:160]})"
            )
            cursor = match.end()
            continue

        addition = (
            f"\n\n**AI-generated image description "
            f"(Groq; verify against image):** {analysis['description']}"
        )
        mermaid = analysis.get("mermaid")
        if analysis["is_diagram"] and isinstance(mermaid, str):
            addition += (
                f"\n\n**AI-generated diagram transcription "
                f"(verify against image):**\n\n```mermaid\n{mermaid}\n```"
            )
        enriched.append(match.group(0) + addition)
        cursor = match.end()

    enriched.append(markdown[cursor:])
    if failed_images:
        warnings.append(
            "Groq image enrichment failed or was skipped for "
            + ", ".join(failed_images)
            + "; original image links were retained"
        )
    return "".join(enriched)


def _vision_enrichment_is_current(
    markdown: str, markdown_dir: Path
) -> bool:
    if not os.environ.get("GROQ_API_KEY"):
        return True
    for match in IMAGE_LINK_PATTERN.finditer(markdown):
        image_uri = unquote(match.group(2).split("#", 1)[0].split("?", 1)[0])
        if image_uri.startswith(("http://", "https://", "data:")):
            continue
        image_path = (markdown_dir / image_uri).resolve()
        try:
            image_path.relative_to(markdown_dir.resolve())
        except ValueError:
            return False
        if not image_path.is_file():
            return False
        try:
            if _image_is_too_small(image_path):
                continue
        except Exception:
            return False
        following_text = markdown[match.end() :]
        if not re.match(
            r"\s*\n\s*\*\*AI-generated image description", following_text
        ):
            return False
    return True


# ----------------------------------------------------------- table markdown
def _table_to_markdown(rows: list[list[str | None]]) -> str:
    columns = max((len(row) for row in rows), default=0)
    if not columns:
        return ""

    def clean(cell: str | None) -> str:
        return "<br>".join(
            escape(
                " ".join(line.split()).replace("|", r"\|"),
                quote=False,
            )
            for line in (cell or "").splitlines()
            if line.strip()
        )

    normalized_rows = [
        [clean(row[index]) if index < len(row) else "" for index in range(columns)]
        for row in rows
    ]
    has_blank_header_stub = bool(
        len(rows) > 1
        and not (rows[0][0] or "").strip()
        and any(cell and cell.strip() for cell in rows[0][1:])
    )
    header_row_count = int(
        bool(_table_header_signature(rows) or has_blank_header_stub)
    )
    while header_row_count < len(rows):
        row = rows[header_row_count]
        if (row[0] or "").strip() or not any(
            cell and cell.strip() for cell in row[1:]
        ):
            break
        header_row_count += 1

    if header_row_count:
        previous_header = ""
        for column in range(1, columns):
            if normalized_rows[0][column]:
                previous_header = normalized_rows[0][column]
            elif previous_header:
                normalized_rows[0][column] = previous_header
        headers = []
        for column in range(columns):
            labels: list[str] = []
            for row in normalized_rows[:header_row_count]:
                label = row[column]
                if label and label not in labels:
                    labels.append(label)
            headers.append("<br>".join(labels))
        data_rows = normalized_rows[header_row_count:]
    else:
        headers = [f"Column {index + 1}" for index in range(columns)]
        data_rows = normalized_rows

    markdown_rows = ["| " + " | ".join(headers) + " |"]
    markdown_rows.append(
        "|" + "|".join([" --- " for _ in range(columns)]) + "|"
    )
    markdown_rows.extend(
        "| " + " | ".join(row) + " |" for row in data_rows
    )
    return "\n".join(markdown_rows)


def _table_to_html(rows: list[list[str | None]]) -> str:
    columns = max((len(row) for row in rows), default=0)
    if not columns:
        return ""

    header_row_count = int(
        bool(_table_header_signature(rows))
        or (
            len(rows) > 1
            and not (rows[0][0] or "").strip()
            and any(cell and cell.strip() for cell in rows[0][1:])
        )
    )
    while header_row_count < len(rows):
        row = rows[header_row_count]
        if (row[0] or "").strip() or not any(
            cell and cell.strip() for cell in row[1:]
        ):
            break
        header_row_count += 1

    def render_row(row: list[str | None], tag: str) -> str:
        cells = []
        for column in range(columns):
            cell = row[column] if column < len(row) else None
            lines = [
                escape(" ".join(line.split()))
                for line in (cell or "").splitlines()
                if line.strip()
            ]
            cells.append(f"<{tag}>{'<br>'.join(lines)}</{tag}>")
        return "<tr>" + "".join(cells) + "</tr>"

    parts = ["<table>"]
    if header_row_count:
        parts.append("<thead>")
        parts.extend(
            render_row(row, "th") for row in rows[:header_row_count]
        )
        parts.append("</thead>")
    if header_row_count < len(rows):
        parts.append("<tbody>")
        parts.extend(
            render_row(row, "td") for row in rows[header_row_count:]
        )
        parts.append("</tbody>")
    parts.append("</table>")
    return "\n".join(parts)


def _table_header_row_count(rows: list[list[str | None]]) -> int:
    if not rows:
        return 0
    first_cell = rows[0][0] if rows[0] else None
    header_row_count = int(
        bool(_table_header_signature(rows))
        or (
            len(rows) > 1
            and not (first_cell or "").strip()
            and any(cell and cell.strip() for cell in rows[0][1:])
        )
    )
    while header_row_count < len(rows):
        row = rows[header_row_count]
        if ((row[0] if row else None) or "").strip() or not any(
            cell and cell.strip() for cell in row[1:]
        ):
            break
        header_row_count += 1
    return header_row_count


def _table_complexity(rows: list[list[str | None]]) -> dict[str, Any]:
    columns = max((len(row) for row in rows), default=0)
    if not rows or not columns:
        return {"is_complex": False, "score": 0.0, "signals": []}

    widths = [len(row) for row in rows]
    populated_cells = [
        cell for row in rows for cell in row if cell and cell.strip()
    ]
    multiline_cells = sum(
        len([line for line in cell.splitlines() if line.strip()]) > 1
        for cell in populated_cells
    )
    blank_ratio = 1 - len(populated_cells) / max(1, sum(widths))
    signals: list[str] = []
    if _table_header_signature(rows):
        signals.append("multilevel-header")
    if multiline_cells:
        signals.append("multiline-cells")
    if len(set(widths)) > 1:
        signals.append("irregular-row-widths")
    if blank_ratio >= 0.2:
        signals.append("sparse-cells")
    if any(
        re.search(
            r"<sup>|(?:^|\n)\s*(?:[a-z]|\*|†|‡)\s*(?:\n|$)",
            cell,
            re.IGNORECASE,
        )
        for cell in populated_cells
    ):
        signals.append("footnote-markers")
    if len(rows) >= 5 and columns >= 4:
        signals.append("large-grid")
    score = min(1.0, len(signals) / 4)
    return {
        "is_complex": len(signals) >= 2,
        "score": score,
        "signals": signals,
    }


def _normalized_table_data(
    rows: list[list[str | None]],
) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    columns = max((len(row) for row in rows), default=0)
    if not rows or not columns:
        return [], [], []

    header_row_count = _table_header_row_count(rows)

    header_matrix: list[list[str]] = []
    for row in rows[:header_row_count]:
        header_matrix.append(
            [
                " ".join(
                    re.sub(
                        r"<sup>\s*(?:[a-z]|\*|†|‡)\s*</sup>",
                        "",
                        re.sub(
                            r"(?m)^\s*(?:[a-z]|\*|†|‡)\s*$",
                            "",
                            row[column] or "",
                            flags=re.IGNORECASE,
                        ),
                        flags=re.IGNORECASE,
                    ).replace("\n", " ").split()
                )
                if column < len(row)
                else ""
                for column in range(columns)
            ]
        )
    for header_row in header_matrix:
        previous = ""
        for column, label in enumerate(header_row):
            if label:
                previous = label
            elif column:
                header_row[column] = previous

    headers: list[str] = []
    for column in range(columns):
        labels: list[str] = []
        for header_row in header_matrix:
            label = header_row[column]
            if label and label not in labels:
                labels.append(label)
        base = " / ".join(labels) if labels else f"Column {column + 1}"
        header = base
        suffix = 1
        while header in headers:
            suffix += 1
            header = f"{base} ({suffix})"
        headers.append(header)

    normalized_rows: list[dict[str, Any]] = []
    normalization_warnings: list[str] = []
    active_group: list[str] = []
    active_group_markers: list[str] = []

    def cell_lines(
        cell: str | None,
    ) -> tuple[list[str], list[list[str]]]:
        lines: list[str] = []
        line_markers: list[list[str]] = []
        pending_markers: list[str] = []
        for raw_line in (cell or "").splitlines() or [cell or ""]:
            line = " ".join(raw_line.split())
            if not line:
                continue
            marker = re.fullmatch(r"([a-z]|\*|†|‡)", line, re.IGNORECASE)
            if marker:
                pending_markers.append(marker.group(1))
                continue
            superscripts = re.findall(
                r"<sup>\s*([a-z]|\*|†|‡)\s*</sup>",
                line,
                re.IGNORECASE,
            )
            line = re.sub(
                r"<sup>\s*(?:[a-z]|\*|†|‡)\s*</sup>",
                "",
                line,
                flags=re.IGNORECASE,
            ).strip()
            if line:
                lines.append(line)
                line_markers.append(pending_markers + superscripts)
                pending_markers = []
        if pending_markers and line_markers:
            line_markers[-1].extend(pending_markers)
        return lines, line_markers

    for source_row_index, source_row in enumerate(
        rows[header_row_count:],
        start=header_row_count,
    ):
        padded = list(source_row[:columns]) + [None] * max(
            0, columns - len(source_row)
        )
        split_cells = [cell_lines(cell) for cell in padded]
        line_cells = [lines for lines, _ in split_cells]
        line_markers = [markers for _, markers in split_cells]
        first_cell = line_cells[0]
        if (
            len(first_cell) == 1
            and first_cell[0].endswith(":")
            and not any(line for cell in line_cells[1:] for line in cell)
        ):
            active_group = [first_cell[0].rstrip(":")]
            active_group_markers = (
                line_markers[0][0] if line_markers[0] else []
            )
            continue

        repeated_lengths = Counter(
            len(cell) for cell in line_cells if len(cell) > 1
        )
        aligned_count, aligned_lines = max(
            ((count, length) for length, count in repeated_lengths.items()),
            default=(0, 0),
        )
        context_lines: list[str] = []
        if aligned_count >= 2 and aligned_lines >= 2:
            if len(first_cell) > aligned_lines:
                context_lines = first_cell[:-aligned_lines]
                line_cells[0] = first_cell[-aligned_lines:]
                if len(line_markers[0]) > aligned_lines:
                    context_marker_lines = line_markers[0][:-aligned_lines]
                    line_markers[0] = line_markers[0][-aligned_lines:]
                else:
                    context_marker_lines = []
                if context_lines:
                    active_group = [
                        item.rstrip(":")
                        for item in context_lines
                        if item.strip()
                    ]
                    active_group_markers = list(
                        dict.fromkeys(
                            marker
                            for marker_line in context_marker_lines
                            for marker in marker_line
                        )
                    )
                if (
                    context_lines
                    and len(active_group) > 1
                    and line_markers[0]
                ):
                    active_group_markers.extend(line_markers[0][0])
                    line_markers[0][0] = []
            matching_columns = sum(
                len(cell) == aligned_lines for cell in line_cells
            )
            if matching_columns < 2:
                aligned_lines = 0
        else:
            aligned_lines = 0
            if (
                len(first_cell) > 1
                and first_cell[0].endswith(":")
                and all(len(cell) <= 1 for cell in line_cells[1:])
            ):
                context_lines = [first_cell[0]]
                active_group = [first_cell[0].rstrip(":")]
                active_group_markers = (
                    line_markers[0][0] if line_markers[0] else []
                )
                line_cells[0] = first_cell[1:]
                line_markers[0] = line_markers[0][1:]

        if aligned_lines:
            for line_index in range(aligned_lines):
                values = [
                    cell[line_index]
                    if len(cell) == aligned_lines
                    else (cell[0] if len(cell) == 1 else "<br>".join(cell))
                    for cell in line_cells
                ]
                normalized_rows.append(
                    {
                        "row_group_path": list(active_group),
                        "row_group_footnote_markers": list(
                            dict.fromkeys(active_group_markers)
                        ),
                        "values": values,
                        "source_row_index": source_row_index,
                        "subrow_index": line_index + 1,
                        "footnote_markers": list(
                            dict.fromkeys(
                                marker
                                for cell_markers in line_markers
                                for marker in (
                                    cell_markers[line_index]
                                    if len(cell_markers) == aligned_lines
                                    else cell_markers[0]
                                    if line_index == 0 and cell_markers
                                    else []
                                )
                            )
                        ),
                    }
                )
        else:
            nonempty_lengths = {
                len(cell) for cell in line_cells if len(cell) > 1
            }
            if nonempty_lengths:
                normalization_warnings.append(
                    "Multiline cell alignment was ambiguous; original cell "
                    "boundaries were retained"
                )
            values = ["\n".join(cell) for cell in line_cells]
            markers = list(
                dict.fromkeys(
                    marker
                    for cell_markers in line_markers
                    for line_marker in cell_markers
                    for marker in line_marker
                )
            )
            normalized_rows.append(
                {
                    "row_group_path": list(active_group),
                    "row_group_footnote_markers": list(
                        dict.fromkeys(active_group_markers)
                    ),
                    "values": values,
                    "source_row_index": source_row_index,
                    "subrow_index": 1,
                    "footnote_markers": markers,
                }
            )
        if first_cell and any(
            label.casefold().startswith("total ")
            for label in first_cell
        ):
            active_group = []
            active_group_markers = []

    return headers, normalized_rows, list(
        dict.fromkeys(normalization_warnings)
    )


def _render_normalized_table(
    headers: list[str],
    rows: list[dict[str, Any]],
) -> str:
    has_groups = any(row["row_group_path"] for row in rows)
    has_footnotes = any(
        row["footnote_markers"] or row["row_group_footnote_markers"]
        for row in rows
    )
    output_headers = (
        (["Row group"] if has_groups else [])
        + headers
        + (["Footnote markers"] if has_footnotes else [])
    )

    def clean(value: str) -> str:
        return "<br>".join(
            escape(
                line.replace("|", r"\|"),
                quote=False,
            )
            for line in value.splitlines() or [value]
        )

    output_rows: list[list[str]] = []
    for row in rows:
        values = list(row["values"])
        if has_groups:
            values.insert(0, " > ".join(row["row_group_path"]))
        if has_footnotes:
            values.append(
                ", ".join(
                    dict.fromkeys(
                        row["row_group_footnote_markers"]
                        + row["footnote_markers"]
                    )
                )
            )
        output_rows.append(values)

    output = [
        "| " + " | ".join(clean(value) for value in output_headers) + " |",
        "|" + "|".join([" --- " for _ in output_headers]) + "|",
    ]
    output.extend(
        "| " + " | ".join(clean(value) for value in row) + " |"
        for row in output_rows
    )
    return "\n".join(output)


def _table_footnote_markers(rows: list[list[str | None]]) -> list[str]:
    markers: list[str] = []
    for row in rows:
        for cell in row:
            if not cell:
                continue
            markers.extend(
                re.findall(
                    r"<sup>\s*([a-z]|\*|†|‡)\s*</sup>",
                    cell,
                    re.IGNORECASE,
                )
            )
            markers.extend(
                re.findall(
                    r"(?m)^\s*([a-z]|\*|†|‡)\s*$",
                    cell,
                    re.IGNORECASE,
                )
            )
    return list(
        dict.fromkeys(marker.casefold() for marker in markers)
    )


def _table_footnote_definitions(
    page_markdown: str,
    markers: set[str],
) -> dict[str, str]:
    definitions: dict[str, str] = {}
    pattern = re.compile(
        r"(?im)^\s*(?:[-*]\s*)?([a-z]|\*|†|‡)\s+(.+?)\s*$"
    )
    for match in pattern.finditer(page_markdown):
        marker = match.group(1).casefold()
        if marker in markers:
            definitions[marker] = match.group(2).strip()
    return definitions


def _pdf_table_artifacts(
    tables_by_page: dict[int, list[dict[str, Any]]],
    markdown_pages: list[str],
) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for page_index, tables in sorted(tables_by_page.items()):
        for table in tables:
            if "merged_into_page" in table:
                continue
            rows = table["rows"]
            raw_rows = table.get("raw_rows", rows)
            headers, normalized_rows, normalization_warnings = (
                _normalized_table_data(rows)
            )
            markers = _table_footnote_markers(rows)
            page_number = page_index + 1
            related_pages = [page_number]
            if isinstance(table.get("continues_to_page"), int):
                related_pages.append(table["continues_to_page"])
            definitions = {}
            for related_page in related_pages:
                if 1 <= related_page <= len(markdown_pages):
                    definitions.update(
                        _table_footnote_definitions(
                            markdown_pages[related_page - 1],
                            set(markers),
                        )
                    )
            artifacts.append(
                {
                    "table_id": f"page-{page_number:04d}-table-{table['table_index']:02d}",
                    "caption": table.get("caption")
                    or _pdf_table_caption(
                        markdown_pages[page_index]
                        if page_index < len(markdown_pages)
                        else "",
                        table["table_index"],
                    ),
                    "source_pages": related_pages,
                    "bbox": list(table["bbox"]),
                    "extractor": table.get("extractor", "pdfplumber"),
                    "structure_score": round(
                        _table_structure_quality(table),
                        3,
                    ),
                    "complexity": _table_complexity(rows),
                    "headers": headers,
                    "header_footnote_markers": _table_footnote_markers(
                        rows[: _table_header_row_count(rows)]
                    ),
                    "normalized_rows": normalized_rows,
                    "footnote_markers": markers,
                    "footnotes": definitions,
                    "unresolved_footnote_markers": [
                        marker for marker in markers
                        if marker not in definitions
                    ],
                    "normalization_warnings": normalization_warnings,
                    "repair_notes": table.get("repair_notes", []),
                    "normalization_status": (
                        "review_required"
                        if normalization_warnings
                        or _table_complexity(rows)["is_complex"]
                        or table.get("repair_notes")
                        else "normalized"
                    ),
                    "raw_rows": raw_rows,
                    "raw_html": table.get("html") or _table_to_html(raw_rows),
                    "alternative": table.get("alternative"),
                    "value_table": table.get("value_table"),
                }
            )
    return artifacts


def _pdf_table_caption(
    page_markdown: str, table_index: int
) -> str | None:
    captions = re.findall(
        r"(?im)^\s*\*{0,2}((?:Table|Exhibit)\s+\d+[^*\n]*)\*{0,2}\s*$",
        page_markdown,
    )
    if table_index <= len(captions):
        return " ".join(captions[table_index - 1].split())
    return None


def _markdown_table_cells(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|") and not line.endswith(r"\|"):
        line = line[:-1]

    cells: list[str] = []
    current: list[str] = []
    escaped = False
    for character in line:
        if escaped:
            current.append(character)
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == "|":
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(character)
    if escaped:
        current.append("\\")
    cells.append("".join(current).strip())
    return cells


def _normalize_markdown_tables(markdown: str) -> str:
    lines = markdown.splitlines()
    in_fenced_block = False
    output: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if re.match(r"^\s*(```|~~~)", line):
            in_fenced_block = not in_fenced_block
            output.append(line)
            index += 1
            continue

        if (
            not in_fenced_block
            and index + 1 < len(lines)
            and re.match(r"^\s*\|.*\|\s*$", line)
            and re.match(
                r"^\s*\|(?:\s*:?-+:?\s*\|)+\s*$",
                lines[index + 1],
            )
        ):
            table_lines = [line, lines[index + 1]]
            index += 2
            while (
                index < len(lines)
                and re.match(r"^\s*\|.*\|\s*$", lines[index])
            ):
                table_lines.append(lines[index])
                index += 1
            expected_columns = len(_markdown_table_cells(table_lines[0]))
            repaired = [table_lines[0], table_lines[1]]
            for table_line in table_lines[2:]:
                cells = _markdown_table_cells(table_line)
                if len(cells) > expected_columns:
                    cells = cells[: expected_columns - 1] + [
                        "<br>".join(cells[expected_columns - 1 :])
                    ]
                cells.extend([""] * (expected_columns - len(cells)))
                cells = [
                    cell.replace("|", r"\|").replace("<br/>", "<br>")
                    for cell in cells
                ]
                repaired.append("| " + " | ".join(cells) + " |")
            output.extend(repaired)
            continue

        output.append(line)
        index += 1
    return "\n".join(output)


def _insert_pdf_tables(
    markdown: str,
    tables: list[tuple[str, str, str]],
) -> tuple[str, set[str]]:
    if not tables:
        return markdown, set()

    table_by_placeholder = {
        placeholder: (label, table_markdown)
        for placeholder, label, table_markdown in tables
    }
    inserted: set[str] = set()
    placeholder_alternation = "|".join(
        re.escape(placeholder) for placeholder in table_by_placeholder
    )
    placeholder_pattern = re.compile(
        r"\*{0,2}(" + placeholder_alternation + r")\*{0,2}"
    )
    standalone_placeholder_pattern = re.compile(
        r"\s*\*{0,2}(" + placeholder_alternation + r")\*{0,2}\s*"
    )
    lines = markdown.splitlines()
    output: list[str] = []
    index = 0
    in_fenced_block = False

    def replace_in_line(line: str) -> tuple[str, list[str]]:
        extracted: list[str] = []

        def replace(match: re.Match[str]) -> str:
            placeholder = match.group(1)
            label, table_markdown = table_by_placeholder[placeholder]
            if placeholder not in inserted:
                inserted.add(placeholder)
                extracted.append(table_markdown)
            return f"**{label} extracted below.**"

        return placeholder_pattern.sub(replace, line), extracted

    def append_table(table_markdown: str, following_content: bool) -> None:
        if output and output[-1].strip():
            output.append("")
        output.extend(table_markdown.splitlines())
        if following_content:
            output.append("")

    while index < len(lines):
        line = lines[index]
        if re.match(r"^\s*(```|~~~)", line):
            in_fenced_block = not in_fenced_block
            output.append(line)
            index += 1
            continue

        if not in_fenced_block and re.match(r"^\s*\|.*\|\s*$", line):
            block: list[str] = []
            extracted_tables: list[str] = []
            while index < len(lines) and re.match(
                r"^\s*\|.*\|\s*$", lines[index]
            ):
                normalized_line, line_tables = replace_in_line(lines[index])
                block.append(normalized_line)
                extracted_tables.extend(line_tables)
                index += 1
            output.extend(block)
            for table_index, table_markdown in enumerate(extracted_tables):
                append_table(
                    table_markdown,
                    table_index + 1 < len(extracted_tables)
                    or (index < len(lines) and bool(lines[index].strip())),
                )
            continue

        if not in_fenced_block:
            standalone_match = standalone_placeholder_pattern.fullmatch(line)
            if standalone_match:
                placeholder = standalone_match.group(1)
                if placeholder not in inserted:
                    inserted.add(placeholder)
                    append_table(
                        table_by_placeholder[placeholder][1],
                        index + 1 < len(lines)
                        and bool(lines[index + 1].strip()),
                    )
                index += 1
                continue

            normalized_line, extracted_tables = replace_in_line(line)
            output.append(normalized_line)
            for table_index, table_markdown in enumerate(extracted_tables):
                append_table(
                    table_markdown,
                    table_index + 1 < len(extracted_tables)
                    or (
                        index + 1 < len(lines)
                        and bool(lines[index + 1].strip())
                    ),
                )
        else:
            output.append(line)
        index += 1

    return "\n".join(output), inserted


# ------------------------------------------------------------- XLSX → markdown
def _xlsx_to_markdown(source: Path) -> str | None:
    from openpyxl import load_workbook
    from openpyxl.utils import get_column_letter

    workbook = load_workbook(source, data_only=False)
    if any(
        worksheet._images or worksheet._charts
        for worksheet in workbook.worksheets
    ):
        workbook.close()
        return None

    def cell_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, bool):
            return "TRUE" if value else "FALSE"
        return " ".join(str(value).split()).replace("|", r"\|")

    def numeric_or_formula(cell: Any) -> bool:
        value = cell.value
        if isinstance(value, bool):
            return False
        if isinstance(value, (int, float)):
            return True
        if cell.data_type == "f":
            return True
        return bool(
            isinstance(value, str)
            and re.fullmatch(
                r"[*+\-]?\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?",
                value.strip(),
            )
        )

    def merged_value(worksheet: Any, row: int, column: int) -> Any:
        value = worksheet.cell(row, column).value
        if value is not None:
            return value
        for merged_range in worksheet.merged_cells.ranges:
            if (
                merged_range.min_row <= row <= merged_range.max_row
                and merged_range.min_col <= column <= merged_range.max_col
            ):
                return worksheet.cell(
                    merged_range.min_row, merged_range.min_col
                ).value
        return None

    sections: list[str] = []
    try:
        for worksheet in workbook.worksheets:
            populated_rows: dict[int, list[int]] = {}
            for row in worksheet.iter_rows():
                columns = [
                    cell.column for cell in row if cell.value is not None
                ]
                if columns:
                    populated_rows[row[0].row] = columns
            data_start = next(
                (
                    row_number
                    for row_number, columns in sorted(populated_rows.items())
                    if sum(
                        numeric_or_formula(
                            worksheet.cell(row_number, column)
                        )
                        for column in columns
                    )
                    >= 3
                    and any(
                        not numeric_or_formula(
                            worksheet.cell(row_number, column)
                        )
                        for column in columns
                    )
                ),
                None,
            )
            if data_start is None:
                return None

            header_rows: list[int] = []
            blank_rows = 0
            row_number = data_start - 1
            while row_number > 0:
                columns = populated_rows.get(row_number, [])
                if not columns:
                    blank_rows += 1
                    if blank_rows > 1:
                        break
                elif len(columns) < 2:
                    break
                else:
                    blank_rows = 0
                    header_rows.append(row_number)
                row_number -= 1
            header_rows.reverse()
            if not header_rows:
                return None

            table_columns = sorted(
                {
                    column
                    for header_row in header_rows
                    for column in populated_rows[header_row]
                }
                | set(populated_rows[data_start])
            )
            if len(table_columns) < 2:
                return None

            headers: list[str] = []
            for column in table_columns:
                labels: list[str] = []
                for header_row in header_rows:
                    label = cell_text(
                        merged_value(worksheet, header_row, column)
                    )
                    if label and (not labels or labels[-1] != label):
                        labels.append(label)
                headers.append(
                    " - ".join(labels) or get_column_letter(column)
                )

            body_rows: list[tuple[int, list[str]]] = []
            table_end = data_start - 1
            for body_row in range(data_start, worksheet.max_row + 1):
                columns = populated_rows.get(body_row, [])
                if not columns:
                    continue
                if len(columns) == 1 and not numeric_or_formula(
                    worksheet.cell(body_row, columns[0])
                ):
                    break
                if not set(columns).intersection(table_columns):
                    break
                values = [
                    cell_text(merged_value(worksheet, body_row, column))
                    for column in table_columns
                ]
                body_rows.append((body_row, values))
                table_end = body_row
            if not body_rows:
                return None

            if any(
                sum(
                    numeric_or_formula(
                        worksheet.cell(row_number, column)
                    )
                    for column in columns
                )
                >= 3
                and any(
                    not numeric_or_formula(
                        worksheet.cell(row_number, column)
                    )
                    for column in columns
                )
                for row_number, columns in populated_rows.items()
                if row_number > table_end
            ):
                return None

            lines = [f"## {worksheet.title}"]
            for title_row in range(1, header_rows[0]):
                title = " ".join(
                    cell_text(worksheet.cell(title_row, column).value)
                    for column in range(1, worksheet.max_column + 1)
                    if worksheet.cell(title_row, column).value is not None
                ).strip()
                if title:
                    lines.extend(["", title])

            lines.extend(
                [
                    "",
                    f"### Table from Excel rows {header_rows[0]}-{table_end}",
                    "",
                    "| Excel row | " + " | ".join(headers) + " |",
                    "| --- | "
                    + " | ".join(["---"] * len(headers))
                    + " |",
                ]
            )
            lines.extend(
                "| "
                + str(body_row)
                + " | "
                + " | ".join(values)
                + " |"
                for body_row, values in body_rows
            )

            notes: list[str] = []
            for note_row in range(table_end + 1, worksheet.max_row + 1):
                columns = populated_rows.get(note_row, [])
                if columns:
                    note = " ".join(
                        cell_text(
                            worksheet.cell(note_row, column).value
                        )
                        for column in columns
                    ).strip()
                    if note:
                        notes.append(note)
            if notes:
                lines.extend(["", *notes])
            sections.append("\n".join(lines))
    finally:
        workbook.close()

    return "\n\n".join(sections) if sections else None


# -------------------------------------------------- PDF images + captions
def _pdf_image_captions(
    page: Any,
) -> list[tuple[tuple[float, ...], str]]:
    blocks = [
        block
        for block in page.get_text("dict").get("blocks", [])
        if block.get("type") == 0
    ]
    text_blocks: list[tuple[tuple[float, ...], str]] = []
    for block in blocks:
        text = " ".join(
            span.get("text", "")
            for line in block.get("lines", [])
            for span in line.get("spans", [])
        )
        text = " ".join(text.split())
        if text:
            text_blocks.append((tuple(block["bbox"]), text))
    text_blocks.sort(key=lambda item: (item[0][1], item[0][0]))

    captions: list[tuple[tuple[float, ...], str]] = []
    index = 0
    while index < len(text_blocks):
        bbox, text = text_blocks[index]
        if not FIGURE_CAPTION_PATTERN.match(text):
            index += 1
            continue

        x0, y0, x1, y1 = bbox
        parts = [text]
        next_index = index + 1
        while next_index < len(text_blocks):
            next_bbox, next_text = text_blocks[next_index]
            overlap = min(x1, next_bbox[2]) - max(x0, next_bbox[0])
            if (
                next_bbox[1] - y1 > 5
                or overlap <= 0
                or not next_text[:1].islower()
                or FIGURE_CAPTION_PATTERN.match(next_text)
            ):
                break
            parts.append(next_text)
            x0 = min(x0, next_bbox[0])
            y1 = max(y1, next_bbox[3])
            x1 = max(x1, next_bbox[2])
            next_index += 1

        captions.append(((x0, y0, x1, y1), " ".join(parts)))
        index = next_index
    return captions


def _caption_for_image(
    image_bbox: tuple[float, ...] | None,
    captions: list[tuple[tuple[float, ...], str]],
    used_captions: set[int],
) -> tuple[int, str] | None:
    if image_bbox is None:
        remaining = [
            (index, caption)
            for index, (_, caption) in enumerate(captions)
            if index not in used_captions
        ]
        return remaining[0] if len(remaining) == 1 else None

    image_x0, image_y0, image_x1, image_y1 = image_bbox
    candidates: list[tuple[float, int, str]] = []
    for index, (caption_bbox, caption) in enumerate(captions):
        if index in used_captions:
            continue
        caption_x0, caption_y0, caption_x1, caption_y1 = caption_bbox
        overlap = min(image_x1, caption_x1) - max(image_x0, caption_x0)
        if overlap <= 0:
            continue
        if caption_y0 >= image_y1:
            distance = caption_y0 - image_y1
        elif caption_y1 <= image_y0:
            distance = (image_y0 - caption_y1) + 30
        else:
            distance = 0
        if distance <= 140:
            candidates.append((distance, index, caption))
    if not candidates:
        return None
    _, index, caption = min(candidates)
    return index, caption


def _insert_image_link_after_caption(
    markdown: str, caption: str, image_link: str
) -> tuple[str, bool]:
    match = FIGURE_CAPTION_PATTERN.match(caption)
    if not match:
        return markdown, False
    identifier = re.match(
        r"^(?:figure|fig\.?|chart|exhibit|diagram|graph|illustration|flowchart)"
        r"\s*(?:no\.?\s*)?\d+[a-z]?",
        caption,
        re.IGNORECASE,
    )
    if not identifier:
        return markdown, False

    lines = markdown.splitlines()
    for index, line in enumerate(lines):
        visible_line = re.sub(r"[*_`]", "", line).strip()
        if not visible_line.lower().startswith(
            identifier.group(0).lower()
        ):
            continue
        end = index
        while end + 1 < len(lines) and lines[end + 1].strip():
            end += 1
        lines[end + 1 : end + 1] = ["", image_link]
        return "\n".join(lines), True
    return markdown, False


def _link_pdf_images(
    page_markdown: str,
    page: Any,
    page_images: list[Path],
    image_prefix: str,
    page_number: int,
) -> tuple[str, list[str]]:
    image_infos = page.get_image_info()
    captions = _pdf_image_captions(page)
    used_captions: set[int] = set()
    warnings: list[str] = []
    image_link_pattern = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")

    for image_path in page_images:
        image_index_match = re.search(r"-\d+-(\d+)$", image_path.stem)
        image_index = (
            int(image_index_match.group(1)) if image_index_match else -1
        )
        image_info = (
            image_infos[image_index]
            if 0 <= image_index < len(image_infos)
            else None
        )
        caption_match = (
            _caption_for_image(
                tuple(image_info["bbox"]), captions, used_captions
            )
            if image_info
            else _caption_for_image(None, captions, used_captions)
        )
        image_uri = f"{image_prefix}/{image_path.name}"
        page_markdown = image_link_pattern.sub(
            lambda match: ""
            if Path(match.group(1).split("#", 1)[0]).name == image_path.name
            else match.group(0),
            page_markdown,
        )

        if caption_match:
            caption_index, caption = caption_match
            used_captions.add(caption_index)
            safe_caption = caption.replace("[", "(").replace("]", ")")
            image_link = (
                f"![{safe_caption} (PDF page {page_number})]({image_uri})"
            )
            page_markdown, inserted = _insert_image_link_after_caption(
                page_markdown, caption, image_link
            )
            if not inserted:
                page_markdown = (
                    page_markdown.rstrip() + "\n\n" + image_link
                )
                warnings.append(
                    f"Image caption on PDF page {page_number} was not located "
                    "in the layout text; its cited image link was appended "
                    "to the page"
                )
        else:
            image_link = f"![Image on PDF page {page_number}]({image_uri})"
            page_markdown = page_markdown.rstrip() + "\n\n" + image_link
            if image_info:
                warnings.append(
                    f"No nearby figure caption was found for an image on "
                    f"PDF page {page_number}; the image is linked with its "
                    "page number only"
                )
    return page_markdown, warnings


# --------------------------------------------------- PDF table machinery
def _table_rows_are_plausible(rows: list[list[str | None]]) -> bool:
    columns = max((len(row) for row in rows), default=0)
    if columns < 2:
        return False

    populated_rows = 0
    for row in rows:
        meaningful_cells = sum(
            bool(cell and any(character.isalnum() for character in cell))
            for cell in row
        )
        if meaningful_cells >= 2:
            populated_rows += 1
    return populated_rows >= 2


def _repair_pdf_table_rows(
    page: Any,
    table: Any,
    rows: list[list[str | None]],
    words: list[dict[str, Any]] | None = None,
) -> tuple[list[list[str | None]], list[int]]:
    columns = max((len(row) for row in rows), default=0)
    if columns < 2:
        return rows, []

    column_edges: list[float] | None = None
    for table_row in table.rows:
        cells = table_row.cells
        if len(cells) != columns or any(cell is None for cell in cells):
            continue
        edges = [float(cells[0][0])]
        edges.extend(float(cell[2]) for cell in cells)
        if all(
            edges[index] < edges[index + 1]
            for index in range(len(edges) - 1)
        ):
            column_edges = edges
            break
    if column_edges is None:
        return rows, []

    if words is None:
        words = page.extract_words()
    repaired_rows = [list(row) for row in rows]
    repaired_indexes: list[int] = []
    table_width = column_edges[-1] - column_edges[0]
    for row_index, table_row in enumerate(table.rows[: len(rows)]):
        cells = table_row.cells
        populated_cells = [cell for cell in cells if cell is not None]
        if (
            len(populated_cells) != 1
            or populated_cells[0][2] - populated_cells[0][0]
            < table_width * 0.75
        ):
            continue

        row_top = min(float(cell[1]) for cell in populated_cells)
        row_bottom = max(float(cell[3]) for cell in populated_cells)
        row_words = [
            word
            for word in words
            if row_top <= float(word["top"])
            and float(word["bottom"]) <= row_bottom
            and column_edges[0] <= float(word["x0"])
            and float(word["x1"]) <= column_edges[-1]
        ]
        if not row_words:
            continue

        text_lines: list[tuple[float, list[dict[str, Any]]]] = []
        for word in sorted(
            row_words,
            key=lambda item: (float(item["top"]), float(item["x0"])),
        ):
            word_top = float(word["top"])
            if not text_lines or abs(text_lines[-1][0] - word_top) > 2:
                text_lines.append((word_top, []))
            text_lines[-1][1].append(word)

        reconstructed = [[] for _ in range(columns)]
        for _, line_words in text_lines:
            line_cells = [[] for _ in range(columns)]
            for word in line_words:
                center = (float(word["x0"]) + float(word["x1"])) / 2
                column = columns - 1
                for candidate in range(columns - 1):
                    if center < column_edges[candidate + 1]:
                        column = candidate
                        break
                line_cells[column].append(str(word["text"]))
            for column, line_cell in enumerate(line_cells):
                if line_cell:
                    reconstructed[column].append(" ".join(line_cell))

        recovered = [
            "\n".join(cell).strip() or None for cell in reconstructed
        ]
        if sum(bool(cell) for cell in recovered) >= 2:
            repaired_rows[row_index] = recovered
            repaired_indexes.append(row_index)
    return repaired_rows, repaired_indexes


def _pdf_table_column_edges(
    table: Any, columns: int
) -> list[float] | None:
    for table_row in table.rows:
        cells = table_row.cells
        if len(cells) != columns or any(cell is None for cell in cells):
            continue
        edges = [float(cells[0][0])]
        edges.extend(float(cell[2]) for cell in cells)
        if all(
            edges[index] < edges[index + 1]
            for index in range(len(edges) - 1)
        ):
            return edges
    return None


def _pdf_words_by_line(
    words: list[dict[str, Any]],
) -> list[tuple[float, list[dict[str, Any]]]]:
    lines: list[tuple[float, list[dict[str, Any]]]] = []
    for word in sorted(
        words,
        key=lambda item: (float(item["top"]), float(item["x0"])),
    ):
        word_top = float(word["top"])
        if not lines or abs(lines[-1][0] - word_top) > 2:
            lines.append((word_top, []))
        lines[-1][1].append(word)
    return lines


def _pdf_context_header_is_plausible(
    header: list[str | None],
) -> bool:
    populated = [
        " ".join((cell or "").split())
        for cell in header
        if cell and cell.strip()
    ]
    if len(populated) < 2:
        return False
    text = " ".join(populated)
    if re.match(
        r"^(?:table|exhibit|schedule|figure|chart)\s+"
        r"(?:of\b|\d)",
        text,
        re.IGNORECASE,
    ):
        return False
    years = [
        cell for cell in populated
        if re.fullmatch(r"(?:19|20)\d{2}", cell)
    ]
    if len(years) >= 2:
        return True
    labels = [
        cell for cell in populated
        if re.search(r"[A-Za-z]", cell)
        and not re.fullmatch(r"(?:19|20)\d{2}", cell)
    ]
    return len(labels) >= 2


def _pdf_context_header_matches_rows(
    header: list[str | None],
    rows: list[list[str | None]],
) -> bool:
    populated = [
        (index, " ".join((cell or "").split()).casefold())
        for index, cell in enumerate(header)
        if cell and cell.strip()
    ]
    if not populated:
        return False
    return any(
        all(
            index < len(row)
            and " ".join((row[index] or "").split()).casefold() == value
            for index, value in populated
        )
        for row in rows[:2]
    )


def _pdf_table_context_header(
    table: Any,
    columns: int,
    words: list[dict[str, Any]],
) -> tuple[list[str | None], dict[int, float]] | None:
    edges = _pdf_table_column_edges(table, columns)
    if edges is None:
        return None

    table_top = float(table.bbox[1])
    candidates: list[
        tuple[float, list[str | None], dict[int, float]]
    ] = []
    for _, line_words in _pdf_words_by_line(words):
        line_words = [
            word
            for word in line_words
            if float(word["bottom"]) <= table_top
            and table_top - float(word["bottom"]) <= 24
            and edges[0] <= (float(word["x0"]) + float(word["x1"])) / 2
            <= edges[-1]
        ]
        if not line_words:
            continue
        grouped: dict[int, list[dict[str, Any]]] = {}
        for word in line_words:
            center = (float(word["x0"]) + float(word["x1"])) / 2
            column = next(
                (
                    index
                    for index in range(columns)
                    if edges[index] <= center < edges[index + 1]
                ),
                columns - 1,
            )
            grouped.setdefault(column, []).append(word)
        if len(grouped) < 2:
            continue

        centers: dict[int, float] = {}
        extents: list[tuple[float, float, float]] = []
        header: list[str | None] = [None] * columns
        for column, cell_words in grouped.items():
            left = min(float(word["x0"]) for word in cell_words)
            right = max(float(word["x1"]) for word in cell_words)
            center = sum(
                (float(word["x0"]) + float(word["x1"])) / 2
                for word in cell_words
            ) / len(cell_words)
            if abs(center - (edges[column] + edges[column + 1]) / 2) > (
                edges[column + 1] - edges[column]
            ) * 0.45:
                break
            centers[column] = center
            extents.append((left, right, center))
            header[column] = " ".join(
                str(word["text"])
                for word in sorted(
                    cell_words, key=lambda item: float(item["x0"])
                )
            )
        else:
            extents.sort()
            separated = all(
                following[0] - previous[1]
                >= max(8, (following[2] - previous[2]) * 0.25)
                for previous, following in zip(extents, extents[1:])
            )
            if separated and _pdf_context_header_is_plausible(header):
                gap = table_top - max(
                    float(word["bottom"]) for word in line_words
                )
                candidates.append((gap, header, centers))

    if not candidates:
        return None
    _, header, centers = min(candidates, key=lambda item: item[0])
    return header, centers


def _rebuild_pdf_table_rows(
    table: Any,
    rows: list[list[str | None]],
    header_centers: dict[int, float],
    words: list[dict[str, Any]],
) -> list[list[str | None]] | None:
    columns = max((len(row) for row in rows), default=0)
    data_centers = sorted(header_centers.items())
    if columns < 3 or len(data_centers) < 2:
        return None
    edges = _pdf_table_column_edges(table, columns)
    if edges is None:
        return None

    first_data_column, first_data_center = data_centers[0]
    label_boundary = edges[1]
    if first_data_column == 0 or first_data_center <= label_boundary:
        return None
    data_positions = [center for _, center in data_centers]
    spacing = min(
        right - left
        for left, right in zip(data_positions, data_positions[1:])
    )
    threshold = spacing * 0.5
    rebuilt_rows: list[list[str | None]] = []
    for row_index, table_row in enumerate(table.rows[: len(rows)]):
        populated_cells = [cell for cell in table_row.cells if cell is not None]
        if (
            len(populated_cells) != 1
            or populated_cells[0][2] - populated_cells[0][0]
            < (edges[-1] - edges[0]) * 0.75
        ):
            rebuilt_rows.append(list(rows[row_index]))
            continue
        row_top = min(float(cell[1]) for cell in populated_cells)
        row_bottom = max(float(cell[3]) for cell in populated_cells)
        row_words = [
            word
            for word in words
            if row_top <= float(word["top"])
            and float(word["bottom"]) <= row_bottom
            and edges[0] <= float(word["x0"])
            and float(word["x1"]) <= edges[-1]
        ]
        reconstructed: list[list[str]] = [[] for _ in range(columns)]
        for _, line_words in _pdf_words_by_line(row_words):
            line_cells: list[list[str]] = [[] for _ in range(columns)]
            for word in line_words:
                center = (float(word["x0"]) + float(word["x1"])) / 2
                nearest_column, nearest_center = min(
                    data_centers,
                    key=lambda item: abs(item[1] - center),
                )
                numeric_token = bool(
                    re.fullmatch(
                        r"[$€£¥₹₩₽¢]|\(?[+-]?\d[\d,]*(?:\.\d+)?%?\)?",
                        str(word["text"]),
                    )
                )
                if (
                    abs(nearest_center - center) <= threshold
                    and (numeric_token or center >= label_boundary)
                ):
                    line_cells[nearest_column].append(str(word["text"]))
                else:
                    line_cells[0].append(str(word["text"]))
            for column, cell_words in enumerate(line_cells):
                if cell_words:
                    reconstructed[column].append(" ".join(cell_words))
        recovered = [
            "\n".join(cell).strip() or None for cell in reconstructed
        ]
        if sum(bool(cell) for cell in recovered) >= 2:
            rebuilt_rows.append(recovered)
        else:
            rebuilt_rows.append(
                list(rows[row_index][:columns])
                + [None] * max(0, columns - len(rows[row_index]))
            )
    return rebuilt_rows


def _pdf_table_caption_from_page(
    page: Any,
    table: Any,
    words: list[dict[str, Any]] | None = None,
) -> str | None:
    table_left, table_top, table_right, _ = map(float, table.bbox)
    table_center = (table_left + table_right) / 2
    candidates: list[tuple[float, str]] = []
    for _, line_words in _pdf_words_by_line(
        page.extract_words() if words is None else words
    ):
        line_bottom = max(float(word["bottom"]) for word in line_words)
        gap = table_top - line_bottom
        if not 0 <= gap <= 90:
            continue
        text = " ".join(
            str(word["text"])
            for word in sorted(
                line_words, key=lambda item: float(item["x0"])
            )
        ).strip()
        if not text:
            continue
        if re.match(
            r"^(?:table|exhibit|schedule|figure|chart)\b", text, re.I
        ):
            if re.match(
                r"^(?:table|exhibit|schedule|figure|chart)\s+of\s+",
                text,
                re.IGNORECASE,
            ):
                continue
            candidates.append((gap, text))
            continue
        letters = re.sub(r"[^A-Za-z]", "", text)
        line_center = (
            min(float(word["x0"]) for word in line_words)
            + max(float(word["x1"]) for word in line_words)
        ) / 2
        if (
            len(text.split()) >= 2
            and letters
            and letters.isupper()
            and not re.search(r"\d", text)
            and abs(line_center - table_center)
            <= (table_right - table_left) * 0.1
        ):
            candidates.append((gap, text))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def _pdfplumber_tables_by_page(
    source: Path,
    warnings: list[str] | None = None,
) -> dict[int, list[dict[str, Any]]]:
    import pdfplumber

    tables_by_page: dict[int, list[dict[str, Any]]] = {}
    with pdfplumber.open(source) as pdf:
        for page_index, page in enumerate(pdf.pages):
            try:
                detected_tables = page.find_tables()
            except Exception as error:
                raise RuntimeError(
                    f"Table detection failed on PDF page "
                    f"{page_index + 1}: {error}"
                ) from error
            tables: list[dict[str, Any]] = []
            page_words = page.extract_words() if detected_tables else []
            for table in detected_tables:
                try:
                    raw_rows = table.extract()
                except Exception as error:
                    raise RuntimeError(
                        f"Table extraction failed on PDF page "
                        f"{page_index + 1}: {error}"
                    ) from error
                rows, repaired_rows = _repair_pdf_table_rows(
                    page, table, raw_rows, page_words,
                )
                columns = max((len(row) for row in rows), default=0)
                context_header = _pdf_table_context_header(
                    table, columns, page_words,
                )
                if context_header is not None:
                    rebuilt_rows = _rebuild_pdf_table_rows(
                        table, rows, context_header[1], page_words,
                    )
                    if rebuilt_rows is not None:
                        repaired_rows = sorted(
                            set(repaired_rows)
                            | {
                                row_index
                                for row_index, (original, rebuilt)
                                in enumerate(zip(rows, rebuilt_rows))
                                if original != rebuilt
                            }
                        )
                        rows = rebuilt_rows
                    header_row = context_header[0]
                    if (
                        _pdf_context_header_is_plausible(header_row)
                        and not _pdf_context_header_matches_rows(
                            header_row, rows
                        )
                    ):
                        rows = [header_row] + rows
                        repaired_rows.append(-1)
                if not _table_rows_are_plausible(rows):
                    columns = max(
                        (len(row) for row in rows), default=0
                    )
                    if warnings is not None and columns >= 2:
                        warnings.append(
                            "Skipped a low-confidence table-like structure "
                            f"on PDF page {page_index + 1}"
                        )
                    continue
                tables.append(
                    {
                        "bbox": table.bbox,
                        "page_height": page.height,
                        "caption": _pdf_table_caption_from_page(
                            page, table, page_words,
                        ),
                        "rows": rows,
                        "raw_rows": raw_rows,
                        "repair_notes": [
                            (
                                "Column headings recovered from PDF text "
                                "above the table"
                                if row_index == -1
                                else
                            "Cells reconstructed from PDF word positions in "
                                f"source row {row_index + 1}"
                            )
                            for row_index in repaired_rows
                        ],
                        "table_index": len(tables) + 1,
                    }
                )
            if tables:
                tables_by_page[page_index] = tables
    return tables_by_page


class _HTMLTableRowsParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell_parts: list[str] | None = None

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag == "tr":
            self._finish_row()
            self._row = []
        elif tag in {"th", "td"} and self._row is not None:
            self._cell_parts = []
        elif tag == "br" and self._cell_parts is not None:
            self._cell_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"th", "td"} and self._row is not None:
            self._finish_cell()
        elif tag == "tr":
            self._finish_row()

    def handle_data(self, data: str) -> None:
        if self._cell_parts is not None:
            self._cell_parts.append(data)

    def close(self) -> None:
        super().close()
        self._finish_row()

    def _finish_cell(self) -> None:
        if self._row is None or self._cell_parts is None:
            return
        text = re.sub(
            r"[ \t\r\f\v]+", " ", "".join(self._cell_parts)
        )
        self._row.append(text.strip())
        self._cell_parts = None

    def _finish_row(self) -> None:
        if self._row is None:
            return
        self._finish_cell()
        if self._row and any(cell for cell in self._row):
            self.rows.append(self._row)
        self._row = None


def _unstructured_table_bbox(
    coordinates: Any,
    page_width: float,
    page_height: float,
) -> tuple[float, float, float, float] | None:
    points = getattr(coordinates, "points", None)
    system = getattr(coordinates, "system", None)
    width = getattr(system, "width", None)
    height = getattr(system, "height", None)
    orientation = getattr(system, "orientation", None)
    orientation_value = getattr(orientation, "value", None)
    if (
        not points
        or not isinstance(width, (int, float))
        or not isinstance(height, (int, float))
        or width <= 0
        or height <= 0
        or not isinstance(orientation_value, (tuple, list))
        or len(orientation_value) != 2
    ):
        return None

    try:
        points_xy = [
            (float(point[0]), float(point[1]))
            for point in points
            if len(point) >= 2
        ]
    except (TypeError, ValueError, IndexError):
        return None
    if len(points_xy) < 3:
        return None

    x_orientation, y_orientation = orientation_value
    x_values = [
        x * page_width / width
        if x_orientation >= 0
        else page_width - x * page_width / width
        for x, _ in points_xy
    ]
    y_values = [
        y * page_height / height
        if y_orientation < 0
        else page_height - y * page_height / height
        for _, y in points_xy
    ]
    bbox = (
        min(x_values),
        min(y_values),
        max(x_values),
        max(y_values),
    )
    return bbox if bbox[2] > bbox[0] and bbox[3] > bbox[1] else None


def _unstructured_tables_by_page(
    source: Path,
    warnings: list[str],
    pages: set[int],
) -> dict[int, list[dict[str, Any]]]:
    try:
        pdf_partition = importlib.import_module(
            "unstructured.partition.pdf"
        )
        partition_pdf = getattr(pdf_partition, "partition_pdf", None)
        if not callable(partition_pdf):
            raise RuntimeError("Unstructured does not expose partition_pdf")
        languages = [
            language
            for language in re.split(
                r"[+,]",
                os.environ.get(
                    "PDF_OCR_LANGUAGES", PDF_OCR_LANGUAGES
                ),
            )
            if language
        ]
        elements = partition_pdf(
            filename=str(source),
            strategy="hi_res",
            infer_table_structure=True,
            languages=languages,
        )
    except Exception as error:
        warnings.append(
            f"Unstructured table fallback failed: {error}; "
            "pdfplumber results were retained"
        )
        return {}

    import pymupdf

    tables_by_page: dict[int, list[dict[str, Any]]] = {}
    with pymupdf.open(source) as document:
        for element in elements:
            if getattr(element, "category", None) != "Table":
                continue
            metadata = getattr(element, "metadata", None)
            page_number = getattr(metadata, "page_number", None)
            if (
                not isinstance(page_number, int)
                or page_number not in {page + 1 for page in pages}
                or page_number > len(document)
            ):
                continue
            page_index = page_number - 1
            page = document[page_index]
            bbox = _unstructured_table_bbox(
                getattr(metadata, "coordinates", None),
                page.rect.width,
                page.rect.height,
            )
            html = getattr(metadata, "text_as_html", None)
            if bbox is None or not isinstance(html, str) or not html.strip():
                warnings.append(
                    f"Unstructured table on PDF page {page_number} had no "
                    "usable coordinates or HTML structure and was skipped"
                )
                continue
            parser = _HTMLTableRowsParser()
            parser.feed(html)
            parser.close()
            if not parser.rows:
                warnings.append(
                    f"Unstructured table on PDF page {page_number} had no "
                    "readable rows and was skipped"
                )
                continue
            page_tables = tables_by_page.setdefault(page_index, [])
            page_tables.append(
                {
                    "bbox": bbox,
                    "page_height": page.rect.height,
                    "rows": parser.rows,
                    "html": html,
                    "table_index": len(page_tables) + 1,
                    "extractor": "unstructured",
                }
            )
    return tables_by_page


def _bbox_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    first_area = max(0.0, first[2] - first[0]) * max(
        0.0, first[3] - first[1]
    )
    second_area = max(0.0, second[2] - second[0]) * max(
        0.0, second[3] - second[1]
    )
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _table_transformer_tables_by_page(
    source: Path,
    warnings: list[str],
    pages: set[int] | None = None,
) -> dict[int, list[dict[str, Any]]]:
    import pymupdf

    # FIX: absolute import only. Relative import fails when run as a script.
    try:
        from table_transformer import TableTransformerExtractor
    except ImportError as exc:
        raise RuntimeError(
            "table_transformer.py not found next to the script; install it "
            "or set PDF_TABLE_BACKEND=pdfplumber"
        ) from exc

    extractor = TableTransformerExtractor()
    tables_by_page: dict[int, list[dict[str, Any]]] = {}
    with pymupdf.open(source) as document:
        for page_index, page in enumerate(document):
            if pages is not None and page_index not in pages:
                continue
            try:
                predictions = extractor.extract_page(page)
            except Exception as error:
                warnings.append(
                    f"Page {page_index + 1} Table Transformer failed: "
                    f"{error}"
                )
                continue
            if predictions:
                tables_by_page[page_index] = [
                    {
                        "bbox": prediction.bbox,
                        "page_height": page.rect.height,
                        "rows": prediction.rows,
                        "html": prediction.html,
                        "confidence": prediction.confidence,
                        "detection_score": prediction.detection_score,
                        "structure_score": prediction.structure_score,
                        "word_coverage": prediction.word_coverage,
                        "table_index": index,
                        "extractor": "table-transformer",
                    }
                    for index, prediction in enumerate(
                        predictions, start=1
                    )
                ]
    return tables_by_page


def _page_has_dot_leader_rows(page: Any) -> bool:
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


def _page_has_aligned_numeric_column(page: Any) -> bool:
    lines: dict[tuple[int, int], list[tuple[Any, ...]]] = {}
    for word in page.get_text("words"):
        if len(word) < 8:
            continue
        y0 = float(word[1])
        if not (page.rect.height * 0.1 <= y0 <= page.rect.height * 0.9):
            continue
        lines.setdefault((int(word[5]), int(word[6])), []).append(word)

    aligned_columns: dict[int, set[str]] = {}
    for words in lines.values():
        words.sort(key=lambda word: float(word[0]))
        if len(words) < 2:
            continue
        value = re.sub(r"[,.;:()\[\]]+$", "", str(words[-1][4]).strip())
        if not re.fullmatch(r"[-+]?\d[\d,]*(?:\.\d+)?%?", value):
            continue
        right_edge_bin = round(float(words[-1][2]) / 12)
        aligned_columns.setdefault(right_edge_bin, set()).add(value)
    return any(
        len(values) >= 5 and len(set(values)) >= 3
        for values in aligned_columns.values()
    )


def _table_transformer_candidate_pages(
    source: Path,
    fallback_pages: set[int],
    requested_pages: set[int] | None = None,
) -> set[int]:
    import pymupdf

    candidates = set(fallback_pages)
    with pymupdf.open(source) as document:
        for page_index, page in enumerate(document):
            if (
                requested_pages is not None
                and page_index not in requested_pages
            ):
                continue
            if page_index in candidates:
                continue
            native_text_chars, image_coverage = _pdf_page_content_metrics(
                page
            )
            if (
                _pdf_page_needs_ocr(native_text_chars, image_coverage)
                or _page_has_dot_leader_rows(page)
                or _page_has_aligned_numeric_column(page)
            ):
                candidates.add(page_index)
    if requested_pages is not None:
        candidates.intersection_update(requested_pages)
    return candidates


def _merge_table_transformer_predictions(
    fallback: dict[int, list[dict[str, Any]]],
    transformed: dict[int, list[dict[str, Any]]],
    replacement_threshold: float = 0.55,
    addition_threshold: float = 0.75,
) -> dict[int, list[dict[str, Any]]]:
    merged = {
        page_index: [dict(table) for table in tables]
        for page_index, tables in fallback.items()
    }
    for page_index, predictions in transformed.items():
        page_tables = merged.setdefault(page_index, [])
        for prediction in predictions:
            matches = [
                (_bbox_iou(table["bbox"], prediction["bbox"]), index)
                for index, table in enumerate(page_tables)
            ]
            overlap, match_index = max(matches, default=(0.0, -1))
            if (
                overlap >= 0.35
                and prediction["confidence"] >= replacement_threshold
            ):
                page_tables[match_index].update(
                    {
                        "rows": prediction["rows"],
                        "html": prediction["html"],
                        "confidence": prediction["confidence"],
                        "detection_score": prediction["detection_score"],
                        "structure_score": prediction["structure_score"],
                        "word_coverage": prediction["word_coverage"],
                        "extractor": "table-transformer",
                    }
                )
            elif (
                overlap < 0.1
                and prediction["confidence"] >= addition_threshold
            ):
                prediction = dict(prediction)
                prediction["table_index"] = len(page_tables) + 1
                page_tables.append(prediction)
    for page_tables in merged.values():
        for index, table in enumerate(page_tables, start=1):
            table["table_index"] = index
    return merged


def _table_structure_quality(table: dict[str, Any]) -> float:
    rows = table.get("rows", [])
    if not rows:
        return 0.0
    columns = max((len(row) for row in rows), default=0)
    if columns < 2:
        return 0.0
    width_consistency = sum(len(row) == columns for row in rows) / len(rows)
    cell_count = sum(len(row) for row in rows)
    populated = sum(
        bool(cell and cell.strip()) for row in rows for cell in row
    )
    density = populated / max(1, cell_count)
    return 0.6 * width_consistency + 0.4 * density


def _unstructured_candidate_pages(
    source: Path,
    tables_by_page: dict[int, list[dict[str, Any]]],
) -> set[int]:
    import pymupdf

    candidates = {
        page_index
        for page_index, tables in tables_by_page.items()
        if any(
            _table_complexity(table["rows"])["is_complex"]
            for table in tables
        )
    }
    with pymupdf.open(source) as document:
        if not tables_by_page:
            return set(range(len(document)))
        for page_index, page in enumerate(document):
            if page_index in tables_by_page:
                continue
            native_text_chars, image_coverage = _pdf_page_content_metrics(
                page
            )
            if (
                _pdf_page_needs_ocr(native_text_chars, image_coverage)
                or _page_has_dot_leader_rows(page)
                or _page_has_aligned_numeric_column(page)
            ):
                candidates.add(page_index)
    return candidates


def _merge_unstructured_fallback(
    primary: dict[int, list[dict[str, Any]]],
    fallback: dict[int, list[dict[str, Any]]],
    warnings: list[str],
) -> dict[int, list[dict[str, Any]]]:
    merged = {
        page: [dict(table) for table in tables]
        for page, tables in primary.items()
    }
    for page_index, candidates in fallback.items():
        page_tables = merged.setdefault(page_index, [])
        for candidate in candidates:
            overlaps = [
                (_bbox_iou(table["bbox"], candidate["bbox"]), index)
                for index, table in enumerate(page_tables)
            ]
            overlap, match_index = max(overlaps, default=(0.0, -1))
            if overlap < 0.2:
                candidate["table_index"] = len(page_tables) + 1
                page_tables.append(candidate)
                warnings.append(
                    f"Unstructured supplied an additional table on PDF page "
                    f"{page_index + 1}; verify it against the source"
                )
                continue

            existing = page_tables[match_index]
            candidate_quality = _table_structure_quality(candidate)
            existing_quality = _table_structure_quality(existing)
            if candidate_quality > existing_quality + 0.1:
                candidate["table_index"] = existing["table_index"]
                candidate["alternative"] = {
                    "extractor": existing.get("extractor", "pdfplumber"),
                    "structure_score": round(existing_quality, 3),
                    "rows": existing["rows"],
                    "html": existing.get("html"),
                }
                page_tables[match_index] = candidate
                warnings.append(
                    f"Unstructured replaced a structurally weaker table "
                    f"on PDF page {page_index + 1}; verify the result"
                )
            else:
                existing["alternative"] = {
                    "extractor": "unstructured",
                    "structure_score": round(candidate_quality, 3),
                    "rows": candidate["rows"],
                    "html": candidate["html"],
                }
                warnings.append(
                    f"Unstructured alternative on PDF page "
                    f"{page_index + 1} was retained for audit; "
                    "pdfplumber output was used"
                )
    for page_tables in merged.values():
        for index, table in enumerate(page_tables, start=1):
            table["table_index"] = index
    return merged


def _pdf_tables_by_page(
    source: Path,
    backend: str,
    warnings: list[str],
    pages: set[int] | None = None,
) -> dict[int, list[dict[str, Any]]]:
    fallback_backend = _configured_pdf_table_fallback()
    pdfplumber_error: Exception | None = None
    try:
        pdfplumber_tables = _pdfplumber_tables_by_page(source, warnings)
    except Exception as error:
        if fallback_backend != "unstructured":
            raise
        pdfplumber_error = error
        pdfplumber_tables = {}
        warnings.append(
            f"pdfplumber table extraction failed: {error}; "
            "trying the configured Unstructured fallback"
        )

    if backend == "pdfplumber" or pdfplumber_error is not None:
        tables = pdfplumber_tables
    else:
        fallback = pdfplumber_tables
        candidate_pages = _table_transformer_candidate_pages(
            source, set(fallback), pages,
        )
        transformed = _table_transformer_tables_by_page(
            source, warnings, candidate_pages,
        )
        if backend == "table-transformer":
            tables = {
                page_index: transformed.get(page_index) or page_tables
                for page_index, page_tables in fallback.items()
            } | {
                page_index: page_tables
                for page_index, page_tables in transformed.items()
                if page_index not in fallback
            }
        else:
            tables = _merge_table_transformer_predictions(
                fallback, transformed,
            )

    if fallback_backend == "unstructured":
        if pdfplumber_error is not None:
            import pymupdf

            with pymupdf.open(source) as document:
                candidate_pages = set(range(len(document)))
        else:
            candidate_pages = _unstructured_candidate_pages(source, tables)
        if candidate_pages:
            unstructured_tables = _unstructured_tables_by_page(
                source, warnings, candidate_pages,
            )
            tables = _merge_unstructured_fallback(
                tables, unstructured_tables, warnings,
            )
        if pdfplumber_error is not None and not tables:
            raise RuntimeError(
                "pdfplumber failed and Unstructured did not return usable "
                f"tables: {pdfplumber_error}"
            )
    return tables


def _mark_pdf_table_continuations(
    tables_by_page: dict[int, list[dict[str, Any]]],
) -> list[tuple[int, int]]:
    continuations: list[tuple[int, int]] = []
    for page_index, previous_tables in tables_by_page.items():
        next_page_index = page_index + 1
        next_tables = tables_by_page.get(next_page_index, [])
        if not next_tables:
            continue

        previous_candidates = sorted(
            (
                table
                for table in previous_tables
                if table["bbox"][3] >= table["page_height"] - 100
            ),
            key=lambda table: table["bbox"][3],
            reverse=True,
        )
        next_candidates = sorted(
            (
                table
                for table in next_tables
                if table["bbox"][1] <= 100
            ),
            key=lambda table: table["bbox"][1],
        )
        for previous in previous_candidates:
            previous_columns = max(
                (len(row) for row in previous["rows"]), default=0
            )
            previous_header = _table_header_signature(previous["rows"])
            if not previous_header:
                continue
            for following in next_candidates:
                following_columns = max(
                    (len(row) for row in following["rows"]), default=0
                )
                if (
                    previous_columns != following_columns
                    or previous_header
                    != _table_header_signature(following["rows"])
                ):
                    continue
                previous["continues_to_page"] = next_page_index + 1
                following["continued_from_page"] = page_index + 1
                following["merged_into_page"] = page_index + 1
                following["merged_into_table"] = previous["table_index"]
                continuations.append((page_index + 1, next_page_index + 1))
                next_candidates.remove(following)
                break
    return continuations


def _merge_pdf_table_continuations(
    tables_by_page: dict[int, list[dict[str, Any]]],
) -> None:
    for page_index, tables in sorted(tables_by_page.items()):
        for table in tables:
            previous_page = table.get("merged_into_page")
            previous_table_index = table.get("merged_into_table")
            if (
                not isinstance(previous_page, int)
                or not isinstance(previous_table_index, int)
            ):
                continue
            previous_tables = tables_by_page.get(previous_page - 1, [])
            if not 1 <= previous_table_index <= len(previous_tables):
                raise RuntimeError(
                    f"PDF table continuation on page {page_index + 1} "
                    "references a missing preceding table"
                )
            previous = previous_tables[previous_table_index - 1]
            previous_header = _table_header_signature(previous["rows"])
            current_header = _table_header_signature(table["rows"])
            if not previous_header or previous_header != current_header:
                raise RuntimeError(
                    f"PDF table continuation on page {page_index + 1} "
                    "has inconsistent headers"
                )
            root_page = previous_page
            root = previous
            while "merged_into_page" in root:
                root_page = root["merged_into_page"]
                root_table_index = root["merged_into_table"]
                root_tables = tables_by_page.get(root_page - 1, [])
                if not 1 <= root_table_index <= len(root_tables):
                    raise RuntimeError(
                        f"PDF table continuation on page {page_index + 1} "
                        "references a missing root table"
                    )
                root = root_tables[root_table_index - 1]
            table["merged_into_page"] = root_page
            table["merged_into_table"] = root["table_index"]
            continuation_rows = table["rows"][1:]
            root["rows"].extend(continuation_rows)
            table["merged_row_count"] = len(continuation_rows)


def _table_header_signature(
    rows: list[list[str | None]],
) -> tuple[str, ...]:
    if len(rows) < 2:
        return ()
    columns = max((len(row) for row in rows[:2]), default=0)
    if columns < 2:
        return ()
    signature = tuple(
        " ".join((rows[0][index] or "").split()).casefold()
        if index < len(rows[0])
        else ""
        for index in range(columns)
    )
    populated_signature = [cell for cell in signature if cell]
    if (
        len(populated_signature) >= 2
        and all(
            re.fullmatch(r"(?:19|20)\d{2}", cell)
            for cell in populated_signature
        )
    ):
        return signature
    second_row = [
        " ".join((rows[1][index] or "").split()).casefold()
        if index < len(rows[1])
        else ""
        for index in range(columns)
    ]
    if any(not cell for cell in signature) or len(set(signature)) != columns:
        return ()

    def numeric_ratio(cells: list[str]) -> float:
        populated = [cell for cell in cells if cell]
        if not populated:
            return 0
        numeric = sum(bool(re.search(r"\d", cell)) for cell in populated)
        return numeric / len(populated)

    first_numeric_ratio = numeric_ratio(list(signature))
    second_numeric_ratio = numeric_ratio(second_row)
    second_row_count = sum(bool(cell) for cell in second_row)
    has_sparse_subheader = 0 < second_row_count < columns and len(rows) >= 3
    body_rows = rows[2:] if has_sparse_subheader else rows[1:]
    body_numeric_ratios = [
        numeric_ratio(
            [
                " ".join((row[index] or "").split()).casefold()
                if index < len(row)
                else ""
                for index in range(columns)
            ]
        )
        for row in body_rows
    ]
    has_numeric_body = any(ratio >= 0.5 for ratio in body_numeric_ratios)
    has_structural_header = (
        first_numeric_ratio <= 0.25 and second_numeric_ratio >= 0.5
    ) or (has_sparse_subheader and has_numeric_body) or (
        not has_sparse_subheader
        and has_numeric_body
        and max(body_numeric_ratios, default=0) > first_numeric_ratio
    )
    if not has_structural_header:
        return ()
    if not any(re.search(r"[a-z]", cell) for cell in signature):
        return ()
    return signature


def _normalize_pdf_header(text: str) -> str:
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<!--.*?-->", "", text)
    text = re.sub(r"<[^>]*>", "", text)
    text = text.replace("|", " ")
    text = re.sub(r"[#*_`]", "", text)
    text = text.replace("\u00ad", "")
    text = re.sub(r"\d+", "#", text)
    return " ".join(text.split()).casefold()


def _repeated_pdf_furniture(document: Any) -> set[str]:
    page_count = len(document)
    if page_count < 3:
        return set()
    furniture_pages: dict[str, set[int]] = {}
    for page_index, page in enumerate(document):
        page_furniture: set[str] = set()
        for block in page.get_text("dict").get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                _, y0, _, y1 = line["bbox"]
                if not (
                    y1 <= page.rect.height * PAGE_FURNITURE_EDGE_FRACTION
                    or y0
                    >= page.rect.height
                    * (1 - PAGE_FURNITURE_EDGE_FRACTION)
                ):
                    continue
                text = " ".join(
                    span.get("text", "") for span in line.get("spans", [])
                ).strip()
                normalized = _normalize_pdf_header(text)
                if (
                    len(normalized) < 4 and normalized.count("#") < 2
                ) or normalized.isdigit():
                    continue
                page_furniture.add(normalized)
        for item in page_furniture:
            furniture_pages.setdefault(item, set()).add(page_index)

    minimum_repetitions = max(3, (page_count + 3) // 4)
    return {
        item
        for item, pages in furniture_pages.items()
        if len(pages) >= minimum_repetitions
    }


def _remove_repeated_pdf_furniture(
    markdown: str, repeated_furniture: set[str]
) -> str:
    if not repeated_furniture:
        return markdown
    lines = markdown.splitlines()
    filtered_lines: list[str] = []
    removed_table_row = False
    for line in lines:
        if re.fullmatch(r"\s*\|(?:\s*:?-+:?\s*\|)+\s*", line):
            if removed_table_row:
                removed_table_row = False
                continue
            filtered_lines.append(line)
            continue
        if re.match(r"^\s*\|.*\|\s*$", line):
            cells = [cell for cell in line.strip().strip("|").split("|")]
            normalized_cells = [
                _normalize_pdf_header(cell)
                for cell in cells
                if _normalize_pdf_header(cell)
            ]
            if normalized_cells and all(
                cell in repeated_furniture for cell in normalized_cells
            ):
                removed_table_row = True
                continue
        normalized = _normalize_pdf_header(line)
        if normalized and normalized in repeated_furniture:
            continue
        removed_table_row = False
        filtered_lines.append(line)
    return "\n".join(filtered_lines)


def _format_pdf_footnotes(markdown: str) -> str:
    definition_pattern = re.compile(r"(?m)^>\s*(\d{1,3})\s+(.+)$")
    footnote_numbers = {
        match.group(1) for match in definition_pattern.finditer(markdown)
    }
    if not footnote_numbers:
        return markdown
    markdown = definition_pattern.sub(
        lambda match: f"[^{match.group(1)}]: {match.group(2)}",
        markdown,
    )
    return re.sub(
        r"<sup>\s*(\d{1,3})\s*</sup>",
        lambda match: (
            f"[^{match.group(1)}]"
            if match.group(1) in footnote_numbers
            else match.group(0)
        ),
        markdown,
    )


def _build_toc_headers(document: Any) -> Any:
    import pymupdf4llm

    toc_headers = getattr(pymupdf4llm, "TocHeaders", None)
    if toc_headers is None:
        return None
    return toc_headers(document)


# ============================================================ PDF conversion
def convert_pdf_in_batches(
    source: Path,
    markdown_path: Path,
    image_dir: Path,
    token: str,
) -> tuple[str, list[str]]:
    import pymupdf
    import pymupdf4llm

    warnings = pdf_ocr_warnings(source)
    ocr_backend = _configured_pdf_ocr_backend()
    tables_by_page = _pdf_tables_by_page(
        source,
        _configured_pdf_table_backend(),
        warnings,
    )
    table_continuations = _mark_pdf_table_continuations(tables_by_page)
    _merge_pdf_table_continuations(tables_by_page)
    warnings.extend(
        f"Tables on PDF pages {previous_page} and {next_page} have matching "
        "headers at the page boundary and were joined; a page citation "
        "remains at each continuation page"
        for previous_page, next_page in table_continuations
    )
    progress_dir = markdown_path.parent / ".progress" / token
    progress_dir.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    markdown_pages: list[str] = []
    with pymupdf.open(source) as document:
        total_pages = len(document)
        if total_pages == 0:
            raise RuntimeError("PDF contains no pages")
        if len(tables_by_page) and max(tables_by_page) >= total_pages:
            raise RuntimeError(
                "PDF table detector returned an invalid page index"
            )
        toc_headers = _build_toc_headers(document)
        # PATCH: consistent printed-page labels
        printed_page_labels = consistent_printed_labels(
            [printed_pdf_page_number(page) for page in document]
        )
        repeated_furniture = _repeated_pdf_furniture(document)
        heading_font_sizes = _pdf_heading_font_sizes(document)

        for first_page in range(total_pages):
            page_number = first_page + 1
            batch_name = f"pages_{page_number:04}-{page_number:04}.md"
            checkpoint = progress_dir / batch_name
            cached_markdown: str | None = None
            if checkpoint.is_file():
                try:
                    existing = checkpoint.read_text(encoding="utf-8")
                    if _markdown_images_exist(
                        existing, markdown_path.parent, image_dir
                    ):
                        cached_markdown = existing
                except OSError:
                    cached_markdown = None

            if cached_markdown is None:
                page = document[first_page]
                native_text_chars, image_coverage = (
                    _pdf_page_content_metrics(page)
                )
                native_page_text = page.get_text("text")
                ocr_text = ""
                if _pdf_page_needs_ocr(
                    native_text_chars, image_coverage
                ):
                    if ocr_backend != "none":
                        ocr_text = _run_page_ocr(
                            page,
                            native_page_text,
                            ocr_backend,
                            page_number,
                            warnings,
                        )
                    if ocr_text:
                        warnings = [
                            warning
                            for warning in warnings
                            if not warning.startswith(
                                f"Page {page_number} has "
                            )
                        ]
                    elif ocr_backend == "none":
                        warnings.append(
                            f"Page {page_number} requires OCR, but "
                            "PDF_OCR_BACKEND is set to none"
                        )
                page_tables = tables_by_page.get(first_page, [])
                for table_index, table in enumerate(
                    page_tables, start=1
                ):
                    table_id = f"{page_number:04}_{table_index:02}"
                    placeholder = (
                        f"{TABLE_PLACEHOLDER_PREFIX}{table_id}ZZ"
                    )
                    table["placeholder"] = placeholder
                    page.add_redact_annot(
                        pymupdf.Rect(table["bbox"]),
                        text=placeholder,
                        fontsize=6,
                        fill=(1, 1, 1),
                        text_color=(0, 0, 0),
                        cross_out=False,
                    )
                if page_tables:
                    page.apply_redactions(
                        images=pymupdf.PDF_REDACT_IMAGE_NONE,
                        graphics=pymupdf.PDF_REDACT_LINE_ART_REMOVE_IF_TOUCHED,
                    )

                chunks = pymupdf4llm.to_markdown(
                    document,
                    pages=[first_page],
                    hdr_info=toc_headers,
                    write_images=True,
                    image_path=str(image_dir),
                    image_format="png",
                    page_chunks=True,
                    margins=0,
                    use_ocr=False,
                )
                if len(chunks) != 1:
                    raise RuntimeError(
                        f"PDF Markdown conversion returned {len(chunks)} "
                        f"chunks for page {page_number}"
                    )
                page_markdown = chunks[0].get("text", "")
                page_markdown = re.sub(
                    r"(?m)^\s*(?:#{1,6}\s*)?\*{0,4}\d{1,3}\*{0,4}\s*$",
                    "",
                    page_markdown,
                )
                page_markdown = _remove_repeated_pdf_furniture(
                    page_markdown, repeated_furniture
                )
                rendered_tables: list[tuple[str, str, str]] = []
                for table_index, table in enumerate(
                    page_tables, start=1
                ):
                    placeholder = table["placeholder"]
                    if "merged_into_page" in table:
                        table_markdown = (
                            f"*Table continuation on PDF page {page_number}; "
                            f"{table['merged_row_count']} rows were joined "
                            f"to the table on PDF page "
                            f"{table['merged_into_page']}.*"
                        )
                    else:
                        # PATCH: value-table fast path
                        context = page_markdown.split(placeholder)[0]
                        value = value_table_from_rows(
                            table["rows"], context
                        )
                        if value is not None:
                            table_markdown = value.markdown
                            table["value_table"] = value.records
                            warnings.extend(
                                f"Table on PDF page {page_number}: {w}"
                                for w in value.warnings
                            )
                        else:
                            complexity = _table_complexity(table["rows"])
                            (
                                headers,
                                normalized_rows,
                                normalization_warnings,
                            ) = _normalized_table_data(table["rows"])
                            table["complexity"] = complexity
                            table["normalized_headers"] = headers
                            table["normalized_rows"] = normalized_rows
                            table["normalization_warnings"] = (
                                normalization_warnings
                            )
                            warnings.extend(
                                f"Complex table on PDF page "
                                f"{page_number}: {warning}"
                                for warning in normalization_warnings
                            )
                            if complexity["is_complex"]:
                                table_markdown = _render_normalized_table(
                                    headers, normalized_rows
                                )
                                if (
                                    table.get(
                                        "extractor", "pdfplumber"
                                    )
                                    == "pdfplumber"
                                ):
                                    warnings.append(
                                        f"Complex table on PDF page "
                                        f"{page_number} was normalized "
                                        "into row-oriented Markdown; "
                                        "raw extraction is available in "
                                        "the tables sidecar"
                                    )
                            else:
                                table_markdown = _table_to_markdown(
                                    table["rows"]
                                )
                    if "continues_to_page" in table:
                        table_markdown += (
                            f"\n\n*Merged table spans PDF pages "
                            f"{page_number}–"
                            f"{table['continues_to_page']}.*"
                        )
                    rendered_tables.append(
                        (
                            placeholder,
                            f"Table {table_index} on page {page_number}",
                            table_markdown,
                        )
                    )

                page_markdown, inserted_tables = _insert_pdf_tables(
                    page_markdown, rendered_tables,
                )
                for table_index, (
                    placeholder,
                    _label,
                    table_markdown,
                ) in enumerate(rendered_tables, start=1):
                    if placeholder not in inserted_tables:
                        warnings.append(
                            f"Table {table_index} on page {page_number} "
                            "was moved to the end of its page because its "
                            "placeholder was not found in the layout output"
                        )
                        page_markdown = (
                            page_markdown.rstrip()
                            + "\n\n"
                            + table_markdown
                            + "\n"
                        )

                if ocr_text:
                    page_markdown = (
                        page_markdown.rstrip()
                        + f"\n\n### OCR-only text (PDF page {page_number}; "
                        "verify against source)\n\n"
                        + ocr_text
                        + "\n"
                    )

                page_images = sorted(
                    image_dir.glob(f"{source.name}-{page_number:04d}-*")
                )
                image_prefix = image_dir.relative_to(
                    markdown_path.parent
                ).as_posix()
                page_markdown = page_markdown.replace(
                    f"{image_dir}/", f"{image_prefix}/"
                )
                if page_images:
                    page_markdown, image_warnings = _link_pdf_images(
                        page_markdown,
                        page,
                        page_images,
                        image_prefix,
                        page_number,
                    )
                    warnings.extend(image_warnings)
                atomic_write_text(checkpoint, page_markdown)
                cached_markdown = page_markdown

            normalized_markdown = _normalize_pdf_heading_levels(
                _fence_unmarked_pdf_examples(
                    _normalize_pdf_fenced_blocks(cached_markdown)
                ),
                heading_font_sizes,
            )
            if normalized_markdown != cached_markdown:
                atomic_write_text(checkpoint, normalized_markdown)
                cached_markdown = normalized_markdown

            markdown_pages.append(cached_markdown)
            print(
                f"[PDF] checkpointed page {page_number} of {total_pages}",
                flush=True,
            )

    # PATCH: strip running headers/footers and date-only footer lines
    markdown_pages, _removed = strip_repeated_lines(markdown_pages)
    markdown_pages = [clean_page_markdown(page) for page in markdown_pages]

    raw_markdown = PAGE_BREAK_PLACEHOLDER.join(markdown_pages)
    combined_markdown = format_markdown(
        raw_markdown,
        page_label_prefix="PDF page",
        printed_page_labels=printed_page_labels,
    )
    combined_markdown = _format_pdf_footnotes(combined_markdown)
    if not _markdown_images_exist(
        combined_markdown, markdown_path.parent, image_dir
    ):
        raise RuntimeError(
            "PDF Markdown output references missing images"
        )
    atomic_write_text(markdown_path, combined_markdown)

    # PATCH: write .pages.json sidecar
    atomic_write_text(
        markdown_path.with_suffix(".pages.json"),
        json.dumps(build_page_index(combined_markdown), indent=2) + "\n",
    )

    table_artifacts = _pdf_table_artifacts(
        tables_by_page, markdown_pages
    )
    warnings.extend(
        f"PDF page {table['source_pages'][0]} table {table['table_id']} has "
        "unresolved footnote markers: "
        + ", ".join(table["unresolved_footnote_markers"])
        for table in table_artifacts
        if table["unresolved_footnote_markers"]
    )
    tables_path = markdown_path.with_suffix(".tables.json")
    atomic_write_text(
        tables_path,
        json.dumps(
            {
                "schema_version": 1,
                "source_file": source.name,
                "table_backend": _configured_pdf_table_backend(),
                "fallback_backend": _configured_pdf_table_fallback(),
                "tables": table_artifacts,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
    )
    try:
        shutil.rmtree(progress_dir)
    except FileNotFoundError:
        pass
    except OSError as error:
        warnings.append(
            f"Could not remove completed PDF checkpoints in "
            f"{progress_dir}: {error}"
        )
    return combined_markdown, warnings


def convert_file(
    source: Path,
    input_dir: Path,
    workspace: Path,
    converter: Any | None,
    force: bool = False,
) -> dict[str, Any]:
    relative_path = source.relative_to(input_dir)
    relative_text = relative_path.as_posix()
    source_sha256 = sha256_file(source)
    token = output_token(relative_path, source_sha256)
    output_subdir = relative_path.parent

    record: dict[str, Any] = {
        "source": relative_text,
        "source_sha256": source_sha256,
        "format": source.suffix.lower(),
        "format_version": CONVERSION_FORMAT_VERSION,
        "markdown": {"status": "pending"},
    }
    if source.suffix.lower() == ".pdf":
        record["pdf_ocr"] = _pdf_ocr_configuration()
        record["pdf_table"] = {
            "backend": _configured_pdf_table_backend(),
            "fallback": _configured_pdf_table_fallback(),
        }

    markdown_path = (
        workspace / "output" / "markdown" / output_subdir / f"{token}.md"
    )
    tables_path = markdown_path.with_suffix(".tables.json")
    pages_path = markdown_path.with_suffix(".pages.json")
    image_dir = markdown_path.parent / "images" / token
    warnings: list[str] = []
    try:
        if markdown_path.is_file() and not force:
            existing_markdown = markdown_path.read_text(encoding="utf-8")
            # PATCH: sidecars are advisory; only the markdown and images
            # are required for reuse. Missing sidecars trigger a warning.
            outputs_complete = True
            if (
                existing_markdown.strip()
                and outputs_complete
                and _markdown_images_exist(
                    existing_markdown, markdown_path.parent, image_dir
                )
            ):
                markdown = existing_markdown
                print(
                    f"[Resume] Reusing completed Markdown output for {relative_text}",
                    flush=True,
                )
            elif source.suffix.lower() == ".pdf":
                markdown, warnings = convert_pdf_in_batches(
                    source, markdown_path, image_dir, token
                )
            else:
                markdown = ""
        else:
            markdown = ""

        if not markdown:
            if source.suffix.lower() == ".pdf":
                markdown, warnings = convert_pdf_in_batches(
                    source, markdown_path, image_dir, token
                )
            else:
                if source.suffix.lower() == ".xlsx":
                    markdown = _xlsx_to_markdown(source) or ""
                if not markdown:
                    if converter is None:
                        raise RuntimeError(
                            "Docling is required for Office documents"
                        )
                    from docling_core.types.doc import ImageRefMode

                    result = converter.convert(source)
                    markdown = result.document.export_to_markdown(
                        image_mode=ImageRefMode.REFERENCED,
                        image_dir=image_dir,
                        image_uri_prefix=f"images/{token}/",
                        page_break_placeholder=PAGE_BREAK_PLACEHOLDER,
                    )
                if not markdown.strip():
                    raise ValueError("Docling returned empty Markdown")
                markdown = format_markdown(markdown)
                if source.suffix.lower() == ".pptx":
                    try:
                        slide_paths = _render_pptx_slides(source, image_dir)
                    except Exception as error:
                        warnings.append(
                            f"PowerPoint slide rendering failed: {error}; "
                            "Docling text was retained without slide images"
                        )
                    else:
                        markdown = _append_rendered_slides(
                            markdown, slide_paths, markdown_path.parent
                        )
                if not _markdown_images_exist(
                    markdown, markdown_path.parent, image_dir
                ):
                    raise RuntimeError(
                        "Office Markdown output references missing or invalid images"
                    )
                atomic_write_text(markdown_path, markdown)

        normalized_markdown = _normalize_markdown_tables(markdown)
        if normalized_markdown != markdown:
            markdown = normalized_markdown
            atomic_write_text(markdown_path, markdown)
        enriched_markdown = _enrich_markdown_images(
            markdown, markdown_path.parent, warnings
        )
        if enriched_markdown != markdown:
            markdown = enriched_markdown
            atomic_write_text(markdown_path, markdown)

        image_files = sorted(
            path for path in image_dir.rglob("*") if path.is_file()
        )
        record["markdown"] = {
            "status": "converted",
            "format_version": CONVERSION_FORMAT_VERSION,
            "output": relative_output_path(markdown_path, workspace),
            "output_sha256": hashlib.sha256(
                markdown.encode("utf-8")
            ).hexdigest(),
            "images": [
                relative_output_path(path, workspace) for path in image_files
            ],
        }
        if source.suffix.lower() == ".pdf":
            # PATCH: table and page sidecars are advisory; warn if missing
            if not tables_path.is_file():
                warnings.append(
                    "PDF conversion completed without a .tables.json "
                    "sidecar; table audit data is unavailable"
                )
            else:
                try:
                    tables_content = tables_path.read_text(encoding="utf-8")
                    table_document = json.loads(tables_content)
                    record["tables"] = {
                        "status": "converted",
                        "output": relative_output_path(tables_path, workspace),
                        "output_sha256": hashlib.sha256(
                            tables_content.encode("utf-8")
                        ).hexdigest(),
                        "count": len(table_document.get("tables", [])),
                    }
                except (OSError, json.JSONDecodeError) as error:
                    warnings.append(
                        f"PDF table sidecar could not be validated: {error}"
                    )
            if not pages_path.is_file():
                warnings.append(
                    "PDF conversion completed without a .pages.json "
                    "sidecar; page index is unavailable"
                )
            else:
                try:
                    pages_content = pages_path.read_text(encoding="utf-8")
                    page_index = json.loads(pages_content)
                    record["pages"] = {
                        "status": "converted",
                        "output": relative_output_path(
                            pages_path, workspace
                        ),
                        "output_sha256": hashlib.sha256(
                            pages_content.encode("utf-8")
                        ).hexdigest(),
                        "count": len(page_index)
                        if isinstance(page_index, list)
                        else 0,
                    }
                except (OSError, json.JSONDecodeError) as error:
                    warnings.append(
                        f"PDF page index sidecar could not be validated: "
                        f"{error}"
                    )
        if warnings:
            record["warnings"] = warnings
    except Exception as error:
        record["markdown"] = {"status": "failed", "error": str(error)}

    if record["markdown"]["status"] == "failed":
        record["status"] = "failed"
    elif warnings:
        record["status"] = "converted_with_warnings"
    else:
        record["status"] = "converted"
    return record


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    atomic_write_text(
        path, json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )


def write_conversion_ledger(
    path: Path, records: list[dict[str, Any]]
) -> None:
    converted_by_source = {
        record["source"]: record
        for record in records
        if record.get("status") in {"converted", "converted_with_warnings"}
        and isinstance(record.get("source"), str)
    }
    converted = (
        json.dumps(record, ensure_ascii=False)
        for record in converted_by_source.values()
    )
    atomic_write_text(path, "".join(f"{record}\n" for record in converted))


def read_conversion_ledger(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as ledger:
        for line_number, line in enumerate(ledger, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error}"
                ) from error
            if isinstance(record, dict) and isinstance(
                record.get("source"), str
            ):
                records[record["source"]] = record
    return records


def worker_main(
    relative_source: str,
    workspace: Path | None = None,
    force: bool = False,
) -> int:
    workspace = (workspace or Path(__file__).resolve().parents[1]).resolve()
    input_dir = workspace / "input"
    source = (input_dir / relative_source).resolve()
    try:
        source.relative_to(input_dir.resolve())
    except ValueError:
        print(
            f"__INGEST_RESULT__{json.dumps({'source': relative_source, 'status': 'failed', 'error': 'Input path escapes the input directory'})}",
            flush=True,
        )
        return 1
    if (
        not source.is_file()
        or source.suffix.lower() not in SUPPORTED_EXTENSIONS
    ):
        print(
            f"__INGEST_RESULT__{json.dumps({'source': relative_source, 'status': 'failed', 'error': 'Input file is missing or unsupported'})}",
            flush=True,
        )
        return 1

    converter = None
    if source.suffix.lower() != ".pdf":
        try:
            from docling.document_converter import DocumentConverter
        except ImportError as error:
            print(
                f"__INGEST_RESULT__{json.dumps({'source': relative_source, 'status': 'failed', 'error': f'Docling is not installed: {error}'})}",
                flush=True,
            )
            return 1

        try:
            converter = DocumentConverter()
        except Exception as error:
            print(
                f"__INGEST_RESULT__{json.dumps({'source': relative_source, 'status': 'failed', 'error': f'Could not initialize Docling: {error}'})}",
                flush=True,
            )
            return 1

    record = convert_file(
        source, input_dir, workspace, converter, force=force
    )
    for warning in record.get("warnings", []):
        print(f"[PDF] Warning: {warning}", flush=True)
    print(
        f"__INGEST_RESULT__{json.dumps(record, ensure_ascii=False)}",
        flush=True,
    )
    return int(record["status"] == "failed")


def output_is_available(
    record: dict[str, Any], source: Path, workspace: Path
) -> bool:
    if record.get("status") not in {
        "converted", "converted_with_warnings",
    }:
        return False
    if record.get("format_version") != CONVERSION_FORMAT_VERSION:
        return False
    source_sha256 = sha256_file(source)
    if record.get("source_sha256") != source_sha256:
        return False

    markdown_info = record.get("markdown", record.get("docling", {}))
    if not isinstance(markdown_info, dict):
        return False
    try:
        relative_source = source.resolve().relative_to(
            (workspace / "input").resolve()
        )
    except ValueError:
        return False
    expected_markdown_path = (
        workspace
        / "output"
        / "markdown"
        / relative_source.parent
        / f"{output_token(relative_source, source_sha256)}.md"
    )
    if markdown_info.get("output") != relative_output_path(
        expected_markdown_path, workspace
    ):
        return False
    markdown_path = workspace / markdown_info.get("output", "")
    if markdown_info.get("status") != "converted" or not markdown_path.is_file():
        return False
    image_paths = markdown_info.get("images")
    if not isinstance(image_paths, list):
        return False
    try:
        markdown = markdown_path.read_text(encoding="utf-8")
    except OSError:
        return False
    expected_markdown_hash = markdown_info.get("output_sha256")
    if not isinstance(expected_markdown_hash, str):
        return False
    if (
        hashlib.sha256(markdown.encode("utf-8")).hexdigest()
        != expected_markdown_hash
    ):
        return False

    # PATCH: sidecars are advisory, not required. Verify integrity if present.
    if source.suffix.lower() == ".pdf":
        tables_info = record.get("tables")
        if isinstance(tables_info, dict) and tables_info.get("output"):
            tables_path = workspace / str(tables_info["output"])
            if tables_path.is_file():
                try:
                    tables_content = tables_path.read_text(encoding="utf-8")
                except OSError:
                    return False
                if hashlib.sha256(
                    tables_content.encode("utf-8")
                ).hexdigest() != tables_info.get("output_sha256"):
                    return False
        pages_info = record.get("pages")
        if isinstance(pages_info, dict) and pages_info.get("output"):
            pages_path = workspace / str(pages_info["output"])
            if pages_path.is_file():
                try:
                    pages_content = pages_path.read_text(encoding="utf-8")
                except OSError:
                    return False
                if hashlib.sha256(
                    pages_content.encode("utf-8")
                ).hexdigest() != pages_info.get("output_sha256"):
                    return False

    if (
        os.environ.get("ENABLE_DIAGRAM_CONVERSION", "").strip().lower()
        in {"1", "true", "yes"}
        and os.environ.get("GROQ_API_KEY")
        and not _vision_enrichment_is_current(
            markdown, markdown_path.parent
        )
    ):
        return False
    for path in image_paths:
        if not isinstance(path, str) or not (workspace / path).is_file():
            return False
        try:
            image_reference = (
                (workspace / path)
                .relative_to(markdown_path.parent)
                .as_posix()
            )
        except ValueError:
            return False
        if image_reference not in markdown:
            return False

    return True


def recover_completed_output(
    source: Path, input_dir: Path, workspace: Path
) -> dict[str, Any] | None:
    relative_path = source.relative_to(input_dir)
    relative_source = relative_path.as_posix()
    source_sha256 = sha256_file(source)
    token = output_token(relative_path, source_sha256)
    output_subdir = relative_path.parent
    markdown_path = (
        workspace / "output" / "markdown" / output_subdir / f"{token}.md"
    )
    if not markdown_path.is_file():
        return None

    try:
        markdown = markdown_path.read_text(encoding="utf-8")
    except OSError:
        return None

    image_dir = markdown_path.parent / "images" / token
    image_paths = sorted(
        path for path in image_dir.rglob("*") if path.is_file()
    )
    for image_uri in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", markdown):
        if image_uri.startswith(("http://", "https://", "data:")):
            continue
        image_path = (
            markdown_path.parent / image_uri.split("#", 1)[0]
        ).resolve()
        try:
            image_path.relative_to(image_dir.resolve())
        except ValueError:
            return None
        if not image_path.is_file():
            return None

    record: dict[str, Any] = {
        "source": relative_source,
        "source_sha256": source_sha256,
        "format": source.suffix.lower(),
        "format_version": CONVERSION_FORMAT_VERSION,
        "markdown": {
            "status": "converted",
            "format_version": CONVERSION_FORMAT_VERSION,
            "output": relative_output_path(markdown_path, workspace),
            "output_sha256": hashlib.sha256(
                markdown.encode("utf-8")
            ).hexdigest(),
            "images": [
                relative_output_path(path, workspace) for path in image_paths
            ],
        },
        "status": "converted",
    }
    if source.suffix.lower() == ".pdf":
        record["pdf_ocr"] = _pdf_ocr_configuration()
        record["pdf_table"] = {
            "backend": _configured_pdf_table_backend(),
            "fallback": _configured_pdf_table_fallback(),
        }
        # PATCH: record sidecars if present; do not require them
        tables_path = markdown_path.with_suffix(".tables.json")
        if tables_path.is_file():
            try:
                tables_content = tables_path.read_text(encoding="utf-8")
                table_document = json.loads(tables_content)
                if isinstance(table_document, dict) and isinstance(
                    table_document.get("tables"), list
                ):
                    record["tables"] = {
                        "status": "converted",
                        "output": relative_output_path(
                            tables_path, workspace
                        ),
                        "output_sha256": hashlib.sha256(
                            tables_content.encode("utf-8")
                        ).hexdigest(),
                        "count": len(table_document["tables"]),
                    }
            except (OSError, json.JSONDecodeError):
                pass
        pages_path = markdown_path.with_suffix(".pages.json")
        if pages_path.is_file():
            try:
                pages_content = pages_path.read_text(encoding="utf-8")
                page_index = json.loads(pages_content)
                if isinstance(page_index, list):
                    record["pages"] = {
                        "status": "converted",
                        "output": relative_output_path(
                            pages_path, workspace
                        ),
                        "output_sha256": hashlib.sha256(
                            pages_content.encode("utf-8")
                        ).hexdigest(),
                        "count": len(page_index),
                    }
            except (OSError, json.JSONDecodeError):
                pass

    return (
        record
        if output_is_available(record, source, workspace)
        else None
    )


def run_worker(
    source: Path,
    input_dir: Path,
    workspace: Path,
    force: bool = False,
) -> dict[str, Any]:
    relative_source = source.relative_to(input_dir).as_posix()
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        relative_source,
        "--workspace",
        str(workspace.resolve()),
    ]
    if force:
        command.append("--force")
    print(f"[Worker] Starting {relative_source}", flush=True)
    # PATCH: timeout-guarded subprocess
    result, return_code, timed_out = run_command_with_timeout(
        command, workspace, worker_timeout_seconds(), relative_source
    )
    if timed_out:
        return {
            "source": relative_source,
            "format": source.suffix.lower(),
            "source_sha256": sha256_file(source),
            "status": "failed",
            "error": (
                f"Worker timed out after "
                f"{worker_timeout_seconds():.0f}s"
            ),
        }
    if result is not None:
        return result
    if return_code < 0:
        reason = f"Worker terminated by signal {-return_code}"
    else:
        reason = (
            f"Worker exited with code {return_code} without a result"
        )
    return {
        "source": relative_source,
        "format": source.suffix.lower(),
        "source_sha256": sha256_file(source),
        "status": "failed",
        "error": reason,
    }


def main(
    workspace: Path | None = None,
    source_filter: str | None = None,
    force: bool = False,
) -> int:
    workspace = workspace or Path(__file__).resolve().parents[1]
    input_dir = workspace / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    # PATCH: declare manifest before try so Ctrl-C cannot NameError.
    manifest_path = workspace / "output" / "manifest.json"
    ledger_path = workspace / "output" / "converted.jsonl"
    manifest: dict[str, Any] = {
        "status": "starting",
        "started_at": None,
        "finished_at": None,
        "records": [],
    }

    config_error = validate_environment()
    if config_error:
        print(config_error, file=sys.stderr)
        return 2

    sources = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if source_filter is not None:
        selected_source = (input_dir / source_filter).resolve()
        try:
            selected_source.relative_to(input_dir.resolve())
        except ValueError:
            print(
                "Source path must stay inside the input directory.",
                file=sys.stderr,
            )
            return 2
        if (
            not selected_source.is_file()
            or selected_source.suffix.lower()
            not in SUPPORTED_EXTENSIONS
        ):
            print(
                "Source file is missing or unsupported.", file=sys.stderr
            )
            return 2
        sources = [selected_source]

    try:
        previous_records: dict[str, dict[str, Any]] = {}
        ledger_records: dict[str, dict[str, Any]] = {}
        if manifest_path.is_file():
            try:
                previous = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
                previous_records.update(
                    {
                        record["source"]: record
                        for record in previous.get("records", [])
                        if isinstance(record, dict)
                        and isinstance(record.get("source"), str)
                    }
                )
            except (OSError, json.JSONDecodeError, TypeError) as error:
                print(f"[Manifest] Could not reuse prior results: {error}")
        if ledger_path.is_file():
            try:
                ledger_records = read_conversion_ledger(ledger_path)
                for source_name, record in ledger_records.items():
                    current = previous_records.get(source_name)
                    if (
                        not current
                        or current.get("status") != "converted"
                    ):
                        previous_records[source_name] = record
            except ValueError as error:
                print(
                    f"[Ledger] Corrupt ledger ignored: {error}"
                )
                backup_path = ledger_path.with_name(
                    ledger_path.name
                    + ".corrupt-"
                    + datetime.now(timezone.utc).strftime(
                        "%Y%m%dT%H%M%S%fZ"
                    )
                    + ".bak"
                )
                try:
                    ledger_path.replace(backup_path)
                    print(
                        f"[Ledger] Corrupt ledger backed up to {backup_path}"
                    )
                except OSError as backup_error:
                    print(
                        f"[Ledger] Could not back up corrupt ledger: "
                        f"{backup_error}"
                    )
            except OSError as error:
                print(
                    f"[Ledger] Could not reuse converted records: {error}"
                )

        records: list[dict[str, Any]] = []
        pending_sources: list[Path] = []
        for source in sources:
            relative_source = source.relative_to(input_dir).as_posix()
            previous_record = (
                previous_records.get(relative_source)
                or ledger_records.get(relative_source)
            )
            current_hash = sha256_file(source)
            ok = bool(previous_record) and output_is_available(
                previous_record, source, workspace
            )
            action = plan_source(
                previous_record, current_hash, ok, force=force
            )

            if action == "reuse":
                records.append(previous_record)
                print(f"[Resume] Reusing {relative_source}")
                continue

            if action == "convert":
                recovered = recover_completed_output(
                    source, input_dir, workspace
                )
                if recovered:
                    records.append(recovered)
                    print(
                        f"[Resume] Recovered {relative_source}"
                    )
                    continue

            if action == "reconvert_changed" and previous_record:
                for removed_path in remove_stale_outputs(
                    previous_record, workspace
                ):
                    print(f"[Cleanup] removed {removed_path}")
                # If you maintain a vector index, delete this document's
                # vectors keyed by relative_source here.

            ledger_records.pop(relative_source, None)
            pending_sources.append(source)
            print(f"[{action}] {relative_source}")

        started_at = datetime.now(timezone.utc).isoformat()
        manifest = {
            "status": "running",
            "started_at": started_at,
            "finished_at": None,
            "records": records,
        }
        for record in records:
            if (
                record.get("status")
                in {"converted", "converted_with_warnings"}
                and isinstance(record.get("source"), str)
            ):
                ledger_records[record["source"]] = record
        write_manifest(manifest_path, manifest)
        write_conversion_ledger(
            ledger_path, list(ledger_records.values())
        )

        failures = 0
        warnings_count = 0
        for source in pending_sources:
            try:
                record = run_worker(
                    source, input_dir, workspace, force=True
                )
            except Exception as error:
                record = {
                    "source": source.relative_to(input_dir).as_posix(),
                    "format": source.suffix.lower(),
                    "status": "failed",
                    "error": (
                        f"Could not start conversion worker: {error}"
                    ),
                }
            records.append(record)
            if record.get("status") in {
                "converted", "converted_with_warnings",
            }:
                ledger_records[record["source"]] = record
            manifest["records"] = records
            write_manifest(manifest_path, manifest)
            write_conversion_ledger(
                ledger_path, list(ledger_records.values())
            )
            print(f"[{record['status']}] {record['source']}", flush=True)
            if record["status"] == "failed":
                failures += 1
            elif record["status"] == "converted_with_warnings":
                warnings_count += 1

        reused = len(sources) - len(pending_sources)
        manifest["status"] = (
            "completed_with_errors"
            if failures or warnings_count
            else "completed"
        )
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["summary"] = {
            "discovered": len(sources),
            "reused": reused,
            "converted": len(pending_sources)
            - failures
            - warnings_count,
            "warnings": warnings_count,
            "failed": failures,
        }
        write_manifest(manifest_path, manifest)
        print(
            f"Finished: {manifest['summary']['converted']} converted, "
            f"{reused} reused, {warnings_count} with warnings, "
            f"{failures} failed. "
            f"Manifest: {manifest_path}"
        )
        return int(failures > 0 or warnings_count > 0)
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_manifest(manifest_path, manifest)
        print("\nInterrupted. Completed files are saved in the manifest.")
        return 130


def entrypoint() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker_parser = argparse.ArgumentParser(add_help=False)
        worker_parser.add_argument(
            "--worker", dest="relative_source", required=True
        )
        worker_parser.add_argument("--workspace", type=Path)
        worker_parser.add_argument("--force", action="store_true")
        worker_args = worker_parser.parse_args(sys.argv[1:])
        return worker_main(
            worker_args.relative_source,
            worker_args.workspace,
            force=worker_args.force,
        )
    parser = argparse.ArgumentParser(
        description="Convert newly added documents under input/."
    )
    parser.add_argument(
        "--source",
        help="Convert only this path relative to input/ (for example, venclexta.pdf).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reconvert inputs even when existing outputs are reusable.",
    )
    args = parser.parse_args()
    return main(source_filter=args.source, force=args.force)


if __name__ == "__main__":
    raise SystemExit(entrypoint())