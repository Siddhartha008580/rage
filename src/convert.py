import argparse
from collections import Counter
import hashlib
from html import escape, unescape
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


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx"}
CONVERSION_FORMAT_VERSION = 26
PAGE_BREAK_PLACEHOLDER = "\n\n<!-- DOC_PAGE_BREAK -->\n\n"
GROQ_VISION_MODEL = "qwen/qwen3.8-27b"
GROQ_IMAGE_BATCH_SIZE = 3
DIAGRAM_CONFIDENCE_THRESHOLD = 0.55
MAX_VISION_IMAGE_BYTES = 20 * 1024 * 1024
MAX_VISION_BATCH_BYTES = 20 * 1024 * 1024
MIN_VISION_IMAGE_DIMENSION = 32
SLIDE_RENDER_DPI = 144
IMAGE_LINK_PATTERN = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
REPEATED_HEADER_TOP_Y = 155
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


def format_markdown(
    markdown: str,
    first_page: int = 1,
    page_label_prefix: str = "Page",
    printed_page_labels: list[str | None] | None = None,
) -> str:
    page_number = first_page

    def replace_page_break(_match: re.Match[str]) -> str:
        nonlocal page_number
        printed_index = page_number - first_page
        printed_page = (
            printed_page_labels[printed_index]
            if printed_page_labels and printed_index < len(printed_page_labels)
            else None
        )
        marker = f"\n\n> **{page_label_prefix} {page_number}"
        if printed_page:
            marker += f" (printed page {printed_page})"
        marker += "**\n\n"
        page_number += 1
        return marker

    markdown = re.sub(
        re.escape(PAGE_BREAK_PLACEHOLDER),
        replace_page_break,
        markdown,
    )
    return markdown


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
    if not nonspace or sum(character.isalpha() for character in nonspace) / len(nonspace) < 0.55:
        return False
    lines = normalized.splitlines()
    indents = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
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
            if re.fullmatch(rf"\s*{re.escape(fence[0])}{{{len(fence)},}}\s*", lines[closing_index]):
                break
            closing_index += 1
        if closing_index == len(lines) or info:
            result.extend(lines[index:closing_index + (closing_index < len(lines))])
            index = closing_index + (closing_index < len(lines))
            continue

        content = "\n".join(lines[index + 1:closing_index])
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
            result.extend(lines[index:closing_index + 1])
        index = closing_index + 1
    return "\n".join(result)


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
        if letters and all(character.isupper() for character in letters):
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
    backend = os.environ.get(
        "PDF_OCR_BACKEND", PDF_OCR_BACKEND
    ).strip().lower()
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
        content = html if category == "Table" and isinstance(html, str) else str(element)
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
            with tempfile.TemporaryDirectory(prefix="document-page-ocr-") as temp_dir:
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
        warnings.append(
            f"Page {page_number} {backend} OCR failed: {error}"
        )
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pdf_ocr_configuration() -> dict[str, str]:
    return {
        "backend": os.environ.get(
            "PDF_OCR_BACKEND", PDF_OCR_BACKEND
        ).strip().lower(),
        "languages": os.environ.get(
            "PDF_OCR_LANGUAGES", PDF_OCR_LANGUAGES
        ).strip(),
    }


def _configured_pdf_table_backend() -> str:
    backend = os.environ.get("PDF_TABLE_BACKEND", "pdfplumber").strip().lower()
    if backend not in {"pdfplumber", "hybrid", "table-transformer"}:
        raise ValueError(
            "PDF_TABLE_BACKEND must be pdfplumber, hybrid, or table-transformer"
        )
    return backend


def output_token(relative_path: Path, source_sha256: str) -> str:
    path_hash = hashlib.sha256(relative_path.as_posix().encode("utf-8")).hexdigest()[:10]
    content_hash = source_sha256[:12]
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", relative_path.stem).strip("._")
    ocr_suffix = ""
    table_suffix = ""
    if relative_path.suffix.lower() == ".pdf":
        configuration = {
            "ocr": _pdf_ocr_configuration(),
            "table_backend": _configured_pdf_table_backend(),
        }
        configuration_json = json.dumps(configuration, sort_keys=True).encode("utf-8")
        configuration_hash = hashlib.sha256(configuration_json).hexdigest()[:8]
        ocr_suffix = f"-ocr-{configuration_hash}"
        table_suffix = f"-table-{configuration['table_backend']}"
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


def _markdown_images_exist(markdown: str, markdown_dir: Path, image_root: Path) -> bool:
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
    if diagram_type not in {"flowchart", "architecture", "sequence", "org_chart", "other"}:
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
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
        temperature=0.1,
        max_completion_tokens=1500 * len(image_paths),
        response_format={"type": "json_object"},
    )
    response_content = response.choices[0].message.content
    if not isinstance(response_content, str):
        raise ValueError("Groq returned an empty image-analysis response")
    result = json.loads(response_content)
    if not isinstance(result, dict) or not isinstance(result.get("images"), list):
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


def _render_pptx_slides(source: Path, image_dir: Path) -> list[Path]:
    import pymupdf

    office_binary = shutil.which("libreoffice") or shutil.which("soffice")
    if office_binary is None:
        raise RuntimeError(
            "LibreOffice is not installed; install LibreOffice Impress to render slides"
        )

    image_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="document-slide-render-") as temp_dir:
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
            raise RuntimeError("LibreOffice timed out while rendering slides") from error
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(
                f"LibreOffice could not render slides"
                + (f": {detail[:300]}" if detail else "")
            )

        rendered_pdf = pdf_dir / f"{source.stem}.pdf"
        if not rendered_pdf.is_file():
            raise RuntimeError("LibreOffice did not produce the expected slide PDF")

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
    image_prefix = slide_paths[0].parent.relative_to(markdown_dir).as_posix()
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
    return markdown.rstrip() + "\n\n" + "\n".join(slide_section).rstrip() + "\n"


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
        "1",
        "true",
        "yes",
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
            markdown[match.end():],
        ):
            continue
        image_uri = unquote(match.group(2).split("#", 1)[0].split("?", 1)[0])
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
                raise ValueError("Groq returned the wrong number of image analyses")
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
        enriched.append(markdown[cursor:match.start()])
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

        next_text = markdown[match.end():]
        if re.match(r"\s*\n\s*\*\*AI-generated image description", next_text):
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
            failed_images.append(f"{image_path.name} ({str(error)[:160]})")
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
        following_text = markdown[match.end():]
        if not re.match(
            r"\s*\n\s*\*\*AI-generated image description", following_text
        ):
            return False
    return True


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
    markdown_rows.append("|" + "|".join([" --- " for _ in range(columns)]) + "|")
    markdown_rows.extend("| " + " | ".join(row) + " |" for row in data_rows)
    return "\n".join(markdown_rows)


def _normalize_markdown_tables(markdown: str) -> str:
    lines = markdown.splitlines()
    in_fenced_block = False
    for index, line in enumerate(lines):
        if re.match(r"^\s*(```|~~~)", line):
            in_fenced_block = not in_fenced_block
        elif not in_fenced_block and re.match(r"^\s*\|.*\|\s*$", line):
            lines[index] = re.sub(
                r"<br\s*/?>", "<br>", line, flags=re.IGNORECASE
            )
    return "\n".join(lines)


def _xlsx_to_markdown(source: Path) -> str | None:
    from openpyxl import load_workbook
    from openpyxl.utils import get_column_letter

    workbook = load_workbook(source, data_only=False)
    if any(worksheet._images or worksheet._charts for worksheet in workbook.worksheets):
        workbook.close()
        return None

    def cell_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (datetime,)):
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
                    cell.column
                    for cell in row
                    if cell.value is not None
                ]
                if columns:
                    populated_rows[row[0].row] = columns
            data_start = next(
                (
                    row_number
                    for row_number, columns in sorted(populated_rows.items())
                    if sum(
                        numeric_or_formula(worksheet.cell(row_number, column))
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
                    numeric_or_formula(worksheet.cell(row_number, column))
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
                    "| --- | " + " | ".join(["---"] * len(headers)) + " |",
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
                        cell_text(worksheet.cell(note_row, column).value)
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


def _pdf_image_captions(page: Any) -> list[tuple[tuple[float, ...], str]]:
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
        if not visible_line.lower().startswith(identifier.group(0).lower()):
            continue
        end = index
        while end + 1 < len(lines) and lines[end + 1].strip():
            end += 1
        lines[end + 1:end + 1] = ["", image_link]
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
        image_index = int(image_index_match.group(1)) if image_index_match else -1
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
                page_markdown = page_markdown.rstrip() + "\n\n" + image_link
                warnings.append(
                    f"Image caption on PDF page {page_number} was not located "
                    "in the layout text; its cited image link was appended "
                    "to the page"
                )
        else:
            image_link = (
                f"![Image on PDF page {page_number}]({image_uri})"
            )
            page_markdown = page_markdown.rstrip() + "\n\n" + image_link
            if image_info:
                warnings.append(
                    f"No nearby figure caption was found for an image on "
                    f"PDF page {page_number}; the image is linked with its "
                    "page number only"
                )
    return page_markdown, warnings


def _pdfplumber_tables_by_page(source: Path) -> dict[int, list[dict[str, Any]]]:
    import pdfplumber

    tables_by_page: dict[int, list[dict[str, Any]]] = {}
    with pdfplumber.open(source) as pdf:
        for page_index, page in enumerate(pdf.pages):
            try:
                detected_tables = page.find_tables()
            except Exception as error:
                raise RuntimeError(
                    f"Table detection failed on PDF page {page_index + 1}: {error}"
                ) from error
            tables: list[dict[str, Any]] = []
            for table in detected_tables:
                try:
                    rows = table.extract()
                except Exception as error:
                    raise RuntimeError(
                        f"Table extraction failed on PDF page "
                        f"{page_index + 1}: {error}"
                    ) from error
                columns = max((len(row) for row in rows), default=0)
                if not rows or (len(rows) < 3 and columns < 2):
                    continue
                if not any(cell and cell.strip() for row in rows for cell in row):
                    continue
                tables.append(
                    {
                        "bbox": table.bbox,
                        "page_height": page.height,
                        "rows": rows,
                        "table_index": len(tables) + 1,
                    }
                )
            if tables:
                tables_by_page[page_index] = tables
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
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _table_transformer_tables_by_page(
    source: Path,
    warnings: list[str],
    pages: set[int] | None = None,
) -> dict[int, list[dict[str, Any]]]:
    import pymupdf

    try:
        from .table_transformer import TableTransformerExtractor
    except ImportError:
        from table_transformer import TableTransformerExtractor

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
                    f"Page {page_index + 1} Table Transformer failed: {error}"
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
                    for index, prediction in enumerate(predictions, start=1)
                ]
    return tables_by_page


def _page_has_dot_leader_rows(page: Any) -> bool:
    page_numbers: list[str] = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            text = "".join(
                span.get("text", "")
                for span in line.get("spans", [])
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
        if not (
            page.rect.height * 0.1 <= y0 <= page.rect.height * 0.9
        ):
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
            if requested_pages is not None and page_index not in requested_pages:
                continue
            if page_index in candidates:
                continue
            native_text_chars, image_coverage = _pdf_page_content_metrics(page)
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
            if overlap >= 0.35 and prediction["confidence"] >= replacement_threshold:
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
            elif overlap < 0.1 and prediction["confidence"] >= addition_threshold:
                prediction = dict(prediction)
                prediction["table_index"] = len(page_tables) + 1
                page_tables.append(prediction)
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
    if backend == "pdfplumber":
        return _pdfplumber_tables_by_page(source)

    fallback = _pdfplumber_tables_by_page(source)
    candidate_pages = _table_transformer_candidate_pages(
        source,
        set(fallback),
        pages,
    )
    transformed = _table_transformer_tables_by_page(
        source,
        warnings,
        candidate_pages,
    )
    if backend == "table-transformer":
        return {
            page_index: transformed.get(page_index) or tables
            for page_index, tables in fallback.items()
        } | {
            page_index: tables
            for page_index, tables in transformed.items()
            if page_index not in fallback
        }
    return _merge_table_transformer_predictions(fallback, transformed)


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
                    or previous_header != _table_header_signature(following["rows"])
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


def _table_header_signature(rows: list[list[str | None]]) -> tuple[str, ...]:
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
        numeric = sum(
            bool(re.search(r"\d", cell))
            for cell in populated
        )
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
        first_numeric_ratio <= 0.25
        and second_numeric_ratio >= 0.5
    ) or (
        has_sparse_subheader
        and has_numeric_body
    ) or (
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
                    or y0 >= page.rect.height * (1 - PAGE_FURNITURE_EDGE_FRACTION)
                ):
                    continue
                text = " ".join(
                    span.get("text", "")
                    for span in line.get("spans", [])
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


def _printed_pdf_page_number(page: Any) -> str | None:
    candidates: list[tuple[float, str]] = []
    page_width = page.rect.width
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            x0, y0, x1, y1 = line["bbox"]
            if y0 < 75 or y1 > REPEATED_HEADER_TOP_Y:
                continue
            text = "".join(
                span.get("text", "")
                for span in line.get("spans", [])
            ).strip()
            if not re.fullmatch(r"\d{1,4}", text):
                continue
            if x0 > page_width * 0.35 and x1 < page_width * 0.65:
                continue
            candidates.append((y0, text))
    return min(candidates)[1] if candidates else None


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
            raise RuntimeError("PDF table detector returned an invalid page index")
        toc_headers = _build_toc_headers(document)
        printed_page_labels = [
            _printed_pdf_page_number(page) for page in document
        ]
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
                native_text_chars, image_coverage = _pdf_page_content_metrics(page)
                native_page_text = page.get_text("text")
                ocr_text = ""
                if _pdf_page_needs_ocr(native_text_chars, image_coverage):
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
                            if not warning.startswith(f"Page {page_number} has ")
                        ]
                    elif ocr_backend == "none":
                        warnings.append(
                            f"Page {page_number} requires OCR, but PDF_OCR_BACKEND "
                            "is set to none"
                        )
                page_tables = tables_by_page.get(first_page, [])
                for table_index, table in enumerate(page_tables, start=1):
                    table_id = f"{page_number:04}_{table_index:02}"
                    placeholder = f"{TABLE_PLACEHOLDER_PREFIX}{table_id}ZZ"
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
                for table_index, table in enumerate(page_tables, start=1):
                    table_id = f"{page_number:04}_{table_index:02}"
                    placeholder = table["placeholder"]
                    if "merged_into_page" in table:
                        table_markdown = (
                            f"*Table continuation on PDF page {page_number}; "
                            f"{table['merged_row_count']} rows were joined to "
                            f"the table on PDF page {table['merged_into_page']}.*"
                        )
                    else:
                        table_markdown = table.get("html") or _table_to_markdown(
                            table["rows"]
                        )
                    if "continues_to_page" in table:
                        table_markdown += (
                            f"\n\n*Merged table spans PDF pages {page_number}–"
                            f"{table['continues_to_page']}.*"
                        )
                    placeholder_pattern = re.compile(
                        rf"\*{{0,2}}{re.escape(placeholder)}\*{{0,2}}"
                    )
                    page_markdown, replacement_count = placeholder_pattern.subn(
                        table_markdown, page_markdown, count=1
                    )
                    if not replacement_count:
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
                _normalize_pdf_fenced_blocks(cached_markdown),
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

    raw_markdown = (
        PAGE_BREAK_PLACEHOLDER.join(markdown_pages) + PAGE_BREAK_PLACEHOLDER
    )
    combined_markdown = format_markdown(
        raw_markdown,
        page_label_prefix="PDF page",
        printed_page_labels=printed_page_labels,
    )
    combined_markdown = _format_pdf_footnotes(combined_markdown)
    if not _markdown_images_exist(combined_markdown, markdown_path.parent, image_dir):
        raise RuntimeError("PDF Markdown output references missing images")
    atomic_write_text(markdown_path, combined_markdown)
    return combined_markdown, warnings


def convert_file(
    source: Path,
    input_dir: Path,
    workspace: Path,
    converter: Any | None,
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
        record["pdf_table"] = {"backend": _configured_pdf_table_backend()}

    markdown_path = (
        workspace / "output" / "markdown" / output_subdir / f"{token}.md"
    )
    image_dir = markdown_path.parent / "images" / token
    warnings: list[str] = []
    try:
        if markdown_path.is_file():
            existing_markdown = markdown_path.read_text(encoding="utf-8")
            if existing_markdown.strip() and _markdown_images_exist(
                existing_markdown, markdown_path.parent, image_dir
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
        image_files = sorted(path for path in image_dir.rglob("*") if path.is_file())
        record["markdown"] = {
            "status": "converted",
            "format_version": CONVERSION_FORMAT_VERSION,
            "output": relative_output_path(markdown_path, workspace),
            "output_sha256": hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
            "images": [
                relative_output_path(path, workspace) for path in image_files
            ],
        }
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
    atomic_write_text(path, json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def write_conversion_ledger(path: Path, records: list[dict[str, Any]]) -> None:
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
            if isinstance(record, dict) and isinstance(record.get("source"), str):
                records[record["source"]] = record
    return records


def worker_main(relative_source: str, workspace: Path | None = None) -> int:
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
    if not source.is_file() or source.suffix.lower() not in SUPPORTED_EXTENSIONS:
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

    record = convert_file(source, input_dir, workspace, converter)
    for warning in record.get("warnings", []):
        print(f"[PDF] Warning: {warning}", flush=True)
    print(f"__INGEST_RESULT__{json.dumps(record, ensure_ascii=False)}", flush=True)
    return int(record["status"] == "failed")


def output_is_available(record: dict[str, Any], source: Path, workspace: Path) -> bool:
    if record.get("status") not in {"converted", "converted_with_warnings"}:
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
    if hashlib.sha256(markdown.encode("utf-8")).hexdigest() != expected_markdown_hash:
        return False
    if (
        os.environ.get("ENABLE_DIAGRAM_CONVERSION", "").strip().lower()
        in {"1", "true", "yes"}
        and os.environ.get("GROQ_API_KEY")
        and not _vision_enrichment_is_current(markdown, markdown_path.parent)
    ):
        return False
    for path in image_paths:
        if not isinstance(path, str) or not (workspace / path).is_file():
            return False
        try:
            image_reference = (
                (workspace / path).relative_to(markdown_path.parent).as_posix()
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
    image_paths = sorted(path for path in image_dir.rglob("*") if path.is_file())
    for image_uri in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", markdown):
        if image_uri.startswith(("http://", "https://", "data:")):
            continue
        image_path = (markdown_path.parent / image_uri.split("#", 1)[0]).resolve()
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
            "output_sha256": hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
            "images": [
                relative_output_path(path, workspace) for path in image_paths
            ],
        },
        "status": "converted",
    }
    if source.suffix.lower() == ".pdf":
        record["pdf_ocr"] = _pdf_ocr_configuration()
        record["pdf_table"] = {"backend": _configured_pdf_table_backend()}

    return record if output_is_available(record, source, workspace) else None


def run_worker(source: Path, input_dir: Path, workspace: Path) -> dict[str, Any]:
    relative_source = source.relative_to(input_dir).as_posix()
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        relative_source,
        "--workspace",
        str(workspace.resolve()),
    ]
    print(f"[Worker] Starting {relative_source}", flush=True)
    process = subprocess.Popen(
        command,
        cwd=workspace,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    result: dict[str, Any] | None = None
    assert process.stdout is not None
    try:
        for line in process.stdout:
            line = line.rstrip()
            if line.startswith("__INGEST_RESULT__"):
                try:
                    result = json.loads(line.removeprefix("__INGEST_RESULT__"))
                except json.JSONDecodeError as error:
                    print(f"[Worker] Invalid result for {relative_source}: {error}")
            elif line:
                print(f"[{relative_source}] {line}", flush=True)
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    finally:
        process.stdout.close()
    return_code = process.wait()

    if result is not None:
        return result
    if return_code < 0:
        reason = f"Worker terminated by signal {-return_code}"
    else:
        reason = f"Worker exited with code {return_code} without a result"
    return {
        "source": relative_source,
        "format": source.suffix.lower(),
        "source_sha256": sha256_file(source),
        "status": "failed",
        "error": reason,
    }


def main(workspace: Path | None = None, source_filter: str | None = None) -> int:
    workspace = workspace or Path(__file__).resolve().parents[1]
    input_dir = workspace / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    sources = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and not path.is_symlink()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if source_filter is not None:
        selected_source = (input_dir / source_filter).resolve()
        try:
            selected_source.relative_to(input_dir.resolve())
        except ValueError:
            print("Source path must stay inside the input directory.", file=sys.stderr)
            return 2
        if not selected_source.is_file() or selected_source.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print("Source file is missing or unsupported.", file=sys.stderr)
            return 2
        sources = [selected_source]
    try:
        manifest_path = workspace / "output" / "manifest.json"
        ledger_path = workspace / "output" / "converted.jsonl"
        previous_records: dict[str, dict[str, Any]] = {}
        ledger_records: dict[str, dict[str, Any]] = {}
        if manifest_path.is_file():
            try:
                previous = json.loads(manifest_path.read_text(encoding="utf-8"))
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
                    if not current or current.get("status") != "converted":
                        previous_records[source_name] = record
            except (OSError, ValueError) as error:
                print(f"[Ledger] Could not reuse converted records: {error}")

        records: list[dict[str, Any]] = []
        pending_sources: list[Path] = []
        skipped = 0
        for source in sources:
            relative_source = source.relative_to(input_dir).as_posix()
            previous_record = previous_records.get(relative_source)
            if not previous_record and relative_source in ledger_records:
                previous_record = ledger_records[relative_source]
            if previous_record and output_is_available(
                previous_record, source, workspace
            ):
                records.append(previous_record)
                print(f"[Resume] Reusing {relative_source}")
            elif previous_record:
                current_hash = sha256_file(source)
                previous_hash = previous_record.get("source_sha256")
                if previous_hash != current_hash:
                    reason = "Input changed since it was first processed"
                    records.append(
                        {
                            "source": relative_source,
                            "source_sha256": current_hash,
                            "format": source.suffix.lower(),
                            "status": "skipped",
                            "error": (
                                f"{reason}; existing paths are not reconverted. "
                                "Add the document under a new input path to convert it."
                            ),
                        }
                    )
                    skipped += 1
                    print(f"[Skip] {relative_source}: {reason}; not reconverting")
                elif previous_record.get("format_version") != CONVERSION_FORMAT_VERSION:
                    pending_sources.append(source)
                    print(
                        f"[Reconvert] {relative_source}: conversion format "
                        f"version changed",
                        flush=True,
                    )
                elif (
                    os.environ.get("ENABLE_DIAGRAM_CONVERSION", "")
                    .strip()
                    .lower()
                    in {"1", "true", "yes"}
                    and os.environ.get("GROQ_API_KEY")
                    and isinstance(previous_record.get("markdown"), dict)
                    and isinstance(
                        previous_record["markdown"].get("output"), str
                    )
                ):
                    previous_markdown_path = (
                        workspace / previous_record["markdown"]["output"]
                    )
                    try:
                        previous_markdown = previous_markdown_path.read_text(
                            encoding="utf-8"
                        )
                    except OSError:
                        previous_markdown = ""
                    if previous_markdown and not _vision_enrichment_is_current(
                        previous_markdown, previous_markdown_path.parent
                    ):
                        pending_sources.append(source)
                        print(
                            f"[Re-enrich] {relative_source}: image descriptions "
                            "are missing for the enabled vision feature",
                            flush=True,
                        )
                        continue
                    reason = "Previous conversion is incomplete or unsuccessful"
                    records.append(
                        {
                            "source": relative_source,
                            "format": source.suffix.lower(),
                            "source_sha256": current_hash,
                            "status": "skipped",
                            "error": (
                                "Previous conversion is incomplete or unsuccessful; "
                                "existing paths are not reconverted. Add the document "
                                "under a new input path to convert it."
                            ),
                        }
                    )
                    skipped += 1
                    print(
                        f"[Skip] {relative_source}: {reason}; not reconverting"
                    )
                else:
                    reason = "Previous conversion is incomplete or unsuccessful"
                    records.append(
                        {
                            "source": relative_source,
                            "format": source.suffix.lower(),
                            "source_sha256": current_hash,
                            "status": "skipped",
                            "error": (
                                "Previous conversion is incomplete or unsuccessful; "
                                "existing paths are not reconverted. Add the document "
                                "under a new input path to convert it."
                            ),
                        }
                    )
                    skipped += 1
                    print(
                        f"[Skip] {relative_source}: {reason}; not reconverting"
                    )
            else:
                recovered_record = recover_completed_output(
                    source, input_dir, workspace
                )
                if recovered_record:
                    records.append(recovered_record)
                    print(f"[Resume] Recovered completed outputs for {relative_source}")
                else:
                    pending_sources.append(source)

        started_at = datetime.now(timezone.utc).isoformat()
        manifest: dict[str, Any] = {
            "status": "running",
            "started_at": started_at,
            "finished_at": None,
            "records": records,
        }
        for record in records:
            if (
                record.get("status") in {"converted", "converted_with_warnings"}
                and isinstance(record.get("source"), str)
            ):
                ledger_records[record["source"]] = record
        write_manifest(manifest_path, manifest)
        write_conversion_ledger(ledger_path, list(ledger_records.values()))

        failures = 0
        warnings = 0
        for source in pending_sources:
            try:
                record = run_worker(source, input_dir, workspace)
            except Exception as error:
                record = {
                    "source": source.relative_to(input_dir).as_posix(),
                    "format": source.suffix.lower(),
                    "status": "failed",
                    "error": f"Could not start conversion worker: {error}",
                }
            records.append(record)
            if record.get("status") in {"converted", "converted_with_warnings"}:
                ledger_records[record["source"]] = record
            manifest["records"] = records
            write_manifest(manifest_path, manifest)
            write_conversion_ledger(ledger_path, list(ledger_records.values()))
            print(f"[{record['status']}] {record['source']}", flush=True)
            if record["status"] == "failed":
                failures += 1
            elif record["status"] == "converted_with_warnings":
                warnings += 1

        reused = len(sources) - len(pending_sources) - skipped
        manifest["status"] = (
            "completed_with_errors"
            if failures or warnings or skipped
            else "completed"
        )
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["summary"] = {
            "discovered": len(sources),
            "reused": reused,
            "converted": len(pending_sources) - failures - warnings,
            "warnings": warnings,
            "failed": failures,
            "skipped": skipped,
        }
        write_manifest(manifest_path, manifest)
        print(
            f"Finished: {manifest['summary']['converted']} converted, "
            f"{reused} reused, {skipped} skipped, {warnings} with warnings, "
            f"{failures} failed. "
            f"Manifest: {manifest_path}"
        )
        return int(failures > 0 or warnings > 0 or skipped > 0)
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_manifest(manifest_path, manifest)
        print("\nInterrupted. Completed files are saved in the manifest.")
        return 130


def entrypoint() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker_parser = argparse.ArgumentParser(add_help=False)
        worker_parser.add_argument("--worker", dest="relative_source", required=True)
        worker_parser.add_argument("--workspace", type=Path)
        worker_args = worker_parser.parse_args(sys.argv[1:])
        return worker_main(worker_args.relative_source, worker_args.workspace)
    parser = argparse.ArgumentParser(
        description="Convert newly added documents under input/."
    )
    parser.add_argument(
        "--source",
        help="Convert only this path relative to input/ (for example, venclexta.pdf).",
    )
    args = parser.parse_args()
    return main(source_filter=args.source)


if __name__ == "__main__":
    raise SystemExit(entrypoint())
