import argparse
from dataclasses import dataclass, field
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


PAGE_MARKER = re.compile(
    r"^>\s*\*\*PDF page\s+(\d+)(?:\s+\(printed page\s+([^)]+)\))?\*\*\s*$"
)
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
MAX_PARENT_TOKENS = 1200
CHUNKER_VERSION = 4
TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
TABLE_SEPARATOR = re.compile(r"^\s*\|(?:\s*:?-+:?\s*\|)+\s*$")
HTML_TABLE_OPEN = re.compile(r"<table\b", re.I)
HTML_TABLE_CLOSE = re.compile(r"</table\s*>", re.I)
HTML_TABLE_ROW = re.compile(r"<tr\b[^>]*>.*?</tr\s*>", re.I | re.S)
HTML_TABLE_HEAD = re.compile(r"<thead\b[^>]*>.*?</thead\s*>", re.I | re.S)
TABLE_CAPTION = re.compile(r"^\*{0,2}Table\s+\d+\b.*\*{0,2}$", re.I | re.S)
IMAGE = re.compile(r"^\s*!\[[^\]]*\]\([^)]+\)\s*$")
AI_DESCRIPTION = re.compile(r"^\s*\*\*AI-generated image description\b", re.I)
OCR_HEADING = re.compile(r"^\s*#{1,6}\s+OCR-only text\b", re.I)
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Block:
    text: str
    page: int
    printed_page: str | None
    kind: str = "prose"
    heading_level: int | None = None
    heading_text: str | None = None


@dataclass
class Section:
    heading: str
    path: list[str]
    blocks: list[Block] = field(default_factory=list)


def estimate_tokens(text: str) -> int:
    return len(TOKEN.findall(text))


def _classify_block(text: str) -> str:
    stripped = text.strip()
    if OCR_HEADING.match(stripped):
        return "ocr_text"
    if AI_DESCRIPTION.match(stripped):
        return "ai_generated_description"
    if IMAGE.match(stripped):
        return "figure"
    return "prose"


def parse_markdown(markdown: str) -> list[Block]:
    blocks: list[Block] = []
    current_page = 1
    printed_page: str | None = None
    lines: list[str] = []
    block_page = current_page
    block_printed_page = printed_page
    in_html_table = False

    def flush() -> None:
        nonlocal lines
        text = "\n".join(lines).strip()
        if text:
            heading_match = HEADING.match(text)
            if heading_match:
                blocks.append(
                    Block(
                        text=text,
                        page=block_page,
                        printed_page=block_printed_page,
                        kind="heading",
                        heading_level=len(heading_match.group(1)),
                        heading_text=heading_match.group(2).strip(),
                    )
                )
            elif all(TABLE_ROW.match(line) for line in text.splitlines()) or (
                HTML_TABLE_OPEN.search(text) and HTML_TABLE_CLOSE.search(text)
            ):
                blocks.append(
                    Block(
                        text=text,
                        page=block_page,
                        printed_page=block_printed_page,
                        kind="table",
                    )
                )
            else:
                blocks.append(
                    Block(
                        text=text,
                        page=block_page,
                        printed_page=block_printed_page,
                        kind=_classify_block(text),
                    )
                )
        lines = []

    for line in markdown.splitlines():
        marker = PAGE_MARKER.match(line.strip())
        if marker:
            flush()
            page_number = int(marker.group(1))
            marker_printed_page = marker.group(2)
            if marker_printed_page is not None:
                for block in reversed(blocks):
                    if block.page != page_number:
                        break
                    if block.printed_page is None:
                        block.printed_page = marker_printed_page
            current_page = int(marker.group(1)) + 1
            printed_page = None
            block_page = current_page
            block_printed_page = printed_page
            continue

        if in_html_table:
            lines.append(line)
            if HTML_TABLE_CLOSE.search(line):
                flush()
                in_html_table = False
                block_page = current_page
                block_printed_page = printed_page
            continue

        if HTML_TABLE_OPEN.search(line):
            flush()
            block_page = current_page
            block_printed_page = printed_page
            lines.append(line)
            if HTML_TABLE_CLOSE.search(line):
                flush()
                block_page = current_page
                block_printed_page = printed_page
            else:
                in_html_table = True
            continue

        heading_match = HEADING.match(line)
        is_table_line = TABLE_ROW.match(line) is not None
        previous_is_table = bool(lines) and all(
            TABLE_ROW.match(existing) for existing in lines
        )
        if lines and (
            heading_match
            or (is_table_line != previous_is_table)
            or (not line.strip() and previous_is_table)
        ):
            flush()
            block_page = current_page
            block_printed_page = printed_page

        if not line.strip():
            flush()
            block_page = current_page
            block_printed_page = printed_page
            continue
        if not lines:
            block_page = current_page
            block_printed_page = printed_page
        lines.append(line)

    flush()
    return blocks


def _split_long_text(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    sentences = [sentence.strip() for sentence in SENTENCE_BOUNDARY.split(text) if sentence.strip()]
    pieces: list[str] = []
    pending = ""
    for sentence in sentences or [text]:
        if estimate_tokens(sentence) > max_tokens:
            if pending:
                pieces.append(pending)
                pending = ""
            words = sentence.split()
            start = 0
            while start < len(words):
                end = start
                while end < len(words):
                    candidate = " ".join(words[start : end + 1])
                    if estimate_tokens(candidate) > max_tokens and end > start:
                        break
                    end += 1
                    if estimate_tokens(" ".join(words[start:end])) > max_tokens:
                        break
                if end == start:
                    end += 1
                pieces.append(" ".join(words[start:end]))
                if end >= len(words):
                    break
                start = max(start + 1, end - overlap_tokens)
            continue
        candidate = f"{pending}\n\n{sentence}".strip() if pending else sentence
        if pending and estimate_tokens(candidate) > max_tokens:
            pieces.append(pending)
            pending = sentence
        else:
            pending = candidate
    if pending:
        pieces.append(pending)
    return pieces


def _table_children(
    block: Block,
    max_tokens: int,
    caption: str | None = None,
) -> list[tuple[str, str, Block]]:
    caption_prefix = f"{caption}\n\n" if caption else ""
    open_match = re.search(r"<table\b[^>]*>", block.text, re.I)
    close_match = re.search(r"</table\s*>", block.text, re.I)
    if open_match and close_match:
        table_open = open_match.group(0)
        head_match = HTML_TABLE_HEAD.search(block.text)
        table_head = head_match.group(0) if head_match else ""
        all_rows = HTML_TABLE_ROW.findall(block.text)
        header_rows = (
            HTML_TABLE_ROW.findall(table_head)
            if table_head
            else []
        )
        body_rows = list(all_rows)
        for header_row in header_rows:
            try:
                body_rows.remove(header_row)
            except ValueError:
                pass
        tbody_open = "<tbody>"
        tbody_close = "</tbody>"

        def render_table(rows: list[str]) -> str:
            parts = [table_open]
            if table_head:
                parts.append(table_head)
            if rows:
                parts.extend([tbody_open, *rows, tbody_close])
            parts.append("</table>")
            return "\n".join(parts)

        if estimate_tokens(block.text) <= max_tokens or not body_rows:
            return [(caption_prefix + block.text, "table", block)]

        children: list[tuple[str, str, Block]] = []
        current_rows: list[str] = []
        for row in body_rows:
            candidate = render_table(current_rows + [row])
            if current_rows and estimate_tokens(candidate) > max_tokens:
                children.append(
                    (caption_prefix + render_table(current_rows), "table", block)
                )
                current_rows = [row]
            else:
                current_rows.append(row)
        if current_rows:
            children.append(
                (caption_prefix + render_table(current_rows), "table", block)
            )
        return children

    rows = block.text.splitlines()
    if len(rows) < 2:
        return [(caption_prefix + block.text, "table", block)]

    separator_index = next(
        (index for index, row in enumerate(rows[1:], 1) if TABLE_SEPARATOR.match(row)),
        None,
    )
    if separator_index is None:
        return [(caption_prefix + block.text, "table", block)]

    table_prefix = ([caption, ""] if caption else []) + rows[: separator_index + 1]
    data_rows = rows[separator_index + 1 :]
    children: list[tuple[str, str, Block]] = []
    current_rows: list[str] = []
    for row in data_rows:
        candidate_rows = current_rows + [row]
        candidate = "\n".join(table_prefix + candidate_rows)
        if current_rows and estimate_tokens(candidate) > max_tokens:
            child_text = "\n".join(table_prefix + current_rows)
            children.append((child_text, "table", block))
            current_rows = [row]
        else:
            current_rows = candidate_rows
    if current_rows:
        children.append(("\n".join(table_prefix + current_rows), "table", block))
    if not children:
        children.append(("\n".join(rows), "table", block))
    return children


def _child_chunks(
    blocks: list[Block],
    max_tokens: int,
    overlap_tokens: int,
) -> list[tuple[str, str, int, int, str | None, str | None]]:
    result: list[tuple[str, str, int, int, str | None, str | None]] = []
    pending: list[Block] = []
    ocr_heading: Block | None = None
    ocr_heading_included = False

    def append_chunk(text: str, kind: str, chunk_blocks: list[Block]) -> None:
        pages = [block.page for block in chunk_blocks]
        printed_pages = [
            block.printed_page for block in chunk_blocks if block.printed_page
        ]
        result.append(
            (
                text,
                kind,
                min(pages),
                max(pages),
                printed_pages[0] if printed_pages else None,
                printed_pages[-1] if printed_pages else None,
            )
        )

    def flush_pending() -> None:
        nonlocal pending
        if not pending:
            return
        text = "\n\n".join(block.text for block in pending)
        if estimate_tokens(text) <= max_tokens:
            append_chunk(text, "prose", pending)
        else:
            for piece in _split_long_text(text, max_tokens, overlap_tokens):
                append_chunk(piece, "prose", pending)
        pending = []

    for block in blocks:
        if block.kind == "heading" and OCR_HEADING.match(block.text):
            flush_pending()
            ocr_heading = block
            ocr_heading_included = False
            continue
        if ocr_heading is not None and (
            block.kind == "heading" or block.kind == "ai_generated_description"
        ):
            ocr_heading = None
            ocr_heading_included = False
        if ocr_heading is not None and block.kind == "table":
            flush_pending()
            table_chunks = _table_children(block, max_tokens)
            for index, (text, kind, source_block) in enumerate(table_chunks):
                if not ocr_heading_included and index == 0:
                    text = f"{ocr_heading.text}\n\n{text}"
                append_chunk(text, kind, [ocr_heading, source_block])
            ocr_heading_included = True
            continue
        if ocr_heading is not None and block.kind != "figure":
            flush_pending()
            text = block.text
            source_blocks = [block]
            if not ocr_heading_included:
                text = f"{ocr_heading.text}\n\n{text}"
                source_blocks.insert(0, ocr_heading)
            if estimate_tokens(text) <= max_tokens:
                append_chunk(text, "ocr_text", source_blocks)
            else:
                for piece in _split_long_text(text, max_tokens, overlap_tokens):
                    append_chunk(piece, "ocr_text", source_blocks)
            ocr_heading_included = True
            continue
        if block.kind == "table":
            caption = None
            caption_block = None
            if pending and TABLE_CAPTION.match(pending[-1].text.strip()):
                caption_block = pending.pop()
                caption = caption_block.text
            flush_pending()
            for text, kind, source_block in _table_children(
                block, max_tokens, caption
            ):
                chunk_blocks = [source_block]
                if caption_block is not None:
                    chunk_blocks.insert(0, caption_block)
                append_chunk(text, kind, chunk_blocks)
            continue
        if block.kind in {"figure", "ai_generated_description", "ocr_text"}:
            flush_pending()
            append_chunk(block.text, block.kind, [block])
            continue
        candidate = "\n\n".join([item.text for item in pending] + [block.text])
        if pending and estimate_tokens(candidate) > max_tokens:
            flush_pending()
        pending.append(block)
    flush_pending()
    return result


def _warnings_for_pages(
    warnings: list[str],
    page_start: int,
    page_end: int,
) -> tuple[list[str], list[int], list[str]]:
    page_warnings: list[str] = []
    warning_pages: list[int] = []
    document_warnings: list[str] = []
    for warning in warnings:
        page_match = re.search(r"\bPage\s+(\d+)\b", warning, re.IGNORECASE)
        if page_match is None:
            document_warnings.append(warning)
            continue
        page = int(page_match.group(1))
        if page_start <= page <= page_end:
            page_warnings.append(warning)
            warning_pages.append(page)
    return page_warnings, warning_pages, document_warnings


def build_document_records(
    markdown: str,
    source: str,
    source_hash: str,
    converter_version: int | None,
    warnings: list[str],
    max_tokens: int = 180,
    overlap_tokens: int = 20,
) -> list[dict[str, Any]]:
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be between zero and max_tokens - 1")

    source_path = Path(source)
    domain = source_path.parts[0] if len(source_path.parts) > 1 else "uncategorized"
    document_id = hashlib.sha256(f"{source}\0{source_hash}".encode()).hexdigest()[:20]
    blocks = parse_markdown(markdown)
    sections: list[Section] = []
    heading_stack: list[tuple[int, str]] = []
    current_section = Section(heading="Document preamble", path=[])

    for block in blocks:
        if block.kind == "heading":
            if current_section.blocks or current_section.heading != "Document preamble":
                sections.append(current_section)
            assert block.heading_level is not None and block.heading_text is not None
            while heading_stack and heading_stack[-1][0] >= block.heading_level:
                heading_stack.pop()
            heading_stack.append((block.heading_level, block.heading_text))
            current_section = Section(
                heading=block.heading_text,
                path=[heading for _, heading in heading_stack],
                blocks=[block],
            )
        else:
            current_section.blocks.append(block)
    if current_section.blocks or not sections:
        sections.append(current_section)

    records: list[dict[str, Any]] = []
    parent_index = 0
    for section in sections:
        if not section.blocks:
            continue
        chunks = _child_chunks(section.blocks, max_tokens, overlap_tokens)
        parent_groups: list[list[tuple[str, str, int, int, str | None, str | None]]] = []
        current_group: list[tuple[str, str, int, int, str | None, str | None]] = []
        current_tokens = 0
        for chunk in chunks:
            chunk_tokens = estimate_tokens(chunk[0])
            if current_group and current_tokens + chunk_tokens > MAX_PARENT_TOKENS:
                parent_groups.append(current_group)
                current_group = []
                current_tokens = 0
            current_group.append(chunk)
            current_tokens += chunk_tokens
        if current_group:
            parent_groups.append(current_group)

        section_prefix = " > ".join(section.path) or "Document preamble"
        section_part_count = len(parent_groups)
        for section_part, parent_chunks in enumerate(parent_groups, 1):
            parent_index += 1
            parent_id = f"{document_id}:p{parent_index:04d}"
            parent_content = "\n\n".join(chunk[0] for chunk in parent_chunks)
            pages = [
                page
                for chunk in parent_chunks
                for page in (chunk[2], chunk[3])
            ]
            printed_pages = [
                printed_page
                for chunk in parent_chunks
                for printed_page in (chunk[4], chunk[5])
                if printed_page
            ]
            page_start = min(pages)
            page_end = max(pages)
            page_warnings, warning_pages, document_warnings = _warnings_for_pages(
                warnings,
                page_start,
                page_end,
            )
            child_ids = [
                f"{parent_id}:c{child_index:04d}"
                for child_index in range(1, len(parent_chunks) + 1)
            ]
            records.append(
                {
                    "id": parent_id,
                    "record_type": "parent",
                    "document_id": document_id,
                    "domain": domain,
                    "source_file": source,
                    "source_hash": source_hash,
                    "converter_version": converter_version,
                    "chunker_version": CHUNKER_VERSION,
                    "heading": section.heading,
                    "section_path": section.path,
                    "section_part": section_part,
                    "section_parts": section_part_count,
                    "page_start": page_start,
                    "page_end": page_end,
                    "printed_page_start": printed_pages[0] if printed_pages else None,
                    "printed_page_end": printed_pages[-1] if printed_pages else None,
                    "content_type": "section",
                    "token_count": estimate_tokens(parent_content),
                    "content": parent_content,
                    "embedding_text": f"{section_prefix}\n\n{parent_content}",
                    "child_ids": child_ids,
                    "warnings": page_warnings + document_warnings,
                    "warning_pages": warning_pages,
                    "document_warnings": document_warnings,
                }
            )
            for child_index, chunk in enumerate(parent_chunks, 1):
                (
                    content,
                    content_type,
                    page_start,
                    page_end,
                    printed_page_start,
                    printed_page_end,
                ) = chunk
                page_warnings, warning_pages, document_warnings = _warnings_for_pages(
                    warnings,
                    page_start,
                    page_end,
                )
                child_id = child_ids[child_index - 1]
                records.append(
                    {
                        "id": child_id,
                        "record_type": "child",
                        "document_id": document_id,
                        "parent_id": parent_id,
                        "domain": domain,
                        "source_file": source,
                        "source_hash": source_hash,
                        "converter_version": converter_version,
                        "chunker_version": CHUNKER_VERSION,
                        "heading": section.heading,
                        "section_path": section.path,
                        "section_part": section_part,
                        "section_parts": section_part_count,
                        "page_start": page_start,
                        "page_end": page_end,
                        "printed_page_start": printed_page_start,
                        "printed_page_end": printed_page_end,
                        "content_type": content_type,
                        "token_count": estimate_tokens(content),
                        "content": content,
                        "embedding_text": f"{section_prefix}\n\n{content}",
                        "warnings": page_warnings + document_warnings,
                        "warning_pages": warning_pages,
                        "document_warnings": document_warnings,
                    }
                )
    return records


def _read_ledger(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as ledger:
        for line_number, line in enumerate(ledger, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if isinstance(record, dict):
                records.append(record)
    return records


def export_chunks(
    workspace: Path,
    output_path: Path,
    source_filter: str | None = None,
    max_tokens: int = 180,
    overlap_tokens: int = 20,
) -> int:
    ledger_path = workspace / "output" / "converted.jsonl"
    if not ledger_path.is_file():
        raise FileNotFoundError(f"Conversion ledger not found: {ledger_path}")

    ledger_records = _read_ledger(ledger_path)
    if source_filter is not None:
        ledger_records = [
            record
            for record in ledger_records
            if record.get("source") == source_filter
        ]
        if not ledger_records:
            raise ValueError(f"No conversion ledger entry found for {source_filter}")

    output_records: list[dict[str, Any]] = []
    missing_outputs: list[str] = []
    for record in ledger_records:
        markdown_info = record.get("markdown")
        if not isinstance(markdown_info, dict) or markdown_info.get("status") != "converted":
            continue
        relative_output = markdown_info.get("output")
        source = record.get("source")
        source_hash = record.get("source_sha256")
        if not isinstance(relative_output, str) or not isinstance(source, str):
            continue
        if not isinstance(source_hash, str):
            raise ValueError(f"Missing source hash for converted document {source}")
        markdown_path = workspace / relative_output
        if not markdown_path.is_file():
            missing_outputs.append(relative_output)
            continue
        markdown = markdown_path.read_text(encoding="utf-8")
        output_records.extend(
            build_document_records(
                markdown=markdown,
                source=source,
                source_hash=source_hash,
                converter_version=record.get("format_version"),
                warnings=record.get("warnings", []),
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
            )
        )
    if missing_outputs:
        raise FileNotFoundError(
            "Converted Markdown output is missing: " + ", ".join(missing_outputs)
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as destination:
            for record in output_records:
                destination.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    print(
        f"Wrote {len(output_records)} records "
        f"({sum(record['record_type'] == 'parent' for record in output_records)} parents, "
        f"{sum(record['record_type'] == 'child' for record in output_records)} children) "
        f"to {output_path}"
    )
    return len(output_records)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export converted Markdown as parent-child JSONL chunks."
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project workspace directory (defaults to the project root).",
    )
    parser.add_argument(
        "--source",
        help="Limit export to one source path relative to input/, e.g. regulations/hipaa.pdf.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSONL path (defaults to output/chunks.jsonl under the workspace).",
    )
    parser.add_argument(
        "--max-child-tokens",
        type=int,
        default=180,
        help="Approximate maximum child chunk size (default: 180).",
    )
    parser.add_argument(
        "--overlap-tokens",
        type=int,
        default=20,
        help="Approximate overlap only when a long paragraph must be split (default: 20).",
    )
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    output_path = args.output or workspace / "output" / "chunks.jsonl"
    if not output_path.is_absolute():
        output_path = workspace / output_path
    try:
        export_chunks(
            workspace=workspace,
            output_path=output_path,
            source_filter=args.source,
            max_tokens=args.max_child_tokens,
            overlap_tokens=args.overlap_tokens,
        )
    except (OSError, ValueError) as error:
        print(f"Chunk export failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
