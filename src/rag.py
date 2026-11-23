"""rag.py — chunk + index + retrieve for the ingest pipeline.

Chunking: content-type-aware & parser-fault-tolerant.
  - prose: section/page parents with overlapping child windows
  - table: whole markdown table as one chunk (with cross-page caption/header recovery)
  - table_row: one self-describing chunk per data row (linked to parent table)

Usage:
  python src/rag.py chunks                    # preview, no embedding
  python src/rag.py build                     # embed + persist
  python src/rag.py query "question"
  python src/rag.py query "question" --table  # tables/table_rows only
  python src/rag.py query "question" --page 9
  python src/rag.py stats
  python src/rag.py inspect <chunk_id>
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile
from typing import Any

try:
    from .spectrag import (
        ChunkGraph,
        build_chunk_graph,
        generate_utility_questions,
        load_utility_questions,
    )
except ImportError:
    from spectrag import (
        ChunkGraph,
        build_chunk_graph,
        generate_utility_questions,
        load_utility_questions,
    )

DEFAULT_OUTPUT = Path("output")
DEFAULT_INDEX = Path("index")
COLLECTION = "documents"
INDEX_MANIFEST = "index_manifest.json"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
PARENT_MAX_TOKENS = 900
CHILD_MAX_TOKENS = 260
CHILD_OVERLAP_TOKENS = 45
EMBED_BATCH_SIZE = int(os.environ.get("RAG_EMBED_BATCH_SIZE", "16"))
INDEX_VERSION = "v5"

NOISE_LINE_RE = re.compile(
    r"^(?:"
    r"\d{1,2}/\d{1,2}/\d{2,4}(?:,\s*\d{1,2}:\d{2}(?::\d{2})?\s*(?:[AP]M)?)?"
    r"|VerDate\b.*"
    r"|Jkt\s+\d+.*"
    r"|PO\s+\d+.*"
    r"|Frm\s+\d+.*"
    r"|Fmt\s+\d+.*"
    r"|Sfmt\s+\d+.*"
    r"|Q:\\.*"
    r")",
    re.IGNORECASE,
)
CFR_RUNNING_HEADER_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?§\s*\d+(?:\.\d+)?\s+21\s+CFR\s+Ch\.\s*[IVX]+(?:\s|\()",
    re.IGNORECASE,
)
TOC_LINK_RE = re.compile(
    r"^<u>\s*table\s+of\s+contents\s*</u>$", re.IGNORECASE
)
TOC_DOT_LEADER_RE = re.compile(r"\.{4,}\s*\d+\s*$")
INLINE_TAG_RE = re.compile(
    r"</?(?:sup|sub|mark|u|em|b|strong|span|div)>", re.IGNORECASE
)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
MD_EMPHASIS_STRIP_RE = re.compile(r"^[\s*_`~]+|[\s*_`~]+$")
BOLD_LINE_RE = re.compile(r"^(?:\*\*|__)\s*\S.*?\S\s*(?:\*\*|__)$")
DECIMAL_OUTLINE_RE = re.compile(
    r"^(\d{1,2}(?:\.\d{1,2})+)\b[\s.:–—\-]+([A-Z][^\n]{2,140})$"
)
MAJOR_OUTLINE_RE = re.compile(
    r"^((?:\d{1,2}|[IVXLC]{1,5})\.?)\s+([A-Z][^\n]{2,120})$"
)
SENTENCE_END_RE = re.compile(r"[.;!?]\s*$")
HEADING_WRAP_TRAILERS = (
    " and",
    " or",
    " for",
    " with",
    " in",
    " of",
    " to",
    " by",
    " on",
    " at",
    " from",
    ":",
    "-",
    "–",
    "—",
    "/",
)
TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
TABLE_SEP_RE = re.compile(r"^\s*\|(?:\s*:?-+:?\s*\|)+\s*$")
ESCAPED_PIPE_SPLIT_RE = re.compile(r"(?<!\\)\|")
GENERIC_HEADER_RE = re.compile(
    r"^(?:col(?:umn)?\s*\d+|field\s*\d+|unnamed:\s*\d+|c\d+|none|-+)?$",
    re.IGNORECASE,
)
TOKEN_SPAN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
CAPTION_RE = re.compile(
    r"^\s*\*{0,2}((?:Table|Exhibit|Figure)\s+\d+[A-Za-z0-9.\-]*[^\n]*?)\*{0,2}\s*$",
    re.IGNORECASE,
)
FTS_TERM_RE = re.compile(r"[A-Za-z0-9]+(?:[.\-/][A-Za-z0-9]+)*", re.UNICODE)
FTS_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "could", "did",
    "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is",
    "it", "its", "itself", "just", "me", "more", "most", "my", "myself",
    "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or",
    "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she",
    "should", "so", "some", "such", "than", "that", "the", "their", "theirs",
    "them", "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "we",
    "were", "what", "when", "where", "which", "while", "who", "whom", "why",
    "will", "with", "would", "you", "your", "yours", "yourself", "yourselves",
}


# ---------------------------------------------------------------- noise
def strip_noise(text: str) -> str:
    out = []
    for line in text.splitlines():
        plain = re.sub(r"<[^>]+>", "", line).strip()
        if not plain:
            out.append("")
            continue
        if NOISE_LINE_RE.match(plain) or CFR_RUNNING_HEADER_RE.match(plain):
            continue
        if plain.lower() == "table of contents":
            continue
        if TOC_LINK_RE.match(line.strip()):
            continue
        out.append(INLINE_TAG_RE.sub("", line))
    return "\n".join(out).strip()


# ---------------------------------------------------------------- headings
def _clean_heading_candidate(raw: str) -> str:
    cleaned = INLINE_TAG_RE.sub("", raw).strip()
    cleaned = re.sub(r"\*{1,3}|_{1,3}", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return MD_EMPHASIS_STRIP_RE.sub("", cleaned).strip()


def _upper_ratio(text: str) -> float:
    alpha = [ch for ch in text if ch.isalpha()]
    if not alpha:
        return 0.0
    return sum(1 for ch in alpha if ch.isupper()) / len(alpha)


def _looks_like_wrapped_continuation(line: str) -> bool:
    stripped = line.strip()
    if not stripped or TABLE_ROW_RE.match(stripped) or HEADING_RE.match(stripped):
        return False
    if TOC_DOT_LEADER_RE.search(stripped):
        return False
    cleaned = _clean_heading_candidate(stripped)
    if not cleaned or len(cleaned) > 85:
        return False
    if re.match(r"^(?:\d{1,2}(?:\.\d{1,2})*\b|[-*•])", cleaned):
        return False
    if SENTENCE_END_RE.search(cleaned):
        return False
    return bool(re.match(r"^[A-Z0-9(]", cleaned))


def _detect_heading_line(
    line: str,
    *,
    prev_blank: bool = True,
    next_blank: bool = True,
) -> tuple[int, str] | None:
    stripped = line.strip()
    if not stripped or TABLE_ROW_RE.match(stripped):
        return None
    if TOC_DOT_LEADER_RE.search(stripped):
        return None

    # 1. Explicit Markdown '#' heading
    md_match = HEADING_RE.match(stripped)
    if md_match:
        level = len(md_match.group(1))
        title = _clean_heading_candidate(md_match.group(2))
        if title and not TOC_DOT_LEADER_RE.search(title):
            return level, title
        return None

    # Parser-dropped structural headings must start after a blank line (or top of page)
    if not prev_blank or len(stripped) > 165:
        return None

    is_bold_line = bool(BOLD_LINE_RE.match(stripped))
    cleaned = _clean_heading_candidate(stripped)
    if not cleaned or len(cleaned) < 3 or len(cleaned) > 140:
        return None
    if SENTENCE_END_RE.search(cleaned):
        return None

    # 2. Multi-level decimal outline heading: e.g. "1.2 System Architecture"
    dec_match = DECIMAL_OUTLINE_RE.match(cleaned)
    if dec_match:
        sec_num, title_rest = dec_match.group(1), dec_match.group(2).strip()
        words = title_rest.split()
        first_word = re.split(r"[\s,:;()\-/]+", title_rest, maxsplit=1)[0]
        # Require proper word start and typographic/layout support
        if (
            len(first_word) >= 2
            and first_word[0].isupper()
            and len(words) <= 16
            and (is_bold_line or next_blank or _upper_ratio(title_rest) >= 0.25)
        ):
            segments = sec_num.split(".")
            level = min(6, 1 + len(segments))
            return level, f"{sec_num} {title_rest}"

    # 3. Top-level numbered or Roman numeral heading: e.g. "1 INTRODUCTION", "**2. Overview**"
    maj_match = MAJOR_OUTLINE_RE.match(cleaned)
    if maj_match:
        sec_token = maj_match.group(1).rstrip(".")
        title_rest = maj_match.group(2).strip()
        if len(title_rest.split()) <= 14 and (
            is_bold_line or _upper_ratio(title_rest) >= 0.72
        ):
            return 2, f"{sec_token} {title_rest}"

    # 4. Standalone ALL-CAPS or Bold structural heading bounded by blank lines
    if next_blank and len(cleaned.split()) <= 12:
        alpha_count = sum(1 for ch in cleaned if ch.isalpha())
        if alpha_count >= 3 and _upper_ratio(cleaned) >= 0.85:
            return 2, cleaned
        if is_bold_line and cleaned[0].isupper():
            return 3, cleaned

    return None


def _extract_headings(text: str) -> list[tuple[int, int, str]]:
    """Extract both Markdown '#' headings and parser-dropped structural headings."""
    lines = text.splitlines(keepends=True)
    offsets = _line_offsets(text)
    headings: list[tuple[int, int, str]] = []
    i = 0
    while i < len(lines):
        prev_blank = (i == 0) or (not lines[i - 1].strip())
        next_blank = (i + 1 >= len(lines)) or (not lines[i + 1].strip())
        detected = _detect_heading_line(
            lines[i], prev_blank=prev_blank, next_blank=next_blank
        )
        if detected is None:
            i += 1
            continue
        level, title = detected
        start_offset = offsets[i]

        # Stitch wrapped multi-line headings (up to 2 continuation lines)
        wraps = 0
        while (
            wraps < 2
            and i + 1 < len(lines)
            and (
                title.lower().endswith(HEADING_WRAP_TRAILERS)
                or (
                    lines[i].strip().startswith("**")
                    and lines[i + 1].strip().startswith("**")
                )
            )
            and _looks_like_wrapped_continuation(lines[i + 1])
            and _detect_heading_line(
                lines[i + 1],
                prev_blank=False,
                next_blank=(i + 2 >= len(lines)) or (not lines[i + 2].strip()),
            )
            is None
        ):
            continuation = _clean_heading_candidate(lines[i + 1])
            if not continuation:
                break
            title = f"{title} {continuation}".strip()
            i += 1
            wraps += 1
        headings.append((start_offset, level, title))
        i += 1
    return headings


# ---------------------------------------------------------------- pages
@dataclass
class PageSpan:
    pdf_page: int
    printed_page: str
    char_start: int
    char_end: int


def load_pages(md_path: Path) -> list[PageSpan]:
    p = md_path.with_suffix(".pages.json")
    if not p.is_file():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    return [
        PageSpan(
            pdf_page=int(it["pdf_page"]),
            printed_page=str(it.get("printed_page") or ""),
            char_start=int(it["char_start"]),
            char_end=int(it["char_end"]),
        )
        for it in data
    ]


def source_name(md_path: Path) -> str:
    t = md_path.with_suffix(".tables.json")
    if t.is_file():
        try:
            return json.loads(t.read_text(encoding="utf-8")).get(
                "source_file", md_path.stem
            )
        except json.JSONDecodeError:
            pass
    return md_path.stem.split("--")[0] + ".pdf"


def _resolve_ledger_output_path(ledger_path: Path, output_rel: str) -> Path:
    """Resolve an output path recorded in converted.jsonl across directory layouts."""
    candidate = Path(output_rel)
    if candidate.is_absolute():
        return candidate.resolve()
    ledger_dir = ledger_path.resolve().parent
    for base in (ledger_dir, ledger_dir.parent, Path.cwd().resolve()):
        resolved = (base / candidate).resolve()
        if resolved.exists():
            return resolved
    return (ledger_dir / candidate).resolve()


def select_markdown_files(output_dir: Path) -> list[Path]:
    candidates = sorted(
        path
        for path in output_dir.rglob("*.md")
        if ".progress" not in path.parts
    )
    grouped: dict[str, list[Path]] = {}
    for path in candidates:
        relative = path.relative_to(output_dir)
        source = (relative.parent / source_name(path)).as_posix()
        grouped.setdefault(source, []).append(path)

    ledger_path = output_dir.parent / "converted.jsonl"
    active_outputs: dict[str, Path] = {}
    if ledger_path.is_file():
        with ledger_path.open(encoding="utf-8") as ledger:
            for line_number, line in enumerate(ledger, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid conversion ledger JSON at "
                        f"{ledger_path}:{line_number}: {error}"
                    ) from error
                if not isinstance(record, dict):
                    continue
                source = record.get("source")
                markdown = record.get("markdown")
                output = (
                    markdown.get("output")
                    if isinstance(markdown, dict)
                    and markdown.get("status") == "converted"
                    else None
                )
                if (
                    isinstance(source, str)
                    and isinstance(output, str)
                    and record.get("status")
                    in {"converted", "converted_with_warnings"}
                ):
                    active_outputs[source] = _resolve_ledger_output_path(
                        ledger_path, output
                    )

    selected: list[Path] = []
    ambiguous: list[str] = []
    for source, source_paths in sorted(grouped.items()):
        if len(source_paths) == 1:
            selected.extend(source_paths)
            continue

        active_path = active_outputs.get(source)
        matching = [
            path
            for path in source_paths
            if active_path is not None and path.resolve() == active_path
        ]
        if len(matching) == 1:
            selected.extend(matching)
            continue
        ambiguous.append(
            f"{source}: "
            + ", ".join(
                path.relative_to(output_dir).as_posix()
                for path in source_paths
            )
        )

    if ambiguous:
        details = "\n".join(f"  - {item}" for item in ambiguous)
        raise ValueError(
            "Multiple Markdown conversions exist for the same source, but "
            f"{ledger_path} does not identify one active output. "
            "Reconvert those sources or update the ledger before indexing:\n"
            f"{details}"
        )
    return sorted(selected)


# ---------------------------------------------------------------- chunk
@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    domain: str
    pdf_page: int
    printed_page: str
    section: str
    subsection: str
    section_path: list[str]
    chunk_type: str
    row_label: str
    caption: str
    citation: str
    char_start: int
    char_end: int
    parent_id: str = ""
    parent_text: str = ""

    def metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "source": self.source,
            "domain": self.domain,
            "pdf_page": self.pdf_page,
            "printed_page": self.printed_page,
            "section": self.section,
            "subsection": self.subsection,
            "chunk_type": self.chunk_type,
            "row_label": self.row_label,
            "caption": self.caption,
            "citation": self.citation,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "index_version": INDEX_VERSION,
        }
        if self.section_path:
            metadata["section_path"] = self.section_path
        if self.parent_id:
            metadata["parent_id"] = self.parent_id
            metadata["parent_text"] = self.parent_text
        return metadata


def _chroma_safe_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Ensure metadata values conform to ChromaDB scalar constraints."""
    safe: dict[str, Any] = {}
    for key, value in meta.items():
        if isinstance(value, list):
            safe[key] = " > ".join(str(v) for v in value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        else:
            safe[key] = str(value)
    return safe


def _cells(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith(r"\|"):
        s = s[:-1]
    return [
        c.replace(r"\|", "|").strip()
        for c in ESCAPED_PIPE_SPLIT_RE.split(s)
    ]


def _line_offsets(text: str) -> list[int]:
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _update_heading_stack(
    stack: list[tuple[int, str]], level: int, heading: str
) -> None:
    while stack and stack[-1][0] >= level:
        stack.pop()
    stack.append((level, heading))


def _section_path_at(
    pos: int,
    headings: list[tuple[int, int, str]],
    heading_context: list[tuple[int, str]],
) -> list[str]:
    stack = list(heading_context)
    for hpos, level, htext in headings:
        if hpos > pos:
            break
        _update_heading_stack(stack, level, htext)
    return [heading for _, heading in stack]


def _caption_near(
    text: str,
    start: int,
    *,
    leading_context: str = "",
) -> str:
    """Look at the lines right before the table (including prior page tail) for 'Table N...'."""
    before = text[max(0, start - 450) : start]
    if start < 350 and leading_context:
        before = leading_context[-350:] + "\n" + before
    for line in reversed(before.splitlines()):
        m = CAPTION_RE.match(line.strip())
        if m:
            return _clean_heading_candidate(m.group(1))
    return ""


def _is_generic_table_header(header: list[str]) -> bool:
    if not header:
        return True
    non_empty = [cell.strip() for cell in header if cell.strip()]
    if not non_empty:
        return True
    return all(GENERIC_HEADER_RE.match(cell) for cell in non_empty)


def _token_windows(
    text: str,
    start: int,
    end: int,
    max_tokens: int,
    overlap_tokens: int = 0,
) -> list[tuple[int, int]]:
    spans = list(TOKEN_SPAN_RE.finditer(text, start, end))
    windows: list[tuple[int, int]] = []
    token_start = 0
    while token_start < len(spans):
        token_stop = min(token_start + max_tokens, len(spans))
        if token_stop < len(spans):
            minimum_stop = token_start + max(1, int(max_tokens * 0.7))
            for candidate_stop in range(token_stop, minimum_stop - 1, -1):
                previous = spans[candidate_stop - 1]
                following = spans[candidate_stop]
                gap = text[previous.end() : following.start()]
                if previous.group() in {".", "!", "?"} or "\n" in gap:
                    token_stop = candidate_stop
                    break
        window_start = spans[token_start].start()
        window_end = spans[token_stop - 1].end()
        if text[window_start:window_end].strip():
            windows.append((window_start, window_end))
        if token_stop == len(spans):
            break
        token_start = max(token_start + 1, token_stop - overlap_tokens)
    return windows


def _format_scope_prefix(
    source: str,
    pdf_page: int,
    section: str,
    subsection: str,
    section_path: list[str],
    caption: str = "",
) -> str:
    cite = f"[{source} p.{pdf_page}]"
    scope = cite
    if section_path:
        scope += f" § {' > '.join(section_path)}"
    elif section:
        scope += f" § {section}"
        if subsection:
            scope += f" > {subsection}"
    if caption and caption.lower() not in scope.lower():
        scope += f" — {caption}"
    return scope


def _make_chunk(
    *,
    body: str,
    page: PageSpan,
    source: str,
    section: str,
    subsection: str,
    section_path: list[str],
    domain: str,
    chunk_type: str,
    row_label: str,
    caption: str,
    char_start: int,
    char_end: int,
    ordinal: int,
    document_id: str,
    parent_id: str = "",
    parent_text: str = "",
) -> Chunk:
    cite = f"[{source} p.{page.pdf_page}]"
    scope = _format_scope_prefix(
        source=source,
        pdf_page=page.pdf_page,
        section=section,
        subsection=subsection,
        section_path=section_path,
        caption=caption if chunk_type == "table" else "",
    )
    prefix = scope + "\n"
    clean_body = body.strip()
    full_text = prefix + clean_body if not clean_body.startswith(cite) else clean_body

    enriched_parent_text = parent_text.strip()
    if enriched_parent_text and not enriched_parent_text.startswith(cite):
        enriched_parent_text = prefix + enriched_parent_text

    cid = (
        f"{Path(source).stem}--{document_id}--p{page.pdf_page}"
        f"--{chunk_type}--{ordinal}"
    )
    return Chunk(
        chunk_id=cid,
        text=full_text,
        source=source,
        domain=domain,
        pdf_page=page.pdf_page,
        printed_page=page.printed_page,
        section=section,
        subsection=subsection,
        section_path=section_path,
        chunk_type=chunk_type,
        row_label=row_label,
        caption=caption,
        citation=cite,
        char_start=char_start,
        char_end=char_end,
        parent_id=parent_id,
        parent_text=enriched_parent_text,
    )


def chunk_page(
    text: str,
    page: PageSpan,
    source: str,
    document_id: str = "",
    domain: str = "uncategorized",
    heading_context: list[tuple[int, str]] | None = None,
    *,
    leading_context: str = "",
    carryover_table: dict[str, Any] | None = None,
) -> list[Chunk]:
    text = strip_noise(text)
    if not text.strip():
        return []
    if not document_id:
        document_id = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]

    lines = text.splitlines(keepends=True)
    offsets = _line_offsets(text)

    # Detect tables (line ranges)
    tables: list[tuple[int, int]] = []
    i = 0
    while i < len(lines) - 1:
        if TABLE_ROW_RE.match(lines[i]) and TABLE_SEP_RE.match(lines[i + 1]):
            start = i
            j = i + 2
            while j < len(lines) and TABLE_ROW_RE.match(lines[j]):
                j += 1
            tables.append((start, j))
            i = j
        else:
            i += 1

    headings = _extract_headings(text)
    initial_heading_context = list(heading_context) if heading_context else []

    heading_offsets = [heading[0] for heading in headings]
    atoms: list[dict[str, Any]] = []

    def add_prose_atoms(start: int, end: int) -> None:
        boundaries = [
            offset for offset in heading_offsets if start < offset < end
        ]
        points = [start, *boundaries, end]
        atoms.extend(
            {"kind": "prose", "start": left, "end": right}
            for left, right in zip(points, points[1:])
            if text[left:right].strip()
        )

    cursor = 0
    for ts, te in tables:
        if cursor < ts:
            add_prose_atoms(offsets[cursor], offsets[ts])
        atoms.append(
            {
                "kind": "table",
                "start": offsets[ts],
                "end": offsets[te],
                "header": _cells(lines[ts]),
                "body": [
                    (_cells(lines[row_line]), offsets[row_line], offsets[row_line + 1])
                    for row_line in range(ts + 2, te)
                ],
            }
        )
        cursor = te
    if cursor < len(lines):
        add_prose_atoms(offsets[cursor], offsets[-1])

    chunks: list[Chunk] = []
    for atom in atoms:
        section_path = _section_path_at(
            atom["start"], headings, initial_heading_context
        )
        section = section_path[0] if section_path else ""
        subsection = " > ".join(section_path[1:])

        if atom["kind"] == "prose":
            parent_windows = _token_windows(
                text, atom["start"], atom["end"], PARENT_MAX_TOKENS
            )
            for parent_ordinal, (parent_start, parent_end) in enumerate(
                parent_windows
            ):
                parent_text = text[parent_start:parent_end].strip()
                parent_id = (
                    f"{document_id}--p{page.pdf_page}--parent"
                    f"{atom['start']:08d}-{parent_ordinal:04}"
                )
                child_windows = _token_windows(
                    text,
                    parent_start,
                    parent_end,
                    CHILD_MAX_TOKENS,
                    CHILD_OVERLAP_TOKENS,
                )
                for s, e in child_windows:
                    body = text[s:e].strip()
                    if not body:
                        continue
                    chunks.append(
                        _make_chunk(
                            body=body,
                            page=page,
                            source=source,
                            section=section,
                            subsection=subsection,
                            section_path=section_path,
                            domain=domain,
                            chunk_type="prose",
                            row_label="",
                            caption="",
                            char_start=s,
                            char_end=e,
                            ordinal=len(chunks),
                            document_id=document_id,
                            parent_id=parent_id,
                            parent_text=parent_text,
                        )
                    )
            continue

        caption = _caption_near(
            text,
            atom["start"],
            leading_context=leading_context if atom["start"] < 350 else "",
        )
        header = list(atom["header"])
        body_entries: list[tuple[list[str], int, int]] = list(atom["body"])
        if not header or not body_entries:
            continue

        # Recover caption/header for tables split across a page boundary
        if (
            atom["start"] < 280
            and carryover_table
            and carryover_table.get("pdf_page") == page.pdf_page - 1
        ):
            if not caption and carryover_table.get("caption"):
                caption = str(carryover_table["caption"])
            prev_header = carryover_table.get("header") or []
            if (
                len(prev_header) == len(header)
                and _is_generic_table_header(header)
                and not _is_generic_table_header(prev_header)
            ):
                header = list(prev_header)

        # Whole-table chunk
        md = []
        if caption:
            md.append(caption)
        md.append("| " + " | ".join(header) + " |")
        md.append("|" + "|".join(["---"] * len(header)) + "|")
        for row, _, _ in body_entries:
            cells = row + [""] * max(0, len(header) - len(row))
            md.append("| " + " | ".join(cells[: len(header)]) + " |")
        table_markdown = "\n".join(md)
        table_parent_id = (
            f"{document_id}--p{page.pdf_page}--tableparent"
            f"{atom['start']:08d}"
        )
        bounded_table_parent = (
            table_markdown if len(table_markdown) <= 4200 else table_markdown[:4200]
        )
        chunks.append(
            _make_chunk(
                body=table_markdown,
                page=page,
                source=source,
                section=section,
                subsection=subsection,
                section_path=section_path,
                domain=domain,
                chunk_type="table",
                row_label="",
                caption=caption,
                char_start=atom["start"],
                char_end=atom["end"],
                ordinal=len(chunks),
                document_id=document_id,
                parent_id=table_parent_id,
                parent_text=bounded_table_parent,
            )
        )

        # One chunk per data row (carrying forward spanning divider group rows safely)
        active_group_label = ""
        for row, row_start, row_end in body_entries:
            padded = row + [""] * max(0, len(header) - len(row))
            cells = [c.strip() for c in padded[: len(header)]]
            first_cell = cells[0] if cells else ""
            rest_empty = not any(cells[1:])

            # A row with text only in column 0 and empty remaining columns is a spanning group header
            if first_cell and rest_empty and len(header) > 1:
                active_group_label = first_cell
                continue

            fields = [
                f"{header[index]}: {cell}"
                for index, cell in enumerate(cells)
                if cell
            ]
            if not fields:
                continue
            if active_group_label and not first_cell:
                fields.insert(0, f"{header[0] or 'Group'}: {active_group_label}")
            elif active_group_label and first_cell != active_group_label:
                fields.insert(0, f"Group: {active_group_label}")

            row_lines = []
            if caption:
                row_lines.append(caption)
            if section_path:
                row_lines.append(f"Section: {' > '.join(section_path)}")
            row_lines.extend(
                [
                    f"Columns: {' | '.join(header)}",
                    f"Row: {' | '.join(fields)}",
                ]
            )
            row_label = first_cell or active_group_label
            chunks.append(
                _make_chunk(
                    body="\n".join(row_lines),
                    page=page,
                    source=source,
                    section=section,
                    subsection=subsection,
                    section_path=section_path,
                    domain=domain,
                    chunk_type="table_row",
                    row_label=row_label,
                    caption=caption,
                    char_start=row_start,
                    char_end=row_end,
                    ordinal=len(chunks),
                    document_id=document_id,
                    parent_id=table_parent_id,
                    parent_text=bounded_table_parent,
                )
            )

        if carryover_table is not None:
            carryover_table.clear()
            trailing_text = text[atom["end"] :].strip()
            has_heading_after = any(hpos >= atom["end"] for hpos, _, _ in headings)
            # Only carry over to the next page if this table ended near the page bottom
            if len(trailing_text) < 160 and not has_heading_after:
                carryover_table.update(
                    {
                        "pdf_page": page.pdf_page,
                        "header": header,
                        "caption": caption,
                        "section_path": section_path,
                    }
                )

    return chunks


def build_chunks(
    output_dir: Path,
    markdown_files: list[Path] | None = None,
) -> list[Chunk]:
    out: list[Chunk] = []
    files = (
        markdown_files
        if markdown_files is not None
        else select_markdown_files(output_dir)
    )
    for md_path in files:
        text = md_path.read_text(encoding="utf-8")
        pages = load_pages(md_path)
        src = source_name(md_path)
        relative_path = md_path.relative_to(output_dir)
        domain = (
            relative_path.parts[0]
            if len(relative_path.parts) > 1
            else "uncategorized"
        )
        document_id = hashlib.sha256(
            relative_path.as_posix().encode("utf-8")
        ).hexdigest()[:12]
        if not pages:
            pages = [PageSpan(1, "", 0, len(text))]

        heading_context: list[tuple[int, str]] = []
        carryover_table: dict[str, Any] = {}
        previous_page_clean = ""

        for page in pages:
            page_text = text[page.char_start : page.char_end]
            page_chunks = chunk_page(
                page_text,
                page,
                src,
                document_id,
                domain,
                heading_context,
                leading_context=previous_page_clean,
                carryover_table=carryover_table,
            )
            out.extend(page_chunks)

            cleaned_page = strip_noise(page_text)
            for _, level, heading in _extract_headings(cleaned_page):
                _update_heading_stack(heading_context, level, heading)
            if cleaned_page:
                previous_page_clean = cleaned_page

    deduplicated: list[Chunk] = []
    seen_tables: set[tuple[str, int, str, str]] = set()
    removed_duplicates = 0
    for chunk in out:
        if chunk.chunk_type not in {"table", "table_row"}:
            deduplicated.append(chunk)
            continue
        key = (chunk.source, chunk.pdf_page, chunk.chunk_type, chunk.text)
        if key in seen_tables:
            removed_duplicates += 1
            continue
        seen_tables.add(key)
        deduplicated.append(chunk)
    if removed_duplicates:
        print(
            f"[chunks] removed {removed_duplicates} exact duplicate "
            "table chunk(s) within active documents"
        )
    return deduplicated


# ---------------------------------------------------------------- index
@lru_cache(maxsize=1)
def _embedding_model() -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL)


def _encode_texts(
    texts: list[str],
) -> list[list[float]]:
    if not texts:
        return []
    model = _embedding_model()
    try:
        embeddings = model.encode(
            texts,
            batch_size=max(1, EMBED_BATCH_SIZE),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    except RuntimeError:
        embeddings = model.encode(
            texts,
            batch_size=1,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    return embeddings.tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed search queries with BGE's asymmetric query instruction prefix."""
    if not texts:
        return []
    prefixed = [
        text
        if text.startswith(BGE_QUERY_PREFIX)
        else f"{BGE_QUERY_PREFIX}{text}"
        for text in texts
    ]
    return _encode_texts(prefixed)


def _embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed indexed passages without a query prompt."""
    return _encode_texts(texts)


def _validate_index_manifest(index_dir: Path, query_dimension: int) -> None:
    manifest_path = index_dir / INDEX_MANIFEST
    if not manifest_path.is_file():
        raise RuntimeError(
            f"Embedding manifest is missing from {index_dir}; rebuild the index "
            "with `python src/rag.py build` before querying."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Could not read embedding manifest at {manifest_path}; rebuild the index."
        ) from error
    if not isinstance(manifest, dict):
        raise RuntimeError(
            f"Embedding manifest at {manifest_path} is invalid; rebuild the index."
        )

    expected = {
        "embedding_model": EMBED_MODEL,
        "embedding_dimension": query_dimension,
        "index_version": INDEX_VERSION,
    }
    mismatches = [
        f"{key}: index has {manifest.get(key)!r}, runtime expects {value!r}"
        for key, value in expected.items()
        if manifest.get(key) != value
    ]
    if mismatches:
        raise RuntimeError(
            "The index is incompatible with the configured embedding model or "
            "index format ("
            + "; ".join(mismatches)
            + "). Rebuild it with `python src/rag.py build`."
        )


def build_index(
    chunks: list[Chunk],
    index_dir: Path,
    utility_questions_path: Path | None = None,
) -> None:
    import chromadb
    import numpy as np

    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / INDEX_MANIFEST).unlink(missing_ok=True)
    client = chromadb.PersistentClient(path=str(index_dir))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    coll = client.create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine"}
    )

    BATCH = max(1, EMBED_BATCH_SIZE)
    n = len(chunks)
    indexed_embeddings: list[np.ndarray] = []
    for i in range(0, n, BATCH):
        batch = chunks[i : i + BATCH]
        embeddings = np.asarray(
            _embed_documents([chunk.text for chunk in batch]),
            dtype=np.float32,
        )
        indexed_embeddings.extend(embeddings)
        coll.add(
            ids=[c.chunk_id for c in batch],
            documents=[c.text for c in batch],
            metadatas=[_chroma_safe_metadata(c.metadata()) for c in batch],
            embeddings=embeddings.tolist(),
        )
        print(f"  indexed {min(i + BATCH, n)}/{n}", flush=True)

    utility_embeddings = None
    if utility_questions_path is not None and indexed_embeddings:
        questions_by_id = load_utility_questions(utility_questions_path)
        non_empty_pairs = [
            (idx, "\n".join(questions_by_id[chunk.chunk_id]))
            for idx, chunk in enumerate(chunks)
            if questions_by_id.get(chunk.chunk_id)
        ]
        if non_empty_pairs:
            dim = int(indexed_embeddings[0].shape[0])
            utility_embeddings = np.zeros((len(chunks), dim), dtype=np.float32)
            valid_indices, valid_questions = zip(*non_empty_pairs)
            utility_embeddings[list(valid_indices)] = np.asarray(
                embed_texts(list(valid_questions)), dtype=np.float32
            )

    graph = build_chunk_graph(
        chunks,
        np.asarray(indexed_embeddings, dtype=np.float32),
        utility_embeddings=utility_embeddings,
    )
    graph.save(index_dir)
    build_lexical_index(chunks, index_dir)
    embedding_dimension = (
        int(indexed_embeddings[0].shape[0])
        if indexed_embeddings
        else int(_embedding_model().get_sentence_embedding_dimension())
    )
    manifest = {
        "embedding_model": EMBED_MODEL,
        "embedding_dimension": embedding_dimension,
        "index_version": INDEX_VERSION,
        "chunk_count": len(chunks),
    }
    (index_dir / INDEX_MANIFEST).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def get_index_chunks(
    chunk_ids: list[str], index_dir: Path
) -> list[dict[str, Any]]:
    """Fetch stored chunk text and metadata in the requested ID order."""
    if not chunk_ids:
        return []
    import chromadb

    client = chromadb.PersistentClient(path=str(index_dir))
    collection = client.get_collection(COLLECTION)
    result = collection.get(ids=chunk_ids, include=["documents", "metadatas"])
    by_id = {
        chunk_id: {
            "id": chunk_id,
            "text": text,
            "metadata": metadata or {},
        }
        for chunk_id, text, metadata in zip(
            result["ids"], result["documents"], result["metadatas"], strict=True
        )
    }
    return [by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_id]


def build_lexical_index(chunks: list[Chunk], index_dir: Path) -> None:
    """Build an on-disk FTS5 index for fast lexical retrieval."""
    index_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".lexical-index-",
        suffix=".sqlite3",
        dir=index_dir,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    destination = index_dir / "lexical.sqlite3"
    try:
        with sqlite3.connect(temporary_path) as connection:
            connection.execute(
                """
                CREATE VIRTUAL TABLE lexical_chunks USING fts5(
                    id UNINDEXED,
                    text,
                    source UNINDEXED,
                    domain UNINDEXED,
                    pdf_page UNINDEXED,
                    chunk_type UNINDEXED,
                    metadata UNINDEXED,
                    tokenize = 'unicode61'
                )
                """
            )
            connection.executemany(
                """
                INSERT INTO lexical_chunks
                    (id, text, source, domain, pdf_page, chunk_type, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.chunk_id,
                        chunk.text,
                        chunk.source,
                        chunk.domain,
                        chunk.pdf_page,
                        chunk.chunk_type,
                        json.dumps(
                            chunk.metadata(),
                            ensure_ascii=False,
                        ),
                    )
                    for chunk in chunks
                ],
            )
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _index_where(
    page: int | None,
    table_only: bool,
    domain: str | None,
) -> dict[str, Any] | None:
    filters: list[dict[str, Any]] = []
    if page is not None:
        filters.append({"pdf_page": page})
    if table_only:
        filters.append({"chunk_type": {"$in": ["table", "table_row"]}})
    if domain is not None:
        filters.append({"domain": domain})
    if len(filters) > 1:
        return {"$and": filters}
    return filters[0] if filters else None


def _sanitize_fts_terms(terms: list[str]) -> list[str]:
    all_tokens: list[str] = []
    filtered_tokens: list[str] = []
    seen: set[str] = set()
    for raw in terms:
        if not raw:
            continue
        for match in FTS_TERM_RE.finditer(str(raw)):
            token = match.group(0).strip(".-/")
            if not token:
                continue
            if len(token) == 1 and not token.isdigit():
                continue
            lowered = token.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            escaped = token.replace('"', '""')
            all_tokens.append(escaped)
            if lowered not in FTS_STOPWORDS:
                filtered_tokens.append(escaped)
    return filtered_tokens if filtered_tokens else all_tokens


def query_lexical_index(
    index_dir: Path,
    terms: list[str],
    top_k: int,
    page: int | None = None,
    table_only: bool = False,
    domain: str | None = None,
) -> list[dict[str, Any]]:
    """Return top BM25 matches from the persisted SQLite FTS5 index."""
    clean_terms = _sanitize_fts_terms(terms)
    if not clean_terms:
        return []
    lexical_path = index_dir / "lexical.sqlite3"
    if not lexical_path.is_file():
        raise FileNotFoundError(
            f"Lexical index not found at {lexical_path}; rebuild the index "
            "with `python src/rag.py build`."
        )

    match_query = " OR ".join(f'"{term}"' for term in clean_terms)
    filters = ["lexical_chunks MATCH ?"]
    parameters: list[Any] = [match_query]
    if page is not None:
        filters.append("pdf_page = ?")
        parameters.append(page)
    if table_only:
        filters.append("chunk_type IN (?, ?)")
        parameters.extend(["table", "table_row"])
    if domain is not None:
        filters.append("domain = ?")
        parameters.append(domain)
    parameters.append(top_k)

    with sqlite3.connect(lexical_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT id, text, metadata, bm25(lexical_chunks) AS score
            FROM lexical_chunks
            WHERE """
            + " AND ".join(filters)
            + " ORDER BY score, id LIMIT ?",
            parameters,
        ).fetchall()
    return [
        {
            "id": row["id"],
            "text": row["text"],
            "metadata": json.loads(row["metadata"]),
            "bm25_score": row["score"],
        }
        for row in rows
    ]


def query_index(
    query: str,
    index_dir: Path,
    top_k: int = 5,
    page: int | None = None,
    table_only: bool = False,
    domain: str | None = None,
) -> list[dict[str, Any]]:
    import chromadb

    query_embeddings = embed_texts([query])
    if not query_embeddings or not query_embeddings[0]:
        return []
    _validate_index_manifest(index_dir, len(query_embeddings[0]))

    client = chromadb.PersistentClient(path=str(index_dir))
    coll = client.get_collection(COLLECTION)

    where = _index_where(page, table_only, domain)

    kwargs: dict[str, Any] = {
        "query_embeddings": query_embeddings,
        "n_results": top_k,
    }
    if where:
        kwargs["where"] = where
    res = coll.query(**kwargs)

    hits = []
    for i, cid in enumerate(res["ids"][0]):
        hits.append(
            {
                "id": cid,
                "distance": res["distances"][0][i] if res.get("distances") else None,
                "text": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
            }
        )
    return hits


# ---------------------------------------------------------------- cli
def cmd_chunks(output_dir: Path) -> None:
    markdown_files = select_markdown_files(output_dir)
    candidate_count = sum(
        1
        for path in output_dir.rglob("*.md")
        if ".progress" not in path.parts
    )
    if len(markdown_files) < candidate_count:
        print(
            f"[chunks] using {len(markdown_files)} active documents; "
            f"skipping {candidate_count - len(markdown_files)} stale "
            "conversion(s)"
        )
    chunks = build_chunks(output_dir, markdown_files)
    Path("chunks.jsonl").write_text(
        "\n".join(json.dumps(c.__dict__, ensure_ascii=False) for c in chunks),
        encoding="utf-8",
    )
    breakdown: dict[str, int] = {}
    for c in chunks:
        breakdown[c.chunk_type] = breakdown.get(c.chunk_type, 0) + 1
    print(f"chunks: {len(chunks)}  breakdown: {breakdown}")
    print("wrote chunks.jsonl")
    for c in chunks[:3]:
        print(f"\n[{c.chunk_id}] type={c.chunk_type} page={c.pdf_page}")
        print(c.text[:280] + ("…" if len(c.text) > 280 else ""))


def cmd_utility_questions(
    output_dir: Path, destination: Path, batch_size: int
) -> None:
    markdown_files = select_markdown_files(output_dir)
    chunks = build_chunks(output_dir, markdown_files)
    generated = generate_utility_questions(
        chunks, destination, batch_size=batch_size
    )
    print(
        f"utility questions: {generated} generated, "
        f"{len(chunks) - generated} already present in {destination}"
    )


def cmd_build(
    output_dir: Path,
    index_dir: Path,
    utility_questions_path: Path | None = None,
) -> None:
    print(f"[build] scanning {output_dir}")
    markdown_files = select_markdown_files(output_dir)
    candidate_count = sum(
        1
        for path in output_dir.rglob("*.md")
        if ".progress" not in path.parts
    )
    if len(markdown_files) < candidate_count:
        print(
            f"[build] using {len(markdown_files)} active documents; "
            f"skipping {candidate_count - len(markdown_files)} stale "
            "conversion(s)"
        )
    chunks = build_chunks(output_dir, markdown_files)
    if not chunks:
        print("ERROR: no chunks; is output/ populated?", flush=True)
        return
    breakdown: dict[str, int] = {}
    for c in chunks:
        breakdown[c.chunk_type] = breakdown.get(c.chunk_type, 0) + 1
    print(f"[build] {len(chunks)} chunks  {breakdown}")
    Path("chunks.jsonl").write_text(
        "\n".join(json.dumps(c.__dict__, ensure_ascii=False) for c in chunks),
        encoding="utf-8",
    )
    print(f"[build] embedding into {index_dir}")
    build_index(chunks, index_dir, utility_questions_path)
    print("[build] done")


def cmd_query(
    q: str,
    index_dir: Path,
    top_k: int,
    page: int | None,
    table_only: bool,
    domain: str | None = None,
) -> None:
    hits = query_index(q, index_dir, top_k, page, table_only, domain)
    if not hits:
        print("no results")
        return
    for i, h in enumerate(hits, 1):
        m = h["metadata"]
        d = f"{h['distance']:.4f}" if h["distance"] is not None else "?"
        print(f"\n--- [{i}] d={d}  {m['citation']}  type={m['chunk_type']}")
        if m["section"]:
            print(
                f"    § {m['section']}"
                + (f" > {m['subsection']}" if m["subsection"] else "")
            )
        if m["caption"]:
            print(f"    caption: {m['caption']}")
        context_text = m.get("parent_text") or h["text"]
        print("    " + context_text[:350].replace("\n", " ")[:350])


def cmd_stats(index_dir: Path) -> None:
    import chromadb

    client = chromadb.PersistentClient(path=str(index_dir))
    coll = client.get_collection(COLLECTION)
    print(f"chunks: {coll.count()}")


def cmd_inspect(chunk_id: str) -> None:
    path = Path("chunks.jsonl")
    if not path.is_file():
        print("no chunks.jsonl; run `python rag.py chunks` first")
        return
    for line in path.open(encoding="utf-8"):
        d = json.loads(line)
        if d["chunk_id"] == chunk_id:
            print(json.dumps(d, indent=2, ensure_ascii=False))
            return
    print(f"chunk {chunk_id} not found")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("chunks")
    c.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)

    u = sub.add_parser(
        "utility-questions",
        help="Generate resumable SpectRAG utility questions using OpenRouter.",
    )
    u.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    u.add_argument(
        "--questions-output",
        type=Path,
        default=Path("output/utility-questions.jsonl"),
    )
    u.add_argument("--batch-size", type=int, default=8)

    b = sub.add_parser("build")
    b.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    b.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX)
    b.add_argument(
        "--utility-questions",
        type=Path,
        help="Optional JSONL records with chunk_id and generated question list.",
    )

    q = sub.add_parser("query")
    q.add_argument("query")
    q.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX)
    q.add_argument("--top-k", type=int, default=5)
    q.add_argument("--page", type=int)
    q.add_argument("--table", action="store_true")
    q.add_argument("--domain", help="Limit results to a document folder/domain.")

    s = sub.add_parser("stats")
    s.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX)

    i = sub.add_parser("inspect")
    i.add_argument("chunk_id")

    a = p.parse_args()
    if a.cmd == "chunks":
        cmd_chunks(a.output)
    elif a.cmd == "utility-questions":
        if a.batch_size < 1:
            p.error("--batch-size must be positive")
        cmd_utility_questions(a.output, a.questions_output, a.batch_size)
    elif a.cmd == "build":
        cmd_build(a.output, a.index_dir, a.utility_questions)
    elif a.cmd == "query":
        cmd_query(a.query, a.index_dir, a.top_k, a.page, a.table, a.domain)
    elif a.cmd == "stats":
        cmd_stats(a.index_dir)
    elif a.cmd == "inspect":
        cmd_inspect(a.chunk_id)


if __name__ == "__main__":
    main()