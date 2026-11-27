"""rag.py — chunk + index + retrieve for the ingest pipeline.
Chunking: content-type-aware, discourse-preserving & parser-fault-tolerant.
  - prose: section/page parents with syntactic, list-aware overlapping child windows
           and cross-page sentence stitching
  - table: whole markdown table as one chunk (with cross-page caption/header recovery,
           unit-row merging, and table footnote capture)
  - table_row: one self-describing chunk per data row (linked to parent table + footnotes)
Usage:
  python src/rag.py chunks                    # preview, no embedding
  python src/rag.py build                     # embed + persist
    python src/rag.py add                       # incrementally sync sources
    python src/rag.py delete source.pdf         # remove one source
  python src/rag.py query "question"
  python src/rag.py query "question" --table  # tables/table_rows only
  python src/rag.py query "question" --page 9
  python src/rag.py stats
  python src/rag.py inspect <chunk_id>
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import threading
import time
from typing import Any
if __package__:
    from .locking import file_lock
    from .spectrag import (
        ChunkGraph,
        build_chunk_graph,
        generate_utility_questions,
        load_utility_questions,
    )
else:
    from locking import file_lock
    from spectrag import (
        ChunkGraph,
        build_chunk_graph,
        generate_utility_questions,
        load_utility_questions,
    )
DEFAULT_OUTPUT = Path(os.environ.get("RAG_OUTPUT_DIR", "output"))
DEFAULT_INDEX = Path(os.environ.get("RAG_INDEX_DIR", "index"))
COLLECTION = "documents"
INDEX_MANIFEST = "index_manifest.json"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
PARENT_MAX_TOKENS = 900
CHILD_MAX_TOKENS = 260
CHILD_OVERLAP_TOKENS = 55
EMBED_BATCH_SIZE = int(os.environ.get("RAG_EMBED_BATCH_SIZE", "32"))
INDEX_VERSION = "v5"
SQLITE_BUSY_TIMEOUT_MS = 30000
_EMBEDDER_LOCK = threading.RLock()
_CHROMA_LOCK = threading.RLock()
_CACHE_LOCK = threading.RLock()
_BUILD_LOCK = threading.RLock()
_CHROMA_CLIENTS: dict[str, Any] = {}
_MANIFEST_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_CHUNKS_BY_ID_CACHE: dict[str, tuple[float, dict[str, dict[str, Any]]]] = {}


@contextmanager
def _index_file_lock(index_dir: Path):
    with file_lock(index_dir / ".index.lock"):
        yield


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
    r"|Page\s+\d+\s+of\s+\d+\s*$"
    r")",
    re.IGNORECASE,
)
CFR_RUNNING_HEADER_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?§\s*\d+(?:\.\d+)?\s+\d+\s+CFR\s+Ch\.\s*[IVXLC]+(?:\s|\()",
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
    r"^(\d{1,3}(?:\.\d{1,3})+)\b[\s.:–—\-]+([A-Z0-9][^\n]{2,140})$"
)
NAMED_STRUCTURAL_HEADING_RE = re.compile(
    r"^((?:Section|Article|Chapter|Part|Subpart|Appendix|Annex|Schedule|Exhibit|Rule|Item|Clause)\s+"
    r"[A-Za-z0-9]+(?:\.[A-Za-z0-9\-()]+)*|§+\s*\d+(?:\.[A-Za-z0-9\-()]+)*)"
    r"(?:[\s.:–—\-]+([A-Z0-9][^\n]{1,130}))?$",
    re.IGNORECASE,
)
MAJOR_OUTLINE_RE = re.compile(
    r"^((?:\d{1,2}|[IVXLC]{1,5})\.?)\s+([A-Z][^\n]{2,120})$"
)
SENTENCE_END_RE = re.compile(r"[.;!?]\s*$")
INCOMPLETE_PAGE_TAIL_RE = re.compile(
    r"[,:;–—\-(\[/]\s*$|\b(?:and|or|to|of|in|for|with|by|that|which|than|as|if|unless|except)\s*$",
    re.IGNORECASE,
)
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
UNIT_CELL_RE = re.compile(
    r"^\s*(?:\([^()\n]{1,30}\)|\[[^\[\]\n]{1,30}\]|(?:in\s+)?(?:\$|%|bps|ms|s|kg|mg|g|L|mL|Hz|kHz|MHz|GHz|W|kW|MW|V|mV|A|mA|°C|°F|K|ppm|ppb|thousands|millions|billions))\s*$",
    re.IGNORECASE,
)
TOKEN_SPAN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
CAPTION_RE = re.compile(
    r"^\s*\*{0,2}((?:(?:Appendix\s+)?Table|Exhibit|Figure|Schedule|Matrix|Chart)\s+"
    r"(?:[A-Z0-9]+(?:\.[A-Za-z0-9\-]+)*|[IVXLC]+)\b[^\n]*?)\*{0,2}\s*$",
    re.IGNORECASE,
)
SCALE_UNIT_LINE_RE = re.compile(
    r"^\s*[\*(\[]*\s*(?:in\s+(?:thousands|millions|billions|percent|percentages)|"
    r"amounts?\s+in\b|values?\s+in\b|figures?\s+in\b|all\s+data\s+in\b|units?\s*:|"
    r"\$\s*in\s+(?:thousands|millions|billions)|except\s+per\s+share)\b[^\n]{0,100}[\*)\]]*\s*$",
    re.IGNORECASE,
)
TABLE_FOOTNOTE_LINE_RE = re.compile(
    r"^\s*(?:[*†‡§#]{1,3}|\(\d{1,2}\)|\[\d{1,2}\]|\d{1,2}/|"
    r"\([a-z]\)|Note\s*:|Notes\s*:|Source\s*:|Abbreviations?\s*:)\s+\S+",
    re.IGNORECASE,
)
LIST_ITEM_START_RE = re.compile(
    r"^\s*(?:[-*•‣▪]|\(\d{1,3}\)|\([a-zA-Z]\)|\([ivxlcIVXLC]{1,5}\)|\d{1,3}[.)]|[a-zA-Z][.)])\s+"
)
DISCOURSE_CONTINUATION_RE = re.compile(
    r"^\s*(?:However|Provided\s+that|Except|Unless|Notwithstanding|Furthermore|"
    r"Moreover|In\s+addition|Specifically|For\s+example|Note\s*:|Caution\s*:|Warning\s*:)\b",
    re.IGNORECASE,
)
ABBREVIATIONS = frozenset(
    {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "fig", "figs",
        "eq", "eqs", "sec", "secs", "art", "no", "nos", "vol", "vols",
        "pp", "ch", "pt", "dept", "est", "approx", "min", "max", "avg",
        "vs", "v", "etc", "al", "eg", "ie", "cf", "inc", "ltd", "co", "corp",
    }
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
# ---------------------------------------------------------------- atomic I/O & SQLite helpers
def _atomic_write_text(target_path: Path, content: str) -> None:
    """Write text atomically via a temporary file + fsync + os.replace."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(target_path.parent),
            prefix=f".{target_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target_path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
def _atomic_write_json(target_path: Path, payload: Any) -> None:
    """Serialize JSON atomically so concurrent readers never observe partial files."""
    _atomic_write_text(
        target_path,
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    )
def _connect_sqlite(
    db_path: Path,
    *,
    readonly: bool = False,
) -> sqlite3.Connection:
    """Open a SQLite connection configured for concurrent multi-threaded access."""
    if readonly:
        uri = f"{db_path.resolve().as_uri()}?mode=ro"
        conn = sqlite3.connect(
            uri,
            uri=True,
            timeout=SQLITE_BUSY_TIMEOUT_MS / 1000.0,
            check_same_thread=False,
        )
    else:
        conn = sqlite3.connect(
            str(db_path),
            timeout=SQLITE_BUSY_TIMEOUT_MS / 1000.0,
            check_same_thread=False,
        )
    conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS};")
    conn.execute("PRAGMA temp_store=MEMORY;")
    if not readonly:
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
        except sqlite3.OperationalError:
            pass
    return conn
def _get_chroma_client(index_dir: Path, *, reset: bool = False) -> Any:
    """Return a thread-safe cached ChromaDB PersistentClient for index_dir."""
    import chromadb
    key = str(index_dir.resolve())
    with _CHROMA_LOCK:
        if reset or key not in _CHROMA_CLIENTS:
            _CHROMA_CLIENTS[key] = chromadb.PersistentClient(path=str(index_dir))
        return _CHROMA_CLIENTS[key]
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
    if SENTENCE_END_RE.search(cleaned) and not NAMED_STRUCTURAL_HEADING_RE.match(cleaned):
        return None
    # 2. Universal named structural heading: "Section 4.2 ...", "Article IV ...", "§ 240.14a-8 ..."
    named_match = NAMED_STRUCTURAL_HEADING_RE.match(cleaned)
    if named_match:
        prefix_part = named_match.group(1).strip()
        rest_part = (named_match.group(2) or "").strip()
        words = cleaned.split()
        if len(words) <= 18 and (is_bold_line or next_blank or not rest_part or rest_part[0].isupper()):
            low_pref = prefix_part.lower()
            if low_pref.startswith(("part", "chapter", "article", "appendix", "annex", "schedule", "exhibit")):
                level = 2
            elif "." in prefix_part:
                level = min(6, 2 + prefix_part.count("."))
            else:
                level = 3
            full_heading = f"{prefix_part} {rest_part}".strip() if rest_part else prefix_part
            return level, full_heading
    # 3. Multi-level decimal outline heading: e.g. "1.2 System Architecture"
    dec_match = DECIMAL_OUTLINE_RE.match(cleaned)
    if dec_match:
        sec_num, title_rest = dec_match.group(1), dec_match.group(2).strip()
        words = title_rest.split()
        first_word = re.split(r"[\s,:;()\-/]+", title_rest, maxsplit=1)[0]
        if (
            len(first_word) >= 2
            and first_word[0].isupper()
            and len(words) <= 16
            and (is_bold_line or next_blank or _upper_ratio(title_rest) >= 0.25)
        ):
            segments = sec_num.split(".")
            level = min(6, 1 + len(segments))
            return level, f"{sec_num} {title_rest}"
    # 4. Top-level numbered or Roman numeral heading: e.g. "1 INTRODUCTION", "**2. Overview**"
    maj_match = MAJOR_OUTLINE_RE.match(cleaned)
    if maj_match:
        sec_token = maj_match.group(1).rstrip(".")
        title_rest = maj_match.group(2).strip()
        if len(title_rest.split()) <= 14 and (
            is_bold_line or _upper_ratio(title_rest) >= 0.72
        ):
            return 2, f"{sec_token} {title_rest}"
    # 5. Standalone ALL-CAPS or Bold structural heading bounded by blank lines
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


def _document_id(md_path: Path, output_dir: Path) -> str:
    relative_path = md_path.relative_to(output_dir).as_posix()
    return hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:12]


def _source_fingerprint(md_path: Path) -> str:
    digest = hashlib.sha256()
    for path in (
        md_path,
        md_path.with_suffix(".pages.json"),
        md_path.with_suffix(".tables.json"),
    ):
        digest.update(path.name.encode("utf-8"))
        if path.is_file():
            digest.update(b"\0")
            digest.update(path.read_bytes())
        else:
            digest.update(b"\1")
    return digest.hexdigest()


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
    """Look at the lines right before the table (including prior page tail) for 'Table N...' and scale/unit descriptors."""
    before = text[max(0, start - 500) : start]
    if start < 350 and leading_context:
        before = leading_context[-350:] + "\n" + before
    lines = [line.strip() for line in before.splitlines() if line.strip()]
    caption_part = ""
    unit_part = ""
    for line in reversed(lines[-6:]):
        if not unit_part and SCALE_UNIT_LINE_RE.match(line):
            unit_part = _clean_heading_candidate(line)
            continue
        m = CAPTION_RE.match(line)
        if m:
            caption_part = _clean_heading_candidate(m.group(1))
            break
    if caption_part and unit_part and unit_part.lower() not in caption_part.lower():
        return f"{caption_part} ({unit_part})"
    return caption_part or unit_part
def _table_footnotes_after(text: str, end: int) -> str:
    """Capture immediate table footnotes (*, †, [1], Note:, Source:) following a table."""
    after = text[end : min(len(text), end + 650)]
    footnotes: list[str] = []
    blank_streak = 0
    for line in after.splitlines():
        stripped = line.strip()
        if not stripped:
            blank_streak += 1
            if blank_streak > 1 and footnotes:
                break
            continue
        blank_streak = 0
        if TABLE_ROW_RE.match(stripped) or HEADING_RE.match(stripped):
            break
        if TABLE_FOOTNOTE_LINE_RE.match(stripped) and len(stripped) <= 260:
            footnotes.append(stripped)
            if len(footnotes) >= 5:
                break
        else:
            break
    return "\n".join(footnotes)
def _is_generic_table_header(header: list[str]) -> bool:
    if not header:
        return True
    non_empty = [cell.strip() for cell in header if cell.strip()]
    if not non_empty:
        return True
    return all(GENERIC_HEADER_RE.match(cell) for cell in non_empty)
def _is_unit_qualifier_row(row: list[str]) -> bool:
    """Return True if a table's first body row consists purely of unit/scale qualifiers like '($ in M)' or '(%)'."""
    non_empty = [c.strip() for c in row if c.strip()]
    if not non_empty:
        return False
    return all(UNIT_CELL_RE.match(c) for c in non_empty)
def _is_real_sentence_break(
    spans: list[re.Match[str]],
    candidate_stop: int,
    text: str,
) -> bool:
    """Verify that a punctuation span is a true sentence end, not a decimal (3.14) or abbreviation (Fig. 1)."""
    if candidate_stop <= 0 or candidate_stop >= len(spans):
        return False
    prev = spans[candidate_stop - 1]
    foll = spans[candidate_stop]
    tok = prev.group()
    if tok not in {".", "!", "?", ";"}:
        return False
    gap = text[prev.end() : foll.start()]
    # Must be followed by whitespace (rules out decimals like 3.14 or § 240.14a)
    if not gap or not gap[0].isspace():
        return False
    if tok == "." and candidate_stop >= 2:
        word_before = spans[candidate_stop - 2]
        if word_before.end() == prev.start():
            w_low = word_before.group().lower()
            if w_low in ABBREVIATIONS or (len(w_low) == 1 and w_low.isalpha()):
                return False
    next_tok = foll.group()
    return bool(
        next_tok
        and (
            next_tok[0].isupper()
            or next_tok[0].isdigit()
            or next_tok[0] in {'"', "'", "(", "[", "§"}
        )
    )
def _token_windows(
    text: str,
    start: int,
    end: int,
    max_tokens: int,
    overlap_tokens: int = 0,
) -> list[tuple[int, int]]:
    """Slice text into token windows using a 4-tier syntactic & discourse boundary hierarchy."""
    spans = list(TOKEN_SPAN_RE.finditer(text, start, end))
    if not spans:
        return []
    windows: list[tuple[int, int]] = []
    token_start = 0
    while token_start < len(spans):
        token_stop = min(token_start + max_tokens, len(spans))
        if token_stop < len(spans):
            minimum_stop = token_start + max(1, int(max_tokens * 0.65))
            best_tier1: int | None = None  # Paragraph break (\n\n) not ending in ':'
            best_tier2: int | None = None  # Clean sentence break not before 'However/Unless/Except'
            best_tier3: int | None = None  # Single line break (\n) not ending in ':'
            best_tier4: int | None = None  # Fallback sentence/newline break
            for candidate_stop in range(token_stop, minimum_stop - 1, -1):
                previous = spans[candidate_stop - 1]
                following = spans[candidate_stop]
                gap = text[previous.end() : following.start()]
                prev_tok = previous.group()
                ends_with_colon = prev_tok == ":"
                if "\n\n" in gap:
                    if not ends_with_colon and best_tier1 is None:
                        best_tier1 = candidate_stop
                        break
                    if best_tier4 is None:
                        best_tier4 = candidate_stop
                if _is_real_sentence_break(spans, candidate_stop, text):
                    lookahead = text[following.start() : min(end, following.start() + 45)]
                    if not DISCOURSE_CONTINUATION_RE.match(lookahead) and best_tier2 is None:
                        best_tier2 = candidate_stop
                    elif best_tier4 is None:
                        best_tier4 = candidate_stop
                if "\n" in gap:
                    if not ends_with_colon and best_tier3 is None:
                        best_tier3 = candidate_stop
                    elif best_tier4 is None:
                        best_tier4 = candidate_stop
            chosen = best_tier1 or best_tier2 or best_tier3 or best_tier4
            if chosen is not None:
                token_stop = chosen
        window_start = spans[token_start].start()
        window_end = spans[token_stop - 1].end()
        if text[window_start:window_end].strip():
            windows.append((window_start, window_end))
        if token_stop == len(spans):
            break
        next_start = max(token_start + 1, token_stop - overlap_tokens)
        # Snap overlap start to the beginning of a sentence or line if nearby
        snap_limit = min(token_stop, next_start + 12)
        for cand in range(next_start, snap_limit):
            if cand > 0:
                gap = text[spans[cand - 1].end() : spans[cand].start()]
                if "\n" in gap or _is_real_sentence_break(spans, cand, text):
                    next_start = cand
                    break
        token_start = next_start
    return windows
def _extract_governing_leadin(parent_text: str) -> str:
    """If a parent block begins with a governing premise ending in ':' before a list, extract it."""
    lines = [ln.strip() for ln in parent_text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return ""
    for idx in range(min(3, len(lines) - 1)):
        line = lines[idx]
        if line.endswith(":") and 15 <= len(line) <= 240:
            if LIST_ITEM_START_RE.match(lines[idx + 1]) or idx == 0:
                return line
    return ""
def _extract_incomplete_page_tail(text: str) -> str:
    """If a page ends mid-sentence, return the trailing incomplete sentence fragment."""
    cleaned = text.strip()
    if not cleaned:
        return ""
    last_line = cleaned.splitlines()[-1].strip()
    if not last_line or TABLE_ROW_RE.match(last_line) or HEADING_RE.match(last_line):
        return ""
    if SENTENCE_END_RE.search(last_line) and not INCOMPLETE_PAGE_TAIL_RE.search(last_line):
        return ""
    # Extract from the last sentence boundary in the final 320 characters
    tail_window = cleaned[-320:]
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", tail_window)
    fragment = parts[-1].strip()
    if 10 <= len(fragment) <= 280 and not SENTENCE_END_RE.search(fragment):
        return re.sub(r"\s+", " ", fragment)
    return ""
def _extract_leading_sentence_completion(text: str) -> str:
    """If a page starts by finishing the previous page's sentence, extract that completion."""
    cleaned = strip_noise(text)
    if not cleaned:
        return ""
    first_lines: list[str] = []
    for line in cleaned.splitlines():
        s = line.strip()
        if not s:
            if first_lines:
                break
            continue
        if TABLE_ROW_RE.match(s) or HEADING_RE.match(s):
            return ""
        first_lines.append(s)
        if SENTENCE_END_RE.search(s) or len(" ".join(first_lines)) >= 240:
            break
    if not first_lines:
        return ""
    joined = " ".join(first_lines)
    match = re.search(r"^(.+?[.!?])(?:\s|$)", joined)
    snippet = match.group(1).strip() if match else joined[:220].strip()
    return snippet if len(snippet) >= 5 else ""
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
    trailing_context: str = "",
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
    # Cross-page sentence stitching fragments
    prior_page_tail = _extract_incomplete_page_tail(leading_context) if leading_context else ""
    next_page_head = (
        _extract_leading_sentence_completion(trailing_context)
        if trailing_context and _extract_incomplete_page_tail(text)
        else ""
    )
    chunks: list[Chunk] = []
    for atom_idx, atom in enumerate(atoms):
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
                # Stitch cross-page sentence fragments into boundary parents
                if (
                    atom_idx == 0
                    and parent_ordinal == 0
                    and prior_page_tail
                    and not any(hpos <= parent_start for hpos, _, _ in headings)
                ):
                    parent_text = f"{prior_page_tail} {parent_text}"
                if (
                    atom_idx == len(atoms) - 1
                    and parent_ordinal == len(parent_windows) - 1
                    and next_page_head
                ):
                    parent_text = f"{parent_text} {next_page_head}"
                governing_leadin = _extract_governing_leadin(parent_text)
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
                for child_idx, (s, e) in enumerate(child_windows):
                    body = text[s:e].strip()
                    if not body:
                        continue
                    if (
                        atom_idx == 0
                        and parent_ordinal == 0
                        and child_idx == 0
                        and prior_page_tail
                        and not any(hpos <= s for hpos, _, _ in headings)
                    ):
                        body = f"{prior_page_tail} {body}"
                    if (
                        atom_idx == len(atoms) - 1
                        and parent_ordinal == len(parent_windows) - 1
                        and child_idx == len(child_windows) - 1
                        and next_page_head
                    ):
                        body = f"{body} {next_page_head}"
                    # If a subsequent child window starts mid-list, attach the governing lead-in
                    if (
                        child_idx > 0
                        and governing_leadin
                        and governing_leadin not in body
                        and LIST_ITEM_START_RE.match(body)
                    ):
                        body = f"[Context: {governing_leadin}]\n{body}"
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
        footnotes = _table_footnotes_after(text, atom["end"])
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
        # Merge unit-qualifier sub-header row (e.g. "($ in millions) | (%)") into column headers
        if len(body_entries) > 1 and _is_unit_qualifier_row(body_entries[0][0]):
            unit_row = body_entries[0][0]
            header = [
                f"{h} {unit_row[idx].strip()}".strip()
                if idx < len(unit_row) and unit_row[idx].strip()
                else h
                for idx, h in enumerate(header)
            ]
            body_entries = body_entries[1:]
        # Whole-table chunk
        md = []
        if caption:
            md.append(caption)
        md.append("| " + " | ".join(header) + " |")
        md.append("|" + "|".join(["---"] * len(header)) + "|")
        for row, _, _ in body_entries:
            cells = row + [""] * max(0, len(header) - len(row))
            md.append("| " + " | ".join(cells[: len(header)]) + " |")
        if footnotes:
            md.append(f"Footnotes:\n{footnotes}")
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
            if footnotes and len(footnotes) <= 280:
                row_lines.append(f"Footnotes: {footnotes.replace(chr(10), ' ; ')}")
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
        document_id = _document_id(md_path, output_dir)
        if not pages:
            pages = [PageSpan(1, "", 0, len(text))]
        heading_context: list[tuple[int, str]] = []
        carryover_table: dict[str, Any] = {}
        previous_page_clean = ""
        for page_idx, page in enumerate(pages):
            page_text = text[page.char_start : page.char_end]
            next_page_text = (
                text[pages[page_idx + 1].char_start : pages[page_idx + 1].char_end]
                if page_idx + 1 < len(pages)
                else ""
            )
            page_chunks = chunk_page(
                page_text,
                page,
                src,
                document_id,
                domain,
                heading_context,
                leading_context=previous_page_clean,
                trailing_context=next_page_text,
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


def _build_source_inventory(
    output_dir: Path,
    markdown_files: list[Path],
    chunks: list[Chunk],
) -> dict[str, dict[str, Any]]:
    paths_by_id = {
        _document_id(path, output_dir): path for path in markdown_files
    }
    chunk_ids_by_document: dict[str, list[str]] = {
        document_id: [] for document_id in paths_by_id
    }
    for chunk in chunks:
        match = re.search(r"--([0-9a-f]{12})--p\d+--", chunk.chunk_id)
        if match and match.group(1) in chunk_ids_by_document:
            chunk_ids_by_document[match.group(1)].append(chunk.chunk_id)
    inventory: dict[str, dict[str, Any]] = {}
    for document_id, path in paths_by_id.items():
        relative_path = path.relative_to(output_dir).as_posix()
        inventory[relative_path] = {
            "document_id": document_id,
            "fingerprint": _source_fingerprint(path),
            "source": source_name(path),
            "chunk_ids": chunk_ids_by_document[document_id],
        }
    return inventory


# ---------------------------------------------------------------- index
@lru_cache(maxsize=1)
def _embedding_model_cached() -> Any:
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)
def _embedding_model() -> Any:
    with _EMBEDDER_LOCK:
        return _embedding_model_cached()
def _encode_texts(
    texts: list[str],
) -> list[list[float]]:
    if not texts:
        return []
    with _EMBEDDER_LOCK:
        model = _embedding_model()

        def encode(batch: list[str], batch_size: int):
            for attempt in range(3):
                try:
                    return model.encode(
                        batch,
                        batch_size=batch_size,
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                    )
                except (OSError, TimeoutError, RuntimeError):
                    if attempt == 2:
                        raise
                    time.sleep(0.25 * (2**attempt))

        try:
            embeddings = encode(texts, max(1, EMBED_BATCH_SIZE))
        except RuntimeError:
            embeddings = encode(texts, 1)
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
    cache_key = str(manifest_path.resolve())
    try:
        mtime = manifest_path.stat().st_mtime
        with _CACHE_LOCK:
            cached = _MANIFEST_CACHE.get(cache_key)
            if cached is not None and cached[0] == mtime:
                manifest = cached[1]
            else:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if isinstance(manifest, dict):
                    _MANIFEST_CACHE[cache_key] = (mtime, manifest)
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
    source_inventory: dict[str, dict[str, Any]] | None = None,
) -> None:
    import numpy as np
    with _BUILD_LOCK, _index_file_lock(index_dir):
        index_dir.mkdir(parents=True, exist_ok=True)
        (index_dir / INDEX_MANIFEST).unlink(missing_ok=True)
        client = _get_chroma_client(index_dir, reset=True)
        with _CHROMA_LOCK:
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
            with _CHROMA_LOCK:
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
            "sources": source_inventory or {},
            "utility_questions_path": (
                str(utility_questions_path.resolve())
                if utility_questions_path is not None
                else None
            ),
        }
        _atomic_write_json(index_dir / INDEX_MANIFEST, manifest)
        with _CACHE_LOCK:
            _MANIFEST_CACHE.pop(str((index_dir / INDEX_MANIFEST).resolve()), None)
            _CHUNKS_BY_ID_CACHE.pop(str((index_dir / "chunks.json").resolve()), None)


def _collection_snapshot(collection: Any) -> tuple[list[dict[str, Any]], Any]:
    import numpy as np

    result = collection.get(include=["embeddings", "documents", "metadatas"])
    ids = result.get("ids") or []
    documents = result.get("documents") or [""] * len(ids)
    metadatas = result.get("metadatas") or [{}] * len(ids)
    raw_embeddings = result.get("embeddings")
    if not ids:
        return [], np.empty((0, 0), dtype=np.float32)
    if raw_embeddings is None:
        raise RuntimeError("The vector collection did not return stored embeddings")
    embeddings = np.asarray(raw_embeddings, dtype=np.float32)
    records: list[dict[str, Any]] = []
    for chunk_id, text, metadata in zip(ids, documents, metadatas, strict=True):
        fields = dict(metadata or {})
        section_path = fields.get("section_path")
        if isinstance(section_path, str):
            fields["section_path"] = [
                part for part in section_path.split(" > ") if part
            ]
        records.append(
            {
                "chunk_id": str(chunk_id),
                "text": str(text or ""),
                **fields,
            }
        )
    if embeddings.ndim != 2 or embeddings.shape[0] != len(records):
        raise RuntimeError("The vector collection returned an invalid embedding set")
    return records, embeddings


def _rebuild_graph_from_collection(
    index_dir: Path,
    collection: Any,
    invalidated_ids: set[str],
    utility_questions_path: Path | None,
) -> None:
    import numpy as np

    records, embeddings = _collection_snapshot(collection)
    previous_graph = None
    try:
        previous_graph = ChunkGraph.load(index_dir)
    except Exception:
        pass

    old_utility: dict[str, Any] = {}
    if previous_graph is not None and previous_graph.utility_embeddings is not None:
        old_utility = dict(
            zip(previous_graph.ids, previous_graph.utility_embeddings, strict=True)
        )

    questions_by_id: dict[str, list[str]] = {}
    if utility_questions_path is not None and utility_questions_path.is_file():
        questions_by_id = load_utility_questions(utility_questions_path)

    utility_embeddings = None
    if old_utility:
        dimension = len(next(iter(old_utility.values())))
        utility_embeddings = np.zeros((len(records), dimension), dtype=np.float32)
        query_indices: list[int] = []
        query_texts: list[str] = []
        for index, record in enumerate(records):
            chunk_id = record["chunk_id"]
            if chunk_id in old_utility and chunk_id not in invalidated_ids:
                utility_embeddings[index] = old_utility[chunk_id]
            elif questions_by_id.get(chunk_id):
                query_indices.append(index)
                query_texts.append("\n".join(questions_by_id[chunk_id]))
        if query_texts:
            for index, vector in zip(
                query_indices, embed_texts(query_texts), strict=True
            ):
                utility_embeddings[index] = vector
    elif questions_by_id:
        query_indices = [
            index
            for index, record in enumerate(records)
            if questions_by_id.get(record["chunk_id"])
        ]
        query_texts = [
            "\n".join(questions_by_id[records[index]["chunk_id"]])
            for index in query_indices
        ]
        if query_texts:
            vectors = embed_texts(query_texts)
            utility_embeddings = np.zeros(
                (len(records), len(vectors[0])), dtype=np.float32
            )
            for index, vector in zip(query_indices, vectors, strict=True):
                utility_embeddings[index] = vector

    graph = build_chunk_graph(
        records,
        embeddings,
        utility_embeddings=utility_embeddings,
    )
    graph.save(index_dir)


def _update_lexical_index(
    index_dir: Path,
    deleted_ids: set[str],
    chunks: list[Chunk],
) -> None:
    lexical_path = index_dir / "lexical.sqlite3"
    if not lexical_path.is_file():
        raise FileNotFoundError(
            f"Lexical index not found at {lexical_path}; run `python src/rag.py build`."
        )
    with _connect_sqlite(lexical_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        if deleted_ids:
            connection.executemany(
                "DELETE FROM lexical_chunks WHERE id = ?",
                ((chunk_id,) for chunk_id in deleted_ids),
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
                    json.dumps(chunk.metadata(), ensure_ascii=False),
                )
                for chunk in chunks
            ],
        )


def _apply_index_delta(
    index_dir: Path,
    manifest: dict[str, Any],
    sources: dict[str, dict[str, Any]],
    old_chunk_ids: set[str],
    new_chunks: list[Chunk],
    utility_questions_path: Path | None,
) -> None:
    client = _get_chroma_client(index_dir)
    collection = client.get_collection(COLLECTION)
    new_ids = {chunk.chunk_id for chunk in new_chunks}
    if new_chunks:
        for start in range(0, len(new_chunks), max(1, EMBED_BATCH_SIZE)):
            batch = new_chunks[start : start + max(1, EMBED_BATCH_SIZE)]
            embeddings = _embed_documents([chunk.text for chunk in batch])
            collection.upsert(
                ids=[chunk.chunk_id for chunk in batch],
                documents=[chunk.text for chunk in batch],
                metadatas=[_chroma_safe_metadata(chunk.metadata()) for chunk in batch],
                embeddings=embeddings,
            )
    deleted_ids = old_chunk_ids - new_ids
    if deleted_ids:
        collection.delete(ids=sorted(deleted_ids))
    _update_lexical_index(index_dir, old_chunk_ids, new_chunks)
    _rebuild_graph_from_collection(
        index_dir,
        collection,
        old_chunk_ids | new_ids,
        utility_questions_path,
    )
    manifest["sources"] = sources
    manifest["chunk_count"] = int(collection.count())
    if utility_questions_path is not None:
        manifest["utility_questions_path"] = str(utility_questions_path.resolve())
    _atomic_write_json(index_dir / INDEX_MANIFEST, manifest)
    with _CACHE_LOCK:
        _MANIFEST_CACHE.pop(str((index_dir / INDEX_MANIFEST).resolve()), None)
        _CHUNKS_BY_ID_CACHE.pop(str((index_dir / "chunks.json").resolve()), None)


def _load_chunks_by_id_cache(index_dir: Path) -> dict[str, dict[str, Any]] | None:
    """Load and mtime-cache `chunks.json` if present for lock-free chunk hydration."""
    chunks_path = index_dir / "chunks.json"
    if not chunks_path.is_file():
        return None
    cache_key = str(chunks_path.resolve())
    try:
        mtime = chunks_path.stat().st_mtime
        with _CACHE_LOCK:
            cached = _CHUNKS_BY_ID_CACHE.get(cache_key)
            if cached is not None and cached[0] == mtime:
                return cached[1]
        raw = json.loads(chunks_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            return None
        by_id: dict[str, dict[str, Any]] = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            cid = item.get("id") or item.get("chunk_id")
            if not cid:
                continue
            meta = item.get("metadata")
            if not isinstance(meta, dict):
                meta = {
                    k: v
                    for k, v in item.items()
                    if k not in {"id", "chunk_id", "text"}
                }
            by_id[str(cid)] = {
                "id": str(cid),
                "text": str(item.get("text") or ""),
                "metadata": meta,
            }
        with _CACHE_LOCK:
            _CHUNKS_BY_ID_CACHE[cache_key] = (mtime, by_id)
        return by_id
    except Exception:
        return None
def get_index_chunks(
    chunk_ids: list[str], index_dir: Path
) -> list[dict[str, Any]]:
    """Fetch stored chunk text and metadata in the requested ID order."""
    if not chunk_ids:
        return []
    cached_by_id = _load_chunks_by_id_cache(index_dir)
    if cached_by_id is not None and all(cid in cached_by_id for cid in chunk_ids):
        return [cached_by_id[cid] for cid in chunk_ids if cid in cached_by_id]
    with _CHROMA_LOCK:
        client = _get_chroma_client(index_dir)
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
        with _connect_sqlite(temporary_path, readonly=False) as connection:
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
            # Switch back to DELETE journal mode before atomic file replacement so no stray .wal/.shm files remain
            connection.execute("PRAGMA journal_mode=DELETE;")
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
    def _add_token(tok: str) -> None:
        if not tok or (len(tok) == 1 and not tok.isdigit()):
            return
        low = tok.lower()
        if low in seen:
            return
        seen.add(low)
        escaped = tok.replace('"', '""')
        all_tokens.append(escaped)
        if low not in FTS_STOPWORDS:
            filtered_tokens.append(escaped)
    for raw in terms:
        if not raw:
            continue
        for match in FTS_TERM_RE.finditer(str(raw)):
            token = match.group(0).strip(".-/")
            _add_token(token)
            # Also index sub-parts of compound identifiers (e.g. "240.14a-8" -> "240", "14a", "8")
            if any(sep in token for sep in (".", "-", "/")):
                for sub in re.split(r"[.\-/]+", token):
                    if len(sub) >= 2 or sub.isdigit():
                        _add_token(sub)
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
    if top_k <= 0:
        return []
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
    parameters.append(int(top_k))
    with _connect_sqlite(lexical_path, readonly=True) as connection:
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
    if top_k <= 0:
        return []
    query_embeddings = embed_texts([query])
    if not query_embeddings or not query_embeddings[0]:
        return []
    _validate_index_manifest(index_dir, len(query_embeddings[0]))
    where = _index_where(page, table_only, domain)
    kwargs: dict[str, Any] = {
        "query_embeddings": query_embeddings,
        "n_results": int(top_k),
    }
    if where:
        kwargs["where"] = where
    with _CHROMA_LOCK:
        client = _get_chroma_client(index_dir)
        coll = client.get_collection(COLLECTION)
        res = coll.query(**kwargs)
    hits = []
    ids_list = (res.get("ids") or [[]])[0]
    docs_list = (res.get("documents") or [[]])[0]
    metas_list = (res.get("metadatas") or [[]])[0]
    dists_list = (res.get("distances") or [[]])[0] if res.get("distances") else []
    for i, cid in enumerate(ids_list):
        hits.append(
            {
                "id": cid,
                "distance": dists_list[i] if i < len(dists_list) else None,
                "text": docs_list[i] if i < len(docs_list) else "",
                "metadata": metas_list[i] if i < len(metas_list) and metas_list[i] else {},
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
    _atomic_write_text(
        Path("chunks.jsonl"),
        "\n".join(json.dumps(c.__dict__, ensure_ascii=False) for c in chunks),
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
    _atomic_write_text(
        Path("chunks.jsonl"),
        "\n".join(json.dumps(c.__dict__, ensure_ascii=False) for c in chunks),
    )
    print(f"[build] embedding into {index_dir}")
    source_inventory = _build_source_inventory(
        output_dir, markdown_files, chunks
    )
    build_index(
        chunks,
        index_dir,
        utility_questions_path,
        source_inventory=source_inventory,
    )
    print("[build] done")


def _read_index_manifest(index_dir: Path) -> dict[str, Any]:
    manifest_path = index_dir / INDEX_MANIFEST
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Could not read index manifest at {manifest_path}; run `build` first."
        ) from error
    if not isinstance(value, dict):
        raise RuntimeError(f"Index manifest at {manifest_path} is invalid.")
    return value


def _utility_questions_path(
    manifest: dict[str, Any], requested_path: Path | None
) -> Path | None:
    if requested_path is not None:
        return requested_path
    saved_path = manifest.get("utility_questions_path")
    return Path(saved_path) if isinstance(saved_path, str) and saved_path else None


def cmd_add(
    output_dir: Path,
    index_dir: Path,
    utility_questions_path: Path | None = None,
) -> None:
    bootstrap_build = False
    with _BUILD_LOCK, _index_file_lock(index_dir):
        try:
            manifest = _read_index_manifest(index_dir)
        except RuntimeError:
            manifest = {}
        existing_sources = manifest.get("sources")
        if not isinstance(existing_sources, dict):
            bootstrap_build = True
        else:
            _validate_index_manifest(
                index_dir, int(manifest.get("embedding_dimension", 0))
            )
            markdown_files = select_markdown_files(output_dir)
            current_paths = {
                path.relative_to(output_dir).as_posix(): path
                for path in markdown_files
            }
            changed_paths = [
                path
                for relative_path, path in current_paths.items()
                if not isinstance(existing_sources.get(relative_path), dict)
                or existing_sources[relative_path].get("fingerprint")
                != _source_fingerprint(path)
            ]
            removed_paths = set(existing_sources) - set(current_paths)
            if not changed_paths and not removed_paths:
                print("[add] index is up to date")
                return

            sources = dict(existing_sources)
            old_chunk_ids: set[str] = set()
            new_chunks: list[Chunk] = []
            for path in changed_paths:
                relative_path = path.relative_to(output_dir).as_posix()
                old_record = sources.pop(relative_path, {})
                old_ids = old_record.get("chunk_ids", [])
                if "chunk_ids" not in old_record or not isinstance(old_ids, list):
                    raise RuntimeError(
                        f"Invalid chunk inventory for {relative_path}; run `build`."
                    )
                old_chunk_ids.update(str(chunk_id) for chunk_id in old_ids)
                document_chunks = build_chunks(output_dir, [path])
                new_chunks.extend(document_chunks)
                record = _build_source_inventory(
                    output_dir, [path], document_chunks
                )[relative_path]
                sources[relative_path] = record

            for relative_path in removed_paths:
                record = sources.pop(relative_path)
                old_ids = record.get("chunk_ids", [])
                if "chunk_ids" not in record or not isinstance(old_ids, list):
                    raise RuntimeError(
                        f"Invalid chunk inventory for {relative_path}; run `build`."
                    )
                old_chunk_ids.update(str(chunk_id) for chunk_id in old_ids)

            questions_path = _utility_questions_path(
                manifest, utility_questions_path
            )
            _apply_index_delta(
                index_dir,
                manifest,
                sources,
                old_chunk_ids,
                new_chunks,
                questions_path,
            )
            print(
                f"[add] updated {len(changed_paths)} source(s), "
                f"removed {len(removed_paths)}, added {len(new_chunks)} chunk(s)"
            )
    if bootstrap_build:
        print("[add] source inventory missing; running one full build to initialize it")
        cmd_build(output_dir, index_dir, utility_questions_path)


def cmd_delete(source: str, index_dir: Path) -> None:
    with _BUILD_LOCK, _index_file_lock(index_dir):
        manifest = _read_index_manifest(index_dir)
        sources = manifest.get("sources")
        if not isinstance(sources, dict):
            raise RuntimeError(
                "Index has no source inventory; run `python src/rag.py build` first."
            )
        matches = [
            key
            for key, record in sources.items()
            if key == source
            or (isinstance(record, dict) and record.get("source") == source)
        ]
        if not matches:
            raise ValueError(f"Source not found in index: {source}")
        if len(matches) > 1:
            raise ValueError(
                f"Source name {source!r} is ambiguous; use one of: "
                + ", ".join(matches)
            )
        source_key = matches[0]
        record = sources.pop(source_key)
        chunk_ids = record.get("chunk_ids", [])
        if "chunk_ids" not in record or not isinstance(chunk_ids, list):
            raise RuntimeError(f"Invalid chunk inventory for {source_key}; run `build`.")
        _validate_index_manifest(
            index_dir, int(manifest.get("embedding_dimension", 0))
        )
        _apply_index_delta(
            index_dir,
            manifest,
            sources,
            {str(chunk_id) for chunk_id in chunk_ids},
            [],
            _utility_questions_path(manifest, None),
        )
        print(f"[delete] removed {source_key}")


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
    with _CHROMA_LOCK:
        client = _get_chroma_client(index_dir)
        coll = client.get_collection(COLLECTION)
        count = coll.count()
    print(f"chunks: {count}")
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
    a = sub.add_parser("add", help="Incrementally sync changed and removed sources.")
    a.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    a.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX)
    a.add_argument("--utility-questions", type=Path)
    d = sub.add_parser("delete", help="Remove one indexed source.")
    d.add_argument("source", help="Manifest source path or unique source filename.")
    d.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX)
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
    elif a.cmd == "add":
        cmd_add(a.output, a.index_dir, a.utility_questions)
    elif a.cmd == "delete":
        cmd_delete(a.source, a.index_dir)
    elif a.cmd == "query":
        cmd_query(a.query, a.index_dir, a.top_k, a.page, a.table, a.domain)
    elif a.cmd == "stats":
        cmd_stats(a.index_dir)
    elif a.cmd == "inspect":
        cmd_inspect(a.chunk_id)
if __name__ == "__main__":
    main()