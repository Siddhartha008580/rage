"""rag.py — chunk + index + retrieve for the ingest pipeline.

Chunking: content-type-aware.
  - prose: fixed-window with overlap, snapped to line boundaries, section-tracked
  - table: whole markdown table as one chunk
  - table_row: one self-describing chunk per data row

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
import hashlib
import json
import os
import re
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT = Path("output")
DEFAULT_INDEX = Path("index")
COLLECTION = "documents"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_CHUNK_CHARS = 1600
PROSE_OVERLAP = 200
INDEX_VERSION = "v2"

NOISE_LINE_RE = re.compile(
    r"^\d{1,2}/\d{1,2}/\d{2,4}(?:,\s*\d{1,2}:\d{2}(?::\d{2})?\s*(?:[AP]M)?)?",
    re.IGNORECASE,
)
TOC_LINK_RE = re.compile(
    r"^<u>\s*table\s+of\s+contents\s*</u>$", re.IGNORECASE
)
INLINE_TAG_RE = re.compile(
    r"</?(?:sup|sub|mark|u|em|b|strong|span|div)>", re.IGNORECASE
)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
TABLE_SEP_RE = re.compile(r"^\s*\|(?:\s*:?-+:?\s*\|)+\s*$")
CAPTION_RE = re.compile(
    r"^\s*\*{0,2}((?:Table|Exhibit|Figure)\s+\d+[^\n]*?)\*{0,2}\s*$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------- noise
def strip_noise(text: str) -> str:
    out = []
    for line in text.splitlines():
        plain = re.sub(r"<[^>]+>", "", line).strip()
        if not plain:
            out.append("")
            continue
        if NOISE_LINE_RE.match(plain):
            continue
        if plain.lower() == "table of contents":
            continue
        if TOC_LINK_RE.match(line.strip()):
            continue
        out.append(INLINE_TAG_RE.sub("", line))
    return "\n".join(out).strip()


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
        workspace = output_dir.resolve().parent.parent
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
                    active_outputs[source] = (workspace / output).resolve()

    selected: list[Path] = []
    ambiguous: list[str] = []
    for source, source_paths in sorted(grouped.items()):
        if len(source_paths) == 1:
            selected.extend(source_paths)
            continue

        active_path = active_outputs.get(source)
        matching = [
            path for path in source_paths
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

    def metadata(self) -> dict[str, Any]:
        metadata = {
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
        return metadata


def _cells(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


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


def _caption_near(text: str, start: int) -> str:
    """Look at the ~3 lines before the table for a 'Table N. ...' line."""
    before = text[max(0, start - 400) : start]
    for line in reversed(before.splitlines()):
        m = CAPTION_RE.match(line)
        if m:
            return m.group(1).strip()
    return ""


def _chunk_prose(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Fixed-window with overlap, snapped to line boundaries."""
    out: list[tuple[int, int]] = []
    pos = start
    while pos < end:
        stop = min(pos + MAX_CHUNK_CHARS, end)
        if stop < end:
            nl = text.rfind("\n", pos + MAX_CHUNK_CHARS // 2, stop)
            if nl > pos:
                stop = nl
        if text[pos:stop].strip():
            out.append((pos, stop))
        if stop >= end:
            break
        pos = max(stop - PROSE_OVERLAP, pos + 1)
    return out


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
) -> Chunk:
    cite = f"[{source} p.{page.pdf_page}]"
    scope = cite
    if section_path:
        scope += f" § {' > '.join(section_path)}"
    elif section:
        scope += f" § {section}"
        if subsection:
            scope += f" > {subsection}"
    prefix = scope + "\n"
    cid = (
        f"{Path(source).stem}--{document_id}--p{page.pdf_page}"
        f"--{chunk_type}--{ordinal}"
    )
    return Chunk(
        chunk_id=cid,
        text=prefix + body.strip(),
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
    )


def chunk_page(
    text: str,
    page: PageSpan,
    source: str,
    document_id: str = "",
    domain: str = "uncategorized",
    heading_context: list[tuple[int, str]] | None = None,
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

    headings = [
        (m.start(), len(m.group(1)), m.group(2).strip())
        for m in HEADING_RE.finditer(text)
    ]
    initial_heading_context = heading_context or []

    # Build atoms in order
    atoms: list[dict[str, Any]] = []
    cursor = 0
    for ts, te in tables:
        if cursor < ts:
            atoms.append(
                {"kind": "prose",
                 "start": offsets[cursor], "end": offsets[ts]}
            )
        atoms.append(
            {
                "kind": "table",
                "start": offsets[ts],
                "end": offsets[te],
                "header": _cells(lines[ts]),
                "body": [_cells(l) for l in lines[ts + 2 : te]],
            }
        )
        cursor = te
    if cursor < len(lines):
        atoms.append(
            {"kind": "prose",
             "start": offsets[cursor], "end": offsets[-1]}
        )

    chunks: list[Chunk] = []
    for atom in atoms:
        section_path = _section_path_at(
            atom["start"], headings, initial_heading_context
        )
        section = section_path[0] if section_path else ""
        subsection = " > ".join(section_path[1:])

        if atom["kind"] == "prose":
            for s, e in _chunk_prose(text, atom["start"], atom["end"]):
                body = text[s:e].strip()
                if body:
                    chunks.append(_make_chunk(
                        body=body, page=page, source=source,
                        section=section, subsection=subsection,
                        section_path=section_path, domain=domain,
                        chunk_type="prose", row_label="", caption="",
                        char_start=s, char_end=e, ordinal=len(chunks),
                        document_id=document_id,
                    ))
            continue

        caption = _caption_near(text, atom["start"])
        header = atom["header"]
        body_rows = atom["body"]
        if not header or not body_rows:
            continue

        # Whole-table chunk
        md = ["| " + " | ".join(header) + " |"]
        md.append("|" + "|".join(["---"] * len(header)) + "|")
        for row in body_rows:
            cells = row + [""] * (len(header) - len(row))
            md.append("| " + " | ".join(cells[: len(header)]) + " |")
        chunks.append(_make_chunk(
            body="\n".join(md), page=page, source=source,
            section=section, subsection=subsection,
            section_path=section_path, domain=domain,
            chunk_type="table", row_label="", caption=caption,
            char_start=atom["start"], char_end=atom["end"],
            ordinal=len(chunks),
            document_id=document_id,
        ))

        # One chunk per row
        for row in body_rows:
            cells = row + [""] * max(0, len(header) - len(row))
            fields = [
                f"{header[index]}: {cell}"
                for index, cell in enumerate(cells[: len(header)])
                if cell
            ]
            if not fields:
                continue
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
            chunks.append(_make_chunk(
                body="\n".join(row_lines), page=page, source=source,
                section=section, subsection=subsection,
                section_path=section_path, domain=domain,
                chunk_type="table_row",
                row_label=(row[0] if row else ""),
                caption=caption,
                char_start=atom["start"], char_end=atom["end"],
                ordinal=len(chunks),
                document_id=document_id,
            ))

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
        document_headings = [
            (match.start(), len(match.group(1)), match.group(2).strip())
            for match in HEADING_RE.finditer(text)
        ]
        heading_context: list[tuple[int, str]] = []
        heading_index = 0
        for page in pages:
            while (
                heading_index < len(document_headings)
                and document_headings[heading_index][0] < page.char_start
            ):
                _, level, heading = document_headings[heading_index]
                _update_heading_stack(heading_context, level, heading)
                heading_index += 1
            page_text = text[page.char_start : page.char_end]
            out.extend(
                chunk_page(
                    page_text,
                    page,
                    src,
                    document_id,
                    domain,
                    heading_context,
                )
            )
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
def _embed_fn():
    from chromadb.utils import embedding_functions
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )


def build_index(chunks: list[Chunk], index_dir: Path) -> None:
    import chromadb

    index_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(index_dir))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    coll = client.create_collection(COLLECTION, embedding_function=_embed_fn())

    BATCH = 64
    n = len(chunks)
    for i in range(0, n, BATCH):
        batch = chunks[i : i + BATCH]
        coll.add(
            ids=[c.chunk_id for c in batch],
            documents=[c.text for c in batch],
            metadatas=[c.metadata() for c in batch],
        )
        print(f"  indexed {min(i+BATCH, n)}/{n}", flush=True)
    build_lexical_index(chunks, index_dir)


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


def query_lexical_index(
    index_dir: Path,
    terms: list[str],
    top_k: int,
    page: int | None = None,
    table_only: bool = False,
    domain: str | None = None,
) -> list[dict[str, Any]]:
    """Return top BM25 matches from the persisted SQLite FTS5 index."""
    if not terms:
        return []
    lexical_path = index_dir / "lexical.sqlite3"
    if not lexical_path.is_file():
        raise FileNotFoundError(
            f"Lexical index not found at {lexical_path}; rebuild the index "
            "with `python src/rag.py build`."
        )

    match_query = " OR ".join(
        f'"{term}"' for term in terms
    )
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

    client = chromadb.PersistentClient(path=str(index_dir))
    coll = client.get_collection(COLLECTION, embedding_function=_embed_fn())

    where = _index_where(page, table_only, domain)

    kwargs: dict[str, Any] = {"query_texts": [query], "n_results": top_k}
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


def cmd_build(output_dir: Path, index_dir: Path) -> None:
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
    build_index(chunks, index_dir)
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
            print(f"    § {m['section']}"
                  + (f" > {m['subsection']}" if m["subsection"] else ""))
        if m["caption"]:
            print(f"    caption: {m['caption']}")
        print("    " + h["text"][:350].replace("\n", " ")[:350])


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

    b = sub.add_parser("build")
    b.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    b.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX)

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
    elif a.cmd == "build":
        cmd_build(a.output, a.index_dir)
    elif a.cmd == "query":
        cmd_query(a.query, a.index_dir, a.top_k, a.page, a.table, a.domain)
    elif a.cmd == "stats":
        cmd_stats(a.index_dir)
    elif a.cmd == "inspect":
        cmd_inspect(a.chunk_id)


if __name__ == "__main__":
    main()