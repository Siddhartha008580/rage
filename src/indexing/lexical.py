"""SQLite FTS5 lexical index build, update, and query operations."""
from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Protocol


SQLITE_BUSY_TIMEOUT_MS = 30000

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


class ChunkLike(Protocol):
    chunk_id: str
    text: str
    source: str
    domain: str
    pdf_page: int
    chunk_type: str

    def metadata(self) -> dict[str, Any]: ...


def connect_sqlite(
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


def sanitize_fts_terms(terms: list[str]) -> list[str]:
    """Tokenize and deduplicate query terms for FTS5 MATCH queries."""
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


def update_lexical_index(
    index_dir: Path,
    deleted_ids: set[str],
    chunks: list[ChunkLike],
) -> None:
    """Incrementally delete and insert chunks in the persisted FTS5 index."""
    lexical_path = index_dir / "lexical.sqlite3"
    if not lexical_path.is_file():
        raise FileNotFoundError(
            f"Lexical index not found at {lexical_path}; run `python src/rag.py build`."
        )
    with connect_sqlite(lexical_path) as connection:
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


def build_lexical_index(chunks: list[ChunkLike], index_dir: Path) -> None:
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
        with connect_sqlite(temporary_path, readonly=False) as connection:
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
                        json.dumps(chunk.metadata(), ensure_ascii=False),
                    )
                    for chunk in chunks
                ],
            )
            # Switch back to DELETE journal mode before atomic file replacement so no stray .wal/.shm files remain
            connection.execute("PRAGMA journal_mode=DELETE;")
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


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
    clean_terms = sanitize_fts_terms(terms)
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
    with connect_sqlite(lexical_path, readonly=True) as connection:
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
