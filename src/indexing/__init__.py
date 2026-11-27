"""Indexing subpackage for lexical and vector index management."""

from .lexical import (
    FTS_STOPWORDS,
    FTS_TERM_RE,
    build_lexical_index,
    connect_sqlite,
    query_lexical_index,
    sanitize_fts_terms,
    update_lexical_index,
)

__all__ = [
    "FTS_STOPWORDS",
    "FTS_TERM_RE",
    "build_lexical_index",
    "connect_sqlite",
    "query_lexical_index",
    "sanitize_fts_terms",
    "update_lexical_index",
]
