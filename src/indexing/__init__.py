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

from .vector import (
    BGE_QUERY_PREFIX,
    EMBED_BATCH_SIZE,
    EMBED_MODEL,
    embed_documents,
    embed_texts,
    embedding_model,
    embedding_model_cached,
    encode_texts,
    get_chroma_client,
)

__all__ = [
    "BGE_QUERY_PREFIX",
    "EMBED_BATCH_SIZE",
    "EMBED_MODEL",
    "FTS_STOPWORDS",
    "FTS_TERM_RE",
    "build_lexical_index",
    "connect_sqlite",
    "embed_documents",
    "embed_texts",
    "embedding_model",
    "embedding_model_cached",
    "encode_texts",
    "get_chroma_client",
    "query_lexical_index",
    "sanitize_fts_terms",
    "update_lexical_index",
]
