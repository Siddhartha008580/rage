"""Vector embedding and ChromaDB client management."""
from __future__ import annotations

import os
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

EMBED_MODEL = "BAAI/bge-base-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
EMBED_BATCH_SIZE = int(os.environ.get("RAG_EMBED_BATCH_SIZE", "32"))

_EMBEDDER_LOCK = threading.RLock()
_CHROMA_LOCK = threading.RLock()
_CHROMA_CLIENTS: dict[str, Any] = {}


def get_chroma_client(index_dir: Path, *, reset: bool = False) -> Any:
    """Return a thread-safe cached ChromaDB PersistentClient for index_dir."""
    import chromadb

    key = str(index_dir.resolve())
    with _CHROMA_LOCK:
        if reset or key not in _CHROMA_CLIENTS:
            _CHROMA_CLIENTS[key] = chromadb.PersistentClient(path=str(index_dir))
        return _CHROMA_CLIENTS[key]


@lru_cache(maxsize=1)
def embedding_model_cached() -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL)


def embedding_model() -> Any:
    with _EMBEDDER_LOCK:
        return embedding_model_cached()


def encode_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    with _EMBEDDER_LOCK:
        model = embedding_model()

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
    return encode_texts(prefixed)


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed indexed passages without a query prompt."""
    return encode_texts(texts)
