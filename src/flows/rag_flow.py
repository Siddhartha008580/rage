"""End-to-end RAG Query Flow:
Query -> Hybrid Retrieval (Vector + Lexical + SpectRAG + Cross-Encoder Rerank) -> LLM Response.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any


def run_rag_flow(
    query: str,
    index_dir: Path = Path("index"),
    *,
    top_k: int = 10,
    mode: str | None = None,
    retrieve_only: bool = False,
) -> dict[str, Any]:
    """Retrieve relevant chunks from *index_dir* and generate a grounded response."""
    if __package__:
        from .. import chat_rag as chat_mod
    else:
        import chat_rag as chat_mod  # type: ignore[no-redef]

    index_path = Path(index_dir)
    retrieval_mode = mode or os.environ.get("RAG_RETRIEVAL_MODE", "hybrid")

    chunks = chat_mod.retrieve_chunks(
        query,
        index_dir=index_path,
        top_k=top_k,
        mode=retrieval_mode,
    )

    if retrieve_only:
        return {
            "query": query,
            "mode": retrieval_mode,
            "chunks": chunks,
            "answer": None,
        }

    context = chat_mod.build_context(chunks) if hasattr(chat_mod, "build_context") else "\n\n".join(
        c.get("text", "") for c in chunks
    )
    answer = chat_mod.generate_answer(query, context) if hasattr(chat_mod, "generate_answer") else None

    return {
        "query": query,
        "mode": retrieval_mode,
        "chunks": chunks,
        "answer": answer,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run end-to-end RAG query: retrieve from index -> generate response."
    )
    parser.add_argument("query", help="User question to answer")
    parser.add_argument("--index-dir", type=Path, default=Path("index"), help="Folder containing the built index")
    parser.add_argument("--top-k", type=int, default=10, help="Number of chunks to retrieve")
    parser.add_argument("--mode", default=None, help="Retrieval mode (e.g. hybrid, vector, lexical)")
    parser.add_argument("--retrieve-only", action="store_true", help="Return retrieved chunks without LLM generation")
    args = parser.parse_args(argv)

    result = run_rag_flow(
        query=args.query,
        index_dir=args.index_dir,
        top_k=args.top_k,
        mode=args.mode,
        retrieve_only=args.retrieve_only,
    )
    if args.retrieve_only or result["answer"] is None:
        print(f"Retrieved {len(result['chunks'])} chunks (mode={result['mode']}).")
    else:
        print(result["answer"])


if __name__ == "__main__":
    main()
