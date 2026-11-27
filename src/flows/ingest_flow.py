"""End-to-end Data Ingestion Flow:
Documents (PDF/DOCX/PPTX/XLSX) -> Markdown conversion -> Chunking -> Embeddings -> Vector/Lexical/SpectRAG Index.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def run_ingest_flow(
    input_dir: Path,
    output_dir: Path = Path("output"),
    index_dir: Path = Path("index"),
    *,
    rebuild_index: bool = True,
) -> dict[str, Any]:
    """Process all documents in *input_dir*, chunk them, and build the vector/lexical/SpectRAG index."""
    if __package__:
        from .. import ingest as ingest_mod
        from .. import rag as rag_mod
    else:
        import ingest as ingest_mod  # type: ignore[no-redef]
        import rag as rag_mod  # type: ignore[no-redef]

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    index_path = Path(index_dir)

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    # Step 1: Convert documents to structured Markdown (native + Docling/MinerU routing, table/figure extraction)
    ingest_mod.main(["--input-dir", str(input_path), "--output-dir", str(output_path)])

    # Step 2: Chunk, embed (BAAI/bge-base-en-v1.5), and persist ChromaDB + SQLite FTS5 + SpectRAG graph
    if rebuild_index:
        summary = rag_mod.build_index(output_dir=output_path, index_dir=index_path)
    else:
        summary = rag_mod.add_sources(output_dir=output_path, index_dir=index_path)

    return {
        "input_dir": str(input_path),
        "output_dir": str(output_path),
        "index_dir": str(index_path),
        "index_summary": summary,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run end-to-end document ingestion: convert -> chunk -> embed -> index."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("input"), help="Folder containing source documents")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Folder for converted Markdown")
    parser.add_argument("--index-dir", type=Path, default=Path("index"), help="Folder for vector/lexical/SpectRAG index")
    parser.add_argument("--incremental", action="store_true", help="Incrementally sync instead of full rebuild")
    args = parser.parse_args(argv)

    result = run_ingest_flow(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        index_dir=args.index_dir,
        rebuild_index=not args.incremental,
    )
    print(f"Ingestion complete. Index persisted at: {result['index_dir']}")


if __name__ == "__main__":
    main()
