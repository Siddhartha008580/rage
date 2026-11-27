# `rage` — Local-First Hybrid RAG, Ingestion & MCP Pipeline

`rage` is a local-first **Retrieval-Augmented Generation (RAG)**, **document ingestion**, **MCP server**, and **document generation** system. It converts PDFs and Office documents (`.pdf`, `.docx`, `.pptx`, `.xlsx`) into structured Markdown with table/figure extraction, builds a hybrid **vector + lexical + spectral graph (SpectRAG)** index, and exposes retrieval and generation via CLI, interactive chat, and an MCP server.

---

## Architecture Overview

```text
input/ (PDF, DOCX, PPTX, XLSX)
  │
  ▼  [Ingestion & Conversion]
  │  src/ingest.py, src/convertstable.py, src/routing.py,
  │  src/backends/ (native, docling_cli, mineru, hyperlink_reinject),
  │  src/tables/ (reconcile, rich_html), src/table_transformer.py
  ▼
output/ (Structured Markdown + extracted tables/images)
  │
  ▼  [Chunking & Indexing]
  │  src/chunk.py, src/rag.py, src/spectrag.py,
  │  src/indexing/ (lexical.py, vector.py, io.py)
  ▼
index/ (ChromaDB vector store, SQLite FTS5 lexical.sqlite3, spectrag_graph.npz, index_manifest.json)
  │
  ├──▶ [Interactive Chat]  src/chat_rag.py (launcher: src/chat_mcp.py)
  ├──▶ [MCP Server]        src/mcp_server.py, src/mcp_tools.py
  ├──▶ [Doc Generation]    src/docgen.py
  └──▶ [Evaluation]        scripts/eval_retrieval.py, scripts/eval_answers.py, scripts/trace_failure.py
```

---

## Prerequisites

- **Python**: 3.10 or newer
- **API Keys**:
  - `OPENROUTER_API_KEY` — required for LLM-generated answers, SpectRAG planning, document generation, and utility-question generation.
  - `GROQ_API_KEY` — optional, for diagram/image vision enrichment.
- **Optional System Dependencies** (Ubuntu/Debian):
  - `libgl1` for PDF inference dependencies
  - `tesseract-ocr` for OCR fallback on scanned pages

---

## Setup & Installation

1. **Create and activate a virtual environment**:

   **Linux / macOS**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   **Windows (PowerShell)**:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. **Install the package and dependencies**:
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -e ".[dev]"
   ```
   *(Alternatively, install runtime dependencies via `pip install -r requirements.txt`.)*

3. **Configure environment variables**:

   **Linux / macOS**:
   ```bash
   export OPENROUTER_API_KEY="your-openrouter-key"
   export GROQ_API_KEY="your-groq-key"  # optional
   export RAG_RETRIEVAL_MODE="hybrid"
   export RAG_INDEX_DIR="index"
   export RAG_DOCS_DIR="documents"
   export OPENROUTER_MODEL="nvidia/nemotron-3-ultra-550b-a55b:free"
   export RAG_PLANNER_MODEL="$OPENROUTER_MODEL"
   ```

   **Windows (PowerShell)**:
   ```powershell
   $env:OPENROUTER_API_KEY="your-openrouter-key"
   $env:RAG_RETRIEVAL_MODE="hybrid"
   $env:RAG_INDEX_DIR="index"
   $env:RAG_DOCS_DIR="documents"
   ```
   > **Note**: `src/docgen.py` also loads a project-root `.env` file if present.

---

## Running the Project

### 1. Ingestion (Convert Documents to Markdown)
Place `.pdf`, `.docx`, `.pptx`, or `.xlsx` files under `input/` (subfolders become domain names), then run:
```bash
python src/ingest.py
```
Routing thresholds and Docling/native heuristics are configured in `config/routing.yaml`.

### 2. Build & Manage the Index
```bash
# Preview chunks without embedding
python src/rag.py chunks

# Build full index (ChromaDB + SQLite FTS5 + SpectRAG graph)
python src/rag.py build

# Incrementally add/sync new or updated sources
python src/rag.py add

# Remove a specific source from the index
python src/rag.py delete source.pdf

# Inspect index statistics or a specific chunk
python src/rag.py stats --index-dir index
python src/rag.py inspect <chunk_id>

# Direct retrieval query
python src/rag.py query "your question"
python src/rag.py query "your question" --table
python src/rag.py query "your question" --page 9
```

### 3. Interactive Chat
```bash
python src/chat_rag.py
# Or via the MCP-routed compatibility launcher:
python src/chat_mcp.py
```

### 4. MCP Server
Run with `stdio` for local MCP clients:
```bash
python src/mcp_server.py
```
Or run with Streamable HTTP on localhost:
```bash
python src/mcp_server.py --http --port 8000
```

### 5. Document Generation
```bash
python src/docgen.py --help
```

---

## Testing & Benchmarks

Run the test suite:
```bash
make test
# or directly:
python -m pytest -q tests/
```

Run retrieval and answer evaluations:
```bash
make benchmark-baseline   # vector-only retrieval eval
make benchmark-hybrid     # hybrid retrieval eval
make bench-diff           # answer evaluation
```

---

## Key Project Files

| Path | Purpose |
| :--- | :--- |
| `src/ingest.py`, `src/convertstable.py` | Document ingestion, PDF/Office conversion, table & figure extraction |
| `src/routing.py`, `config/routing.yaml` | Page-level confidence scoring and native vs. Docling routing |
| `src/chunk.py`, `src/rag.py` | Content-aware chunking, index lifecycle, and CLI |
| `src/indexing/` | Modularized lexical (`lexical.py`), vector (`vector.py`), and atomic I/O (`io.py`) |
| `src/spectrag.py` | Weighted chunk graph and spectral retrieval features |
| `src/chat_rag.py`, `src/chat_mcp.py` | Hybrid retrieval, cross-encoder reranking, and interactive chat |
| `src/mcp_server.py`, `src/mcp_tools.py` | FastMCP server (stdio / Streamable HTTP) |
| `src/docgen.py` | Grounded Markdown document generation |
| `src/locking.py` | Cross-platform advisory file locking (POSIX `fcntl` / Windows `msvcrt`) |
| `scripts/` | Retrieval/answer evaluation and failure tracing scripts |
| `tests/` | Unit and contract test suite |
