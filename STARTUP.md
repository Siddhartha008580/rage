# Startup Guide

This guide initializes the local RAG system from a fresh checkout. It covers document conversion, index creation, interactive chat, direct retrieval, MCP startup, and the most common environment settings.

Run every command below from the project root (`rage/`) unless a command says otherwise.

## 1. Prerequisites

- Python 3.10 or newer
- `pip` and `venv`
- Enough disk space for Python packages and downloaded embedding/layout models
- An OpenRouter API key for LLM-generated answers, agentic/SpectRAG planning, document generation, and utility-question generation
- Optional: a Groq API key for diagram/image enrichment
- Optional: Tesseract for OCR fallback
- Optional: `libgl1` for PDF inference dependencies on Ubuntu/Debian

Check Python:

```bash
python3 --version
```

## 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

After activation, `python` should resolve to the virtual environment. If VS Code is open in this workspace, select `.venv/bin/python` as the Python interpreter.

## 3. Install optional system dependencies

For PDF processing on Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y libgl1
```

Install Tesseract only if scanned pages require OCR:

```bash
sudo apt-get install -y tesseract-ocr
```

Verify it is available:

```bash
command -v tesseract
```

## 4. Configure API keys

LLM features require an OpenRouter key:

```bash
export OPENROUTER_API_KEY="your-openrouter-key"
```

Optional Groq key for enabled diagram/image analysis:

```bash
export GROQ_API_KEY="your-groq-key"
```

Keep secrets out of source control. The chat and MCP processes must be started after exporting these variables in the same shell.

A typical shell configuration is:

```bash
export OPENROUTER_API_KEY="..."
export RAG_RETRIEVAL_MODE="hybrid"
export RAG_INDEX_DIR="index"
export RAG_DOCS_DIR="documents"
export OPENROUTER_MODEL="nvidia/nemotron-3-ultra-550b-a55b:free"
export RAG_PLANNER_MODEL="$OPENROUTER_MODEL"
```

The repository's document generator also looks for a project-root `.env` file. Shell environment variables take precedence for the processes that use them.

## 5. Start with an existing index

A complete index can already exist under `index/`. Verify it before rebuilding:

```bash
python src/rag.py stats --index-dir index
ls -l index/index_manifest.json index/chroma.sqlite3 index/lexical.sqlite3
```

If `stats` succeeds and the required files exist, the system may already be ready. Skip to [Run interactive chat](#8-run-interactive-chat).

## 6. Add documents

Place PDFs or supported source files under `input/`. Folder names directly below `input/` become domains, for example:

```text
input/
  medical/
    study.pdf
  standards/
    standard.pdf
  financial/
    report.pdf
```

The existing `input/` directory may contain domain folders or individual files. Additions are processed by the ingestion script.

## 7. Build from documents

### 7.1 Convert documents

Convert all newly added documents:

```bash
python src/ingest.py
```

Convert one source only:

```bash
python src/ingest.py --source medical/study.pdf
```

Reconvert an existing source:

```bash
python src/ingest.py --source medical/study.pdf --force
```

Converted Markdown is written under `output/markdown/`. Conversion manifests and table sidecars are stored alongside the output. The first Docling or MinerU run may download model weights.

The default PDF table backend is Unstructured. Common alternatives are:

```bash
PDF_TABLE_BACKEND=pdfplumber python src/ingest.py --source medical/study.pdf
PDF_TABLE_FALLBACK=camelot python src/ingest.py --source medical/study.pdf
PDF_OCR_BACKEND=tesseract PDF_OCR_LANGUAGES=eng python src/ingest.py --source medical/study.pdf
```

Useful ingestion variables:

| Variable | Purpose | Default |
|---|---|---|
| `DOCLING_ENABLED` | Enable or disable Docling extraction | Enabled |
| `DOCLING_DISABLE` | Disable Docling explicitly | Unset |
| `DOCLING_OCR` | Enable OCR inside Docling | Disabled |
| `MINERU_ENABLED` | Enable MinerU fallback | Enabled |
| `MINERU_TIER` | `flash`, `basic`, `standard`, or `advanced` | `standard` |
| `PDF_OCR_BACKEND` | `tesseract`, `paddle`, `unstructured`, `auto`, or `none` | `tesseract` |
| `PDF_TABLE_BACKEND` | `pdfplumber`, `unstructured`, or `hybrid` | `unstructured` |
| `PDF_TABLE_FALLBACK` | `none`, `unstructured`, or `camelot` | `none` |
| `ENABLE_DIAGRAM_CONVERSION` | Enable Groq-backed diagram conversion | Disabled |
| `VISION_CACHE_DIR` | Directory for cached image analysis | `~/.cache/vision` |

Changing extraction settings changes the conversion output name. Conversion state is resumable, but changing backend settings should be followed by index rebuilding.

#### 7.1.1 Enable Groq image enrichment

Image enrichment requires a Groq API key and the diagram-conversion flag:

```bash
export GROQ_API_KEY="your-groq-key"
export ENABLE_DIAGRAM_CONVERSION=true
python src/ingest.py
```

The Groq integration uses `qwen/qwen3.8-27b` to describe supported images found in converted Markdown. AI-generated descriptions are added beside image links, cached, and included in the generated Markdown and later RAG index.

To process or force-reprocess one source:

```bash
export GROQ_API_KEY="your-groq-key"
export ENABLE_DIAGRAM_CONVERSION=true
python src/ingest.py --source medical/study.pdf --force
```

The default cache directory is `~/.cache/vision`. Use a custom cache location if needed:

```bash
export VISION_CACHE_DIR="/path/to/vision-cache"
```

If `ENABLE_DIAGRAM_CONVERSION=true` but `GROQ_API_KEY` is unset, ingestion continues without changing the image links and records a warning. After image-enriched Markdown is generated, rebuild the RAG index:

```bash
python src/rag.py build --output output/markdown --index-dir index
```

### 7.2 Generate utility questions, optionally

Utility questions improve graph retrieval. They require `OPENROUTER_API_KEY`:

```bash
python src/rag.py utility-questions
```

Review `output/utility-questions.jsonl` before indexing high-stakes corpora. Utility-question generation skips chunk IDs already present in the file.

### 7.3 Build vector, lexical, and graph indexes

Build the complete index from converted Markdown:

```bash
python src/rag.py build --output output/markdown --index-dir index
```

To include reviewed utility questions:

```bash
python src/rag.py build \
  --output output/markdown \
  --index-dir index \
  --utility-questions output/utility-questions.jsonl
```

The build replaces the `documents` Chroma collection, creates `index/lexical.sqlite3`, and writes `index/spectrag_graph.npz` and `index/index_manifest.json`. Preserve existing index files if they must be retained, and rebuild after changing chunks or metadata.

After the initial build, synchronize changed, added, and removed Markdown sources without re-embedding unchanged documents:

```bash
python src/rag.py add --output output/markdown --index-dir index
```

An older index manifest without a source inventory triggers one full build to initialize it. Incremental updates rewrite the SpectRAG graph from persisted vectors, so graph generation remains part of update time. Remove one source explicitly by its path relative to the Markdown directory or by a unique source filename:

```bash
python src/rag.py delete medical/study--backend-v36.md --index-dir index
```

Inspect the resulting index:

```bash
python src/rag.py stats --index-dir index
```

## 8. Run interactive chat

Set the key and start the interactive prompt:

```bash
export OPENROUTER_API_KEY="your-openrouter-key"
python src/chat_rag.py
```

Enter a question at the prompt. Type `quit` or `exit` to stop.

Ask a single question:

```bash
python src/chat_rag.py "What does this standard say about caching?" --top-k 5
```

For evidence chunks and retrieval diagnostics:

```bash
python src/chat_rag.py "Which pages support this conclusion?" \
  --show-chunks \
  --domain standards \
  --top-k 10
```

Useful chat options:

```text
--retrieval-mode {hybrid,spectrag,agentic}
--top-k 10
--candidate-k 50
--mmr-lambda 0.7
--domain DOMAIN
--table
--page PAGE
--context-token-budget 12000
--show-chunks
```

Example:

```bash
python src/chat_rag.py "Compare the warning requirements across documents" \
  --retrieval-mode spectrag \
  --top-k 15 \
  --show-chunks
```

The first run may download the cross-encoder reranker model.

## 9. Run direct retrieval without LLM generation

Dense and lexical retrieval can be checked with:

```bash
python src/rag.py query "warning requirements" --index-dir index --top-k 10
```

Add filters:

```bash
python src/rag.py query "warning requirements" \
  --index-dir index \
  --domain standards \
  --table \
  --top-k 10
```

## 10. Start the MCP server

### 10.1 Standard I/O for a local MCP client

```bash
python src/mcp_server.py
```

This mode communicates over standard input/output and is intended for MCP clients that launch and manage the server process.

### 10.2 Streamable HTTP on localhost

```bash
python src/mcp_server.py --http --host 127.0.0.1 --port 8000
```

The HTTP server is restricted to loopback hosts. Open or configure clients at:

```text
http://127.0.0.1:8000/mcp
```

Because HTTP mode is loopback-only, use a secured reverse proxy before exposing it beyond the local machine.

### 10.3 Test MCP tool routing

```bash
python src/mcp_tools.py "Find the requirements for tax-free wine removal"
```

For document generation routing:

```bash
python src/mcp_tools.py "Create a briefing about diabetic retinopathy warnings"
```

Document generation writes Markdown/PDF output under `documents/` by default. Override the output directory with:

```bash
export RAG_DOCS_DIR="/absolute/path/to/documents"
```

Generated output paths are constrained to this directory.

## 11. Run tests

The Makefile runs the test suite:

```bash
make test
```

Or run pytest directly:

```bash
python -m pytest -q tests/
```

The test suite does not require every external model or service, although individual integration-style tests may need additional system dependencies.

## 12. Retrieval and environment reference

| Variable | Purpose | Default |
|---|---|---|
| `OPENROUTER_API_KEY` | Required for answer generation and optional planning | Unset |
| `OPENROUTER_MODEL` | Answer model | Free Nemotron model |
| `RAG_PLANNER_MODEL` | Agentic/SpectRAG planner model | `OPENROUTER_MODEL` |
| `RAG_RERANKER_MODEL` | Local cross-encoder reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `RAG_INDEX_DIR` | Index directory | `index` |
| `RAG_DOCS_DIR` | Generated document directory | `documents` |
| `RAG_RETRIEVAL_MODE` | `hybrid`, `spectrag`, or `agentic` | `hybrid` |
| `RAG_REASONING_EFFORT` | `none`, `low`, `medium`, or `high` | `high` |
| `RAG_EMBED_BATCH_SIZE` | Dense embedding batch size | `32` |
| `RAG_OPENROUTER_TIMEOUT` | OpenRouter timeout in seconds | `90` |
| `RAG_OPENROUTER_MAX_RETRIES` | OpenRouter retry count | `4` |
| `RAG_AGENTIC_CONFIDENCE_THRESHOLD` | Agentic answer-confidence threshold | `2.5` |
| `GROQ_API_KEY` | Optional diagram/image enrichment | Unset |

## 13. Validation checklist

1. Activate `.venv`.
2. Export `OPENROUTER_API_KEY` for answer generation.
3. Confirm `index/index_manifest.json` exists for a prebuilt index.
4. Run `python src/rag.py stats --index-dir index`.
5. Run a direct retrieval query.
6. Run `python src/chat_rag.py "..." --show-chunks`.
7. Start MCP with `python src/mcp_server.py` or HTTP mode.

For a clean rebuild:

```bash
python src/ingest.py
python src/rag.py build --output output/markdown --index-dir index
python src/rag.py stats --index-dir index
python src/chat_rag.py
```

## 14. Troubleshooting

**Missing OpenRouter key**

Set `OPENROUTER_API_KEY` in the shell before starting chat, MCP, utility-question generation, or document generation.

**Missing index manifest**

Run:

```bash
python src/rag.py build --output output/markdown --index-dir index
```

Indexes built before a manifest existed may need rebuilding once.

**Missing lexical index**

Rebuild both the vector and lexical indexes with `python src/rag.py build ...`; dense-only fallback is not used automatically.

**MinerU or Docling download failures**

Check disk space and network access. Model weights may download on first use. For Docling-only extraction, disable MinerU explicitly:

```bash
MINERU_ENABLED=0 python src/ingest.py
```

**OCR or PDF inference failures**

Install `libgl1` and Tesseract, then retry:

```bash
sudo apt-get install -y libgl1 tesseract-ocr
PDF_OCR_BACKEND=tesseract python src/ingest.py --force
```

**No sources returned**

Run `python src/rag.py stats`, verify the source appears under `output/markdown/`, and rebuild the index if conversion or metadata changed.

**MCP stdio server prints into the protocol stream**

The server suppresses third-party Hugging Face and tokenizer progress output. Client launchers should pass API keys and `RAG_INDEX_DIR` through the MCP server environment.
