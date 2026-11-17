# RAG retrieval

The chat path combines dense retrieval from Chroma with lexical BM25 retrieval
from a SQLite FTS5 index. It fuses the two ranked lists with Reciprocal Rank
Fusion (`k=60`), reranks the fused candidates with a Sentence Transformers
cross-encoder, and selects a diverse final context with lexical Jaccard MMR.
The best reranked result is placed last in the prompt context.

## Build and query

Build both the vector collection and lexical index from the active converted
Markdown outputs. The build also writes `index/spectrag_graph.npz` with the
weighted chunk graph and spectral coordinates:

```bash
python src/rag.py build --output output/markdown --index-dir index
```

The build replaces the `documents` Chroma collection, the adjacent
`index/lexical.sqlite3`, and `index/spectrag_graph.npz`; back up these index
artifacts first if they must be preserved.
The first chat query may download the default cross-encoder model.

## SpectRAG mode

The current hybrid/MMR retriever remains the default. To use graph expansion,
spectral scoring, and token-budgeted context selection:

```bash
python src/chat_rag.py "How did revenue relate to operating costs?" \
  --retrieval-mode spectrag --context-token-budget 12000 --top-k 15
```

`--top-k` is the maximum number of context chunks; the rendered evidence
context is also limited by `--context-token-budget`. Token counting uses the
`cl100k_base` tokenizer. SpectRAG seeds with up to 20 hybrid results, expands
at most two graph hops (200 candidates maximum), reranks the expanded set, and
combines personalized PageRank and spectral proximity before selection.
`RAG_RETRIEVAL_MODE=spectrag` selects this mode by default for chat and MCP.

Graph edges combine semantic similarity, document adjacency, section
hierarchy, heuristic named-entity overlap, and resolvable source-file
citations. Signal weights follow the SpectRAG configuration and are
renormalized when a signal is unavailable. Utility-question edges and query
matching are enabled when pre-generated questions are supplied at index time:

```bash
python src/rag.py build --utility-questions output/utility-questions.jsonl
```

The JSONL file contains one record per chunk ID, for example:

```json
{"chunk_id":"chunk-id","questions":["Which risks does the report identify?","What evidence supports the risk assessment?"]}
```

Generate resumable questions with OpenRouter, then build the index with them:

```bash
python src/rag.py utility-questions
python src/rag.py build --utility-questions output/utility-questions.jsonl
```

This command requires `OPENROUTER_API_KEY`; it checkpoints each batch and skips
chunk IDs already present in the output file. Review generated questions before
indexing for high-stakes corpora. Rebuild the graph after changing chunks,
metadata, or utility questions. The configured token budget covers rendered
retrieved evidence, not the system prompt, query, or generated answer.

Set the OpenRouter API key, then query the corpus:

```bash
export OPENROUTER_API_KEY="your-key"
python src/chat_rag.py
```

The interactive prompt accepts multiple questions; type `quit` or `exit` to
close it. After each question, choose `h` for hybrid retrieval, `s` for
SpectRAG, or `d` to generate a citation-grounded PDF with the document
generator. Press Enter to use the configured `RAG_RETRIEVAL_MODE`, or `c` to
cancel that question. PDF generation requires `OPENROUTER_API_KEY` and writes
the result under `documents/`. You can also send a single question directly:

```bash
python src/chat_rag.py "quarterly net sales" \
  --domain financial --table --index-dir index --show-chunks
```

Available controls:

- `--top-k`: final context size (default 5).
- `--candidate-k`: dense and BM25 candidates fused and reranked (default 50;
  must be at least `--top-k`).
- `--mmr-lambda`: relevance/diversity balance from 0 to 1 (default 0.7).
- `--domain`, `--table`, and `--page`: restrict both retrieval channels.
- `RAG_RERANKER_MODEL`: override the default cross-encoder model.
- `OPENROUTER_MODEL`: override the answer model.
- `RAG_INDEX_DIR`: set the default index path.

Example:

```bash
python src/chat_rag.py "What does this standard say about caching?" \
  --domain standards --top-k 5 --candidate-k 50 --mmr-lambda 0.7
```

## Implementation notes

The lexical index is built together with Chroma so queries do not download and
rescore every document from the vector store. Rebuild both indexes after
changing chunks or metadata. A missing lexical index is an error; run the build
command rather than silently falling back to dense-only retrieval.

The metadata domain currently comes from the first folder below
`output/markdown` (for example, `financial`, `medical`, or `standards`). The
reranker and diversity settings are starting defaults, not measured quality
claims. Evaluate with labeled queries across each domain before tuning or
shipping production thresholds.

## Retrieval-only evaluation

To compare retrieval modes without OpenRouter generation calls, provide a JSON
array of labeled queries with `id`, `type`, and `query` fields. Optionally add
`expected_sources` and `expected_terms`; every listed source and term must be
present in the retrieved chunks for that query to count as a hit. For example:

```json
[
  {
    "id": "Q-01",
    "type": "single-fact",
    "query": "What was the reported revenue?",
    "expected_sources": ["annual-report.pdf"],
    "expected_terms": ["47,310"]
  }
]
```

Compare hybrid, SpectRAG, and agentic retrieval, saving per-query results and
summaries:

```bash
python scripts/eval_retrieval.py eval_selection.json \
  --modes hybrid spectrag agentic --top-k 10 --out-dir .baseline
```

The evaluator checks expected sources, terms, and pages directly in retrieved
chunks. Its side-by-side summary reports hit rates and the best retrieval-only
hit rate for this eval set; this measures evidence coverage, not answer quality,
latency, or cost. Hybrid and SpectRAG make no OpenRouter calls. Agentic mode
uses the planner LLM for its evidence-gap audit when `OPENROUTER_API_KEY` is
set, so it may make one planner call per query with initially retrieved chunks.
Without that key, agentic mode skips the gap audit and only measures its initial
retrieval pass. Retrieval or input errors are surfaced instead of being counted
as retrieval misses.
