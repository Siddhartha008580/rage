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
close it. With hybrid selected, direct lookups and table-focused questions use
hybrid retrieval, comparisons use SpectRAG, and complex logical or multi-hop
questions use agentic retrieval. High-stakes questions still receive the
claim-to-evidence check regardless of retrieval mode and are withheld if that
check fails or cannot be interpreted. Agentic retrieval reports when its
evidence-gap audit is skipped, fails, or cannot confirm that follow-up
retrieval closed a gap; these diagnostics do not prove answer correctness.
Queries containing more than four numbered requests are rejected so they can
be split into smaller batches rather than silently under-retrieved.
`--retrieval-mode` can select another primary mode. In interactive chat, choose
document generation to call `docgen` directly and write a citation-grounded PDF
under `documents/`. The standalone MCP server separately exposes a
`create_document` tool. `src/chat_mcp.py` is a compatibility launcher for the
same chat and CLI; it does not run a separate answer pipeline.
You can also send a single question directly:

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
Each successful build writes `index/index_manifest.json` with the embedding
model, vector dimension, index version, and chunk count. Dense queries verify
these values before searching and ask for a rebuild if the manifest is missing
or incompatible. Existing indexes without a manifest must be rebuilt once.

The metadata domain currently comes from the first folder below
`output/markdown` (for example, `financial`, `medical`, or `standards`). The
reranker and diversity settings are starting defaults, not measured quality
claims. Evaluate with labeled queries across each domain before tuning or
shipping production thresholds.
Retrieved chunks retain their original metadata through normalization so
table-row deduplication can use source, page, and character-span fields.
Citation verification checks each source-page citation separately, including
comma- or semicolon-separated citations grouped inside one pair of brackets.
For long regulatory PDFs, the chunker removes common CFR running headers and
printer slugs before carrying section headings across page boundaries. The
document generator retrieves a larger candidate set for each outline query and
reserves final context slots across queries, instead of letting one aspect
dominate the evidence bundle. If generation is still truncated after one
recovery attempt, it omits that section rather than publishing a partial body.
Questions that mention CFR or regulations receive the claim-to-evidence check.

The active indexer uses `BAAI/bge-base-en-v1.5` for dense embeddings. It is an
English-focused 768-dimensional model. Indexed passages are embedded without
an instruction prefix; search queries use the BGE query instruction. Prose is indexed as overlapping
260-token child chunks with 45-token overlap, grouped under same-page,
same-section parents of up to 900 tokens. Search and reranking operate on child
text, while answer context expands each selected child to its parent text.
Table-row chunks include their caption, section, column headers, and row values;
answer context uses the self-contained row rather than repeating the entire
table for every selected row. Multiple rows from one table can therefore
contribute evidence to the same answer. Rebuild the index after changing the
embedding model or chunker; the build replaces the existing Chroma collection
and graph.

## Retrieval-only evaluation

To compare retrieval modes without OpenRouter generation calls, provide a JSON
array of labeled queries with `id`, `type`, and `query` fields. Optionally add
`expected_sources` and `expected_terms`; every listed source and term must be
present in the retrieved chunks for that query to count as a hit. Queries marked
`"status": "blocked"` are retained in the set but excluded from scoring until
their sources or gold evidence are available. For example:

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

For the CFR corpus, use its source-matched evaluation set rather than the
financial/medical/Federal Register questions in `eval_selection.json`:

```bash
python scripts/eval_retrieval.py eval_cfr.json \
  --modes hybrid spectrag logical --top-k 10 --out-dir .cfr-eval
```

The CFR set includes a question grounded in 27 CFR § 24.291 on tax-free wine
removal for vinegar production. Its gold answer and source pages are included
for review; the retrieval evaluator scores evidence coverage, not answer
correctness.

The evaluator checks expected sources, terms, and pages directly in retrieved
chunks. Its side-by-side summary reports hit rates and the best retrieval-only
hit rate for this eval set; this measures evidence coverage, not answer quality,
latency, or cost. Hybrid and SpectRAG make no OpenRouter calls. Agentic mode
uses the planner LLM for its evidence-gap audit when `OPENROUTER_API_KEY` is
set, so it may make one planner call per query with initially retrieved chunks.
Without that key, agentic mode skips the gap audit and only measures its initial
retrieval pass. Retrieval or input errors are surfaced instead of being counted
as retrieval misses.

## Answer-level evaluation

For a new corpus or evaluator set, use `eval_retrieval.py` first to measure
evidence coverage. To check generated answers against a changing set, provide
the same JSON records with optional `expected_sources`, `expected_terms`,
`expected_pages`, and `expected_answer_terms`. The answer evaluator runs the
selected retrieval mode, generates each answer, forces the claim-support check,
and reports missing expectations, invalid citations, and truncation:

```bash
python scripts/eval_answers.py evaluator-set.json \
  --retrieval-mode hybrid --top-k 10 --out .answer-eval/results.json
```

`expected_terms` are checked in retrieved evidence; `expected_answer_terms`
are checked in the generated answer. The script exits nonzero if any case
misses its expectations, has an invalid citation, is incomplete, or fails the
claim-support check. It makes generation and verification API calls, so it is
intended for explicit evaluation runs, not every interactive query. Rebuild the
index when the source documents change, then run both retrieval and answer
evaluations on the supplied set; a retrieval hit alone does not prove a correct
answer.
