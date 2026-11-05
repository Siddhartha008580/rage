# RAG retrieval

The chat path combines dense retrieval from Chroma with lexical BM25 retrieval
from a SQLite FTS5 index. It fuses the two ranked lists with Reciprocal Rank
Fusion (`k=60`), reranks the fused candidates with a Sentence Transformers
cross-encoder, and selects a diverse final context with lexical Jaccard MMR.
The best reranked result is placed last in the prompt context.

## Build and query

Build both the vector collection and lexical index from the active converted
Markdown outputs:

```bash
python src/rag.py build --output output/markdown --index-dir index
```

The build replaces the `documents` Chroma collection and the adjacent
`index/lexical.sqlite3`; back up either index first if it must be preserved.
The first chat query may download the default cross-encoder model.

Set the OpenRouter API key, then query the corpus:

```bash
export OPENROUTER_API_KEY="your-key"
python src/chat_rag.py
```

The interactive prompt accepts multiple questions; type `quit` or `exit` to
close it. You can also send a single question directly:

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
