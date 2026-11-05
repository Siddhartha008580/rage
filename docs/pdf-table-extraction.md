# PDF table extraction

PDF conversion uses pdfplumber by default. Tables are emitted as compact
Markdown pipe tables, including complex tables normalized into row-oriented
records for retrieval. The original extracted rows, an HTML alternative, and
the best-effort normalized rows are written to a sibling
`.tables.json` sidecar. The sidecar includes the page and bounding box,
caption when detected, extractor, structural score, complexity signals,
normalized rows, footnote references, and any coordinate-based row repairs.
When a row spans multiple detected columns, cell text is reassigned only when
the PDF word positions and a complete header grid establish the column
boundaries; when nearby column headings are available, their word positions
also guide recovery of merged rows. Only merged rows are rebuilt, and the
original row remains in the sidecar. Captions detected next to a table are
stored with its sidecar record. Complex tables are marked
`normalization_status: review_required`. Use the source PDF to verify
normalized values; ambiguous multiline alignment is retained as extracted
and recorded in `normalization_warnings`. Sparse detections without multiple
rows containing meaningful values across columns are excluded to avoid
emitting vector diagrams as tables. Unfenced HTTP exchanges and multi-rule
ABNF blocks are fenced when their syntax is unambiguous; other text is left
unchanged.

Unstructured can be enabled as an optional fallback:

```bash
PDF_TABLE_FALLBACK=unstructured python src/ingest.py --source medical/d1.pdf
```

The fallback is attempted for complex tables and pages where a table may have
been missed. It only replaces an overlapping pdfplumber result when its
structure score is meaningfully higher; otherwise its result is retained in
the sidecar as an alternative and the pdfplumber table remains in Markdown.
Non-overlapping Unstructured tables are added as additional detected tables.
Fallback errors are recorded as conversion warnings and do not discard
successful pdfplumber output.

Install Unstructured's PDF extras in the same Python environment before
enabling the fallback:

```bash
python -m pip install "unstructured[pdf]>=0.26"
```

The fallback is intentionally opt-in because it has additional Python and
system dependencies. Its configuration is included in output naming, so
changing the setting produces a distinct conversion output.
