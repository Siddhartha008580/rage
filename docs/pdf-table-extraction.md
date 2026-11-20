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
stored with its sidecar record. Multiline text remains inside its original
cell instead of being guessed into additional rows. Blank row labels are
carried forward in the normalized representation, while the original
extracted rows remain unchanged in the sidecar. Sparse detections without
multiple rows containing meaningful values across columns are excluded to
avoid emitting vector diagrams as tables. Unfenced HTTP exchanges and
multi-rule ABNF blocks are fenced when their syntax is unambiguous; other text
is left unchanged.

Unstructured or Camelot can be enabled as an optional fallback:

```bash
PDF_TABLE_FALLBACK=unstructured python src/ingest.py --source medical/d1.pdf
PDF_TABLE_FALLBACK=camelot python src/ingest.py --source medical/d1.pdf
```

The fallback is attempted for complex tables and pages where a table may have
been missed. Camelot is a local, free parser for text-based PDFs; it does not
perform OCR and will not recover tables from scanned pages by itself. It only
replaces an overlapping pdfplumber result when its structure score is
meaningfully higher; otherwise its result is retained in the sidecar as an
alternative and the pdfplumber table remains in Markdown. Non-overlapping
fallback tables are added as additional detected tables. Fallback errors are
recorded as conversion warnings and do not discard successful pdfplumber output.

Camelot is included in `requirements.txt`. Install it in the Python environment
used by ingestion before enabling that fallback:

```bash
python -m pip install "camelot-py>=2.0,<3"
```

Unstructured remains an optional fallback and requires its PDF extras:

```bash
python -m pip install "unstructured[pdf]>=0.18.32"
```

On Ubuntu/Debian, the PDF inference stack also needs the OpenGL runtime
library. Install it in the system environment used for ingestion:

```bash
sudo apt-get update
sudo apt-get install -y libgl1
```

Install Tesseract when OCR is needed:

```bash
sudo apt-get install -y tesseract-ocr
```

Fallbacks are intentionally opt-in because they have additional Python and
system dependencies. Their configuration is included in output naming, so
changing the setting produces a distinct conversion output. No cloud API is
used by these local fallback options.

## Docling page fallback

PDF pages with substantial image coverage and sparse text, aligned numeric
columns, repeated table-of-contents dot leaders, or a clear two-column text
layout can be routed through Docling's layout-aware PDF pipeline when no table
was detected on the page. Pages with detected tables stay on the standard
PyMuPDF and pdfplumber pipeline so extracted tables and cross-page
continuations are retained. Docling is optional at runtime; if it is
unavailable, the standard pipeline is used and a conversion warning is
recorded when fallback pages were requested.

```bash
# Disable the fallback
DOCLING_DISABLE=1 python src/ingest.py

# Enable Docling OCR on pages routed to the fallback
DOCLING_OCR=1 python src/ingest.py
```

Docling OCR is disabled by default. Its enabled/disabled and OCR settings are
included in PDF output naming, so changing them causes a distinct conversion
output rather than reusing pages produced with different settings. The first
use may require downloading Docling's models.
