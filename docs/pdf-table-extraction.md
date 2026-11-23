# PDF table extraction

PDF conversion uses Unstructured by default for table detection. Tables are emitted as compact
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

Unstructured is the default table extractor. pdfplumber remains available as
an explicit alternative (`PDF_TABLE_BACKEND=pdfplumber`); Camelot can be used
as an optional fallback:

```bash
PDF_TABLE_BACKEND=pdfplumber python src/ingest.py --source medical/d1.pdf
PDF_TABLE_FALLBACK=camelot python src/ingest.py --source medical/d1.pdf
```

Camelot is a local, free parser for text-based PDFs; it does not perform OCR
and will not recover tables from scanned pages by itself. Fallback tables are
merged with the selected primary backend and errors are recorded as conversion
warnings.

Camelot is included in `requirements.txt`. Install it in the Python environment
used by ingestion before enabling that fallback:

```bash
python -m pip install "camelot-py>=2.0,<3"
```

Unstructured is installed with the PDF extras:

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

The backend configuration is included in output naming, so changing a setting
produces a distinct conversion output. No cloud API is used by these local
options.

## Docling and MinerU page extraction

Docling is the default PDF text and layout extractor for every page. MinerU is
enabled by default as a fallback when Docling cannot produce Markdown. PyMuPDF
is used only for low-level page and rendering utilities, not to extract PDF
content. If both layout engines fail on a page, conversion fails with the page
number rather than silently using the lower-quality text path.

```bash
# Use MinerU as the all-page primary extractor
DOCLING_ENABLED=0 python src/ingest.py --source medical/d1.pdf

# Enable OCR in Docling
DOCLING_OCR=1 python src/ingest.py
```

Docling OCR is disabled by default. MinerU receives one-page PDF copies, has a
configurable timeout, and its Markdown passes through native hyperlink
re-injection. `MINERU_TIER` accepts
`flash`, `basic`, `standard` (default), or `advanced`;
`MINERU_TIMEOUT_SECONDS` defaults to 300 and `MINERU_MAX_PAGES` defaults to 20.
The first use may download model weights. Backend and OCR settings are included
in output naming, so changing them causes a distinct conversion output.
At least one of Docling or MinerU must be enabled; disabling both is an explicit
error rather than a silent switch to PyMuPDF text extraction.
