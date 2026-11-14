# Document ingestion

This package discovers `.pdf` and `.docx` sources in `data/documents/flex/`,
`data/documents/bass/`, and `data/documents/saffire/`, then writes normalized
UTF-8 JSON Lines files beneath `data/processed/`. Run it from the project root:

```powershell
python -m pip install -r requirements.txt
python -m src.ingestion
python -m pytest -q
```

PDFs produce one record per original page. Pages with fewer than 40
non-whitespace extracted characters are retained and logged as warnings. No
OCR is attempted. DOCX files produce one record containing ordered body
paragraphs and tables, along with Word core properties. The `source_id` is
shared by all records from a source document and is derived from the SHA-256
checksum of the original file bytes. Each record also includes that
`content_sha256`, so identical source bytes have the same identity even at a
different path. `source_locator` is `page:N` for PDFs and `document` for DOCX
files; `page_number` is an integer for PDF pages and `null` for DOCX records.
Available PDF and DOCX core properties are retained in `document_metadata`
without adding empty metadata values.

Explicit labels are configured in `data/document_metadata.json`, using
source-relative paths as keys. Unlisted files are labeled `unknown`, and their
evidence status is `null`: classification is not inferred from document
contents. The configured values are validated before ingestion.
Optional `source_url` and `source_accession` fields may be supplied in the
same configuration; they remain `null` unless explicitly provided.

PDF text normalization only joins tab-plus-line-break extraction artifacts,
collapses repeated horizontal whitespace, and trims spaces around line breaks.
It preserves paragraph breaks and does not rewrite extracted words, numbers,
units, or symbols.

Extraction failures are logged and reported while other source files continue
to process. Re-running ingestion replaces only the corresponding normalized
JSONL outputs; original sources are read-only. This foundation does not
implement OCR, structured-data ingestion, chunking, embeddings, a vector
store, retrieval, or language-model calls.
