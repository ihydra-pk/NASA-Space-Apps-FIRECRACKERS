"""Command-line entry point: ``python -m src.ingestion``."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .pipeline import IngestionConfig, ingest_documents, load_metadata_overrides


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract NASA PDF and DOCX source documents.")
    parser.add_argument("--documents-dir", type=Path, default=Path("data/documents"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--metadata", type=Path, default=Path("data/document_metadata.json"))
    parser.add_argument("--min-pdf-text-chars", type=int, default=40)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    overrides = load_metadata_overrides(args.metadata) if args.metadata.is_file() else {}
    result = ingest_documents(
        IngestionConfig(
            documents_dir=args.documents_dir,
            output_dir=args.output_dir,
            metadata_overrides=overrides,
            min_pdf_text_chars=args.min_pdf_text_chars,
        )
    )
    print(
        f"Processed {result.source_documents} source document(s): "
        f"{result.pdf_pages} PDF page(s), {result.docx_documents} DOCX document(s), "
        f"{len(result.records)} normalized record(s)."
    )
    if result.errors:
        print(f"{len(result.errors)} source document(s) could not be processed.")
        for error in result.errors:
            print(f"  - {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
