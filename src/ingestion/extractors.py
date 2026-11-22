"""Format-specific PDF and DOCX text extraction."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from docx import Document as WordDocument
from docx.document import Document as WordDocumentObject
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ExtractedPage:
    """Extracted PDF page text and page-scoped warnings."""

    page_number: int
    text: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ExtractedDocx:
    """Extracted Word text and document-level core properties."""

    text: str
    metadata: dict[str, str | None]


def normalize_pdf_whitespace(text: str) -> str:
    """Normalize extraction-only whitespace without editing content or numbers.

    Pypdf can expose tabs followed by line breaks where a single visual line
    was split into positioned text fragments. Join only that explicit artifact;
    preserve other single and blank line breaks because they may delimit lines,
    paragraphs, or table rows in the source.
    """

    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    text = re.sub(r"[ \t]*\t[ \t]*\n[ \t]*", " ", text)
    text = text.replace("\t", " ")
    text = re.sub(r"[ \f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _metadata_value(value: object) -> str | None:
    """Convert an existing metadata value to text, omitting empty values."""

    if value is None:
        return None
    if hasattr(value, "isoformat"):
        value = value.isoformat()
    normalized = str(value).strip()
    return normalized or None


def extract_pdf(path: Path, *, min_text_chars: int = 40) -> list[ExtractedPage]:
    """Extract one text record per PDF page, retaining original page numbers.

    Pages with fewer than ``min_text_chars`` non-whitespace characters are
    retained and flagged. This identifies blank/image-only pages without
    incorrectly discarding them or attempting OCR.
    """

    if min_text_chars < 0:
        raise ValueError("min_text_chars must be zero or greater")

    reader = PdfReader(str(path), strict=False)
    if reader.is_encrypted:
        raise ValueError(f"PDF is encrypted and cannot be extracted: {path}")

    pages: list[ExtractedPage] = []
    for page_number, page in enumerate(reader.pages, start=1):
        warnings: list[str] = []
        try:
            text = normalize_pdf_whitespace(page.extract_text() or "")
        except Exception as exc:  # A damaged page should not block other pages.
            text = ""
            warning = f"page {page_number}: text extraction failed ({exc})"
            warnings.append(warning)
            logger.warning("%s: %s", path, warning)

        char_count = sum(not character.isspace() for character in text)
        if char_count < min_text_chars:
            warning = (
                f"page {page_number}: very little or no extractable text "
                f"({char_count} non-whitespace characters)"
            )
            warnings.append(warning)
            logger.warning("%s: %s", path, warning)

        pages.append(ExtractedPage(page_number, text, tuple(warnings)))

    return pages


def extract_pdf_metadata(reader: PdfReader) -> dict[str, str]:
    """Return available PDF core properties using stable normalized names."""

    properties = reader.metadata
    if properties is None:
        return {}

    metadata: dict[str, str] = {}
    for field_name, pdf_key in (
        ("title", "/Title"),
        ("author", "/Author"),
        ("created", "/CreationDate"),
        ("modified", "/ModDate"),
        ("producer", "/Producer"),
        ("subject", "/Subject"),
    ):
        value = _metadata_value(properties.get(pdf_key))
        if value is not None:
            metadata[field_name] = value
    return metadata


def _iter_document_blocks(document: WordDocumentObject):
    """Yield paragraphs and tables in their original body order."""

    body = document.element.body
    for child in body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, document)
        elif isinstance(child, CT_Tbl):
            yield Table(child, document)


def _table_to_text(table: Table) -> str:
    """Render table rows and cells as readable, tab-separated text."""

    return "\n".join("\t".join(cell.text.strip() for cell in row.cells) for row in table.rows)


def extract_docx(path: Path) -> ExtractedDocx:
    """Extract ordered Word paragraphs/tables and preserve core properties."""

    document = WordDocument(str(path))
    blocks: list[str] = []
    table_number = 0
    for block in _iter_document_blocks(document):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if text:
                blocks.append(text)
        else:
            table_number += 1
            table_text = _table_to_text(block)
            if table_text:
                blocks.append(f"[Table {table_number}]\n{table_text}")

    properties = document.core_properties
    metadata: dict[str, str | None] = {}
    for name in (
        "author",
        "category",
        "comments",
        "content_status",
        "created",
        "identifier",
        "keywords",
        "language",
        "last_modified_by",
        "modified",
        "revision",
        "subject",
        "title",
        "version",
    ):
        value = _metadata_value(getattr(properties, name))
        if value is not None:
            metadata[name] = value

    return ExtractedDocx("\n\n".join(blocks), metadata)
