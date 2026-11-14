"""Normalized document records produced by the ingestion pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

DocumentType = Literal[
    "experiment_metadata",
    "test_plan",
    "science_report",
    "publication",
    "data_description",
    "engineering",
    "unknown",
]
EvidenceStatus = Literal["planned", "observed", "interpreted", "metadata"]
FileType = Literal["pdf", "docx"]


@dataclass(frozen=True, slots=True)
class NormalizedDocument:
    """One extractable unit: one PDF page or one complete DOCX document.

    A PDF source produces one record per page. Its ``source_id`` is shared by
    every page, while ``page_number`` identifies the original page. A DOCX
    produces one record with ``page_number=None`` and preserves its Word core
    properties in ``document_metadata``.
    """

    source_id: str
    content_sha256: str
    experiment: str
    filename: str
    file_type: FileType
    source_path: str
    document_type: DocumentType
    evidence_status: EvidenceStatus | None
    text: str
    source_locator: str
    page_number: int | None = None
    document_metadata: dict[str, str | None] = field(default_factory=dict)
    extraction_warnings: list[str] = field(default_factory=list)
    source_url: str | None = None
    source_accession: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation with stable field names."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """Summary of a directory ingestion run."""

    source_documents: int
    records: tuple[NormalizedDocument, ...]
    processed_paths: tuple[str, ...]
    errors: tuple[str, ...] = ()

    @property
    def pdf_pages(self) -> int:
        return sum(record.file_type == "pdf" for record in self.records)

    @property
    def docx_documents(self) -> int:
        return sum(record.file_type == "docx" for record in self.records)
