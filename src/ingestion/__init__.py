"""Source-document ingestion for NASA FireLens.

This package only extracts and normalizes PDF and DOCX documents. It does not
chunk text, generate embeddings, perform retrieval, or call a language model.
"""

from .models import (
    DocumentType,
    EvidenceStatus,
    IngestionResult,
    NormalizedDocument,
)
from .pipeline import IngestionConfig, MetadataOverride, ingest_documents, ingest_file

__all__ = [
    "DocumentType",
    "EvidenceStatus",
    "IngestionConfig",
    "IngestionResult",
    "MetadataOverride",
    "NormalizedDocument",
    "ingest_documents",
    "ingest_file",
]
