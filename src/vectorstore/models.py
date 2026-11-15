"""Indexing and retrieval result models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class IndexResult:
    input_records: int
    indexed: int
    upserted: int
    duplicates: int
    rejected: int
    failures: tuple[str, ...] = ()

    @property
    def unique_chunk_ids(self) -> int:
        return self.indexed + self.upserted


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """One ranked match with citation-ready source metadata."""

    rank: int
    chunk_id: str
    similarity_score: float
    text: str
    experiment: str
    filename: str
    source_locator: str
    page_number: int | None
    document_type: str
    evidence_status: str | None
    source_id: str
    content_sha256: str
    source_path: str
    metadata: dict[str, Any] = field(default_factory=dict)
    # Set by hybrid search only. ``similarity_score`` is always the cosine
    # similarity; these record why a chunk was ranked where it was.
    semantic_rank: int | None = None
    keyword_rank: int | None = None
    keyword_score: float | None = None

    @classmethod
    def from_record(
        cls,
        rank: int,
        score: float,
        record: dict[str, Any],
        *,
        semantic_rank: int | None = None,
        keyword_rank: int | None = None,
        keyword_score: float | None = None,
    ) -> "RetrievalResult":
        return cls(
            semantic_rank=semantic_rank,
            keyword_rank=keyword_rank,
            keyword_score=keyword_score,
            rank=rank,
            chunk_id=record["chunk_id"],
            similarity_score=score,
            text=record["text"],
            experiment=record["experiment"],
            filename=record["filename"],
            source_locator=record["source_locator"],
            page_number=record["page_number"],
            document_type=record["document_type"],
            evidence_status=record["evidence_status"],
            source_id=record["source_id"],
            content_sha256=record["content_sha256"],
            source_path=record["source_path"],
            metadata={key: value for key, value in record.items() if key != "embedding"},
        )


@dataclass(frozen=True, slots=True)
class SearchResult:
    query: str
    provider: str
    model_name: str
    dimension: int
    similarity_metric: str
    matches: tuple[RetrievalResult, ...]
    mode: str = "semantic"
    # Experiment labels the question named, when ranking was limited to them.
    scoped_experiments: tuple[str, ...] = ()
