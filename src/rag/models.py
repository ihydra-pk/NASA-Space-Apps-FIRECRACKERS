"""Structured, citation-ready evidence context models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

EvidenceStatus = Literal["planned", "observed", "interpreted", "metadata", "unknown"]


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    rank: int
    similarity_score: float
    chunk_id: str
    source_id: str
    content_sha256: str
    experiment: str
    filename: str
    file_type: str
    source_path: str
    document_type: str
    evidence_status: EvidenceStatus
    source_locator: str
    page_number: int | None
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RAGContext:
    query: str
    retrieval_count: int
    evidence_items: tuple[EvidenceItem, ...]
    retrieval_configuration: dict[str, Any]
    context_text: str
    context_characters: int
    estimated_context_tokens: int
    truncation_occurred: bool = False
    dropped_evidence_count: int = 0
    dropped_chunk_ids: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "retrieval_count": self.retrieval_count,
            "evidence_items": [item.to_dict() for item in self.evidence_items],
            "retrieval_configuration": dict(self.retrieval_configuration),
            "context_text": self.context_text,
            "context_characters": self.context_characters,
            "estimated_context_tokens": self.estimated_context_tokens,
            "truncation_occurred": self.truncation_occurred,
            "dropped_evidence_count": self.dropped_evidence_count,
            "dropped_chunk_ids": list(self.dropped_chunk_ids),
        }
