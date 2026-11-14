"""Typed structured answer and citation models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from src.rag.models import EvidenceStatus

AnswerStatus = Literal["grounded", "insufficient_evidence"]


@dataclass(frozen=True, slots=True)
class AnswerCitation:
    chunk_id: str
    source_id: str
    source_locator: str
    page_number: int | None
    evidence_status: EvidenceStatus

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvidenceReference:
    """A trace reference copied from one supplied RAG evidence item."""

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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AnswerResponse:
    answer: str
    status: AnswerStatus
    citations: tuple[AnswerCitation, ...]
    evidence_summary: tuple[EvidenceReference, ...]
    query: str
    model_name: str
    provider: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "status": self.status,
            "citations": [citation.to_dict() for citation in self.citations],
            "evidence_summary": [reference.to_dict() for reference in self.evidence_summary],
            "query": self.query,
            "model_name": self.model_name,
            "provider": self.provider,
        }
