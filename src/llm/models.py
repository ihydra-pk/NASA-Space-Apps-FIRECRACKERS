"""Typed structured answer and citation models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from src.rag.models import EvidenceStatus

# "partial": the evidence answers part of the question; `limitations` says
# which part is not covered.
AnswerStatus = Literal["grounded", "partial", "insufficient_evidence"]
# "direct": stated in a cited chunk. "inference": reasoned from the cited
# chunks together, though none of them states it outright.
FindingSupport = Literal["direct", "inference"]


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
class AnswerFinding:
    """One claim the NASA evidence supports, with the chunks it rests on."""

    statement: str
    support: FindingSupport
    chunk_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"statement": self.statement, "support": self.support, "chunk_ids": list(self.chunk_ids)}


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
    findings: tuple[AnswerFinding, ...] = ()
    # General combustion science that explains the findings. It carries no
    # citations and is never attributed to NASA documents.
    scientific_context: str = ""
    limitations: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "status": self.status,
            "findings": [finding.to_dict() for finding in self.findings],
            "scientific_context": self.scientific_context,
            "limitations": self.limitations,
            "citations": [citation.to_dict() for citation in self.citations],
            "evidence_summary": [reference.to_dict() for reference in self.evidence_summary],
            "query": self.query,
            "model_name": self.model_name,
            "provider": self.provider,
        }
