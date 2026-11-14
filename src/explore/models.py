"""Request, unified evidence, and response models for Explore."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, TypeAlias

from src.experimental_data.matching import MatchResult
from src.llm.models import AnswerResponse
from src.rag.models import EvidenceStatus, RAGContext

EvidenceMode = Literal["document_only", "experimental_only", "combined", "insufficient"]


@dataclass(frozen=True, slots=True)
class ExploreRequest:
    question: str
    conditions: dict[str, object] | None = None
    # Explicit unit declarations prevent silent conversions. Supported input
    # units are documented in README; matcher comparisons use source units.
    condition_units: dict[str, str] | None = None

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "ExploreRequest":
        """Build a request from a frontend JSON object with strict shape checks."""

        if not isinstance(value, dict):
            raise ValueError("Explore request must be a JSON object")
        extra = set(value) - {"question", "conditions", "condition_units"}
        if extra:
            raise ValueError(f"Unexpected Explore request field(s): {', '.join(sorted(extra))}")
        question = value.get("question")
        conditions = value.get("conditions")
        units = value.get("condition_units")
        if not isinstance(question, str):
            raise ValueError("question must be a string")
        if conditions is not None and not isinstance(conditions, dict):
            raise ValueError("conditions must be an object or null")
        if units is not None and not isinstance(units, dict):
            raise ValueError("condition_units must be an object or null")
        if units is not None and not all(isinstance(key, str) and isinstance(unit, str) for key, unit in units.items()):
            raise ValueError("condition_units keys and values must be strings")
        return cls(question, conditions, units)


@dataclass(frozen=True, slots=True)
class DocumentEvidence:
    evidence_type: Literal["document"]
    experiment: str
    source_id: str
    source_file: str
    source_locator: str
    page_number: int | None
    record_identifier: str
    evidence_status: EvidenceStatus
    original_text: str
    similarity_score: float
    provenance: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExperimentalEvidence:
    evidence_type: Literal["experimental_condition"]
    experiment: str | None
    source_id: str
    source_file: str
    source_locator: str
    page_number: int | None
    record_identifier: str
    experiment_id: str | None
    evidence_status: EvidenceStatus
    original_data: dict[str, object]
    match_type: str
    parameter_matches: tuple[dict[str, object], ...]
    provenance: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


UnifiedEvidence: TypeAlias = DocumentEvidence | ExperimentalEvidence


@dataclass(frozen=True, slots=True)
class ExploreResponse:
    question: str
    answer: AnswerResponse
    document_evidence: tuple[DocumentEvidence, ...]
    experimental_evidence: tuple[ExperimentalEvidence, ...]
    rag_context: RAGContext
    condition_match: MatchResult | None
    requested_conditions: dict[str, object]
    normalized_conditions: dict[str, object]
    requested_condition_units: dict[str, str]
    condition_units: dict[str, str]
    condition_availability: dict[str, dict[str, object]]
    condition_match_summary: dict[str, object]
    evidence_mode: EvidenceMode
    status: Literal["grounded", "partial", "insufficient_evidence"]
    max_serialized_condition_matches: int = 25
    warnings: tuple[str, ...] = ()
    # How the router read the question; None when it was not used.
    routing: dict[str, object] | None = None

    @property
    def evidence(self) -> tuple[UnifiedEvidence, ...]:
        """Backward-compatible combined evidence view; JSON exposes typed arrays."""

        return (*self.document_evidence, *self.experimental_evidence)

    @property
    def citations(self) -> tuple[dict[str, object], ...]:
        return tuple(citation.to_dict() for citation in self.answer.citations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer.to_dict(),
            "status": self.status,
            "evidence_mode": self.evidence_mode,
            "routing": self.routing,
            "document_evidence": [item.to_dict() for item in self.document_evidence],
            "experimental_evidence": [item.to_dict() for item in self.experimental_evidence],
            "rag_context": self.rag_context.to_dict(),
            "condition_match": _match_result_dict(
                self.condition_match,
                limit=self.max_serialized_condition_matches,
            ),
            "requested_conditions": self.requested_conditions,
            "normalized_conditions": self.normalized_conditions,
            "requested_condition_units": self.requested_condition_units,
            "condition_units": self.condition_units,
            "condition_availability": self.condition_availability,
            "condition_match_summary": self.condition_match_summary,
            "citations": [citation.to_dict() for citation in self.answer.citations],
            "warnings": list(self.warnings),
        }


def _match_result_dict(result: MatchResult | None, *, limit: int) -> dict[str, object] | None:
    if result is None:
        return None
    status_counts = {
        "exact_match": sum(match.match_type == "exact_match" for match in result.matches),
        "close_match": sum(match.match_type == "close_match" for match in result.matches),
        "partial_match": sum(match.match_type == "partial_match" for match in result.matches),
        "no_match": sum(match.match_type == "no_match" for match in result.matches),
        "unavailable": sum(match.match_type == "parameter_unavailable" for match in result.matches),
    }
    relevant = [match for match in result.matches if match.match_type != "no_match"]
    return {
        "requested_conditions": result.requested_conditions,
        "tolerances": result.tolerances,
        "match_type": result.match_type,
        "candidate_count": len(result.matches),
        "status_counts": status_counts,
        "matches_truncated": len(relevant) > limit,
        "matches": [
            {
                "record": match.record.to_dict(),
                "match_type": match.match_type,
                "parameters": [asdict(parameter) for parameter in match.parameters],
            }
            for match in relevant[:limit]
        ],
        "unmatched_candidates": [
            {
                "record_id": match.record.record_id,
                "experiment_id": match.record.experiment_id,
                "source_id": match.record.source_id,
                "source_file": match.record.source_file,
                "source_locator": match.record.source_locator,
                "evidence_status": match.record.evidence_status,
                "match_type": match.match_type,
                "parameters": [asdict(parameter) for parameter in match.parameters],
            }
            for match in result.matches
            if match.match_type == "no_match"
        ][:limit],
        "unmatched_candidates_truncated": status_counts["no_match"] > limit,
    }
