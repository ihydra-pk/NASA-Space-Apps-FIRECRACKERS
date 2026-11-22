"""Convert existing RAG and experimental matching results without score blending."""

from __future__ import annotations

from dataclasses import asdict

from src.experimental_data.matching import MatchResult
from src.rag.models import RAGContext

from .models import DocumentEvidence, ExperimentalEvidence, UnifiedEvidence


def matching_experimental_matches(result: MatchResult | None):
    """Select condition rows that actually satisfy every comparable request.

    The existing matcher reports `partial_match` when some requested fields
    agree and others do not or are unavailable. Preserve those useful partial
    candidates, but do not turn pure no-match candidates into evidence.
    """

    if result is None:
        return ()
    return tuple(
        match
        for match in result.matches
        if match.match_type in {"exact_match", "close_match", "partial_match"}
    )


def document_evidence(context: RAGContext) -> tuple[DocumentEvidence, ...]:
    return tuple(
        DocumentEvidence(
            evidence_type="document",
            experiment=item.experiment,
            source_id=item.source_id,
            source_file=item.source_path,
            source_locator=item.source_locator,
            page_number=item.page_number,
            record_identifier=item.chunk_id,
            evidence_status=item.evidence_status,
            original_text=item.text,
            similarity_score=item.similarity_score,
            provenance={
                "chunk_id": item.chunk_id,
                "content_sha256": item.content_sha256,
                "filename": item.filename,
                "file_type": item.file_type,
                "source_path": item.source_path,
                "document_type": item.document_type,
                "retrieval_rank": item.rank,
            },
        )
        for item in context.evidence_items
    )


def experimental_evidence(
    result: MatchResult | None,
    *,
    limit: int,
) -> tuple[ExperimentalEvidence, ...]:
    if result is None:
        return ()
    # No-match candidates remain available on MatchResult. The unified evidence
    # list carries records with at least one relevant/partial match only.
    selected = matching_experimental_matches(result)[:limit]
    return tuple(
        ExperimentalEvidence(
            evidence_type="experimental_condition",
            experiment=match.record.experiment,
            source_id=match.record.source_id,
            source_file=match.record.source_file,
            source_locator=match.record.source_locator,
            page_number=_source_page(match.record.source_locator),
            record_identifier=match.record.record_id,
            experiment_id=match.record.experiment_id,
            evidence_status=match.record.evidence_status,
            original_data=match.record.to_dict(),
            match_type=match.match_type,
            parameter_matches=tuple(asdict(parameter) for parameter in match.parameters),
            provenance={
                "content_sha256": match.record.content_sha256,
                "source_url": match.record.source_url,
                "table_number": match.record.table_number,
                "record_id": match.record.record_id,
                "source_locator": match.record.source_locator,
            },
        )
        for match in selected
    )


def unified_evidence(
    context: RAGContext,
    match_result: MatchResult | None,
    *,
    max_condition_evidence: int,
) -> tuple[UnifiedEvidence, ...]:
    return (
        *document_evidence(context),
        *experimental_evidence(match_result, limit=max_condition_evidence),
    )


def _source_page(locator: str) -> int | None:
    # Experimental source locators have the explicit shape page:N:... .
    parts = locator.split(":")
    if len(parts) >= 2 and parts[0] == "page":
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None
