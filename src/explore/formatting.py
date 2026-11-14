"""Build a separate, provenance-rich structured condition block for grounded generation."""

from __future__ import annotations

import json
import re
from src.experimental_data.matching import MatchResult
from src.rag.models import RAGContext

from .evidence import matching_experimental_matches


def format_condition_evidence(
    result: MatchResult | None,
    *,
    max_records: int,
    max_characters: int,
) -> tuple[str, tuple[str, ...]]:
    """Render structured matches; similarity scores are never included here."""

    if result is None:
        return "", ()
    records: list[dict[str, object]] = []
    omitted: list[str] = []
    for match in matching_experimental_matches(result):
        if len(records) >= max_records:
            omitted.append(match.record.record_id)
            continue
        records.append(
            {
                "record_id": match.record.record_id,
                "experiment": match.record.experiment,
                "experiment_id": match.record.experiment_id,
                "evidence_status": match.record.evidence_status,
                "match_type": match.match_type,
                "source_file": match.record.source_file,
                "source_locator": match.record.source_locator,
                "source_id": match.record.source_id,
                "content_sha256": match.record.content_sha256,
                "original_data": {
                    key: value
                    for key, value in {
                        "fuel": match.record.fuel,
                        "oxygen_concentration": match.record.oxygen_concentration,
                        "pressure": match.record.pressure,
                        "flow_velocity": match.record.flow_velocity,
                        "suppressant": match.record.suppressant,
                        "suppressant_concentration": match.record.suppressant_concentration,
                        "diluent": match.record.diluent,
                        "fiber_support": match.record.fiber_support,
                        "units": match.record.units,
                    }.items()
                    if value is not None
                },
                "parameter_matches": [
                    {
                        "parameter": item.parameter,
                        "match_type": item.match_type,
                        "requested_value": item.requested_value,
                        "experimental_value": item.experimental_value,
                        "unit": item.unit,
                        "difference": item.difference,
                        "tolerance": item.tolerance,
                    }
                    for item in match.parameters
                ],
            }
        )
    block = (
        "STRUCTURED EXPERIMENT-CONDITION MATCHES (not document similarity)\n"
        "-----------------------------------------------------------------\n"
        "Evidence status is source-authored; planned conditions are not observed results.\n"
        "These records are condition-match evidence, not document chunks; use record IDs for their provenance.\n"
        f"Overall matcher status: {result.match_type}\n"
        f"Matcher tolerances: {json.dumps(result.tolerances, sort_keys=True)}\n"
        f"Matching records: {json.dumps(records, ensure_ascii=False, sort_keys=True)}"
    )
    if omitted:
        block += f"\nAdditional matching record IDs omitted from this prompt for size: {json.dumps(omitted)}"
    while len(block) > max_characters and records:
        removed = records.pop()
        omitted.append(str(removed["record_id"]))
        block = (
            "STRUCTURED EXPERIMENT-CONDITION MATCHES (not document similarity)\n"
            "-----------------------------------------------------------------\n"
            "Evidence status is source-authored; planned conditions are not observed results.\n"
            f"Overall matcher status: {result.match_type}\n"
            f"Matcher tolerances: {json.dumps(result.tolerances, sort_keys=True)}\n"
            f"Matching records: {json.dumps(records, ensure_ascii=False, sort_keys=True)}"
        )
        if omitted:
            block += f"\nAdditional matching record IDs omitted from this prompt for size: {json.dumps(omitted)}"
    if len(block) > max_characters:
        block = (
            "STRUCTURED EXPERIMENT-CONDITION MATCHES omitted from LLM prompt due to size limit. "
            f"Matcher status: {result.match_type}. Evidence remains available in Explore response."
        )
    return block, tuple(omitted)


def add_condition_evidence(context: RAGContext, block: str) -> RAGContext:
    """Copy an existing context while retaining document evidence/score identity."""

    if not block:
        return context
    text = f"{context.context_text}\n\n{block}"
    return RAGContext(
        query=context.query,
        retrieval_count=context.retrieval_count,
        evidence_items=context.evidence_items,
        retrieval_configuration=dict(context.retrieval_configuration),
        context_text=text,
        context_characters=len(text),
        estimated_context_tokens=len(re.findall(r"\w+|[^\w\s]", text, re.UNICODE)),
        truncation_occurred=context.truncation_occurred,
        dropped_evidence_count=context.dropped_evidence_count,
        dropped_chunk_ids=context.dropped_chunk_ids,
    )
