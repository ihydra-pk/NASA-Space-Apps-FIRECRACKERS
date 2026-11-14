"""Deterministic human-readable formatting for retrieved evidence."""

from __future__ import annotations

from .models import EvidenceItem


def format_context(
    query: str,
    evidence_items: tuple[EvidenceItem, ...] | list[EvidenceItem],
    *,
    include_similarity_scores: bool = True,
    include_full_provenance: bool = True,
    dropped_chunk_ids: tuple[str, ...] | list[str] = (),
) -> str:
    """Format evidence metadata and source text without changing the text.

    Full provenance affects display only. Structured evidence always retains
    all required source identifiers, hashes, location, and status fields.
    """

    sections = ["QUESTION\n--------\n" + query]
    for item in evidence_items:
        lines = [
            f"EVIDENCE {item.rank}",
            "----------",
            f"Rank: {item.rank}",
        ]
        if include_similarity_scores:
            lines.append(f"Similarity: {item.similarity_score:.6f}")
        lines.extend(
            [
                f"Experiment: {item.experiment}",
                f"Document: {item.filename}",
                f"Source locator: {item.source_locator}",
                f"Evidence status: {item.evidence_status}",
                f"Document type: {item.document_type}",
                f"Source ID: {item.source_id}",
                f"Chunk ID: {item.chunk_id}",
            ]
        )
        if include_full_provenance:
            lines.extend(
                [
                    f"Source path: {item.source_path}",
                    f"File type: {item.file_type}",
                    f"Content SHA-256: {item.content_sha256}",
                    f"Page number: {item.page_number if item.page_number is not None else 'n/a'}",
                ]
            )
        # Insert text exactly as received. No trimming, summarizing, or rewrite.
        sections.append("\n".join(lines) + "\n\n" + item.text)

    if dropped_chunk_ids:
        sections.append(
            "CONTEXT LIMIT\n-------------\n"
            f"Omitted {len(dropped_chunk_ids)} lower-ranked evidence item(s) in full "
            "to respect the configured context size limit. No evidence text was truncated."
        )
    return "\n\n".join(sections)
