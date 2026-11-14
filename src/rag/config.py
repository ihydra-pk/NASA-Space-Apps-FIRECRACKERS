"""Context assembly limits and display options."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RAGContextConfig:
    """Controls retrieval count and deterministic evidence formatting.

    Context limits omit lower-ranked evidence records in full. Scientific text
    is never truncated. Similarity is always preserved as a similarity score;
    it is never renamed to, or interpreted as, confidence.
    """

    top_k: int = 5
    # Room for about a dozen 800-token chunks with their provenance headers.
    max_context_chars: int | None = 60_000
    max_estimated_tokens: int | None = None
    include_similarity_scores: bool = True
    include_full_provenance: bool = True

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if self.max_context_chars is not None and self.max_context_chars <= 0:
            raise ValueError("max_context_chars must be greater than zero or None")
        if self.max_estimated_tokens is not None and self.max_estimated_tokens <= 0:
            raise ValueError("max_estimated_tokens must be greater than zero or None")
        if self.max_context_chars is None and self.max_estimated_tokens is None:
            raise ValueError("At least one context size limit must be configured")
