"""Composition settings for the Explore orchestration layer."""

from __future__ import annotations

from dataclasses import dataclass

from src.rag.config import RAGContextConfig


@dataclass(frozen=True, slots=True)
class ExploreConfig:
    rag: RAGContextConfig = RAGContextConfig()
    max_condition_evidence: int = 25
    max_condition_context_chars: int = 12_000

    def __post_init__(self) -> None:
        if self.max_condition_evidence <= 0:
            raise ValueError("max_condition_evidence must be greater than zero")
        if self.max_condition_context_chars < 128:
            raise ValueError("max_condition_context_chars must be at least 128")
