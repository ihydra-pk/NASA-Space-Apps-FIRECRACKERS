"""Thin service adapter connecting an existing retriever to context building."""

from __future__ import annotations

from typing import Any, Callable

from .config import RAGContextConfig
from .context import build_context
from .models import RAGContext


def retrieve_context(
    query: str,
    retriever: Callable[..., Any],
    config: RAGContextConfig | None = None,
) -> RAGContext:
    """Call the existing retrieval function, then package its evidence.

    ``retriever`` is expected to accept ``(query, top_k=...)`` and return the
    existing ranked search-result model. No stores, documents, or embeddings
    are accessed by this module directly.
    """

    config = config or RAGContextConfig()
    retrieval = retriever(query, top_k=config.top_k)
    return build_context(query, retrieval, config)
