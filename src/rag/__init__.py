"""Evidence-only RAG context packaging; no answer generation or LLM calls."""

from .config import RAGContextConfig
from .context import build_context, format_context
from .models import EvidenceItem, RAGContext
from .service import retrieve_context

__all__ = [
    "EvidenceItem",
    "RAGContext",
    "RAGContextConfig",
    "build_context",
    "format_context",
    "retrieve_context",
]
