"""Embedding generation for provenance-preserving chunk records."""

from .config import EmbeddingConfig
from .models import EmbeddingFailure, EmbeddingRecord, EmbeddingResult
from .pipeline import embed_record, run_embedding
from .providers import (
    EmbeddingProvider,
    HashMockEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
    create_embedding_provider,
)

__all__ = [
    "EmbeddingConfig",
    "EmbeddingFailure",
    "EmbeddingProvider",
    "EmbeddingRecord",
    "EmbeddingResult",
    "HashMockEmbeddingProvider",
    "OpenAICompatibleEmbeddingProvider",
    "create_embedding_provider",
    "embed_record",
    "run_embedding",
]
