"""Provider-aware query embedding and cosine similarity retrieval."""

from __future__ import annotations

import math

from src.embeddings.config import EmbeddingConfig
from src.embeddings.pipeline import _validate_vectors
from src.embeddings.providers import EmbeddingProvider, create_embedding_provider

from .config import VectorStoreConfig
from .models import RetrievalResult, SearchResult
from .store import VectorStore


def cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        raise ValueError(f"Cannot compare vectors with dimensions {len(left)} and {len(right)}")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("Cosine similarity is undefined for a zero vector")
    score = sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
    # Floating-point roundoff can move mathematically bounded cosine scores a
    # few ulps outside [-1, 1].
    return max(-1.0, min(1.0, score))


def search(
    query_text: str,
    store: VectorStore,
    config: VectorStoreConfig,
    *,
    provider: EmbeddingProvider | None = None,
    top_k: int | None = None,
) -> SearchResult:
    """Embed a query with the indexed profile and return ranked evidence."""

    if not isinstance(query_text, str) or not query_text.strip():
        raise ValueError("query_text must not be empty")
    count = config.top_k if top_k is None else top_k
    if count <= 0:
        raise ValueError("top_k must be greater than zero")
    provider = provider or create_embedding_provider(
        EmbeddingConfig(
            provider=config.provider,
            model_name=config.model_name,
            dimension=config.dimension,
            batch_size=1,
        )
    )
    provider_name = getattr(provider, "provider_name", None)
    if provider_name != config.provider:
        raise ValueError(
            f"Query provider {provider_name!r} does not match indexed provider {config.provider!r}"
        )

    profile = store.get_collection_profile()
    expected_profile = (config.provider, config.model_name, config.dimension)
    if profile is not None and profile != expected_profile:
        raise ValueError(f"Query profile {expected_profile} does not match stored profile {profile}")

    embedding_config = EmbeddingConfig(
        provider=config.provider,
        model_name=config.model_name,
        dimension=config.dimension,
        batch_size=1,
    )
    try:
        vectors = _validate_vectors(provider.embed_batch([query_text], embedding_config), 1, embedding_config)
    except Exception as exc:
        raise RuntimeError(f"Could not embed retrieval query: {exc}") from exc
    query_vector = vectors[0]

    scored: list[tuple[float, dict, tuple[float, ...]]] = []
    for record, vector in store.records():
        if len(vector) != config.dimension:
            raise ValueError(
                f"Stored chunk {record.get('chunk_id')!r} has dimension {len(vector)}; "
                f"expected {config.dimension}"
            )
        scored.append((cosine_similarity(query_vector, vector), record, vector))
    scored.sort(key=lambda item: (-item[0], item[1]["chunk_id"]))
    results = tuple(
        RetrievalResult.from_record(rank, score, record)
        for rank, (score, record, _) in enumerate(scored[:count], start=1)
    )
    return SearchResult(
        query=query_text,
        provider=config.provider,
        model_name=config.model_name,
        dimension=config.dimension,
        similarity_metric=config.similarity_metric,
        matches=results,
    )
