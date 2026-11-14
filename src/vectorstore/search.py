"""Provider-aware query embedding, cosine similarity and hybrid retrieval."""

from __future__ import annotations

import math

from src.embeddings.config import EmbeddingConfig
from src.embeddings.pipeline import _validate_vectors
from src.embeddings.providers import EmbeddingProvider, create_embedding_provider

from .config import VectorStoreConfig
from .keyword import bm25_scores
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


def query_text_for_embedding(query_text: str, config: VectorStoreConfig) -> str:
    """The text sent to the embedding model for a query.

    Qwen3 embedding models expect ``Instruct: <task>\nQuery: <question>`` on
    the query side only. Other models, including the mock provider, get the
    question unchanged.
    """

    if config.query_instruction and "qwen3-embedding" in config.model_name.casefold():
        return f"Instruct: {config.query_instruction}\nQuery: {query_text}"
    return query_text


def _rankable(
    scored: list[tuple[float, dict, tuple[float, ...]]], min_chunk_words: int
) -> list[tuple[float, dict, tuple[float, ...]]]:
    """Drop title-only chunks and repeats of a passage already ranked higher.

    The same paper is often filed under several names, so identical text would
    otherwise fill several result slots. ``scored`` is in similarity order, so
    the first copy kept is the best-scoring one.
    """

    kept: list[tuple[float, dict, tuple[float, ...]]] = []
    seen: set[str] = set()
    for item in scored:
        words = item[1]["text"].split()
        if len(words) < min_chunk_words:
            continue
        key = " ".join(words).casefold()
        if key in seen:
            continue
        seen.add(key)
        kept.append(item)
    return kept


SEARCH_MODES = ("semantic", "hybrid")
# Reciprocal rank fusion: each ranking contributes weight / (k + rank). A small
# k keeps the top semantic hits well ahead; the keyword ranking counts for less,
# so it promotes exact-term matches without letting word overlap alone outrank
# a close match in meaning. Chosen by checking a handful of questions against
# the full index; revisit once there is a question set with known answers.
_RRF_K = 10
_KEYWORD_WEIGHT = 0.3


def _fuse_with_keywords(
    query_text: str,
    scored: list[tuple[float, dict, tuple[float, ...]]],
    count: int,
) -> tuple[RetrievalResult, ...]:
    """Re-rank similarity-sorted chunks by fusing semantic and keyword ranks."""

    keyword_scores = bm25_scores(query_text, [record["text"] for _, record, _ in scored])
    by_keyword = sorted(
        (index for index, score in enumerate(keyword_scores) if score > 0),
        key=lambda index: (-keyword_scores[index], scored[index][1]["chunk_id"]),
    )
    keyword_rank = {index: rank for rank, index in enumerate(by_keyword, start=1)}

    def fused(index: int) -> float:
        value = 1.0 / (_RRF_K + index + 1)  # ``scored`` is already in semantic order
        if index in keyword_rank:
            value += _KEYWORD_WEIGHT / (_RRF_K + keyword_rank[index])
        return value

    order = sorted(range(len(scored)), key=lambda index: (-fused(index), scored[index][1]["chunk_id"]))
    return tuple(
        RetrievalResult.from_record(
            rank,
            scored[index][0],
            scored[index][1],
            semantic_rank=index + 1,
            keyword_rank=keyword_rank.get(index),
            keyword_score=keyword_scores[index] if index in keyword_rank else None,
        )
        for rank, index in enumerate(order[:count], start=1)
    )


def search(
    query_text: str,
    store: VectorStore,
    config: VectorStoreConfig,
    *,
    provider: EmbeddingProvider | None = None,
    top_k: int | None = None,
    mode: str = "semantic",
) -> SearchResult:
    """Embed a query with the indexed profile and return ranked evidence.

    ``mode="semantic"`` ranks by cosine similarity alone. ``mode="hybrid"``
    also scores chunk text by keyword (BM25) and orders by reciprocal rank
    fusion of the two rankings. Fusion uses ranks only, so the two score
    scales are never added together and ``similarity_score`` stays cosine.
    """

    if not isinstance(query_text, str) or not query_text.strip():
        raise ValueError("query_text must not be empty")
    if mode not in SEARCH_MODES:
        raise ValueError(f"mode must be one of {SEARCH_MODES}; got {mode!r}")
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
        vectors = _validate_vectors(provider.embed_batch([query_text_for_embedding(query_text, config)], embedding_config), 1, embedding_config)
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
    scored = _rankable(scored, config.min_chunk_words)
    if mode == "hybrid":
        results = _fuse_with_keywords(query_text, scored, count)
    else:
        results = tuple(
            RetrievalResult.from_record(rank, score, record)
            for rank, (score, record, _) in enumerate(scored[:count], start=1)
        )
    return SearchResult(
        mode=mode,
        query=query_text,
        provider=config.provider,
        model_name=config.model_name,
        dimension=config.dimension,
        similarity_metric=config.similarity_metric,
        matches=results,
    )
