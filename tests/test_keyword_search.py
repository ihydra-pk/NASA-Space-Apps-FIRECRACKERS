from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.vectorstore import SQLiteVectorStore, VectorStoreConfig, search
from src.vectorstore.keyword import bm25_scores, tokenize


def _record(chunk_id: str, vector: list[float], text: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "source_id": "nasa-source",
        "content_sha256": hashlib.sha256(b"source").hexdigest(),
        "experiment": "FLEX",
        "filename": "FLEX_Test_Plan.pdf",
        "file_type": "pdf",
        "source_path": "flex/FLEX_Test_Plan.pdf",
        "document_type": "test_plan",
        "evidence_status": "planned",
        "source_locator": "page:2",
        "page_number": 2,
        "text": text,
        "embedding": vector,
        "embedding_model": "test-model",
        "embedding_provider": "test-provider",
        "embedding_dimension": len(vector),
    }


def _config(tmp_path: Path) -> VectorStoreConfig:
    return VectorStoreConfig(
        database_path=tmp_path / "vectors.sqlite3",
        collection="test-collection",
        provider="test-provider",
        model_name="test-model",
        dimension=3,
        top_k=3,
    )


class QueryProvider:
    provider_name = "test-provider"

    def embed_batch(self, texts, embedding_config):
        return [[1.0, 0.0, 0.0]]


def _store(tmp_path: Path) -> tuple[SQLiteVectorStore, VectorStoreConfig]:
    config = _config(tmp_path)
    store = SQLiteVectorStore(config)
    # "near" is closest in meaning; only "exact" names the suppressant.
    store.upsert(_record("near", [1.0, 0.0, 0.0], "Droplets burn in a quiescent chamber."), [1.0, 0.0, 0.0])
    store.upsert(_record("middle", [0.8, 0.6, 0.0], "The chamber is filled before each test."), [0.8, 0.6, 0.0])
    store.upsert(_record("exact", [0.6, 0.8, 0.0], "Tests 201 to 228 add SF6 as suppressant."), [0.6, 0.8, 0.0])
    store.upsert(_record("far", [0.0, 0.0, 1.0], "Crew time is scheduled weekly."), [0.0, 0.0, 1.0])
    return store, config


def test_tokenize_keeps_identifiers_whole():
    assert tokenize("PSI-69 uses SF6 (C7H16)") == ["psi", "69", "uses", "sf6", "c7h16"]


def test_bm25_scores_only_texts_containing_query_terms():
    scores = bm25_scores("SF6 suppressant", ["SF6 is a suppressant", "helium only", "SF6 SF6 SF6"])
    assert scores[1] == 0.0
    assert scores[0] > 0 and scores[2] > 0
    assert scores[0] > scores[2]  # matches both query terms
    assert bm25_scores("", ["text"]) == [0.0]
    assert bm25_scores("query", []) == []


def test_semantic_mode_is_unchanged_and_reports_no_keyword_fields(tmp_path):
    store, config = _store(tmp_path)
    with store:
        result = search("which tests use SF6", store, config, provider=QueryProvider(), top_k=3)
    assert result.mode == "semantic"
    assert [item.chunk_id for item in result.matches] == ["near", "middle", "exact"]
    assert all(item.keyword_rank is None and item.semantic_rank is None for item in result.matches)


def test_hybrid_mode_promotes_exact_identifier_and_keeps_cosine_score(tmp_path):
    store, config = _store(tmp_path)
    with store:
        result = search(
            "which tests use SF6", store, config, provider=QueryProvider(), top_k=3, mode="hybrid"
        )
    assert result.mode == "hybrid"
    by_id = {item.chunk_id: item for item in result.matches}
    exact = by_id["exact"]
    # Semantically third, but the only keyword hit for "SF6", so it moves up.
    assert exact.semantic_rank == 3
    assert exact.keyword_rank == 1
    assert exact.rank < by_id["middle"].rank
    # The reported score is still the cosine similarity, not a blended value.
    assert exact.similarity_score == pytest.approx(0.6)
    assert exact.keyword_score > 0
    assert [item.rank for item in result.matches] == [1, 2, 3]


def test_hybrid_without_any_keyword_hit_falls_back_to_semantic_order(tmp_path):
    store, config = _store(tmp_path)
    with store:
        result = search("zzz", store, config, provider=QueryProvider(), top_k=3, mode="hybrid")
    assert [item.chunk_id for item in result.matches] == ["near", "middle", "exact"]
    assert all(item.keyword_rank is None for item in result.matches)


def test_unknown_mode_is_rejected(tmp_path):
    store, config = _store(tmp_path)
    with store, pytest.raises(ValueError, match="mode must be one of"):
        search("query", store, config, provider=QueryProvider(), mode="keyword")
