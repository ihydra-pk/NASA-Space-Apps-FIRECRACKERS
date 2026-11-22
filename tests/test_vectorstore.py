from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from pathlib import Path

import pytest

from src.embeddings.config import EmbeddingConfig
from src.embeddings.providers import HashMockEmbeddingProvider
from src.vectorstore import (
    SQLiteVectorStore,
    VectorStoreConfig,
    index_embeddings,
    search,
)
from src.vectorstore.pipeline import validate_embedding_record
from src.vectorstore.search import cosine_similarity


def _record(chunk_id: str, vector: list[float], **overrides) -> dict:
    record = {
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
        "text": f"Evidence text for {chunk_id}.",
        "chunk_index": 1,
        "chunk_count": 1,
        "estimated_token_count": 10,
        "document_metadata": {"producer": "NASA"},
        "embedding": vector,
        "embedding_model": "test-model",
        "embedding_provider": "test-provider",
        "embedding_dimension": len(vector),
    }
    record.update(overrides)
    return record


def _config(tmp_path: Path, dimension: int = 3) -> VectorStoreConfig:
    return VectorStoreConfig(
        database_path=tmp_path / "vectors.sqlite3",
        collection="test-collection",
        provider="test-provider",
        model_name="test-model",
        dimension=dimension,
        top_k=3,
    )


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_sqlite_store_creation_and_indexing_valid_records(tmp_path):
    source = tmp_path / "embeddings" / "flex" / "records.jsonl"
    records = [_record("a", [1.0, 0.0, 0.0]), _record("b", [0.0, 1.0, 0.0])]
    _write_jsonl(source, records)
    config = _config(tmp_path)

    result = index_embeddings(source.parent.parent, config)

    assert result.input_records == result.indexed == 2
    assert result.upserted == result.duplicates == result.rejected == 0
    assert result.unique_chunk_ids == 2
    assert config.database_path.is_file()
    with SQLiteVectorStore(config) as store:
        stored = store.records()
    assert {record["chunk_id"] for record, _ in stored} == {"a", "b"}


def test_repeated_indexing_is_idempotent_and_duplicate_ids_are_reported(tmp_path):
    source_dir = tmp_path / "embeddings"
    source = source_dir / "flex" / "records.jsonl"
    record = _record("same", [1.0, 0.0, 0.0])
    _write_jsonl(source, [record])
    config = _config(tmp_path)

    first = index_embeddings(source_dir, config)
    second = index_embeddings(source_dir, config)
    assert first.indexed == 1
    assert second.upserted == 1
    assert second.indexed == 0
    with SQLiteVectorStore(config) as store:
        assert len(store.records()) == 1

    duplicate_input = tmp_path / "duplicate-input"
    _write_jsonl(duplicate_input / "duplicates.jsonl", [record, record])
    duplicate_result = index_embeddings(duplicate_input, config)
    assert duplicate_result.input_records == 2
    assert duplicate_result.upserted == 1
    assert duplicate_result.duplicates == 1
    assert len(duplicate_result.failures) == 1


def test_malformed_records_and_vectors_are_reported(tmp_path):
    input_dir = tmp_path / "embeddings"
    _write_jsonl(
        input_dir / "flex" / "records.jsonl",
        [
            _record("wrong-dimension", [1.0, 0.0]),
            _record("wrong-model", [1.0, 0.0, 0.0], embedding_model="other-model"),
            {"chunk_id": "missing"},
        ],
    )
    (input_dir / "flex" / "bad.jsonl").write_text("not-json\n", encoding="utf-8")

    result = index_embeddings(input_dir, _config(tmp_path))

    assert result.input_records == 4
    assert result.indexed == 0
    assert result.rejected == 4
    assert any("dimension 2" in failure for failure in result.failures)
    assert any("does not match collection model" in failure for failure in result.failures)
    assert any("missing required embedding fields" in failure for failure in result.failures)
    assert any("invalid JSON" in failure for failure in result.failures)


def test_non_numeric_and_non_finite_vectors_are_rejected():
    config = _config(Path("."))
    with pytest.raises(ValueError, match="not numeric"):
        validate_embedding_record(_record("bad", [1.0, "x", 0.0]), config)
    with pytest.raises(ValueError, match="not finite"):
        validate_embedding_record(_record("bad", [1.0, math.inf, 0.0]), config)


def test_store_persists_vectors_and_metadata_across_reload(tmp_path):
    config = _config(tmp_path)
    record = _record("persistent", [0.25, 0.5, 0.75])
    with SQLiteVectorStore(config) as store:
        store.upsert(record, record["embedding"])

    with SQLiteVectorStore(config) as reloaded:
        [(stored_record, vector)] = reloaded.records()
    assert stored_record == record
    assert vector == (0.25, 0.5, 0.75)
    assert stored_record["document_metadata"] == {"producer": "NASA"}
    assert stored_record["evidence_status"] == "planned"


def test_collection_rejects_model_or_dimension_profile_mismatch(tmp_path):
    config = _config(tmp_path)
    with SQLiteVectorStore(config):
        pass
    with pytest.raises(ValueError, match="uses profile"):
        SQLiteVectorStore(
            VectorStoreConfig(
                database_path=config.database_path,
                collection=config.collection,
                provider=config.provider,
                model_name="another-model",
                dimension=config.dimension,
            )
        )


def test_cosine_similarity_and_ranking_top_k_and_traceability(tmp_path):
    assert cosine_similarity((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)) == pytest.approx(1.0)
    assert cosine_similarity((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)) == pytest.approx(0.0)

    config = _config(tmp_path)
    with SQLiteVectorStore(config) as store:
        store.upsert(_record("aligned", [1.0, 0.0, 0.0]), [1.0, 0.0, 0.0])
        store.upsert(_record("diagonal", [0.7, 0.7, 0.0]), [0.7, 0.7, 0.0])
        store.upsert(_record("orthogonal", [0.0, 1.0, 0.0]), [0.0, 1.0, 0.0])
        store.upsert(_record("negative", [-1.0, 0.0, 0.0]), [-1.0, 0.0, 0.0])

        class QueryProvider:
            provider_name = "test-provider"

            def embed_batch(self, texts, embedding_config):
                assert texts == ["query"]
                assert embedding_config.model_name == "test-model"
                return [[1.0, 0.0, 0.0]]

        result = search("query", store, config, provider=QueryProvider(), top_k=2)

    assert [item.chunk_id for item in result.matches] == ["aligned", "diagonal"]
    assert [item.rank for item in result.matches] == [1, 2]
    assert result.matches[0].similarity_score == pytest.approx(1.0)
    first = result.matches[0]
    assert first.source_id == "nasa-source"
    assert first.content_sha256 == hashlib.sha256(b"source").hexdigest()
    assert first.source_path == "flex/FLEX_Test_Plan.pdf"
    assert first.filename == "FLEX_Test_Plan.pdf"
    assert first.source_locator == "page:2"
    assert first.page_number == 2
    assert first.document_type == "test_plan"
    assert first.evidence_status == "planned"
    assert first.text == "Evidence text for aligned."


def test_search_rejects_empty_query_and_provider_dimension_mismatch(tmp_path):
    config = _config(tmp_path)
    with SQLiteVectorStore(config) as store:
        with pytest.raises(ValueError, match="must not be empty"):
            search("  ", store, config, provider=HashMockEmbeddingProvider())

        class WrongDimensionProvider:
            provider_name = "test-provider"

            def embed_batch(self, texts, embedding_config):
                return [[1.0, 0.0]]

        with pytest.raises(RuntimeError, match="expected 3"):
            search("query", store, config, provider=WrongDimensionProvider())


def test_query_provider_failure_is_surfaced(tmp_path):
    config = _config(tmp_path)
    with SQLiteVectorStore(config) as store:
        class FailedProvider:
            provider_name = "test-provider"

            def embed_batch(self, texts, embedding_config):
                raise RuntimeError("provider unavailable")

        with pytest.raises(RuntimeError, match="provider unavailable"):
            search("query", store, config, provider=FailedProvider())


def test_mock_provider_cannot_search_real_provider_collection(tmp_path):
    config = VectorStoreConfig(
        database_path=tmp_path / "real.sqlite3",
        collection="real",
        provider="openai-compatible",
        model_name="qwen-real",
        dimension=3,
    )
    with SQLiteVectorStore(config) as store:
        with pytest.raises(ValueError, match="does not match indexed provider"):
            search("query", store, config, provider=HashMockEmbeddingProvider())


def test_duplicate_conflict_and_real_flex_records_preserve_metadata(tmp_path):
    repository_root = Path(__file__).resolve().parents[1]
    real_dir = (
        repository_root
        / "data"
        / "embeddings"
        / "openai-compatible"
        / "gwdg.qwen3-embedding-4b"
        / "flex"
    )
    if not real_dir.is_dir():
        return
    config = VectorStoreConfig(
        database_path=tmp_path / "real.sqlite3",
        collection="flex-real",
        provider="openai-compatible",
        model_name="gwdg.qwen3-embedding-4b",
        dimension=2560,
    )
    result = index_embeddings(real_dir, config)
    assert result.input_records == result.indexed >= 18
    assert result.rejected == result.duplicates == 0
    with SQLiteVectorStore(config) as store:
        records = store.records()
    assert len(records) == result.indexed
    for record, vector in records:
        assert len(vector) == 2560
        assert record["embedding"] == list(vector)
        assert record["evidence_status"] in {"planned", "metadata"}
        assert record["source_locator"]
