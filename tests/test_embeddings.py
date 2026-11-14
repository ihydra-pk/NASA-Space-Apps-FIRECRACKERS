from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.embeddings import (
    EmbeddingConfig,
    HashMockEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
    embed_record,
    run_embedding,
)


def _chunk(text: str = "A flame test at 1.00 atm and 21% oxygen.", **overrides) -> dict:
    record = {
        "chunk_id": "chunk-001",
        "source_id": "nasa-source-001",
        "content_sha256": hashlib.sha256(b"NASA source").hexdigest(),
        "experiment": "FLEX",
        "filename": "FLEX_Test_Plan.pdf",
        "file_type": "pdf",
        "source_path": "flex/FLEX_Test_Plan.pdf",
        "document_type": "test_plan",
        "evidence_status": "planned",
        "source_locator": "page:1",
        "page_number": 1,
        "text": text,
        "chunk_index": 1,
        "chunk_count": 1,
        "estimated_token_count": 12,
        "document_metadata": {"producer": "NASA"},
        "source_url": None,
        "source_accession": None,
    }
    record.update(overrides)
    return record


def test_embedding_record_preserves_chunk_provenance_and_configuration():
    record = _chunk()
    config = EmbeddingConfig(provider="mock", model_name="test-hash-v1", dimension=24)

    embedded = embed_record(record, config).to_dict()

    for key, value in record.items():
        assert embedded[key] == value
    assert len(embedded["embedding"]) == 24
    assert embedded["embedding_dimension"] == 24
    assert embedded["embedding_provider"] == "mock"
    assert embedded["embedding_model"] == "test-hash-v1"


def test_mock_embeddings_are_deterministic_and_change_with_text():
    config = EmbeddingConfig(dimension=32)
    provider = HashMockEmbeddingProvider()

    first = provider.embed_batch(["oxygen at 21%"], config)[0]
    second = provider.embed_batch(["oxygen at 21%"], config)[0]
    different = provider.embed_batch(["oxygen at 22%"], config)[0]

    assert first == second
    assert first != different
    assert len(first) == 32


class CountingProvider:
    provider_name = "test-provider"

    def __init__(self):
        self.batch_sizes: list[int] = []

    def embed_batch(self, texts, config):
        self.batch_sizes.append(len(texts))
        return [[float(index + 1)] * config.dimension for index, _ in enumerate(texts)]


def test_pipeline_batches_records_and_reads_output_jsonl_back(tmp_path):
    input_dir = tmp_path / "chunks"
    output_dir = tmp_path / "embeddings"
    source = input_dir / "flex" / "plan.jsonl"
    source.parent.mkdir(parents=True)
    source_records = [_chunk(f"Chunk content {index}", chunk_id=f"chunk-{index}") for index in range(5)]
    source.write_text("".join(json.dumps(record) + "\n" for record in source_records), encoding="utf-8")
    input_before = source.read_bytes()
    provider = CountingProvider()
    config = EmbeddingConfig(input_dir, output_dir, provider="test-provider", model_name="fixture", dimension=4, batch_size=2)

    result = run_embedding(config, provider=provider)

    assert result.input_chunks == result.successfully_embedded == 5
    assert result.failed_chunks == 0
    assert provider.batch_sizes == [2, 2, 1]
    output = output_dir / "test-provider" / "fixture" / "flex" / "plan.jsonl"
    saved = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(saved) == 5
    assert saved[0]["chunk_id"] == source_records[0]["chunk_id"]
    assert saved[0]["embedding"] == [1.0] * 4
    assert source.read_bytes() == input_before


def test_empty_text_and_malformed_chunk_are_reported_not_discarded(tmp_path):
    input_dir = tmp_path / "chunks"
    output_dir = tmp_path / "embeddings"
    source = input_dir / "flex" / "plan.jsonl"
    source.parent.mkdir(parents=True)
    records = [_chunk(""), _chunk("valid text", chunk_id="valid"), {"chunk_id": "missing-fields"}]
    source.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    result = run_embedding(EmbeddingConfig(input_dir, output_dir, dimension=6, batch_size=2))

    assert result.input_chunks == 3
    assert result.successfully_embedded == 1
    assert result.failed_chunks == 2
    assert any("text is empty" in failure.error for failure in result.failures)
    assert any("missing required chunk fields" in failure.error for failure in result.failures)
    errors = [
        json.loads(line)
        for line in (
            output_dir / "flex" / "plan.errors.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert len(errors) == 2
    assert all("chunk_record" in failure for failure in errors)


def test_malformed_json_line_is_reported_with_line_number(tmp_path):
    input_dir = tmp_path / "chunks"
    output_dir = tmp_path / "embeddings"
    source = input_dir / "flex" / "bad.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text('{"chunk_id":"ok"}\nnot-json\n', encoding="utf-8")

    result = run_embedding(EmbeddingConfig(input_dir, output_dir, dimension=4))

    assert result.input_chunks == 2
    assert result.failed_chunks == 2
    assert {failure.line_number for failure in result.failures} == {1, 2}
    assert (output_dir / "flex" / "bad.errors.jsonl").is_file()


def test_provider_errors_and_bad_vectors_are_reported(tmp_path):
    input_dir = tmp_path / "chunks"
    output_dir = tmp_path / "embeddings"
    source = input_dir / "flex" / "provider.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(_chunk()) + "\n", encoding="utf-8")

    class FailingProvider:
        provider_name = "failure"

        def embed_batch(self, texts, config):
            raise RuntimeError("provider offline")

    result = run_embedding(
        EmbeddingConfig(input_dir, output_dir, provider="failure", dimension=4),
        provider=FailingProvider(),
    )
    assert result.successfully_embedded == 0
    assert result.failed_chunks == 1
    assert "provider offline" in result.failures[0].error

    class MalformedVectorProvider:
        provider_name = "malformed"

        def embed_batch(self, texts, config):
            return [[1.0]]

    result = run_embedding(
        EmbeddingConfig(input_dir, output_dir, provider="malformed", dimension=4),
        provider=MalformedVectorProvider(),
    )
    assert result.failed_chunks == 1
    assert "expected 4" in result.failures[0].error


def test_non_mock_provider_requires_an_explicit_implementation(tmp_path):
    with pytest.raises(ValueError, match="Unsupported embedding provider"):
        run_embedding(EmbeddingConfig(input_dir=tmp_path, provider="remote"))


def test_real_provider_fails_clearly_without_credentials(monkeypatch):
    monkeypatch.delenv("FIRELENS_EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("UPB_AI_GATEWAY_KEY", raising=False)
    provider = OpenAICompatibleEmbeddingProvider(api_key=None)

    with pytest.raises(RuntimeError, match="FIRELENS_EMBEDDING_API_KEY or OPENAI_API_KEY"):
        provider.embed_batch(["semantic text"], EmbeddingConfig(provider="openai-compatible"))


def test_real_provider_can_use_existing_gateway_environment_credential(monkeypatch):
    monkeypatch.delenv("FIRELENS_EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("UPB_AI_GATEWAY_KEY", "gateway-secret")

    provider = OpenAICompatibleEmbeddingProvider(base_url="https://gateway.example/v1")

    assert provider.api_key == "gateway-secret"


def test_openai_compatible_provider_sends_batches_and_orders_response_vectors():
    import json as json_module

    requests = []
    vectors = [[float(index)] * 3 for index in range(2)]

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json_module.dumps(
                {
                    "data": [
                        {"index": 1, "embedding": vectors[1]},
                        {"index": 0, "embedding": vectors[0]},
                    ]
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return FakeResponse()

    provider = OpenAICompatibleEmbeddingProvider(
        api_key="test-secret",
        base_url="https://embeddings.example/v1/",
        urlopen_func=fake_urlopen,
    )
    config = EmbeddingConfig(
        provider="openai-compatible",
        model_name="text-embedding-3-small",
        dimension=3,
        batch_size=2,
    )

    result = provider.embed_batch(["first text", "second text"], config)

    assert result == vectors
    request, timeout = requests[0]
    assert request.full_url == "https://embeddings.example/v1/embeddings"
    assert request.get_header("Authorization") == "Bearer test-secret"
    assert timeout == 60
    assert json_module.loads(request.data) == {
        "model": "text-embedding-3-small",
        "input": ["first text", "second text"],
        "dimensions": 3,
    }


def test_real_provider_compatible_with_pipeline_records_and_metadata(tmp_path):
    input_dir = tmp_path / "chunks"
    output_dir = tmp_path / "embeddings"
    source = input_dir / "flex" / "real.jsonl"
    source.parent.mkdir(parents=True)
    chunks = [_chunk(f"semantic chunk {index}", chunk_id=f"real-{index}") for index in range(3)]
    source.write_text("".join(json.dumps(chunk) + "\n" for chunk in chunks), encoding="utf-8")

    class DeterministicSemanticProvider:
        provider_name = "openai-compatible"

        def __init__(self):
            self.batches = []

        def embed_batch(self, texts, config):
            self.batches.append(list(texts))
            return [[float(index + 1)] * config.dimension for index in range(len(texts))]

    provider = DeterministicSemanticProvider()
    config = EmbeddingConfig(
        input_dir,
        output_dir,
        provider="openai-compatible",
        model_name="text-embedding-3-small",
        dimension=5,
        batch_size=2,
    )

    result = run_embedding(config, provider=provider)

    assert result.input_chunks == result.successfully_embedded == 3
    assert result.failed_chunks == 0
    assert [len(batch) for batch in provider.batches] == [2, 1]
    assert all(record.embedding_provider == "openai-compatible" for record in result.records)
    assert all(record.embedding_model == "text-embedding-3-small" for record in result.records)
    assert all(record.embedding_dimension == 5 and len(record.embedding) == 5 for record in result.records)
    assert all(record.chunk_record["source_id"] == "nasa-source-001" for record in result.records)
    assert result.output_paths[0].replace("\\", "/").endswith(
        "/openai-compatible/text-embedding-3-small/flex/real.jsonl"
    )


def test_openai_compatible_provider_surfaces_api_errors_without_secrets():
    from urllib.error import HTTPError

    def failing_urlopen(request, timeout):
        raise HTTPError(request.full_url, 503, "Unavailable", {}, None)

    provider = OpenAICompatibleEmbeddingProvider(api_key="do-not-log", urlopen_func=failing_urlopen)

    with pytest.raises(RuntimeError, match="HTTP 503") as error:
        provider.embed_batch(["semantic text"], EmbeddingConfig(provider="openai-compatible"))
    assert "do-not-log" not in str(error.value)


def test_real_flex_chunk_files_embed_with_all_provenance_retained(tmp_path):
    repository_root = Path(__file__).resolve().parents[1]
    chunk_dir = repository_root / "data" / "chunks" / "flex"
    if not chunk_dir.exists():
        return
    inputs = [
        json.loads(line)
        for path in chunk_dir.rglob("*.jsonl")
        # Not splitlines(): chunk text can hold Unicode line separators.
        for line in path.read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]
    result = run_embedding(
        EmbeddingConfig(input_dir=chunk_dir, output_dir=tmp_path / "embeddings", dimension=12)
    )

    assert len(inputs) == 18
    assert result.input_chunks == 18
    assert result.successfully_embedded == 18
    assert result.failed_chunks == 0
    required = (
        "chunk_id", "source_id", "content_sha256", "experiment", "filename",
        "source_path", "document_type", "evidence_status", "source_locator",
        "page_number", "text", "chunk_index", "chunk_count", "estimated_token_count",
    )
    assert all(all(field in record.to_dict() for field in required) for record in result.records)


def test_openai_compatible_provider_waits_and_retries_on_rate_limit() -> None:
    from urllib.error import HTTPError

    calls: list[int] = []
    waits: list[float] = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self) -> bytes:
            return json.dumps({"data": [{"index": 0, "embedding": [0.0, 1.0]}]}).encode("utf-8")

    def rate_limited_twice(request, timeout):
        calls.append(1)
        if len(calls) <= 2:
            raise HTTPError(request.full_url, 429, "Too Many Requests", {"Retry-After": "3"}, None)
        return Response()

    provider = OpenAICompatibleEmbeddingProvider(
        api_key="do-not-log", urlopen_func=rate_limited_twice, sleep_func=waits.append
    )
    config = EmbeddingConfig(provider="openai-compatible", model_name="m", dimension=2)

    assert provider.embed_batch(["text"], config) == [[0.0, 1.0]]
    assert len(calls) == 3
    assert waits == [3.0, 3.0]


def test_openai_compatible_provider_gives_up_after_repeated_rate_limits() -> None:
    from urllib.error import HTTPError

    waits: list[float] = []

    def always_limited(request, timeout):
        raise HTTPError(request.full_url, 429, "Too Many Requests", {}, None)

    provider = OpenAICompatibleEmbeddingProvider(
        api_key="do-not-log",
        urlopen_func=always_limited,
        sleep_func=waits.append,
        rate_limit_retries=3,
    )
    config = EmbeddingConfig(provider="openai-compatible", model_name="m", dimension=2)

    with pytest.raises(RuntimeError, match="HTTP 429"):
        provider.embed_batch(["text"], config)
    assert waits == [5.0, 10.0, 20.0]
