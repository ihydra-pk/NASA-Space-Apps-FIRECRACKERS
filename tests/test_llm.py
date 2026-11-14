from __future__ import annotations

import json

import pytest

from src.llm import (
    LLMConfig,
    LLMResponseError,
    LLMService,
    MockLLMProvider,
    OpenAICompatibleLLMProvider,
)
from src.rag import RAGContext, build_context
from src.vectorstore.models import RetrievalResult, SearchResult


def _result(
    *,
    status="planned",
    chunk_id="chunk-1",
    text="The test plan specifies 21% oxygen as a planned condition.",
):
    record = {
        "chunk_id": chunk_id,
        "source_id": "source-1",
        "content_sha256": "sha256-1",
        "experiment": "FLEX",
        "filename": "FLEX_Test_Plan.pdf",
        "file_type": "pdf",
        "source_path": "flex/FLEX_Test_Plan.pdf",
        "document_type": "test_plan",
        "evidence_status": status,
        "source_locator": "page:2",
        "page_number": 2,
        "text": text,
        "embedding": [0.0],
    }
    return RetrievalResult.from_record(1, 0.812345, record)


def _context(*results):
    search_result = SearchResult(
        query="What does the FLEX source say?",
        provider="openai-compatible",
        model_name="qwen-embed",
        dimension=4,
        similarity_metric="cosine",
        matches=tuple(results),
    )
    return build_context("What does the FLEX source say?", search_result)


def _citation(item):
    return {
        "chunk_id": item.chunk_id,
        "source_id": item.source_id,
        "source_locator": item.source_locator,
        "page_number": item.page_number,
        "evidence_status": item.evidence_status,
    }


def _mock_response(context, *, status="grounded", answer="The supplied source describes a planned condition.", citations=None):
    return {
        "answer": answer,
        "status": status,
        "citations": citations if citations is not None else [_citation(context.evidence_items[0])],
    }


def test_evidence_policy_prompt_has_explicit_scientific_rules():
    context = _context(_result())
    provider = MockLLMProvider(_mock_response(context))
    service = LLMService(LLMConfig(provider="mock", model_name="fixture"), provider)

    service.answer(context.query, context)
    system, user = provider.calls[0]

    for required in (
        "ONLY the supplied evidence",
        "planned ≠ observed",
        "metadata ≠ experimental result",
        "interpreted ≠ direct observation",
        "Similarity scores are retrieval/ranking signals only",
        "Never invent numerical values",
    ):
        assert required in system
    assert context.context_text in user
    assert context.evidence_items[0].text in user


def test_grounded_end_to_end_response_has_validated_citation_and_summary():
    context = _context(_result())
    provider = MockLLMProvider(_mock_response(context))
    service = LLMService(LLMConfig(provider="mock", model_name="fixture"), provider)

    response = service.answer(context.query, context)

    assert response.status == "grounded"
    assert response.citations[0].chunk_id == context.evidence_items[0].chunk_id
    assert response.citations[0].evidence_status == "planned"
    assert response.evidence_summary[0].source_path == "flex/FLEX_Test_Plan.pdf"
    assert response.model_name == "fixture"
    assert "confidence" not in response.to_dict()


def test_empty_context_returns_insufficient_without_calling_llm():
    context = _context()
    provider = MockLLMProvider()
    service = LLMService(LLMConfig(provider="mock"), provider)

    response = service.answer(context.query, context)

    assert response.status == "insufficient_evidence"
    assert "indexed evidence is insufficient" in response.answer
    assert not provider.calls
    assert response.citations == ()


def test_model_insufficient_evidence_response_is_supported():
    context = _context(_result())
    provider = MockLLMProvider(
        _mock_response(
            context,
            status="insufficient_evidence",
            answer="The indexed evidence is insufficient to establish an observed outcome.",
            citations=[],
        )
    )

    response = LLMService(LLMConfig(provider="mock"), provider).answer(context.query, context)

    assert response.status == "insufficient_evidence"
    assert response.citations == ()
    assert response.evidence_summary[0].evidence_status == "planned"


def test_invalid_or_invented_citation_ids_are_rejected():
    context = _context(_result())
    invalid = _citation(context.evidence_items[0])
    invalid["chunk_id"] = "not-retrieved"
    service = LLMService(
        LLMConfig(provider="mock"),
        MockLLMProvider(_mock_response(context, citations=[invalid])),
    )

    with pytest.raises(LLMResponseError, match="not present in supplied RAGContext"):
        service.answer(context.query, context)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_id", "invented-source"),
        ("source_locator", "page:999"),
        ("page_number", 999),
        ("evidence_status", "observed"),
    ],
)
def test_citation_provenance_must_match_supplied_context(field, value):
    context = _context(_result(status="planned"))
    invalid = _citation(context.evidence_items[0])
    invalid[field] = value
    service = LLMService(
        LLMConfig(provider="mock"),
        MockLLMProvider(_mock_response(context, citations=[invalid])),
    )

    with pytest.raises(LLMResponseError, match=f"invalid {field}"):
        service.answer(context.query, context)


def test_grounded_response_requires_citations_and_known_status():
    context = _context(_result())
    no_citations = LLMService(
        LLMConfig(provider="mock"),
        MockLLMProvider(_mock_response(context, citations=[])),
    )
    with pytest.raises(LLMResponseError, match="must cite"):
        no_citations.answer(context.query, context)

    no_status = LLMService(
        LLMConfig(provider="mock"),
        MockLLMProvider({"answer": "Test", "status": "unknown", "citations": []}),
    )
    with pytest.raises(LLMResponseError, match="status"):
        no_status.answer(context.query, context)


def test_invalid_response_json_and_missing_fields_fail_clearly():
    context = _context(_result())
    for raw in ("not json", json.dumps({"answer": "no schema"})):
        service = LLMService(LLMConfig(provider="mock"), MockLLMProvider(raw))
        with pytest.raises(LLMResponseError):
            service.answer(context.query, context)


@pytest.mark.parametrize("status", ["metadata", "observed", "interpreted", "unknown"])
def test_evidence_status_is_preserved_in_prompt_summary_and_citation(status):
    context = _context(_result(status=status))
    provider = MockLLMProvider(_mock_response(context))

    response = LLMService(LLMConfig(provider="mock"), provider).answer(context.query, context)

    assert response.citations[0].evidence_status == status
    assert response.evidence_summary[0].evidence_status == status
    assert f"Evidence status: {status}" in provider.calls[0][1]


def test_planned_is_never_converted_to_observed_by_service():
    context = _context(_result(status="planned"))
    provider = MockLLMProvider(_mock_response(context, answer="A planned condition is described."))

    response = LLMService(LLMConfig(provider="mock"), provider).answer(context.query, context)

    assert response.citations[0].evidence_status == "planned"
    assert response.evidence_summary[0].evidence_status == "planned"
    assert "planned ≠ observed" in provider.calls[0][0]


def test_planned_evidence_cannot_cite_a_claim_that_nasa_observed_results():
    context = _context(_result(status="planned"))
    provider = MockLLMProvider(
        _mock_response(
            context,
            answer="NASA observed that 21% oxygen caused the flame to extinguish.",
        )
    )

    with pytest.raises(LLMResponseError, match="claims an observation"):
        LLMService(LLMConfig(provider="mock"), provider).answer(context.query, context)


def test_answer_does_not_report_similarity_as_confidence():
    context = _context(_result())
    provider = MockLLMProvider(
        _mock_response(context, answer="The evidence has high confidence because its score is high.")
    )

    with pytest.raises(LLMResponseError, match="confidence"):
        LLMService(LLMConfig(provider="mock"), provider).answer(context.query, context)


def test_observation_claim_cannot_be_supported_by_mixed_planned_and_observed_citations():
    context = _context(_result(status="planned"), _result(status="observed", chunk_id="observed"))
    provider = MockLLMProvider(
        _mock_response(
            context,
            answer="NASA observed the result.",
            citations=[_citation(context.evidence_items[0]), _citation(context.evidence_items[1])],
        )
    )

    with pytest.raises(LLMResponseError, match="not exclusively"):
        LLMService(LLMConfig(provider="mock"), provider).answer(context.query, context)


def test_deterministic_mock_provider_produces_repeatable_structured_fixture():
    context = _context(_result())
    provider = MockLLMProvider()
    service = LLMService(LLMConfig(provider="mock"), provider)

    first = service.answer(context.query, context).to_dict()
    second = service.answer(context.query, context).to_dict()

    assert first == second


def test_provider_configuration_reads_environment_without_exposing_secrets(monkeypatch):
    monkeypatch.setenv("FIRELENS_LLM_MODEL", "configured-model")
    monkeypatch.setenv("FIRELENS_LLM_BASE_URL", "https://llm.example/v1")
    monkeypatch.setenv("FIRELENS_LLM_TIMEOUT", "25")
    monkeypatch.setenv("FIRELENS_LLM_TEMPERATURE", "0.2")
    monkeypatch.setenv("FIRELENS_LLM_MAX_OUTPUT_TOKENS", "512")
    monkeypatch.setenv("UPB_AI_GATEWAY_KEY", "test-secret")

    config = LLMConfig.from_env()
    provider = OpenAICompatibleLLMProvider()

    assert config.model_name == "configured-model"
    assert config.base_url == "https://llm.example/v1"
    assert config.timeout_seconds == 25
    assert config.temperature == 0.2
    assert config.max_output_tokens == 512
    assert provider.api_key == "test-secret"
    assert "test-secret" not in repr(config)


def test_openai_compatible_provider_fails_without_credentials(monkeypatch):
    monkeypatch.delenv("FIRELENS_LLM_API_KEY", raising=False)
    monkeypatch.delenv("UPB_AI_GATEWAY_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = OpenAICompatibleLLMProvider(base_url="https://gateway.example/v1")

    with pytest.raises(RuntimeError, match="UPB_AI_GATEWAY_KEY"):
        provider.generate(system_prompt="system", user_prompt="user", config=LLMConfig())


def test_openai_compatible_provider_requests_chat_json_with_configured_options():
    requests = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": '{"ok":true}'}}]}).encode()

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return FakeResponse()

    provider = OpenAICompatibleLLMProvider(
        api_key="secret",
        base_url="https://gateway.example/v1",
        urlopen_func=fake_urlopen,
    )
    config = LLMConfig(model_name="model-x", temperature=0.1, max_output_tokens=99, timeout_seconds=8)

    response = provider.generate(system_prompt="system", user_prompt="user", config=config)

    assert response == '{"ok":true}'
    request, timeout = requests[0]
    payload = json.loads(request.data)
    assert request.full_url == "https://gateway.example/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer secret"
    assert timeout == 8
    assert payload["model"] == "model-x"
    assert payload["temperature"] == 0.1
    assert payload["max_tokens"] == 99
    assert payload["response_format"] == {"type": "json_object"}


def test_live_llm_smoke_is_opt_in_and_requires_gateway_configuration():
    import os

    if os.environ.get("FIRELENS_RUN_LLM_SMOKE") != "1":
        pytest.skip("set FIRELENS_RUN_LLM_SMOKE=1 to opt into a live LLM API call")
    if not (os.environ.get("UPB_AI_GATEWAY_KEY") or os.environ.get("FIRELENS_LLM_API_KEY")):
        pytest.skip("UPB_AI_GATEWAY_KEY or FIRELENS_LLM_API_KEY is required")

    context = _context(_result())
    response = LLMService(LLMConfig.from_env()).answer(context.query, context)
    assert response.status in {"grounded", "insufficient_evidence"}
    if response.status == "grounded":
        assert response.citations


def test_truncated_llm_output_is_reported_as_truncation():
    import pytest

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            body = {"choices": [{"finish_reason": "length", "message": {"content": '{"answer": "cut'}}]}
            return json.dumps(body).encode()

    provider = OpenAICompatibleLLMProvider(
        api_key="secret", base_url="https://gateway.example/v1", urlopen_func=lambda r, timeout: FakeResponse()
    )
    with pytest.raises(RuntimeError, match="cut off at the output token limit"):
        provider.generate(system_prompt="s", user_prompt="u", config=LLMConfig(model_name="m", max_output_tokens=50))


def test_default_output_token_limit_leaves_room_for_long_answers(monkeypatch):
    monkeypatch.delenv("FIRELENS_LLM_MAX_OUTPUT_TOKENS", raising=False)
    assert LLMConfig.from_env().max_output_tokens >= 3000
