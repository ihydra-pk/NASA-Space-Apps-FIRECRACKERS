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


def _mock_response(
    context,
    *,
    status="grounded",
    answer="The supplied source describes a planned condition.",
    chunk_ids=None,
    statement="The source lists 21% oxygen as a planned condition.",
    support="direct",
    **fields,
):
    ids = [context.evidence_items[0].chunk_id] if chunk_ids is None else chunk_ids
    response = {
        "answer": answer,
        "status": status,
        "findings": [{"statement": statement, "support": support, "chunk_ids": ids}] if ids else [],
        "scientific_context": "",
        "limitations": "",
    }
    response.update(fields)
    return response


def _answer(context, provider):
    return LLMService(LLMConfig(provider="mock", model_name="fixture"), provider).answer(context.query, context)


class SequenceProvider:
    """Returns the given replies in order, one per call."""

    provider_name = "mock"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def generate(self, *, system_prompt, user_prompt, config):
        self.calls.append((system_prompt, user_prompt))
        return json.dumps(self.replies.pop(0))


def test_evidence_policy_prompt_has_explicit_scientific_rules():
    context = _context(_result())
    provider = MockLLMProvider(_mock_response(context))

    _answer(context, provider)
    system, user = provider.calls[0]

    for required in (
        "planned ≠ observed",
        "metadata ≠ experimental result",
        "interpreted ≠ direct observation",
        "unknown = not labelled yet",
        "A bibliography or reference-list entry shows that a paper exists",
        "Science Documents: SRD/SCR science requirements written before flight",
        "scientific_context",
        "Similarity scores are retrieval/ranking signals only",
        "Never invent numerical values",
    ):
        assert required in system
    assert "ONLY the supplied evidence" not in system
    assert context.context_text in user
    # Chunk text is sent once, in the context block; the structured items
    # carry identifiers and provenance only.
    assert user.count(context.evidence_items[0].text) == 1


def test_grounded_end_to_end_response_has_validated_citation_and_summary():
    context = _context(_result())
    response = _answer(context, MockLLMProvider(_mock_response(context)))

    assert response.status == "grounded"
    assert response.citations[0].chunk_id == context.evidence_items[0].chunk_id
    assert response.citations[0].evidence_status == "planned"
    assert response.findings[0].support == "direct"
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
            chunk_ids=[],
            scientific_context="In general, oxygen concentration controls the flame temperature.",
        )
    )

    response = _answer(context, provider)

    assert response.status == "insufficient_evidence"
    assert response.citations == ()
    assert response.scientific_context.startswith("In general")
    assert response.evidence_summary[0].evidence_status == "planned"


def test_invented_chunk_ids_are_rejected():
    context = _context(_result())
    provider = MockLLMProvider(_mock_response(context, chunk_ids=["not-retrieved"]))

    with pytest.raises(LLMResponseError, match="not present in supplied RAGContext"):
        _answer(context, provider)


def test_citation_provenance_is_copied_from_context_not_from_model_output():
    context = _context(_result(status="planned"))
    item = context.evidence_items[0]
    reply = _mock_response(context)
    # Provenance the model writes anywhere is ignored, including a citations
    # array in the old format.
    reply["findings"][0].update(source_id="forged", page_number=999, evidence_status="observed")
    reply["citations"] = [{"chunk_id": item.chunk_id, "source_id": "forged", "evidence_status": "observed"}]

    response = _answer(context, MockLLMProvider(reply))

    citation = response.citations[0]
    assert (citation.source_id, citation.source_locator, citation.page_number, citation.evidence_status) == (
        item.source_id,
        item.source_locator,
        item.page_number,
        "planned",
    )


def test_grounded_response_requires_findings_and_known_status():
    context = _context(_result())
    with pytest.raises(LLMResponseError, match="must cite"):
        _answer(context, MockLLMProvider(_mock_response(context, chunk_ids=[])))

    with pytest.raises(LLMResponseError, match="status"):
        _answer(context, MockLLMProvider({"answer": "Test", "status": "unknown", "findings": []}))


def test_invalid_response_json_and_missing_fields_fail_clearly():
    context = _context(_result())
    for raw in ("not json", json.dumps({"answer": "no schema"})):
        with pytest.raises(LLMResponseError):
            _answer(context, MockLLMProvider(raw))


@pytest.mark.parametrize("status", ["metadata", "observed", "interpreted", "unknown"])
def test_evidence_status_is_preserved_in_prompt_summary_and_citation(status):
    context = _context(_result(status=status))
    provider = MockLLMProvider(_mock_response(context))

    response = _answer(context, provider)

    assert response.citations[0].evidence_status == status
    assert response.evidence_summary[0].evidence_status == status
    assert f"Evidence status: {status}" in provider.calls[0][1]


def test_planned_is_never_converted_to_observed_by_service():
    context = _context(_result(status="planned"))
    provider = MockLLMProvider(_mock_response(context, answer="A planned condition is described."))

    response = _answer(context, provider)

    assert response.citations[0].evidence_status == "planned"
    assert response.evidence_summary[0].evidence_status == "planned"
    assert "planned ≠ observed" in provider.calls[0][0]


def test_planned_evidence_cannot_cite_a_claim_that_nasa_observed_results():
    context = _context(_result(status="planned"))
    provider = MockLLMProvider(
        _mock_response(context, answer="NASA observed that 21% oxygen caused the flame to extinguish.")
    )

    with pytest.raises(LLMResponseError, match="claims an observation"):
        _answer(context, provider)


def test_a_finding_cannot_present_planned_evidence_as_an_observation():
    context = _context(_result(status="planned"))
    provider = MockLLMProvider(
        _mock_response(context, statement="The tests measured extinction at 21% oxygen.")
    )

    with pytest.raises(LLMResponseError, match="finding 1 claims an observation"):
        _answer(context, provider)


def test_observation_claim_citing_unlabelled_evidence_is_accepted():
    # Nearly every indexed document is unlabelled; "unknown" is not "planned".
    context = _context(_result(status="unknown", text="Cool flames were observed after hot-flame extinction."))
    chunk_id = context.evidence_items[0].chunk_id
    response = _answer(
        context,
        MockLLMProvider(
            _mock_response(
                context,
                answer=f"NASA found that cool flames follow radiative extinction of the hot flame [{chunk_id}].",
                statement="The experiments showed cool flames after hot-flame extinction.",
            )
        ),
    )

    assert response.status == "grounded"
    assert response.citations[0].evidence_status == "unknown"


def test_answer_does_not_report_similarity_as_confidence():
    context = _context(_result())
    provider = MockLLMProvider(
        _mock_response(context, answer="The evidence has high confidence because its score is high.")
    )

    with pytest.raises(LLMResponseError, match="confidence"):
        _answer(context, provider)


def test_observation_claim_cannot_be_supported_by_mixed_planned_and_observed_citations():
    context = _context(_result(status="planned"), _result(status="observed", chunk_id="observed"))
    provider = MockLLMProvider(
        _mock_response(context, answer="NASA observed the result.", chunk_ids=["chunk-1", "observed"])
    )

    with pytest.raises(LLMResponseError, match="claims an observation"):
        _answer(context, provider)


def test_partial_answer_keeps_inference_and_limitations():
    context = _context(_result(status="unknown"))
    response = _answer(
        context,
        MockLLMProvider(
            _mock_response(
                context,
                status="partial",
                support="inference",
                scientific_context="Lower oxygen generally slows chemical reactions.",
                limitations="The evidence gives no extinction diameters.",
            )
        ),
    )

    assert response.status == "partial"
    assert response.findings[0].support == "inference"
    payload = response.to_dict()
    assert payload["limitations"] == "The evidence gives no extinction diameters."
    assert payload["scientific_context"] == "Lower oxygen generally slows chemical reactions."
    assert payload["findings"][0]["chunk_ids"] == ["chunk-1"]


def test_partial_answer_must_state_its_limitations():
    context = _context(_result())
    with pytest.raises(LLMResponseError, match="limitations"):
        _answer(context, MockLLMProvider(_mock_response(context, status="partial")))


def test_limitations_written_as_a_list_are_joined():
    context = _context(_result())
    response = _answer(
        context,
        MockLLMProvider(_mock_response(context, status="partial", limitations=["No outcomes.", "No temperatures."])),
    )
    assert response.limitations == "No outcomes.\nNo temperatures."


def test_scientific_context_cannot_cite_evidence():
    context = _context(_result())
    provider = MockLLMProvider(
        _mock_response(context, scientific_context="Kinetics explain this [chunk-1].")
    )

    with pytest.raises(LLMResponseError, match="scientific_context"):
        _answer(context, provider)


def test_finding_without_chunk_ids_is_rejected():
    context = _context(_result())
    reply = _mock_response(context)
    reply["findings"].append({"statement": "Flames are hot.", "support": "direct", "chunk_ids": []})

    with pytest.raises(LLMResponseError, match="finding 2 must cite at least one chunk_id"):
        _answer(context, MockLLMProvider(reply))


def test_inline_markers_become_citations_and_unknown_markers_are_rejected():
    context = _context(_result(), _result(chunk_id="chunk-2"))
    response = _answer(
        context,
        MockLLMProvider(_mock_response(context, answer="The plan lists 21% oxygen [chunk-2].", chunk_ids=["chunk-1"])),
    )
    assert [citation.chunk_id for citation in response.citations] == ["chunk-2", "chunk-1"]

    invented = _mock_response(context, answer="The plan lists 21% oxygen [chunk-deadbeef00].")
    with pytest.raises(LLMResponseError, match="'chunk-deadbeef00' not present"):
        _answer(context, MockLLMProvider(invented))
    in_limitations = _mock_response(context, limitations="Only [chunk-deadbeef00] covers methanol.")
    with pytest.raises(LLMResponseError, match="limitations references chunk_id"):
        _answer(context, MockLLMProvider(in_limitations))


def test_limitations_may_cite_supplied_chunks_and_they_become_citations():
    context = _context(_result(), _result(chunk_id="chunk-2"))
    response = _answer(
        context,
        MockLLMProvider(_mock_response(context, status="partial", limitations="[chunk-2, chunk-1] cover methanol only.")),
    )
    assert [citation.chunk_id for citation in response.citations] == ["chunk-1", "chunk-2"]


def test_rejected_reply_is_retried_once_with_the_reason():
    context = _context(_result())
    bad = _mock_response(context, chunk_ids=["not-retrieved"])
    provider = SequenceProvider(bad, _mock_response(context))

    response = _answer(context, provider)

    assert response.status == "grounded"
    assert len(provider.calls) == 2
    retry_prompt = provider.calls[1][1]
    assert "YOUR PREVIOUS REPLY WAS REJECTED" in retry_prompt
    assert "not present in supplied RAGContext" in retry_prompt


def test_second_rejection_is_raised():
    context = _context(_result())
    provider = MockLLMProvider(_mock_response(context, chunk_ids=["not-retrieved"]))

    with pytest.raises(LLMResponseError):
        _answer(context, provider)
    assert len(provider.calls) == 2


def test_default_mock_fixture_cites_the_first_supplied_item():
    context = _context(_result(), _result(chunk_id="chunk-2"))
    response = _answer(context, MockLLMProvider())
    assert response.status == "grounded"
    assert [citation.chunk_id for citation in response.citations] == ["chunk-1"]


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
    assert response.status in {"grounded", "partial", "insufficient_evidence"}
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
