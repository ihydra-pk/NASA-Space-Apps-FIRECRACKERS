from __future__ import annotations

from src.experimental_data.matching import MatchResult
from src.llm.config import LLMConfig
from src.llm.providers import MockLLMProvider
from src.llm.service import LLMService
from src.vectorstore.models import RetrievalResult, SearchResult

from src.explore.models import ExploreRequest
from src.explore.router import QuestionRouter, validate_route
from src.explore.service import ExploreService

CONFIG = LLMConfig(provider="mock", model_name="fixture", base_url="http://localhost/v1")
QUESTION = "Which FLEX tests are planned for methanol at 21% oxygen and 1 atm?"


def test_stated_conditions_and_units_are_kept():
    decision = validate_route(
        {
            "route": "test_data",
            "conditions": {"fuel": "methanol", "oxygen_concentration": 21, "pressure": 1},
            "condition_units": {"oxygen_concentration": "%", "pressure": "atm"},
            "reason": "Asks which planned tests match stated conditions.",
        },
        QUESTION,
    )
    assert decision.route == "test_data"
    assert decision.conditions == {"fuel": "methanol", "oxygen_concentration": 21, "pressure": 1}
    assert decision.condition_units == {"oxygen_concentration": "%", "pressure": "atm"}
    assert decision.ignored == ()
    assert decision.uses_test_data


def test_unverifiable_conditions_are_dropped_and_reported():
    decision = validate_route(
        {
            "route": "both",
            "conditions": {
                "fuel": "methanol",
                "pressure": 3,  # not written in the question
                "oxygen_concentration": "about 21",  # not a number
                "flow_velocity": 21,  # written, but in an unsupported unit
                "temperature": 300,  # not a field in the test data
            },
            "condition_units": {"flow_velocity": "mph"},
        },
        QUESTION,
    )
    assert decision.conditions == {"fuel": "methanol"}
    assert len(decision.ignored) == 4
    assert any("pressure" in note and "not a number written" in note for note in decision.ignored)
    assert any("unsupported unit" in note for note in decision.ignored)


def test_numbers_quoted_as_text_are_accepted():
    decision = validate_route(
        {"route": "test_data", "conditions": {"pressure": "1", "oxygen_concentration": "21"}},
        QUESTION,
    )
    assert decision.conditions == {"pressure": 1.0, "oxygen_concentration": 21.0}
    assert decision.ignored == ()


def test_no_conditions_means_documents_route():
    decision = validate_route({"route": "test_data", "conditions": {}}, "Why do cool flames form?")
    assert decision.route == "documents"
    assert not decision.uses_test_data


def test_malformed_router_output_falls_back_to_documents():
    assert validate_route({"route": "sql"}, QUESTION).fallback
    assert validate_route({"route": "both", "conditions": ["fuel"]}, QUESTION).fallback

    not_json = QuestionRouter(MockLLMProvider("not json"), CONFIG).route(QUESTION)
    assert not_json.fallback and not_json.route == "documents"

    class Unreachable:
        provider_name = "mock"

        def generate(self, **kwargs):
            raise RuntimeError("Could not reach LLM API: timed out")

    unreachable = QuestionRouter(Unreachable(), CONFIG).route(QUESTION)
    assert unreachable.fallback and "timed out" in unreachable.reason


def _retriever(query, *, top_k):
    return SearchResult(
        query=query,
        provider="mock",
        model_name="fixture",
        dimension=2,
        similarity_metric="cosine",
        matches=(
            RetrievalResult(
                rank=1,
                chunk_id="chunk-a",
                similarity_score=0.61,
                text="The FLEX test plan specifies planned test conditions.",
                experiment="FLEX",
                filename="FLEX_Test_Plan.pdf",
                source_locator="page:6",
                page_number=6,
                document_type="test_plan",
                evidence_status="planned",
                source_id="nasa-doc-hash",
                content_sha256="doc-hash",
                source_path="data/documents/flex/FLEX_Test_Plan.pdf",
                metadata={"file_type": "pdf"},
            ),
        ),
    )


class RecordingMatcher:
    def __init__(self):
        self.calls: list[dict[str, object]] = []

    def match_conditions(self, conditions):
        self.calls.append(conditions)
        return MatchResult(
            requested_conditions=conditions, tolerances={}, match_type="no_match", matches=()
        )


def _service(router_reply, matcher):
    llm = LLMService(config=CONFIG, provider=MockLLMProvider())
    router = QuestionRouter(MockLLMProvider(router_reply), CONFIG)
    return ExploreService(_retriever, matcher, llm, router=router)


def test_explore_matches_conditions_the_router_read_from_the_question():
    matcher = RecordingMatcher()
    response = _service(
        {
            "route": "test_data",
            "conditions": {"fuel": "methanol", "oxygen_concentration": 21, "pressure": 1},
            "condition_units": {"oxygen_concentration": "%", "pressure": "atm"},
            "reason": "Stated conditions.",
        },
        matcher,
    ).explore(ExploreRequest(QUESTION))

    # Percent oxygen is converted to the matcher's mole fraction as usual.
    assert matcher.calls == [{"fuel": "methanol", "oxygen_concentration": 0.21, "pressure": 1.0}]
    assert response.requested_conditions == {"fuel": "methanol", "oxygen_concentration": 21, "pressure": 1}
    assert response.routing["route"] == "test_data"
    assert response.to_dict()["routing"]["conditions"]["fuel"] == "methanol"


def test_explore_skips_matching_on_the_documents_route():
    matcher = RecordingMatcher()
    response = _service({"route": "documents", "conditions": {}, "reason": "Conceptual."}, matcher).explore(
        ExploreRequest("Why do cool flames form?")
    )
    assert matcher.calls == []
    assert response.routing["route"] == "documents"
    assert response.condition_match is None


def test_conditions_supplied_by_the_caller_bypass_the_router():
    matcher = RecordingMatcher()
    response = _service({"route": "test_data", "conditions": {"fuel": "heptane"}}, matcher).explore(
        ExploreRequest(QUESTION, {"fuel": "Methanol"})
    )
    assert matcher.calls == [{"fuel": "Methanol"}]
    assert response.routing is None


def test_router_failure_is_reported_as_a_warning_not_an_error():
    matcher = RecordingMatcher()
    response = _service("not json", matcher).explore(ExploreRequest(QUESTION))
    assert matcher.calls == []
    assert response.routing["fallback"] is True
    assert any("fell back to document search" in warning for warning in response.warnings)


def test_router_prompt_mentions_json_in_the_user_message():
    # The gateway rejects JSON mode unless the user message itself says "json".
    provider = MockLLMProvider({"route": "documents", "conditions": {}})
    QuestionRouter(provider, CONFIG).route(QUESTION)
    _, user_prompt = provider.calls[0]
    assert QUESTION in user_prompt
    assert "json" in user_prompt.casefold()
