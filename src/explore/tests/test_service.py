from src.experimental_data.matching import MatchResult, ExperimentMatch, ParameterMatch
from src.experimental_data.models import ExperimentRecord
from src.llm.config import LLMConfig
from src.llm.providers import MockLLMProvider
from src.llm.service import LLMService
from src.vectorstore.models import RetrievalResult, SearchResult

from src.explore.models import ExploreRequest
from src.explore.service import ExploreService, condition_availability, normalize_conditions


def planned_record():
    return ExperimentRecord(
        record_id="plan-row-1",
        kind="test_condition",
        experiment="FLEX",
        experiment_id="23",
        evidence_status="planned",
        source_id="nasa-source-hash",
        source_file="data/documents/flex/FLEX_Test_Plan.pdf",
        source_locator="page:6:table:6:line:24",
        content_sha256="source-hash",
        fuel="methanol",
        oxygen_concentration=0.21,
        pressure=1.0,
        flow_velocity=0.0,
        units={
            "oxygen_concentration": "mole_fraction",
            "pressure": "atm",
            "flow_velocity": "cm/s",
        },
        original_values={"fuel": "CH3OH", "oxygen_concentration": "0.21"},
    )


def test_condition_normalization_converts_declared_supported_units_only():
    result = normalize_conditions(
        {"oxygen_concentration": 21, "pressure": 101.325, "flow_velocity": 1, "temperature": 300},
        {"pressure": "kPa", "flow_velocity": "m/s"},
    )
    assert result.values["oxygen_concentration"] == 0.21
    assert result.values["pressure"] == 1.0
    assert result.values["flow_velocity"] == 100.0
    assert result.values["temperature"] == 300
    assert result.units == {
        "oxygen_concentration": "mole_fraction",
        "pressure": "atm",
        "flow_velocity": "cm/s",
        "temperature": "K",
    }


def test_unified_service_preserves_independent_scores_and_planned_status():
    record = planned_record()
    parameters = (
        ParameterMatch("fuel", "exact_match", "methanol", "methanol", None),
        ParameterMatch("oxygen_concentration", "close_match", 0.208, 0.21, "mole_fraction", 0.002, 0.005),
        ParameterMatch("gravity", "parameter_unavailable", "microgravity", None, None),
    )
    condition_result = MatchResult(
        requested_conditions={"fuel": "methanol", "oxygen_concentration": 0.208},
        tolerances={"oxygen_concentration": 0.005},
        match_type="close_match",
        matches=(ExperimentMatch(record, "close_match", parameters),),
    )

    class ExperimentalService:
        def match_conditions(self, conditions):
            assert abs(conditions["oxygen_concentration"] - 0.208) < 1e-12
            return condition_result

    def retriever(query, *, top_k):
        assert query == "What can NASA evidence tell us?"
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

    llm_provider = MockLLMProvider()
    llm = LLMService(
        config=LLMConfig(provider="mock", model_name="fixture", base_url="http://localhost/v1"),
        provider=llm_provider,
    )
    response = ExploreService(retriever, ExperimentalService(), llm).explore(
        ExploreRequest(
            "What can NASA evidence tell us?",
            {
                "fuel": "Methanol",
                "oxygen_concentration": 20.8,
                "pressure": 1,
                "temperature": 300,
                "gravity": "microgravity",
                "flow_velocity": 0,
            },
        )
    )

    document = response.document_evidence[0]
    experimental = response.experimental_evidence[0]
    assert document.evidence_type == "document"
    assert document.similarity_score == 0.61
    assert experimental.evidence_type == "experimental_condition"
    assert experimental.evidence_status == "planned"
    assert experimental.match_type == "close_match"
    oxygen = next(item for item in experimental.parameter_matches if item["parameter"] == "oxygen_concentration")
    assert oxygen["requested_value"] == 0.208
    assert oxygen["experimental_value"] == 0.21
    assert oxygen["difference"] == 0.002
    assert oxygen["tolerance"] == 0.005
    assert response.normalized_conditions["temperature"] == 300
    assert response.condition_availability["fuel"]["status"] == "AVAILABLE"
    assert response.condition_availability["temperature"]["status"] == "UNAVAILABLE"
    assert response.condition_availability["gravity"]["status"] == "UNAVAILABLE"
    assert response.evidence_mode == "combined"
    assert response.status == response.answer.status
    assert any("'planned' are proposed test conditions" in warning for warning in response.warnings)
    assert "0.61" not in str(experimental.to_dict())
    assert "planned conditions are not observed results" in llm_provider.calls[0][1]


def test_unknown_parameter_and_missing_experimental_db_remain_explicit():
    result = normalize_conditions({"unknown_measurement": 4, "gravity": "microgravity"})
    assert result.values == {"unknown_measurement": 4, "gravity": "microgravity"}


def test_no_loaded_records_is_not_misreported_as_parameter_unsupported():
    availability = condition_availability(
        {"fuel": "methanol", "temperature": 300},
        MatchResult({}, {}, "no_match", ()),
    )
    assert availability["fuel"]["status"] == "NO_DATA"
    assert availability["temperature"]["status"] == "UNAVAILABLE"


def test_frontend_response_is_bounded_and_machine_readable():
    record = planned_record()
    matches = tuple(
        ExperimentMatch(record, "exact_match", (ParameterMatch("fuel", "exact_match", "methanol", "methanol", None),))
        for _ in range(30)
    )
    result = MatchResult({"fuel": "methanol"}, {}, "exact_match", matches)
    from src.explore.models import _match_result_dict

    payload = _match_result_dict(result, limit=5)
    assert payload["candidate_count"] == 30
    assert payload["status_counts"]["exact_match"] == 30
    assert payload["matches_truncated"] is True
    assert len(payload["matches"]) == 5


def _record(status, test_id):
    from dataclasses import replace
    return replace(planned_record(), record_id=f"{status}-{test_id}", experiment_id=test_id, evidence_status=status)


def _match_result(*pairs):
    matches = tuple(
        ExperimentMatch(_record(status, test_id), match_type, ())
        for status, test_id, match_type in pairs
    )
    return MatchResult({"fuel": "methanol"}, {}, matches[0].match_type, matches)


def test_observed_records_move_ahead_of_planned_when_question_asks_for_observed_results():
    from src.explore.service import prefer_observed

    result = _match_result(("planned", "1", "exact_match"), ("planned", "2", "exact_match"),
                           ("observed", "3", "close_match"))
    reordered = prefer_observed("What were the observed burn times?", result)
    assert [m.record.experiment_id for m in reordered.matches] == ["3", "1", "2"]
    assert [m.match_type for m in reordered.matches] == ["close_match", "exact_match", "exact_match"]
    assert {m.record.evidence_status for m in reordered.matches} == {"observed", "planned"}


def test_ranking_is_unchanged_for_plan_questions_and_neutral_questions():
    from src.explore.service import prefer_observed

    result = _match_result(("planned", "1", "exact_match"), ("observed", "3", "close_match"))
    assert prefer_observed("Which conditions are in the test plan?", result) is result
    assert prefer_observed("Tell me about methanol", result) is result
    assert prefer_observed("planned observed burn times", result) is result


def test_planned_warning_only_when_planned_records_are_returned():
    from src.explore.service import _planned_warning

    observed_only = _match_result(("observed", "3", "exact_match"))
    mixed = _match_result(("observed", "3", "exact_match"), ("planned", "1", "exact_match"))
    assert _planned_warning(observed_only, 25) == ()
    assert _planned_warning(mixed, 25) != ()
    assert _planned_warning(mixed, 1) == ()
