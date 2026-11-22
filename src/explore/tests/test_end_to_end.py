"""Offline, realistic Explore scenarios using the real RAG/matcher/LLM adapters."""

from __future__ import annotations

from dataclasses import replace
import json

from src.experimental_data.loaders import parse_test_plan_pages
from src.experimental_data.matching import match_experiments
from src.llm.config import LLMConfig
from src.llm.providers import MockLLMProvider
from src.llm.service import LLMResponseError, LLMService
from src.vectorstore.models import RetrievalResult, SearchResult

from src.explore.models import ExploreRequest
from src.explore.service import ExploreService


def planned_records():
    # Values are copied from the structure of FLEX Test Plan Table 6, including
    # the methanol test 24 condition: 1.00 atm and X_O2 0.21. No outcomes are
    # present in a test-plan row.
    page = "\n".join(
        [
            "Table 6: Quiescent environment LOI test matrix.",
            "Fuel: C7H16, Diluent: N2, No fiber support, Flow: 0.0 cm/s,",
            "Test No. P (atm) XO2",
            "1 0.50 0.20",
            "Fuel: CH3OH",
            "22 1.00 0.10",
            "23 1.00 0.15",
            "24 1.00 0.21",
            "25 1.00 0.30",
        ]
    )
    records, diagnostics = parse_test_plan_pages(
        [(6, page)],
        source_file="data/documents/flex/FLEX_Test_Plan.pdf",
        content_sha256="fixture-source-sha256",
    )
    assert diagnostics.discovered == 5
    assert len(records) == 5
    assert all(record.evidence_status == "planned" for record in records)
    assert all(record.observed_result is None and record.measurement is None for record in records)
    return records


def document_search(*, has_hit=True):
    def retrieve(query, *, top_k):
        matches = ()
        if has_hit:
            matches = (
                RetrievalResult(
                    rank=1,
                    chunk_id="flex-plan-page-6",
                    similarity_score=0.61,
                    text=(
                        "Table 6 is a proposed quiescent-environment FLEX test matrix. "
                        "Test 24 lists methanol, pressure 1.00 atm, and oxygen mole fraction 0.21."
                    ),
                    experiment="FLEX",
                    filename="FLEX_Test_Plan.pdf",
                    source_locator="page:6",
                    page_number=6,
                    document_type="test_plan",
                    evidence_status="planned",
                    source_id="nasa-document-sha256",
                    content_sha256="document-sha256",
                    source_path="data/documents/flex/FLEX_Test_Plan.pdf",
                    metadata={"file_type": "pdf"},
                ),
            )
        return SearchResult(query, "mock", "fixture", 2, "cosine", matches)

    return retrieve


class FixtureExperimentalData:
    def __init__(self, records):
        self.records = records
        self.calls = 0

    def match_conditions(self, conditions):
        self.calls += 1
        return match_experiments(
            self.records,
            conditions,
            absolute_tolerances={
                "oxygen_concentration": 0.005,
                "pressure": 0.01,
                "flow_velocity": 0.1,
            },
        )


def service(records, *, document_hit=True):
    provider = CitationFixtureProvider()
    llm = LLMService(
        config=LLMConfig(provider="mock", model_name="fixture", base_url="http://localhost/v1"),
        provider=provider,
    )
    structured = FixtureExperimentalData(records)
    return ExploreService(document_search(has_hit=document_hit), structured, llm), structured, provider


class CitationFixtureProvider:
    """A deterministic fake provider returning a citation from supplied context."""

    provider_name = "mock"

    def __init__(self):
        self.calls = []

    def generate(self, *, system_prompt, user_prompt, config):
        self.calls.append((system_prompt, user_prompt))
        marker = "STRUCTURED EVIDENCE ITEMS\n-------------------------\n"
        section = user_prompt.split(marker, 1)[1].split("\n\nRequired output format:", 1)[0]
        evidence = json.loads(section)
        item = evidence[0]
        return json.dumps(
            {
                "answer": "Fixture answer: the supplied source identifies a planned condition; it does not report an observation.",
                "status": "grounded",
                "findings": [
                    {
                        "statement": "The source lists a planned test condition.",
                        "support": "direct",
                        "chunk_ids": [item["cite_as"]],
                    }
                ],
                "scientific_context": "",
                "limitations": "",
            }
        )


def test_case_a_exact_flex_condition_and_relevant_document_evidence():
    explore, structured, _ = service(planned_records())
    response = explore.explore(
        ExploreRequest(
            "What does the FLEX test plan say about methanol at 21% oxygen and 1 atm?",
            {"fuel": "Methanol", "oxygen_concentration": 21, "pressure": 1, "flow_velocity": 0},
        )
    )

    assert structured.calls == 1
    assert response.rag_context.retrieval_count == 1
    document = next(item for item in response.evidence if item.evidence_type == "document")
    assert document.source_locator == "page:6"
    assert document.similarity_score == 0.61
    exact = [item for item in response.experimental_evidence if item.match_type == "exact_match"]
    assert len(exact) == 1
    assert exact[0].experiment_id == "24"
    assert exact[0].evidence_status == "planned"
    assert response.condition_match.match_type == "exact_match"
    assert response.answer.status == "grounded"
    assert response.evidence_mode == "combined"
    assert response.document_evidence[0].original_text == (
        "Table 6 is a proposed quiescent-environment FLEX test matrix. "
        "Test 24 lists methanol, pressure 1.00 atm, and oxygen mole fraction 0.21."
    )
    assert response.citations[0]["chunk_id"] == "flex-plan-page-6"
    assert response.citations[0]["source_id"] == "nasa-document-sha256"
    assert response.citations[0]["source_locator"] == "page:6"
    assert response.citations[0]["page_number"] == 6
    assert response.citations[0]["evidence_status"] == "planned"


def test_case_b_unsupported_temperature_and_gravity_are_labeled_unavailable():
    explore, _, _ = service(planned_records())
    response = explore.explore(
        ExploreRequest(
            "Does NASA have evidence for these conditions?",
            {
                "fuel": "Methanol",
                "oxygen_concentration": 21,
                "pressure": 1,
                "temperature": 300,
                "gravity": "microgravity",
            },
        )
    )

    availability = response.to_dict()["condition_availability"]
    assert availability["fuel"]["status"] == "AVAILABLE"
    assert availability["oxygen_concentration"]["status"] == "AVAILABLE"
    assert availability["pressure"]["status"] == "AVAILABLE"
    assert availability["temperature"]["status"] == "UNAVAILABLE"
    assert availability["gravity"]["status"] == "UNAVAILABLE"
    for match in response.condition_match.matches:
        assert any(
            parameter.parameter == "temperature" and parameter.match_type == "parameter_unavailable"
            for parameter in match.parameters
        )
        assert any(
            parameter.parameter == "gravity" and parameter.match_type == "parameter_unavailable"
            for parameter in match.parameters
        )
    unavailable_warning = next(warning for warning in response.warnings if warning.startswith("Unavailable structured conditions:"))
    assert "temperature" in unavailable_warning and "gravity" in unavailable_warning


def test_case_c_close_oxygen_match_reports_source_value_difference_and_tolerance():
    explore, _, _ = service(planned_records())
    response = explore.explore(
        ExploreRequest(
            "Find planned FLEX conditions near 20.8% oxygen.",
            {"fuel": "Methanol", "oxygen_concentration": 20.8, "pressure": 1, "flow_velocity": 0},
        )
    )

    row = next(item for item in response.experimental_evidence if item.experiment_id == "24")
    assert row.experiment_id == "24"
    assert row.match_type == "close_match"
    oxygen = next(item for item in row.parameter_matches if item["parameter"] == "oxygen_concentration")
    assert abs(oxygen["requested_value"] - 0.208) < 1e-12
    assert oxygen["experimental_value"] == 0.21
    assert abs(oxygen["difference"] - 0.002) < 1e-12
    assert oxygen["unit"] == "mole_fraction"
    assert oxygen["tolerance"] == 0.005
    assert row.evidence_status == "planned"


def test_case_d_no_structured_match_does_not_hide_relevant_document_evidence():
    explore, _, _ = service(planned_records())
    response = explore.explore(
        ExploreRequest(
            "What does the FLEX plan say about cellulose at these conditions?",
            {"fuel": "cellulose", "oxygen_concentration": 90, "pressure": 9},
        )
    )

    assert response.condition_match.match_type == "no_match"
    assert response.condition_match.matches
    assert all(match.match_type in {"no_match", "partial_match"} for match in response.condition_match.matches)
    assert response.evidence_mode == "document_only"
    assert response.document_evidence
    assert not response.experimental_evidence
    payload = response.to_dict()
    assert payload["condition_match"]["candidate_count"] == 5
    assert payload["condition_match"]["status_counts"]["no_match"] == 5
    assert payload["condition_match"]["matches"] == []
    assert len(payload["condition_match"]["unmatched_candidates"]) == 5
    assert payload["condition_match"]["unmatched_candidates"][0]["source_locator"].startswith("page:6")


def test_case_e_document_only_query_skips_structured_matcher():
    explore, structured, _ = service(planned_records())
    response = explore.explore(ExploreRequest("What is FLEX?"))
    assert structured.calls == 0
    assert response.condition_match is None
    assert response.condition_availability == {}
    assert response.evidence_mode == "document_only"
    assert not response.experimental_evidence
    assert response.document_evidence


def test_case_f_structured_hit_without_document_evidence_stays_insufficient():
    explore, _, _ = service(planned_records(), document_hit=False)
    response = explore.explore(
        ExploreRequest(
            "What was measured for methanol at these conditions?",
            {"fuel": "Methanol", "oxygen_concentration": 21, "pressure": 1},
        )
    )
    assert response.rag_context.retrieval_count == 0
    assert response.condition_match.match_type == "exact_match"
    assert response.experimental_evidence
    assert response.evidence_mode == "experimental_only"
    assert response.answer.status == "insufficient_evidence"
    assert "insufficient" in response.answer.answer.casefold()
    assert all(item.evidence_status == "planned" for item in response.experimental_evidence)


def test_case_g_weak_document_evidence_and_condition_hit_keep_separate_scores():
    retrieve = document_search(has_hit=True)

    def weak_retrieve(query, *, top_k):
        result = retrieve(query, top_k=top_k)
        match = result.matches[0]
        return SearchResult(
            result.query,
            result.provider,
            result.model_name,
            result.dimension,
            result.similarity_metric,
            (replace(match, similarity_score=0.12),),
        )

    provider = MockLLMProvider()
    llm = LLMService(
        config=LLMConfig(provider="mock", model_name="fixture", base_url="http://localhost/v1"),
        provider=provider,
    )
    response = ExploreService(weak_retrieve, FixtureExperimentalData(planned_records()), llm).explore(
        ExploreRequest(
            "Find the planned methanol condition.",
            {"fuel": "methanol", "oxygen_concentration": 21, "pressure": 1},
        )
    )
    assert response.document_evidence[0].similarity_score == 0.12
    assert response.experimental_evidence
    assert response.evidence_mode == "combined"


def test_case_i_no_document_or_condition_evidence_is_insufficient():
    explore, _, _ = service(planned_records(), document_hit=False)
    response = explore.explore(
        ExploreRequest(
            "What was observed for cellulose at 9 atm?",
            {"fuel": "cellulose", "oxygen_concentration": 90, "pressure": 9},
        )
    )
    assert response.status == "insufficient_evidence"
    assert response.evidence_mode == "insufficient"
    assert response.document_evidence == ()
    assert response.experimental_evidence == ()
    assert response.citations == ()


def test_case_h_planned_only_match_never_becomes_an_observation_or_answered_result():
    explore, _, provider = service(planned_records(), document_hit=False)
    response = explore.explore(
        ExploreRequest(
            "What did NASA observe at this planned methanol condition?",
            {"fuel": "methanol", "oxygen_concentration": 21, "pressure": 1},
        )
    )
    assert response.evidence_mode == "experimental_only"
    assert response.status == "insufficient_evidence"
    assert all(item.evidence_status == "planned" for item in response.experimental_evidence)
    assert all(item.original_data["observed_result"] is None for item in response.experimental_evidence)
    # Existing grounded LLM preflight does not make a model call with no
    # retrieved document chunks, preventing an answer from model memory.
    assert provider.calls == []


def test_frontend_json_contract_has_separate_semantic_and_match_fields():
    explore, _, _ = service(planned_records())
    payload = explore.explore(
        ExploreRequest(
            "Find a FLEX planned condition.",
            {"fuel": "Methanol", "oxygen_concentration": 21, "pressure": 1},
        )
    ).to_dict()
    assert payload["answer"]["status"] == "grounded"
    assert payload["status"] == "grounded"
    assert payload["evidence_mode"] == "combined"
    assert payload["condition_match"]["candidate_count"] == 5
    assert "status_counts" in payload["condition_match"]
    assert "similarity_score" in payload["document_evidence"][0]
    experimental = payload["experimental_evidence"][0]
    assert "match_type" in experimental
    assert "similarity_score" not in experimental
    assert experimental["evidence_status"] == "planned"
    assert isinstance(payload["citations"], list)
    json.dumps(payload, ensure_ascii=False)


def test_partial_match_is_distinct_from_exact_and_close():
    explore, _, _ = service(planned_records())
    response = explore.explore(
        ExploreRequest(
            "Find methanol conditions at 1 atm and include suppressant when available.",
            {"fuel": "methanol", "pressure": 1, "oxygen_concentration": 21, "suppressant": "CO2"},
        )
    )
    row = next(item for item in response.condition_match.matches if item.record.experiment_id == "24")
    assert row.match_type == "partial_match"
    assert any(item.experiment_id == "24" and item.match_type == "partial_match" for item in response.experimental_evidence)
    suppressant = next(item for item in row.parameters if item.parameter == "suppressant")
    assert suppressant.match_type == "parameter_unavailable"


def test_invented_citation_is_rejected_by_existing_grounded_llm_validator():
    class InvalidCitationProvider:
        provider_name = "mock"

        def generate(self, *, system_prompt, user_prompt, config):
            return json.dumps(
                {
                    "answer": "The retrieved evidence supports a response.",
                    "status": "grounded",
                    "findings": [
                        {"statement": "An invented claim.", "support": "direct", "chunk_ids": ["invented-chunk"]}
                    ],
                }
            )

    llm = LLMService(
        config=LLMConfig(provider="mock", model_name="fixture", base_url="http://localhost/v1"),
        provider=InvalidCitationProvider(),
    )
    explore = ExploreService(document_search(), FixtureExperimentalData(planned_records()), llm)
    try:
        explore.explore(ExploreRequest("What is in the FLEX plan?"))
    except LLMResponseError as exc:
        assert "not present in supplied RAGContext" in str(exc)
    else:
        raise AssertionError("Invented citation was not rejected")


def test_claim_of_observation_cannot_be_supported_by_planned_citation():
    class PlannedAsObservedProvider:
        provider_name = "mock"

        def generate(self, *, system_prompt, user_prompt, config):
            marker = "STRUCTURED EVIDENCE ITEMS\n-------------------------\n"
            section = user_prompt.split(marker, 1)[1].split("\n\nRequired output format:", 1)[0]
            item = json.loads(section)[0]
            return json.dumps(
                {
                    "answer": "NASA observed the test outcome at this condition.",
                    "status": "grounded",
                    "findings": [
                        {"statement": "A test condition is listed.", "support": "direct", "chunk_ids": [item["cite_as"]]}
                    ],
                }
            )

    llm = LLMService(
        config=LLMConfig(provider="mock", model_name="fixture", base_url="http://localhost/v1"),
        provider=PlannedAsObservedProvider(),
    )
    explore = ExploreService(document_search(), FixtureExperimentalData(planned_records()), llm)
    try:
        explore.explore(ExploreRequest("What did NASA observe?"))
    except LLMResponseError as exc:
        assert "claims an observation" in str(exc)
    else:
        raise AssertionError("Planned citation incorrectly supported an observation claim")


def test_forged_source_provenance_in_model_output_never_reaches_citations():
    class ForgedProvenanceProvider:
        provider_name = "mock"

        def generate(self, *, system_prompt, user_prompt, config):
            marker = "STRUCTURED EVIDENCE ITEMS\n-------------------------\n"
            section = user_prompt.split(marker, 1)[1].split("\n\nRequired output format:", 1)[0]
            item = json.loads(section)[0]
            return json.dumps(
                {
                    "answer": "The retrieved document describes the plan.",
                    "status": "grounded",
                    "findings": [
                        {
                            "statement": "The plan lists methanol at 21% oxygen.",
                            "support": "direct",
                            "chunk_ids": [item["cite_as"]],
                            "source_id": "forged-source",
                        }
                    ],
                    "citations": [{"chunk_id": item["cite_as"], "source_id": "forged-source"}],
                }
            )

    llm = LLMService(
        config=LLMConfig(provider="mock", model_name="fixture", base_url="http://localhost/v1"),
        provider=ForgedProvenanceProvider(),
    )
    explore = ExploreService(document_search(), FixtureExperimentalData(planned_records()), llm)
    response = explore.explore(ExploreRequest("What does the FLEX plan say?"))
    assert response.citations[0]["source_id"] == "nasa-document-sha256"
    assert response.citations[0]["evidence_status"] == "planned"
