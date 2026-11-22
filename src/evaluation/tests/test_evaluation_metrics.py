from __future__ import annotations

import json

import pytest

from src.evaluation.metrics import (
    DEFAULT_QUESTIONS,
    GoldPage,
    Question,
    load_questions,
    score_question,
    summarize,
)

REPORT = "PSI-39_Reports_CFI Science Final Report.pdf"


def _evidence(chunk_id, filename, page, rank):
    return {
        "record_identifier": chunk_id,
        "page_number": page,
        "provenance": {"filename": filename, "retrieval_rank": rank},
    }


def _response(*, status="grounded", cited=("c9",), answer="Extinction is controlled by chemical kinetics."):
    return {
        "status": status,
        "answer": {"answer": answer, "findings": [{"statement": "Residence time falls as droplets shrink."}]},
        "citations": [{"chunk_id": chunk_id} for chunk_id in cited],
        # Deliberately out of rank order: metrics must sort by retrieval rank.
        "document_evidence": [
            _evidence("c9", REPORT, 9, 3),
            _evidence("c12", REPORT, 12, 1),
            _evidence("x", "other.pdf", 1, 2),
        ],
    }


QUESTION = Question(
    id="q",
    question="What causes cool flame extinction?",
    category="inference",
    gold=(GoldPage(REPORT, 9),),
    must_mention=("kinetic", "residence time", "helium"),
)


def test_committed_question_set_loads_and_has_abstain_questions():
    questions = load_questions(DEFAULT_QUESTIONS)
    assert len(questions) >= 20
    assert any(question.should_abstain for question in questions)
    assert all(question.gold for question in questions if not question.should_abstain)


def test_retrieval_rank_coverage_and_citation_are_scored_from_the_response():
    result = score_question(QUESTION, _response(), seconds=1.234)

    assert result["gold_rank"] == 3
    assert result["hit@5"] is True and result["hit@10"] is True
    assert result["coverage@10"] == 1.0
    assert result["gold_cited"] is True
    assert result["status"] == "grounded"
    # Findings count toward the answer text; "helium" appears nowhere.
    assert result["terms_found"] == ["kinetic", "residence time"]
    assert result["missing_terms"] == ["helium"]
    assert result["retrieved"][0] == f"{REPORT} p.12"
    assert result["seconds"] == 1.23


def test_page_none_gold_matches_any_page_of_the_file():
    gold = GoldPage("readme.docx", None)
    assert gold.matches(("readme.docx", None)) and gold.matches(("readme.docx", 4))
    assert not gold.matches(("other.docx", None))


def test_pipeline_error_is_recorded_without_metrics():
    result = score_question(QUESTION, None, error="LLMResponseError: rejected")
    assert result["error"] == "LLMResponseError: rejected"
    assert result["status"] is None and result["gold_rank"] is None


def test_retrieval_only_scores_ranks_but_not_answers():
    result = score_question(QUESTION, _response(), retrieval_only=True)
    assert result["gold_rank"] == 3
    assert result["status"] is None and result["gold_cited"] is None


def test_summary_separates_abstention_from_answering():
    abstain = Question("a", "Bone density?", "should_abstain", ())
    results = [
        score_question(QUESTION, _response()),
        score_question(QUESTION, _response(status="insufficient_evidence", cited=())),
        score_question(abstain, _response(status="insufficient_evidence", cited=())),
        score_question(QUESTION, None, error="RuntimeError: gateway down"),
    ]
    summary = summarize(results)

    assert summary["questions"] == 4 and summary["errors"] == 1
    assert summary["answered_when_answerable"] == 0.5
    assert summary["abstained_when_unanswerable"] == 1.0
    assert summary["gold_cited"] == 0.5
    assert summary["hit@5"] == 1.0
    assert summary["mrr"] == pytest.approx(1 / 3, abs=1e-3)
    assert summary["statuses"] == {"grounded": 1, "insufficient_evidence": 2, "error": 1}


def test_question_file_rejects_unknown_categories(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text(json.dumps({"id": "x", "question": "?", "category": "trivia", "gold": []}) + "\n")
    with pytest.raises(ValueError, match="unknown category"):
        load_questions(path)
