from __future__ import annotations

import pytest

from src.rag import RAGContextConfig, build_context, retrieve_context
from src.vectorstore.models import RetrievalResult, SearchResult


def _result(rank: int, chunk_id: str, *, status="planned", text=None, **overrides):
    values = {
        "rank": rank,
        "similarity_score": 0.91 - rank / 100,
        "chunk_id": chunk_id,
        "source_id": f"source-{chunk_id}",
        "content_sha256": f"sha-{chunk_id}",
        "experiment": "FLEX",
        "filename": "FLEX_Test_Plan.pdf",
        "file_type": "pdf",
        "source_path": "flex/FLEX_Test_Plan.pdf",
        "document_type": "test_plan",
        "evidence_status": status,
        "source_locator": f"page:{rank}",
        "page_number": rank,
        "text": text if text is not None else f"Exact source text for {chunk_id}.\nSecond line.",
        "metadata": {"author": "NASA"},
    }
    values.update(overrides)
    return values


def _search_result(matches, query="test question"):
    return SearchResult(
        query=query,
        provider="openai-compatible",
        model_name="qwen-test",
        dimension=8,
        similarity_metric="cosine",
        matches=tuple(
            RetrievalResult.from_record(
                item["rank"],
                item["similarity_score"],
                {
                    **item,
                    "embedding": [0.0],
                    "embedding_dimension": 1,
                    "embedding_provider": "mock",
                    "embedding_model": "mock",
                },
            )
            if isinstance(item, dict)
            else item
            for item in matches
        ),
    )


def test_context_creation_preserves_provenance_and_source_text_exactly():
    text = "  LOI = 0.209 (1 − MEC)\n\nTemperature: 0.5 °C.  "
    result = _result(1, "chunk-a", text=text)

    context = build_context("test question", _search_result([result]))

    assert context.query == "test question"
    assert context.retrieval_count == 1
    item = context.evidence_items[0]
    assert item.text == text
    assert item.source_id == "source-chunk-a"
    assert item.content_sha256 == "sha-chunk-a"
    assert item.source_path == "flex/FLEX_Test_Plan.pdf"
    assert item.filename == "FLEX_Test_Plan.pdf"
    assert item.source_locator == "page:1"
    assert item.page_number == 1
    assert item.document_type == "test_plan"
    assert item.evidence_status == "planned"
    assert "Evidence status: planned" in context.context_text
    assert text in context.context_text
    assert context.retrieval_configuration["model_name"] == "qwen-test"


def test_formatting_is_deterministic_and_results_are_sorted_by_rank():
    retrieval = _search_result([_result(3, "third"), _result(1, "first"), _result(2, "second")])

    first = build_context("test question", retrieval)
    second = build_context("test question", retrieval)

    assert first.context_text == second.context_text
    assert [item.chunk_id for item in first.evidence_items] == ["first", "second", "third"]
    assert first.context_text.index("EVIDENCE 1") < first.context_text.index("EVIDENCE 2")


def test_evidence_status_planned_metadata_observed_interpreted_and_unknown_are_preserved():
    statuses = ["planned", "metadata", "observed", "interpreted", "unknown", None]
    results = [
        _result(index, f"chunk-{index}", status=status)
        for index, status in enumerate(statuses, start=1)
    ]

    context = build_context("test question", _search_result(results), RAGContextConfig(top_k=10))

    assert [item.evidence_status for item in context.evidence_items] == [
        "planned", "metadata", "observed", "interpreted", "unknown", "unknown"
    ]
    assert "Evidence status: planned" in context.context_text
    assert "Evidence status: metadata" in context.context_text
    assert "Evidence status: observed" in context.context_text
    assert "Evidence status: interpreted" in context.context_text
    assert "Evidence status: unknown" in context.context_text


def test_duplicate_chunk_ids_keep_first_best_ranked_result():
    retrieval = _search_result(
        [_result(1, "same", text="Best ranked text"), _result(2, "same", text="Duplicate text"), _result(3, "other")]
    )

    context = build_context("test question", retrieval)

    assert [item.chunk_id for item in context.evidence_items] == ["same", "other"]
    assert context.evidence_items[0].text == "Best ranked text"


def test_top_k_limits_unique_evidence_items():
    retrieval = _search_result([_result(index, f"chunk-{index}") for index in range(1, 5)])

    context = build_context("test question", retrieval, RAGContextConfig(top_k=2))

    assert context.retrieval_count == 2
    assert [item.chunk_id for item in context.evidence_items] == ["chunk-1", "chunk-2"]


def test_context_limit_drops_lowest_ranked_whole_without_source_truncation():
    results = [
        _result(1, "short", text="Top evidence is intact."),
        _result(2, "long", text="Lower-ranked complete source evidence. " * 30),
    ]
    retrieval = _search_result(results)
    unlimited = build_context("test question", retrieval)
    budget = len("QUESTION\n--------\ntest question") + 600

    context = build_context(
        "test question", retrieval, RAGContextConfig(top_k=5, max_context_chars=budget)
    )

    assert context.truncation_occurred
    assert context.dropped_evidence_count == 1
    assert context.dropped_chunk_ids == ("long",)
    assert context.context_characters <= budget
    assert [item.chunk_id for item in context.evidence_items] == ["short"]
    assert context.evidence_items[0].text == "Top evidence is intact."
    assert "No evidence text was truncated" in context.context_text


def test_token_limit_also_drops_whole_lower_ranked_items():
    retrieval = _search_result(
        [_result(1, "one", text="One concise sentence."), _result(2, "two", text="many words " * 100)]
    )
    unlimited = build_context("test question", retrieval)
    limit = len(unlimited.context_text.split()) - 5

    context = build_context(
        "test question",
        retrieval,
        RAGContextConfig(max_context_chars=None, max_estimated_tokens=limit),
    )

    assert context.truncation_occurred
    assert [item.chunk_id for item in context.evidence_items] == ["one"]
    assert context.estimated_context_tokens <= limit


def test_empty_retrieval_returns_question_only_context():
    context = build_context("Nothing retrieved", _search_result([], query="Nothing retrieved"))

    assert context.retrieval_count == 0
    assert context.evidence_items == ()
    assert context.context_text == "QUESTION\n--------\nNothing retrieved"
    assert not context.truncation_occurred


def test_invalid_or_missing_retrieval_metadata_fails_clearly():
    with pytest.raises(ValueError, match="similarity_score"):
        build_context(
            "test question",
            {"query": "test question", "matches": [{"rank": 1}]},
        )

    invalid = _result(1, "missing-path")
    invalid["source_path"] = ""
    with pytest.raises(ValueError, match="source_path"):
        build_context("test question", {"query": "test question", "matches": [invalid]})


def test_similarity_is_never_renamed_to_confidence():
    context = build_context("test question", _search_result([_result(1, "score")]))

    assert context.evidence_items[0].similarity_score == 0.9
    assert "Similarity:" in context.context_text
    assert "Confidence" not in context.context_text
    assert "confidence" not in context.to_dict()


def test_similarity_can_be_hidden_from_text_but_remains_structured():
    context = build_context(
        "test question",
        _search_result([_result(1, "hidden-score")]),
        RAGContextConfig(include_similarity_scores=False),
    )

    assert "Similarity:" not in context.context_text
    assert context.evidence_items[0].similarity_score == 0.9


def test_query_is_not_rewritten_and_query_mismatch_is_rejected():
    query = "  FLEX objectives?\n"
    context = build_context(query, _search_result([], query=query))
    assert context.query == query
    assert context.context_text.endswith(query)

    with pytest.raises(ValueError, match="does not match"):
        build_context("different question", _search_result([]))


def test_service_uses_existing_retriever_and_configured_top_k():
    calls = []

    def retriever(query, *, top_k):
        calls.append((query, top_k))
        return _search_result([_result(1, "retrieved")], query=query)

    context = retrieve_context("service query", retriever, RAGContextConfig(top_k=3))

    assert calls == [("service query", 3)]
    assert context.evidence_items[0].chunk_id == "retrieved"
