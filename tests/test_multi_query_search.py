"""Multi-query fusion, reference-list demotion, neighbours and the in-memory cache."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from src.vectorstore import CachedCollection, SQLiteVectorStore, VectorStoreConfig, search, search_many

REFERENCES = (
    "Nayagam, V., Dietrich, D. L., Hicks, M. C., & Williams, F. A. (2015). Cool-flame extinction during "
    "n-alkane droplet combustion in microgravity. Combustion and Flame, 162(5), 2140-2147. Farouk, T. I., "
    "& Dryer, F. L. (2014). Isolated n-heptane droplet combustion in microgravity. Combustion and Flame, "
    "161(2), 565-581."
)


def _record(chunk_id, vector, text, *, filename="PSI-39_Reports_Final Report.pdf", page=1, source="report"):
    return {
        "chunk_id": chunk_id,
        "source_id": source,
        "content_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "experiment": "CFI",
        "filename": filename,
        "file_type": "pdf",
        "source_path": f"cfi/{filename}",
        "document_type": "unknown",
        "evidence_status": None,
        "source_locator": f"page:{page}",
        "page_number": page,
        "chunk_index": 1,
        "chunk_count": 1,
        "text": text,
        "embedding": vector,
        "embedding_model": "test-model",
        "embedding_provider": "test-provider",
        "embedding_dimension": len(vector),
    }


class MappedProvider:
    """Embeds each query text to a fixed vector; records every batch."""

    provider_name = "test-provider"

    def __init__(self, vectors):
        self.vectors = vectors
        self.batches = []

    def embed_batch(self, texts, embedding_config):
        self.batches.append(list(texts))
        return [self.vectors[text] for text in texts]


def _config(tmp_path: Path, **overrides) -> VectorStoreConfig:
    return replace(
        VectorStoreConfig(
            database_path=tmp_path / "vectors.sqlite3",
            collection="test",
            provider="test-provider",
            model_name="test-model",
            dimension=3,
            top_k=2,
        ),
        **overrides,
    )


def _store(config, records):
    store = SQLiteVectorStore(config)
    for record in records:
        store.upsert(record, record["embedding"])
    return store


def test_sub_queries_that_agree_bring_a_passage_into_the_results(tmp_path):
    config = _config(tmp_path)
    records = [
        _record("question-best", [1.0, 0.0, 0.0], "Droplet burning overview."),
        _record("question-second", [0.9, 0.1, 0.0], "Hardware description of the chamber."),
        _record("mechanism", [0.2, 0.0, 1.0], "Extinction is controlled by chemical kinetics."),
        _record("other", [0.0, 1.0, 0.0], "Crew schedule."),
    ]
    provider = MappedProvider(
        {
            "What causes extinction?": [1.0, 0.0, 0.0],
            "extinction mechanism kinetics": [0.0, 0.0, 1.0],
            "why does the flame go out": [0.1, 0.0, 1.0],
        }
    )
    with _store(config, records) as store:
        alone = search("What causes extinction?", store, config, provider=provider)
        fused = search_many(
            ["What causes extinction?", "extinction mechanism kinetics", "why does the flame go out"],
            store,
            config,
            provider=provider,
        )

    assert [item.chunk_id for item in alone.matches] == ["question-best", "question-second"]
    # Both sub-queries rank "mechanism" first; together they lift it past the
    # question's own second hit.
    assert {item.chunk_id for item in fused.matches} == {"question-best", "mechanism"}
    # All queries go to the embedding model in one request.
    assert provider.batches[-1] == ["What causes extinction?", "extinction mechanism kinetics", "why does the flame go out"]
    # The score shown is still the cosine to the user's question.
    mechanism = next(item for item in fused.matches if item.chunk_id == "mechanism")
    assert mechanism.similarity_score == pytest.approx(0.2 / (1.0 * (0.2**2 + 1.0) ** 0.5))
    assert fused.query == "What causes extinction?"


def test_blank_and_repeated_sub_queries_are_ignored(tmp_path):
    config = _config(tmp_path)
    provider = MappedProvider({"Question?": [1.0, 0.0, 0.0]})
    with _store(config, [_record("a", [1.0, 0.0, 0.0], "Text a.")]) as store:
        result = search_many(["Question?", " ", "question?", "QUESTION? "], store, config, provider=provider)
    assert provider.batches == [["Question?"]]
    assert result.query == "Question?"


def test_reference_lists_rank_after_findings_only_when_enabled(tmp_path):
    config = _config(tmp_path)
    records = [
        _record("bibliography", [1.0, 0.0, 0.0], REFERENCES, page=12),
        _record("finding", [0.8, 0.6, 0.0], "Cool flame extinction is controlled primarily by chemical kinetics.", page=9),
        _record("other", [0.0, 0.0, 1.0], "Crew schedule for the week.", page=3),
    ]
    provider = MappedProvider(
        {
            "What causes cool flame extinction?": [1.0, 0.0, 0.0],
            "Which papers report cool flame extinction?": [1.0, 0.0, 0.0],
        }
    )
    with _store(config, records) as store:
        default = search("What causes cool flame extinction?", store, config, provider=provider)
        demoting = replace(config, demote_reference_lists=True)
        demoted = search("What causes cool flame extinction?", store, demoting, provider=provider)
        papers = search("Which papers report cool flame extinction?", store, demoting, provider=provider)

    assert [item.chunk_id for item in default.matches] == ["bibliography", "finding"]
    # Demoted, not removed: it follows every other passage.
    assert [item.chunk_id for item in demoted.matches] == ["finding", "other"]
    assert [item.chunk_id for item in papers.matches] == ["bibliography", "finding"]


def test_next_page_of_a_report_follows_its_hit(tmp_path):
    config = _config(tmp_path, top_k=3, neighbour_chunks=1)
    records = [
        _record("p9", [1.0, 0.0, 0.0], "Cool flame extinction is controlled by kinetics.", page=9),
        _record("p10", [0.0, 0.0, 1.0], "Helium dilution leads to earlier cool flame extinction.", page=10),
        _record("p11", [0.0, 0.1, 1.0], "Sooting behaviour as a function of pressure.", page=11),
        _record("elsewhere-1", [0.9, 0.1, 0.0], "Another report on droplets.", source="other"),
        _record("elsewhere-2", [0.8, 0.2, 0.0], "A third report on droplets.", source="third"),
    ]
    provider = MappedProvider({"q": [1.0, 0.0, 0.0]})
    with _store(config, records) as store:
        result = search("q", store, config, provider=provider)
    # p10 is semantically unrelated but continues the top hit; the lowest hit makes room.
    assert [item.chunk_id for item in result.matches] == ["p9", "p10", "elsewhere-1"]


def test_slides_do_not_pull_in_their_next_slide(tmp_path):
    config = _config(tmp_path, top_k=2, neighbour_chunks=1)
    slides = "PSI-39_Presentations_Cool Flames.pdf"
    records = [
        _record("s1", [1.0, 0.0, 0.0], "Slide one.", filename=slides, page=1, source="slides"),
        _record("s2", [0.0, 0.0, 1.0], "Slide two.", filename=slides, page=2, source="slides"),
        _record("r", [0.9, 0.1, 0.0], "A report page.", source="report"),
    ]
    with _store(config, records) as store:
        result = search("q", store, config, provider=MappedProvider({"q": [1.0, 0.0, 0.0]}))
    assert [item.chunk_id for item in result.matches] == ["s1", "r"]


def test_cached_collection_searches_like_the_store_and_is_read_only(tmp_path):
    config = _config(tmp_path, top_k=3)
    records = [
        _record("a", [1.0, 0.0, 0.0], "Droplets burn in a quiescent chamber."),
        _record("b", [0.6, 0.8, 0.0], "Tests 201 to 228 add SF6 as suppressant."),
        _record("c", [0.0, 0.0, 1.0], "Crew time is scheduled weekly."),
    ]
    provider = MappedProvider({"which tests use SF6": [1.0, 0.0, 0.0]})
    with _store(config, records) as store:
        direct = search("which tests use SF6", store, config, provider=provider, mode="hybrid")
    cached = CachedCollection.load(config)
    from_cache = search("which tests use SF6", cached, config, provider=provider, mode="hybrid")

    assert [(m.chunk_id, m.rank, m.keyword_rank) for m in from_cache.matches] == [
        (m.chunk_id, m.rank, m.keyword_rank) for m in direct.matches
    ]
    assert from_cache.matches[0].similarity_score == pytest.approx(direct.matches[0].similarity_score)
    with pytest.raises(TypeError, match="read-only"):
        cached.upsert(records[0], records[0]["embedding"])


def test_streamed_rows_match_records_and_cache_drops_the_duplicate_vector(tmp_path):
    config = _config(tmp_path)
    records = [_record(name, [1.0, 0.0, 0.0], f"Text {name}.") for name in ("b", "a")]
    with _store(config, records) as store:
        assert list(store.iter_records()) == store.records()
        assert [record["chunk_id"] for record, _ in store.iter_records()] == ["a", "b"]
    cached = CachedCollection.load(config)
    assert all("embedding" not in record for record, _ in cached.records())
    assert list(cached.records()[0][1]) == [1.0, 0.0, 0.0]
