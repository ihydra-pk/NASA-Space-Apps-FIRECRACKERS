"""Provider-aware query embedding, cosine similarity and hybrid retrieval."""

from __future__ import annotations

import math
import operator
import re
from dataclasses import dataclass
from typing import Any, Sequence

from src.embeddings.config import EmbeddingConfig
from src.embeddings.pipeline import _validate_vectors
from src.embeddings.providers import EmbeddingProvider, create_embedding_provider

from .config import VectorStoreConfig
from .keyword import bm25_scores
from .models import RetrievalResult, SearchResult
from .references import asks_for_publications, looks_like_reference_list
from .store import VectorStore

# math.sumprod (Python 3.12+) is a dot product in C, several times faster than
# the equivalent generator that older interpreters fall back to.
_dot = getattr(math, "sumprod", None) or (lambda left, right: sum(map(operator.mul, left, right)))


def cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        raise ValueError(f"Cannot compare vectors with dimensions {len(left)} and {len(right)}")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("Cosine similarity is undefined for a zero vector")
    score = sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
    # Floating-point roundoff can move mathematically bounded cosine scores a
    # few ulps outside [-1, 1].
    return max(-1.0, min(1.0, score))


def query_text_for_embedding(query_text: str, config: VectorStoreConfig) -> str:
    """The text sent to the embedding model for a query.

    Qwen3 embedding models expect ``Instruct: <task>\nQuery: <question>`` on
    the query side only. Other models, including the mock provider, get the
    question unchanged.
    """

    if config.query_instruction and "qwen3-embedding" in config.model_name.casefold():
        return f"Instruct: {config.query_instruction}\nQuery: {query_text}"
    return query_text


Scored = list[tuple[float, dict[str, Any], Sequence[float]]]


def _text_key(record: dict[str, Any]) -> str:
    return " ".join(record["text"].split()).casefold()


def _rankable(scored: Scored, min_chunk_words: int) -> Scored:
    """Drop title-only chunks and repeats of a passage already ranked higher.

    The same paper is often filed under several names, so identical text would
    otherwise fill several result slots. ``scored`` is in similarity order, so
    the first copy kept is the best-scoring one.
    """

    kept: Scored = []
    seen: set[str] = set()
    for item in scored:
        if len(item[1]["text"].split()) < min_chunk_words:
            continue
        key = _text_key(item[1])
        if key in seen:
            continue
        seen.add(key)
        kept.append(item)
    return kept


_WORD = "A-Za-z0-9"
_NAME_SUFFIX = re.compile(r"^(?P<stem>[A-Za-z]{3,})-(?:[IVX]+|\d+|[A-Z])$")


def _alias_tiers(experiment: str, accession: str | None) -> tuple[set[str], set[str]]:
    """(exact names, looser names) for one experiment.

    Exact names are the label's own parts and the NASA ID. Looser names are
    words taken out of a label (``ACME`` from ``ACME s-Flame``) and a name
    without its numbered suffix (``SAFFIRE`` from ``SAFFIRE-I``).
    """

    # "DAFT / DAFT-2" lists two names; "ACME/Flame Design" is one name whose
    # pieces ("ACME", "Flame Design") are only looser names.
    exact = {part for part in re.split(r"\s+/\s+", experiment.strip()) if part}
    looser: set[str] = set()
    for part in exact:
        pieces = [piece.strip() for piece in part.split("/") if piece.strip()]
        if len(pieces) > 1:
            looser.update(pieces)
        for token in re.split(r"[\s/]+", part):
            if "-" in token or (len(token) >= 3 and token.isupper()):
                looser.add(token)
    for alias in exact | looser:
        suffixed = _NAME_SUFFIX.match(alias)
        if suffixed:
            looser.add(suffixed.group("stem"))
    # The ID is added last so that "PSI-47" does not also yield "PSI".
    looser.discard("PSI")
    if accession:
        exact.add(accession.strip())
    exact = {alias for alias in exact if len(alias) >= 3}
    return exact, {alias for alias in looser if len(alias) >= 3} - exact


def experiment_aliases(experiment: str, accession: str | None) -> set[str]:
    """Names a question may use for one experiment, from its stored labels.

    ``"DAFT / DAFT-2"`` with ``"PSI-47"`` gives ``PSI-47, DAFT, DAFT-2``;
    ``"ACME/Flame Design"`` gives ``ACME, Flame Design``; ``"SAFFIRE-I"`` also
    gives ``SAFFIRE``. Ordinary words inside a label (``Flame``, ``Design``)
    are not aliases on their own.
    """

    exact, looser = _alias_tiers(experiment, accession)
    return exact | looser


def _names(query_text: str, alias: str) -> bool:
    """Whether the question uses ``alias`` as a name rather than a word.

    Matching ignores case, but an alias made only of letters (``SAME``,
    ``FLEX``, ``Flame Design``) must be written with a capital, so "burn the
    same" is not read as the SAME experiment. A hyphen is part of a name, so
    ``BASS`` does not match inside ``BASS-II``.
    """

    pattern = re.compile(
        rf"(?<![{_WORD}-]){re.escape(alias)}(?![{_WORD}-])".replace(r"\ ", r"\s+"),
        re.IGNORECASE,
    )
    plain_word = alias.replace(" ", "").isalpha()
    return any(
        not plain_word or any(char.isupper() for char in found.group(0))
        for found in pattern.finditer(query_text)
    )


def named_experiments(query_text: str, records: list[dict]) -> tuple[str, ...]:
    """Experiment labels of ``records`` that the question names, sorted.

    An exact name wins over a looser one: "BASS" means the BASS experiment and
    not BASS-II as well, while "SAFFIRE", which is no experiment's exact name,
    covers SAFFIRE-I and SAFFIRE-III.
    """

    tiers: dict[str, tuple[set[str], set[str]]] = {}
    for record in records:
        experiment = record.get("experiment")
        if isinstance(experiment, str) and experiment.strip():
            exact, looser = _alias_tiers(experiment, record.get("source_accession"))
            known = tiers.setdefault(experiment, (set(), set()))
            known[0].update(exact)
            known[1].update(looser)
    for tier in (0, 1):
        named = sorted(
            experiment
            for experiment, aliases in tiers.items()
            if any(_names(query_text, alias) for alias in aliases[tier])
        )
        if named:
            return tuple(named)
    return ()


SEARCH_MODES = ("semantic", "hybrid")
# Reciprocal rank fusion: each ranking contributes weight / (k + rank). A small
# k keeps the top semantic hits well ahead; the keyword ranking counts for less,
# so it promotes exact-term matches without letting word overlap alone outrank
# a close match in meaning. Chosen by checking a handful of questions against
# the full index; revisit with the benchmark in src/evaluation.
_RRF_K = 10
_KEYWORD_WEIGHT = 0.3
# Fusing several queries works the same way. The user's question counts 1.5
# times as much as each planner sub-query: its own best hits stay on top, but
# each sub-query's best hit still beats the question's hits from rank 7 down,
# and a passage several queries agree on rises above both. At 2.0 a passage
# found only by a sub-query barely reached the top 10 in a check on the full
# index. Tune with the benchmark in src/evaluation.
_QUESTION_WEIGHT = 1.5
_SUBQUERY_WEIGHT = 1.0
# Passages taken from each query's ranking before the rankings are fused.
_CANDIDATES_PER_QUERY = 15
# The next page of a slide deck or of a per-test data sheet is usually about
# something else, so neighbours are only added for prose documents.
_NO_NEIGHBOURS = ("_Presentations_", "_Analyzed Data_")


@dataclass(frozen=True, slots=True)
class _Ranking:
    """One query's ordering of the rankable passages."""

    scored: Scored  # similarity order, after _rankable
    order: list[int]  # indices into ``scored``, best first
    keyword_rank: dict[int, int]  # hybrid mode only
    keyword_scores: list[float]  # hybrid mode only
    hybrid: bool

    def result(self, rank: int, index: int) -> RetrievalResult:
        score, record, _ = self.scored[index]
        if not self.hybrid:
            return RetrievalResult.from_record(rank, score, record)
        return RetrievalResult.from_record(
            rank,
            score,
            record,
            semantic_rank=index + 1,
            keyword_rank=self.keyword_rank.get(index),
            keyword_score=self.keyword_scores[index] if index in self.keyword_rank else None,
        )


def _rank(query_text: str, scored: Scored, mode: str) -> _Ranking:
    if mode != "hybrid":
        return _Ranking(scored, list(range(len(scored))), {}, [], False)
    keyword_scores = bm25_scores(query_text, [record["text"] for _, record, _ in scored])
    by_keyword = sorted(
        (index for index, score in enumerate(keyword_scores) if score > 0),
        key=lambda index: (-keyword_scores[index], scored[index][1]["chunk_id"]),
    )
    keyword_rank = {index: rank for rank, index in enumerate(by_keyword, start=1)}

    def fused(index: int) -> float:
        value = 1.0 / (_RRF_K + index + 1)  # ``scored`` is already in semantic order
        if index in keyword_rank:
            value += _KEYWORD_WEIGHT / (_RRF_K + keyword_rank[index])
        return value

    order = sorted(range(len(scored)), key=lambda index: (-fused(index), scored[index][1]["chunk_id"]))
    return _Ranking(scored, order, keyword_rank, keyword_scores, True)


def _similarities(
    query_vectors: list[tuple[float, ...]],
    entries: list[tuple[dict[str, Any], Sequence[float]]],
    dimension: int,
) -> list[list[float]]:
    """Cosine similarity of every stored vector to each query vector."""

    query_norms = [math.sqrt(_dot(vector, vector)) for vector in query_vectors]
    if not all(query_norms):
        raise ValueError("Cosine similarity is undefined for a zero vector")
    columns: list[list[float]] = [[] for _ in query_vectors]
    for record, vector in entries:
        if len(vector) != dimension:
            raise ValueError(
                f"Stored chunk {record.get('chunk_id')!r} has dimension {len(vector)}; "
                f"expected {dimension}"
            )
        norm = math.sqrt(_dot(vector, vector))
        if norm == 0:
            raise ValueError("Cosine similarity is undefined for a zero vector")
        for column, query, query_norm in zip(columns, query_vectors, query_norms):
            # Roundoff can move a bounded cosine a few ulps outside [-1, 1].
            column.append(max(-1.0, min(1.0, _dot(query, vector) / (query_norm * norm))))
    return columns


def _fuse_queries(rankings: list[_Ranking], depth: int) -> list[int]:
    """Reciprocal rank fusion of each query's top ``depth`` passages.

    Returns indices into the question's ranking. Passages are matched by text,
    so a duplicate that one query kept under another chunk ID still counts.
    """

    question = rankings[0]
    position = {_text_key(record): index for index, (_, record, _) in enumerate(question.scored)}
    fused: dict[int, float] = {}
    for number, ranking in enumerate(rankings):
        weight = _QUESTION_WEIGHT if number == 0 else _SUBQUERY_WEIGHT
        for rank, index in enumerate(ranking.order[:depth], start=1):
            target = position.get(_text_key(ranking.scored[index][1]))
            if target is not None:
                fused[target] = fused.get(target, 0.0) + weight / (_RRF_K + rank)
    return sorted(fused, key=lambda index: (-fused[index], question.scored[index][1]["chunk_id"]))


def _demote_reference_lists(candidates: list[int], scored: Scored, count: int) -> list[int]:
    """Rank reference-list passages after all others (see ``references.py``)."""

    findings: list[int] = []
    references: list[int] = []
    for index in candidates:
        (references if looks_like_reference_list(scored[index][1]["text"]) else findings).append(index)
        if len(findings) == count:
            break
    return findings + references


def _next_in_document(record: dict[str, Any], scored: Scored) -> int | None:
    """The following chunk of the same page, or the first one of the next page."""

    page = record.get("page_number") or 0
    here = (page, record.get("chunk_index") or 0)
    best: tuple[tuple[int, int], int] | None = None
    for index, (_, other, _) in enumerate(scored):
        if other["source_id"] != record["source_id"]:
            continue
        key = (other.get("page_number") or 0, other.get("chunk_index") or 0)
        if here < key and key[0] <= page + 1 and (best is None or key < best[0]):
            best = (key, index)
    return best[1] if best else None


def _with_neighbours(
    selected: list[int], scored: Scored, neighbours: int, count: int, *, skip_references: bool
) -> list[int]:
    """Insert the next passage of the same document after each of the top hits.

    A finding often continues on the next page of a report. The added passage
    sits right after the hit it continues; the lowest-ranked hits make room.
    """

    result = list(selected)
    for parent in selected[:neighbours]:
        record = scored[parent][1]
        if any(tag in record["filename"] for tag in _NO_NEIGHBOURS):
            continue
        neighbour = _next_in_document(record, scored)
        if neighbour is None or neighbour in result:
            continue
        if skip_references and looks_like_reference_list(scored[neighbour][1]["text"]):
            continue
        result.insert(result.index(parent) + 1, neighbour)
    return result[:count]


def _distinct_queries(queries: Sequence[str]) -> list[str]:
    if isinstance(queries, str):
        raise TypeError("queries must be a sequence of query strings")
    texts = list(queries)
    if not texts or not isinstance(texts[0], str) or not texts[0].strip():
        raise ValueError("query_text must not be empty")
    # The question is kept exactly as given: RAG checks it against the result.
    kept = [texts[0]]
    seen = {texts[0].strip().casefold()}
    for text in texts[1:]:
        if not isinstance(text, str):
            raise ValueError("search queries must be strings")
        if text.strip() and text.strip().casefold() not in seen:
            seen.add(text.strip().casefold())
            kept.append(text.strip())
    return kept


def _scope(candidates: list[int], ranking: _Ranking, experiments: tuple[str, ...]) -> list[int]:
    """Put every passage of the named experiments ahead of all others.

    Fused candidates keep their order; the experiment's remaining passages
    follow in the question's own order, so a small experiment that no query
    ranked highly is still searched in full. Other passages come last and only
    fill slots the named experiments leave empty.
    """

    def inside(index: int) -> bool:
        return ranking.scored[index][1].get("experiment") in experiments

    chosen = [index for index in candidates if inside(index)]
    seen = set(chosen)
    chosen += [index for index in ranking.order if index not in seen and inside(index)]
    return chosen + [index for index in candidates if not inside(index)]


def search_many(
    queries: Sequence[str],
    store: VectorStore,
    config: VectorStoreConfig,
    *,
    provider: EmbeddingProvider | None = None,
    top_k: int | None = None,
    mode: str = "semantic",
) -> SearchResult:
    """Rank passages for a question and any planner sub-queries in one pass.

    ``queries[0]`` is the user's question; further entries are sub-queries.
    All are embedded in one request and each is ranked as :func:`search`
    would rank it alone. With several queries the rankings are fused by
    reciprocal rank, the question weighted highest. ``similarity_score`` is
    always the cosine similarity to the question, and ``semantic_rank`` and
    ``keyword_rank`` describe the question's own ranking.

    Two optional search-time rules come from ``config``:
    ``demote_reference_lists`` ranks chunks that are mostly citations after
    all others, unless the question asks about publications, and
    ``neighbour_chunks`` adds the next passage of the same document after each
    of that many top hits.
    """

    texts = _distinct_queries(queries)
    question = texts[0]
    if mode not in SEARCH_MODES:
        raise ValueError(f"mode must be one of {SEARCH_MODES}; got {mode!r}")
    count = config.top_k if top_k is None else top_k
    if count <= 0:
        raise ValueError("top_k must be greater than zero")
    provider = provider or create_embedding_provider(
        EmbeddingConfig(
            provider=config.provider,
            model_name=config.model_name,
            dimension=config.dimension,
            batch_size=1,
        )
    )
    provider_name = getattr(provider, "provider_name", None)
    if provider_name != config.provider:
        raise ValueError(
            f"Query provider {provider_name!r} does not match indexed provider {config.provider!r}"
        )

    profile = store.get_collection_profile()
    expected_profile = (config.provider, config.model_name, config.dimension)
    if profile is not None and profile != expected_profile:
        raise ValueError(f"Query profile {expected_profile} does not match stored profile {profile}")

    embedding_config = EmbeddingConfig(
        provider=config.provider,
        model_name=config.model_name,
        dimension=config.dimension,
        batch_size=len(texts),
    )
    try:
        vectors = _validate_vectors(
            provider.embed_batch([query_text_for_embedding(text, config) for text in texts], embedding_config),
            len(texts),
            embedding_config,
        )
    except Exception as exc:
        raise RuntimeError(f"Could not embed retrieval query: {exc}") from exc

    entries = store.records()
    rankings: list[_Ranking] = []
    for text, scores in zip(texts, _similarities(vectors, entries, config.dimension)):
        scored = sorted(
            ((score, record, vector) for score, (record, vector) in zip(scores, entries)),
            key=lambda item: (-item[0], item[1]["chunk_id"]),
        )
        rankings.append(_rank(text, _rankable(scored, config.min_chunk_words), mode))

    question_ranking = rankings[0]
    if len(rankings) == 1:
        candidates = question_ranking.order
    else:
        candidates = _fuse_queries(rankings, max(count, _CANDIDATES_PER_QUERY))
    scoped: tuple[str, ...] = ()
    if config.scope_to_named_experiments:
        scoped = named_experiments(question, [record for _, record, _ in question_ranking.scored])
    if scoped:
        candidates = _scope(candidates, question_ranking, scoped)
    demote = config.demote_reference_lists and not asks_for_publications(question)
    if demote:
        candidates = _demote_reference_lists(candidates, question_ranking.scored, count)
    selected = candidates[:count]
    if config.neighbour_chunks:
        selected = _with_neighbours(
            selected, question_ranking.scored, config.neighbour_chunks, count, skip_references=demote
        )

    return SearchResult(
        scoped_experiments=scoped,
        mode=mode,
        query=question,
        provider=config.provider,
        model_name=config.model_name,
        dimension=config.dimension,
        similarity_metric=config.similarity_metric,
        matches=tuple(question_ranking.result(rank, index) for rank, index in enumerate(selected, start=1)),
    )


def search(
    query_text: str,
    store: VectorStore,
    config: VectorStoreConfig,
    *,
    provider: EmbeddingProvider | None = None,
    top_k: int | None = None,
    mode: str = "semantic",
) -> SearchResult:
    """Embed a query with the indexed profile and return ranked evidence.

    ``mode="semantic"`` ranks by cosine similarity alone. ``mode="hybrid"``
    also scores chunk text by keyword (BM25) and orders by reciprocal rank
    fusion of the two rankings. Fusion uses ranks only, so the two score
    scales are never added together and ``similarity_score`` stays cosine.
    """

    return search_many([query_text], store, config, provider=provider, top_k=top_k, mode=mode)
