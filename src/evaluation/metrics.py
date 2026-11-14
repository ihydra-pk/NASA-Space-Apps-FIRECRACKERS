"""Retrieval and answer metrics computed from Explore responses.

Gold evidence is named by document filename and page rather than chunk ID, so
the question set survives re-chunking. Every function reads the JSON produced
by ``ExploreResponse.to_dict()``; only the keys present in every version of
that contract are required, so a run on older code can serve as the baseline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CATEGORIES = (
    "single_doc",
    "multi_doc",
    "inference",
    "comparison",
    "numeric_conditions",
    "should_abstain",
)
DEFAULT_QUESTIONS = Path(__file__).with_name("questions.jsonl")

Page = tuple[str, "int | None"]


@dataclass(frozen=True, slots=True)
class GoldPage:
    filename: str
    # None matches every page of the file (DOCX records carry no page).
    page: int | None

    def matches(self, page: Page) -> bool:
        filename, number = page
        return filename == self.filename and (self.page is None or number == self.page)


@dataclass(frozen=True, slots=True)
class Question:
    id: str
    question: str
    category: str
    gold: tuple[GoldPage, ...]
    must_mention: tuple[str, ...] = ()

    @property
    def should_abstain(self) -> bool:
        return self.category == "should_abstain"


def load_questions(path: Path = DEFAULT_QUESTIONS) -> list[Question]:
    questions: list[Question] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        category = raw["category"]
        if category not in CATEGORIES:
            raise ValueError(f"line {line_number}: unknown category {category!r}")
        gold = tuple(GoldPage(item["filename"], item["page"]) for item in raw.get("gold", []))
        if (category == "should_abstain") == bool(gold):
            raise ValueError(f"line {line_number}: gold pages are required exactly when the question is answerable")
        questions.append(
            Question(
                id=raw["id"],
                question=raw["question"],
                category=category,
                gold=gold,
                must_mention=tuple(raw.get("must_mention", [])),
            )
        )
    ids = [question.id for question in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("question ids must be unique")
    return questions


def retrieved_pages(response: dict[str, Any]) -> list[Page]:
    """Document pages the answer model was given, best-ranked first."""

    items = sorted(
        response.get("document_evidence") or [],
        key=lambda item: item["provenance"]["retrieval_rank"],
    )
    return [(item["provenance"]["filename"], item["page_number"]) for item in items]


def cited_pages(response: dict[str, Any]) -> list[Page]:
    by_chunk = {
        item["record_identifier"]: (item["provenance"]["filename"], item["page_number"])
        for item in response.get("document_evidence") or []
    }
    return [by_chunk[citation["chunk_id"]] for citation in response.get("citations") or [] if citation["chunk_id"] in by_chunk]


def first_gold_rank(pages: list[Page], gold: tuple[GoldPage, ...]) -> int | None:
    for rank, page in enumerate(pages, start=1):
        if any(item.matches(page) for item in gold):
            return rank
    return None


def gold_coverage(pages: list[Page], gold: tuple[GoldPage, ...], k: int) -> float:
    """Fraction of gold pages found in the top ``k`` retrieved pages."""

    top = pages[:k]
    return sum(any(item.matches(page) for page in top) for item in gold) / len(gold)


def answer_text(response: dict[str, Any]) -> str:
    answer = response.get("answer") or {}
    parts = [answer.get("answer") or ""]
    parts.extend(finding.get("statement", "") for finding in answer.get("findings") or [])
    return "\n".join(parts)


def score_question(
    question: Question,
    response: dict[str, Any] | None,
    *,
    error: str | None = None,
    seconds: float = 0.0,
    retrieval_only: bool = False,
) -> dict[str, Any]:
    """Score one question; ``response`` is None when the pipeline raised."""

    result: dict[str, Any] = {
        "id": question.id,
        "category": question.category,
        "question": question.question,
        "error": error,
        "seconds": round(seconds, 2),
        "status": None,
        "gold_rank": None,
        "hit@5": None,
        "hit@10": None,
        "coverage@10": None,
        "gold_cited": None,
        "terms_found": None,
        "missing_terms": None,
        "retrieved": [],
    }
    if response is None:
        return result
    pages = retrieved_pages(response)
    result["retrieved"] = [f"{name} p.{page}" for name, page in pages[:10]]
    if question.gold:
        rank = first_gold_rank(pages, question.gold)
        result["gold_rank"] = rank
        result["hit@5"] = rank is not None and rank <= 5
        result["hit@10"] = rank is not None and rank <= 10
        result["coverage@10"] = round(gold_coverage(pages, question.gold, 10), 3)
    if retrieval_only:
        return result
    result["status"] = response.get("status")
    text = answer_text(response).casefold()
    result["terms_found"] = [term for term in question.must_mention if term.casefold() in text]
    result["missing_terms"] = [term for term in question.must_mention if term.casefold() not in text]
    if question.gold:
        result["gold_cited"] = any(item.matches(page) for page in cited_pages(response) for item in question.gold)
    return result


def _rate(values: list[bool]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [item for item in results if item["category"] != "should_abstain" and item["error"] is None]
    abstain = [item for item in results if item["category"] == "should_abstain" and item["error"] is None]
    statuses: dict[str, int] = {}
    for item in results:
        key = item["status"] or ("error" if item["error"] else "not_run")
        statuses[key] = statuses.get(key, 0) + 1
    answered = [item for item in answerable if item["status"] is not None]
    mention = [item for item in answered if item["terms_found"] or item["missing_terms"]]
    return {
        "questions": len(results),
        "errors": sum(item["error"] is not None for item in results),
        "hit@5": _rate([bool(item["hit@5"]) for item in answerable]),
        "hit@10": _rate([bool(item["hit@10"]) for item in answerable]),
        "mrr": round(sum(1 / item["gold_rank"] for item in answerable if item["gold_rank"]) / len(answerable), 3)
        if answerable
        else None,
        "coverage@10": round(sum(item["coverage@10"] for item in answerable) / len(answerable), 3)
        if answerable
        else None,
        "statuses": statuses,
        "answered_when_answerable": _rate([item["status"] != "insufficient_evidence" for item in answered]),
        "abstained_when_unanswerable": _rate(
            [item["status"] == "insufficient_evidence" for item in abstain if item["status"] is not None]
        ),
        "gold_cited": _rate([bool(item["gold_cited"]) for item in answered]),
        # Mean share of each question's must_mention terms found in the answer.
        "terms_mentioned": round(
            sum(len(item["terms_found"]) / (len(item["terms_found"]) + len(item["missing_terms"])) for item in mention)
            / len(mention),
            3,
        )
        if mention
        else None,
        "mean_seconds": round(sum(item["seconds"] for item in results) / len(results), 1) if results else None,
    }
