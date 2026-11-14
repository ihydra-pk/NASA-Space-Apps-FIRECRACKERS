"""Structured, citation-validated answer generation from an existing RAGContext."""

from __future__ import annotations

import json
import re
from typing import Any

from src.rag.models import EvidenceItem, RAGContext

from .config import LLMConfig
from .models import AnswerCitation, AnswerResponse, EvidenceReference
from .policy import EvidencePolicy
from .providers import LLMProvider, create_llm_provider


class LLMResponseError(ValueError):
    """Raised when generated output violates the structured evidence contract."""


def _parse_json(raw_response: str) -> dict[str, Any]:
    if not isinstance(raw_response, str):
        raise LLMResponseError("LLM response content must be a string")
    text = raw_response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip().startswith("```"):
            text = "\n".join(lines[1:-1]).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMResponseError(f"LLM response is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise LLMResponseError("LLM response must be a JSON object")
    return parsed


def _reference(item: EvidenceItem) -> EvidenceReference:
    return EvidenceReference(
        chunk_id=item.chunk_id,
        source_id=item.source_id,
        content_sha256=item.content_sha256,
        experiment=item.experiment,
        filename=item.filename,
        file_type=item.file_type,
        source_path=item.source_path,
        document_type=item.document_type,
        evidence_status=item.evidence_status,
        source_locator=item.source_locator,
        page_number=item.page_number,
    )


def _validate_response(parsed: dict[str, Any], context: RAGContext, config: LLMConfig) -> AnswerResponse:
    answer = parsed.get("answer")
    status = parsed.get("status")
    raw_citations = parsed.get("citations")
    if not isinstance(answer, str) or not answer.strip():
        raise LLMResponseError("LLM response answer must be a non-empty string")
    if status not in {"grounded", "insufficient_evidence"}:
        raise LLMResponseError("LLM response status must be grounded or insufficient_evidence")
    if not isinstance(raw_citations, list):
        raise LLMResponseError("LLM response citations must be a JSON array")

    supplied = {item.chunk_id: item for item in context.evidence_items}
    citations: list[AnswerCitation] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_citations):
        if not isinstance(raw, dict):
            raise LLMResponseError(f"citation {index + 1} must be a JSON object")
        chunk_id = raw.get("chunk_id")
        if not isinstance(chunk_id, str) or chunk_id not in supplied:
            raise LLMResponseError(
                f"citation {index + 1} references chunk_id {chunk_id!r} not present in supplied RAGContext"
            )
        if chunk_id in seen:
            raise LLMResponseError(f"duplicate citation for chunk_id {chunk_id!r}")
        seen.add(chunk_id)
        item = supplied[chunk_id]
        expected = {
            "source_id": item.source_id,
            "source_locator": item.source_locator,
            "page_number": item.page_number,
            "evidence_status": item.evidence_status,
        }
        for field, expected_value in expected.items():
            if raw.get(field) != expected_value:
                raise LLMResponseError(
                    f"citation {chunk_id!r} has invalid {field}; it must match supplied provenance"
                )
        citations.append(
            AnswerCitation(
                chunk_id=item.chunk_id,
                source_id=item.source_id,
                source_locator=item.source_locator,
                page_number=item.page_number,
                evidence_status=item.evidence_status,
            )
        )

    if status == "grounded":
        if not context.evidence_items:
            raise LLMResponseError("Cannot return a grounded answer without supplied evidence")
        if not citations:
            raise LLMResponseError("A grounded answer must cite at least one supplied evidence item")
        cited_statuses = {citation.evidence_status for citation in citations}
        claims_observation = re.search(
            r"\b(?:NASA|FLEX|the experiment|the test|the tests|the results)\s+"
            r"(?:observed|measured|found|demonstrated|showed|recorded)\b",
            answer,
            flags=re.IGNORECASE,
        )
        if claims_observation and cited_statuses != {"observed"}:
            raise LLMResponseError(
                "Answer claims an observation but citations are not exclusively "
                "evidence_status='observed'; planned, metadata, and interpreted "
                "evidence cannot substantiate that claim."
            )
    else:
        if "insufficient" not in answer.casefold():
            raise LLMResponseError(
                "An insufficient_evidence response must explicitly state that evidence is insufficient"
            )
    if re.search(r"\bconfidence\b", answer, flags=re.IGNORECASE):
        raise LLMResponseError(
            "Do not describe retrieval similarity as confidence or include confidence claims"
        )

    # Evidence references are built locally from the exact context, not from
    # model-generated metadata, so the model cannot invent trace identifiers.
    references = tuple(_reference(item) for item in context.evidence_items)
    return AnswerResponse(
        answer=answer.strip(),
        status=status,
        citations=tuple(citations),
        evidence_summary=references,
        query=context.query,
        model_name=config.model_name,
        provider=config.provider,
    )


class LLMService:
    """Generate and validate a grounded structured response from RAGContext only."""

    def __init__(
        self,
        config: LLMConfig | None = None,
        provider: LLMProvider | None = None,
        policy: EvidencePolicy | None = None,
    ) -> None:
        self.config = config or LLMConfig.from_env()
        self.provider = provider or create_llm_provider(self.config)
        self.policy = policy or EvidencePolicy()
        provider_name = getattr(self.provider, "provider_name", None)
        if provider_name != self.config.provider:
            raise ValueError(
                f"Configured LLM provider {self.config.provider!r} does not match "
                f"provider implementation {provider_name!r}"
            )

    def answer(self, question: str, context: RAGContext) -> AnswerResponse:
        decision = self.policy.evaluate(question, context)
        if not context.evidence_items:
            # Avoid asking a model to fill evidence gaps from pretraining.
            return AnswerResponse(
                answer=(
                    "The indexed evidence is insufficient to answer this question: "
                    "no retrieved evidence items were supplied."
                ),
                status="insufficient_evidence",
                citations=(),
                evidence_summary=(),
                query=question,
                model_name=self.config.model_name,
                provider=self.config.provider,
            )

        raw = self.provider.generate(
            system_prompt=decision.system_prompt,
            user_prompt=decision.user_prompt,
            config=self.config,
        )
        return _validate_response(_parse_json(raw), context, self.config)


def answer_question(
    question: str,
    context: RAGContext,
    *,
    config: LLMConfig | None = None,
    provider: LLMProvider | None = None,
) -> AnswerResponse:
    return LLMService(config=config, provider=provider).answer(question, context)
