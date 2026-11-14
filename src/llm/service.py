"""Structured, citation-validated answer generation from an existing RAGContext."""

from __future__ import annotations

import json
import re
from typing import Any

from src.rag.models import EvidenceItem, RAGContext

from .config import LLMConfig
from .models import AnswerCitation, AnswerFinding, AnswerResponse, EvidenceReference
from .policy import EvidencePolicy, evidence_labels
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


_STATUSES = ("grounded", "partial", "insufficient_evidence")
_OBSERVATION_CLAIM = re.compile(
    r"\b(?:NASA|FLEX|the experiments?|the tests?|the results)\s+"
    r"(?:observed|measured|found|demonstrated|showed|recorded)\b",
    re.IGNORECASE,
)
# Statuses whose sources are known not to report results. `unknown` is only
# unlabelled, so it can support an observation claim.
_NOT_RESULTS = frozenset({"planned", "metadata"})
# Real chunk IDs look like this; one that was not supplied is invented.
_CHUNK_ID_SHAPE = re.compile(r"\bchunk-[0-9a-f]{8,}\b")
_BRACKETS = re.compile(r"\[([^\[\]]+)\]")
# The prompt labels evidence E1, E2, ...; a label that was not supplied is invented.
_LABEL_SHAPE = re.compile(r"^E\d+$")


def _expand_labels(text: str, labels: dict[str, str]) -> str:
    """Replace evidence labels inside square brackets with their chunk IDs.

    ``[E2, E5]`` becomes ``[chunk-…, chunk-…]``: the stored answer and the web
    page keep working with chunk IDs; only the model sees the short labels.
    """

    def expand(match: re.Match[str]) -> str:
        parts = re.split(r"([,;\s]+)", match.group(1))
        return "[" + "".join(labels.get(part, part) for part in parts) + "]"

    return _BRACKETS.sub(expand, text)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _text_field(parsed: dict[str, Any], key: str, *, required: bool) -> str:
    value = parsed.get(key)
    if value is None and not required:
        return ""
    if not required and isinstance(value, list) and all(isinstance(part, str) for part in value):
        # Models sometimes write limitations as a list of points.
        value = "\n".join(part.strip() for part in value if part.strip())
    if not isinstance(value, str) or (required and not value.strip()):
        raise LLMResponseError(f"LLM response {key} must be a {'non-empty ' if required else ''}string")
    return value.strip()


def _cited_ids(text: str, supplied: dict[str, EvidenceItem]) -> list[str]:
    """Supplied chunk IDs cited in square brackets, in order of first appearance.

    ``[chunk-a]`` and ``[chunk-a, chunk-b]`` both count; the web page renders
    bracketed IDs as citation links.
    """

    cited = [
        token
        for group in _BRACKETS.findall(text)
        for token in re.split(r"[,;\s]+", group)
        if token in supplied
    ]
    return list(dict.fromkeys(cited))


def _reject_invented_ids(text: str, supplied: dict[str, EvidenceItem], *, where: str) -> None:
    # Runs after _expand_labels, so any label still in brackets was never supplied.
    invented = [token for token in _CHUNK_ID_SHAPE.findall(text) if token not in supplied] + [
        token
        for group in _BRACKETS.findall(text)
        for token in re.split(r"[,;\s]+", group)
        if _LABEL_SHAPE.match(token)
    ]
    if invented:
        raise LLMResponseError(f"{where} references chunk_id {invented[0]!r} not present in supplied RAGContext")


def _check_observation_claim(
    text: str, chunk_ids: list[str] | tuple[str, ...], supplied: dict[str, EvidenceItem], *, where: str
) -> None:
    if not _OBSERVATION_CLAIM.search(text):
        return
    not_results = sorted({supplied[chunk_id].evidence_status for chunk_id in chunk_ids} & _NOT_RESULTS)
    if not_results:
        raise LLMResponseError(
            f"{where} claims an observation but cites {' and '.join(not_results)} evidence; planned "
            "conditions and metadata are not observed results. Describe them as planned or as metadata."
        )


def _findings(
    raw_findings: Any, supplied: dict[str, EvidenceItem], labels: dict[str, str]
) -> tuple[AnswerFinding, ...]:
    if raw_findings is None:
        return ()
    if not isinstance(raw_findings, list):
        raise LLMResponseError("LLM response findings must be a JSON array")
    findings: list[AnswerFinding] = []
    for index, raw in enumerate(raw_findings, start=1):
        if not isinstance(raw, dict):
            raise LLMResponseError(f"finding {index} must be a JSON object")
        statement = raw.get("statement")
        if not isinstance(statement, str) or not statement.strip():
            raise LLMResponseError(f"finding {index} statement must be a non-empty string")
        support = raw.get("support")
        if support not in ("direct", "inference"):
            raise LLMResponseError(f"finding {index} support must be 'direct' or 'inference'")
        chunk_ids = raw.get("chunk_ids")
        if not isinstance(chunk_ids, list) or not chunk_ids:
            raise LLMResponseError(
                f"finding {index} must cite at least one chunk_id; general background belongs in scientific_context"
            )
        chunk_ids = [labels.get(chunk_id, chunk_id) if isinstance(chunk_id, str) else chunk_id for chunk_id in chunk_ids]
        for chunk_id in chunk_ids:
            if not isinstance(chunk_id, str) or chunk_id not in supplied:
                raise LLMResponseError(
                    f"finding {index} references chunk_id {chunk_id!r} not present in supplied RAGContext"
                )
        unique_ids = tuple(dict.fromkeys(chunk_ids))
        _check_observation_claim(statement, unique_ids, supplied, where=f"finding {index}")
        findings.append(AnswerFinding(_expand_labels(statement.strip(), labels), support, unique_ids))
    return tuple(findings)


def _validate_response(parsed: dict[str, Any], context: RAGContext, config: LLMConfig) -> AnswerResponse:
    status = parsed.get("status")
    if status not in _STATUSES:
        raise LLMResponseError("LLM response status must be grounded, partial or insufficient_evidence")
    labels = evidence_labels(context)
    answer = _expand_labels(_text_field(parsed, "answer", required=True), labels)
    scientific_context = _expand_labels(_text_field(parsed, "scientific_context", required=False), labels)
    limitations = _expand_labels(_text_field(parsed, "limitations", required=False), labels)

    supplied = {item.chunk_id: item for item in context.evidence_items}
    findings = _findings(parsed.get("findings"), supplied, labels)
    _reject_invented_ids(answer, supplied, where="answer")
    _reject_invented_ids(limitations, supplied, where="limitations")
    for index, finding in enumerate(findings, start=1):
        _reject_invented_ids(finding.statement, supplied, where=f"finding {index}")
    answer_ids = _cited_ids(answer, supplied)
    finding_ids = [chunk_id for finding in findings for chunk_id in finding.chunk_ids]
    for sentence in _SENTENCE_END.split(answer):
        # A sentence without its own marker is judged by everything the findings cite.
        _check_observation_claim(sentence, _cited_ids(sentence, supplied) or finding_ids, supplied, where="answer")

    if _cited_ids(scientific_context, supplied) or _CHUNK_ID_SHAPE.search(scientific_context):
        raise LLMResponseError(
            "scientific_context is general background and must not cite chunk_ids; "
            "put claims supported by NASA evidence in findings"
        )
    if status in ("grounded", "partial"):
        if not context.evidence_items:
            raise LLMResponseError("Cannot return a grounded answer without supplied evidence")
        if not findings:
            raise LLMResponseError(
                "A grounded or partial answer must cite at least one supplied evidence item in findings"
            )
    if status == "partial" and not limitations:
        raise LLMResponseError("A partial answer must state in limitations what the evidence does not cover")
    if status == "insufficient_evidence" and "insufficient" not in answer.casefold():
        raise LLMResponseError(
            "An insufficient_evidence response must explicitly state that evidence is insufficient"
        )
    prose = "\n".join([answer, scientific_context, limitations, *(finding.statement for finding in findings)])
    if re.search(r"\bconfidence\b", prose, flags=re.IGNORECASE):
        raise LLMResponseError(
            "Do not describe retrieval similarity as confidence or include confidence claims"
        )

    # The model supplies chunk IDs only; provenance is copied from the context,
    # so a citation can never carry an invented source, page or status.
    citations = tuple(
        AnswerCitation(
            chunk_id=chunk_id,
            source_id=supplied[chunk_id].source_id,
            source_locator=supplied[chunk_id].source_locator,
            page_number=supplied[chunk_id].page_number,
            evidence_status=supplied[chunk_id].evidence_status,
        )
        for chunk_id in dict.fromkeys([*answer_ids, *finding_ids, *_cited_ids(limitations, supplied)])
    )
    references = tuple(_reference(item) for item in context.evidence_items)
    return AnswerResponse(
        answer=answer,
        status=status,
        citations=citations,
        evidence_summary=references,
        query=context.query,
        model_name=config.model_name,
        provider=config.provider,
        findings=findings,
        scientific_context=scientific_context,
        limitations=limitations,
    )


def _repair_prompt(user_prompt: str, rejected: str, error: LLMResponseError) -> str:
    return (
        f"{user_prompt}\n\n"
        "YOUR PREVIOUS REPLY WAS REJECTED\n"
        "--------------------------------\n"
        f"Reason: {error}\n"
        f"Rejected reply: {rejected}\n\n"
        "Return a corrected JSON object that fixes this problem and follows every rule in the system prompt."
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
        try:
            return _validate_response(_parse_json(raw), context, self.config)
        except LLMResponseError as error:
            # One repair attempt: the model sees why its reply was rejected.
            # A second failure is raised to the caller unchanged.
            raw = self.provider.generate(
                system_prompt=decision.system_prompt,
                user_prompt=_repair_prompt(decision.user_prompt, raw, error),
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
