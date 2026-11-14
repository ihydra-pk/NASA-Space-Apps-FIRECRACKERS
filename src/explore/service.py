"""Orchestrate existing semantic retrieval, condition matching, and grounded answer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

from src.experimental_data.matching import MatchResult
from src.experimental_data.matching import CATEGORICAL_FIELDS, NUMERIC_FIELDS
from src.experimental_data.service import ExperimentalDataService
from src.llm.models import AnswerResponse
from src.llm.service import LLMService
from src.rag.models import RAGContext
from src.rag.service import retrieve_context

from .config import ExploreConfig
from .evidence import document_evidence, experimental_evidence, matching_experimental_matches
from .formatting import add_condition_evidence, format_condition_evidence
from .models import ExploreRequest, ExploreResponse
from .router import QuestionRouter, RouteDecision

Retriever = Callable[..., Any]
ConditionMatcher = Callable[[dict[str, object]], MatchResult]

_CONDITION_UNITS = {
    "oxygen_concentration": "%",
    "pressure": "atm",
    "flow_velocity": "cm/s",
    "suppressant_concentration": "mole_fraction",
    "temperature": "K",
    "gravity": None,
}


class ExploreInputError(ValueError):
    """Raised for unsupported, malformed, or unit-ambiguous Explore requests."""


@dataclass(frozen=True, slots=True)
class NormalizedConditions:
    values: dict[str, object]
    units: dict[str, str]


def normalize_conditions(
    conditions: dict[str, object],
    declared_units: dict[str, str] | None = None,
) -> NormalizedConditions:
    """Convert only explicitly supported request units into matcher units.

    API defaults follow the project-facing condition convention: oxygen is
    percent (e.g. 21), pressure atm, and flow cm/s. Callers with fractional
    oxygen or other units must declare them in ``condition_units``.
    """

    if not isinstance(conditions, dict):
        raise ExploreInputError("conditions must be an object")
    if declared_units is not None and not isinstance(declared_units, dict):
        raise ExploreInputError("condition_units must be an object")
    declared_units = declared_units or {}
    unknown_unit_keys = set(declared_units) - set(conditions)
    if unknown_unit_keys:
        raise ExploreInputError(
            "condition_units contains parameters absent from conditions: "
            + ", ".join(sorted(unknown_unit_keys))
        )

    values = dict(conditions)
    units: dict[str, str] = {}
    for field, value in conditions.items():
        if field in declared_units and field not in {
            "oxygen_concentration",
            "pressure",
            "flow_velocity",
            "suppressant_concentration",
            "temperature",
        }:
            raise ExploreInputError(f"Units are not applicable to categorical/unsupported condition {field!r}")
        if field not in _CONDITION_UNITS:
            # Unknown fields remain visible and the matcher will classify them
            # as unavailable rather than silently discarding caller intent.
            continue
        unit = declared_units.get(field, _CONDITION_UNITS[field])
        if unit is None:
            continue
        if field == "oxygen_concentration":
            if unit in {"%", "percent", "percentage"}:
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ExploreInputError("oxygen_concentration must be a finite number")
                values[field] = float(value) / 100.0
                units[field] = "mole_fraction"
            elif unit in {"mole_fraction", "fraction"}:
                values[field] = _finite_number(value, field)
                units[field] = "mole_fraction"
            else:
                raise ExploreInputError(f"Unsupported oxygen_concentration unit: {unit!r}")
        elif field == "pressure":
            factors_to_atm = {"atm": 1.0, "kPa": 1 / 101.325, "Pa": 1 / 101325.0}
            if unit not in factors_to_atm:
                raise ExploreInputError(f"Unsupported pressure unit: {unit!r}")
            values[field] = _finite_number(value, field) * factors_to_atm[unit]
            units[field] = "atm"
        elif field == "flow_velocity":
            factors_to_cms = {"cm/s": 1.0, "m/s": 100.0}
            if unit not in factors_to_cms:
                raise ExploreInputError(f"Unsupported flow_velocity unit: {unit!r}")
            values[field] = _finite_number(value, field) * factors_to_cms[unit]
            units[field] = "cm/s"
        elif field == "suppressant_concentration":
            if unit not in {"mole_fraction", "fraction", "%", "percent"}:
                raise ExploreInputError(f"Unsupported suppressant_concentration unit: {unit!r}")
            number = _finite_number(value, field)
            values[field] = number / 100 if unit in {"%", "percent"} else number
            units[field] = "mole_fraction"
        elif field == "temperature":
            # Temperature is not in the structured FLEX matrix. Preserve value
            # and supplied/default unit only; matching reports unavailable.
            if unit not in {"K", "C", "°C", "F", "°F"}:
                raise ExploreInputError(f"Unsupported temperature unit: {unit!r}")
            values[field] = _finite_number(value, field)
            units[field] = unit
    return NormalizedConditions(values, units)


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ExploreInputError(f"{field} must be a finite number")
    return float(value)


def _planned_warning(result: MatchResult | None) -> tuple[str, ...]:
    if result is None or not result.matches:
        return ()
    if any(match.record.evidence_status == "planned" for match in result.matches):
        return (
            "Structured FLEX matches are planned test conditions, not observations or experimental results.",
        )
    return ()


def condition_availability(
    conditions: dict[str, object],
    result: MatchResult | None,
) -> dict[str, dict[str, object]]:
    """Report dataset-level support separately from a record's match outcome."""

    availability: dict[str, dict[str, object]] = {}
    supported = CATEGORICAL_FIELDS | set(NUMERIC_FIELDS)
    for field in conditions:
        if field not in supported:
            availability[field] = {
                "status": "UNAVAILABLE",
                "available_record_count": 0,
                "unit": None,
                "reason": "This parameter is not represented in the structured FLEX dataset.",
            }
            continue
        if result is None or not result.matches:
            availability[field] = {
                "status": "NO_DATA",
                "available_record_count": 0,
                "unit": NUMERIC_FIELDS.get(field),
                "reason": "The parameter is supported by the schema, but no structured records were loaded.",
            }
            continue
        matches_with_value = [
            match for match in result.matches if getattr(match.record, field, None) is not None
        ]
        availability[field] = {
            "status": "AVAILABLE" if matches_with_value else "UNAVAILABLE",
            "available_record_count": len(matches_with_value),
            "unit": (
                next(
                    (match.record.units.get(field) for match in matches_with_value if match.record.units.get(field)),
                    NUMERIC_FIELDS.get(field),
                )
                if field in NUMERIC_FIELDS
                else None
            ),
            "reason": None if matches_with_value else "No source record contains a value for this parameter.",
        }
    return availability


def _match_summary(
    conditions: dict[str, object],
    result: MatchResult | None,
    availability: dict[str, dict[str, object]],
) -> dict[str, object]:
    counts = {name: 0 for name in ("exact_match", "close_match", "partial_match", "no_match", "parameter_unavailable")}
    if result is not None:
        for match in result.matches:
            counts[match.match_type] = counts.get(match.match_type, 0) + 1
    unavailable = [field for field, entry in availability.items() if entry["status"] == "UNAVAILABLE"]
    no_data = [field for field, entry in availability.items() if entry["status"] == "NO_DATA"]
    return {
        "match_type": result.match_type if result is not None else None,
        "matched_count": sum(counts[name] for name in ("exact_match", "close_match", "partial_match")),
        "partial_count": counts["partial_match"],
        "candidate_count": len(result.matches) if result is not None else 0,
        "status_counts": counts,
        "available_parameters": [field for field, entry in availability.items() if entry["status"] == "AVAILABLE"],
        "unavailable_parameters": unavailable,
        "no_data_parameters": no_data,
        "parameter_availability": availability,
        "tolerances": result.tolerances if result is not None else {},
        "requested_conditions": conditions,
    }


class ExploreService:
    """Compose existing RAG and experimental-data interfaces with no score blending."""

    def __init__(
        self,
        retriever: Retriever,
        experimental_data: ExperimentalDataService | ConditionMatcher,
        llm: LLMService,
        *,
        config: ExploreConfig | None = None,
        router: QuestionRouter | None = None,
    ) -> None:
        self.router = router
        self.retriever = retriever
        self.experimental_data = experimental_data
        self.llm = llm
        self.config = config or ExploreConfig()

    def explore(self, request: ExploreRequest) -> ExploreResponse:
        if not isinstance(request.question, str) or not request.question.strip():
            raise ExploreInputError("question must not be empty")
        conditions = {} if request.conditions is None else request.conditions
        if not isinstance(conditions, dict):
            raise ExploreInputError("conditions must be an object")
        condition_units = request.condition_units
        # Conditions chosen by the caller win. Otherwise the router reads them
        # from the question; document retrieval runs on every route because
        # answers are cited against retrieved chunks.
        routing: RouteDecision | None = None
        if self.router is not None and not conditions:
            routing = self.router.route(request.question)
            if routing.uses_test_data:
                conditions, condition_units = routing.conditions, routing.condition_units
        normalized = normalize_conditions(conditions, condition_units)

        # A. Existing retrieval function -> existing RAGContext builder.
        context = retrieve_context(request.question, self.retriever, self.config.rag)

        # B. Existing experimental-data service/matcher. Conditions are matched
        # independently, with no score arithmetic against document similarity.
        if conditions:
            matcher = getattr(self.experimental_data, "match_conditions", self.experimental_data)
            match_result = matcher(normalized.values)
        else:
            match_result = None

        condition_prompt_omitted = False
        available_context_chars = self.config.max_condition_context_chars
        if self.config.rag.max_context_chars is not None:
            available_context_chars = min(
                available_context_chars,
                self.config.rag.max_context_chars - context.context_characters - 2,
            )
        if match_result is not None and available_context_chars >= 300:
            max_records = self.config.max_condition_evidence
            while True:
                condition_block, omitted_condition_ids = format_condition_evidence(
                    match_result,
                    max_records=max_records,
                    max_characters=available_context_chars,
                )
                trial_context = add_condition_evidence(context, condition_block)
                token_limit = self.config.rag.max_estimated_tokens
                if token_limit is None or trial_context.estimated_context_tokens <= token_limit:
                    break
                if max_records == 0:
                    condition_block = ""
                    omitted_condition_ids = tuple(match.record.record_id for match in match_result.matches)
                    break
                max_records -= 1
            condition_prompt_omitted = bool(omitted_condition_ids)
        else:
            condition_block = ""
            condition_prompt_omitted = match_result is not None
        llm_context = add_condition_evidence(context, condition_block)
        answer: AnswerResponse = self.llm.answer(request.question, llm_context)
        doc_evidence = document_evidence(context)
        exp_evidence = experimental_evidence(
            match_result,
            limit=self.config.max_condition_evidence,
        )
        availability = condition_availability(conditions, match_result)
        answer_status = answer.status
        has_documents = bool(doc_evidence)
        has_structured_conditions = bool(matching_experimental_matches(match_result))
        evidence_mode = (
            "combined" if has_documents and has_structured_conditions
            else "document_only" if has_documents
            else "experimental_only" if has_structured_conditions
            else "insufficient"
        )

        warnings = list(_planned_warning(match_result))
        if routing is not None and routing.fallback:
            warnings.append(f"Question routing fell back to document search. {routing.reason}")
        if routing is not None and routing.ignored:
            warnings.append(
                "The router dropped conditions it could not verify: " + "; ".join(routing.ignored) + "."
            )
        if match_result is not None and not match_result.matches:
            warnings.append("No structured condition records were available to compare.")
        if match_result is not None and any(
            parameter.match_type == "parameter_unavailable"
            for match in match_result.matches
            for parameter in match.parameters
        ):
            warnings.append(
                "One or more requested conditions are unavailable in the structured FLEX test-plan data."
            )
        unavailable_fields = [
            name for name, item in availability.items() if item["status"] == "UNAVAILABLE"
        ]
        if unavailable_fields:
            warnings.append(
                "Unavailable structured conditions: " + ", ".join(sorted(unavailable_fields)) + "."
            )
        no_data_fields = [
            name for name, item in availability.items() if item["status"] == "NO_DATA"
        ]
        if no_data_fields:
            warnings.append(
                "No structured records loaded for supported conditions: "
                + ", ".join(sorted(no_data_fields))
                + "."
            )
        if context.truncation_occurred:
            warnings.append(
                f"RAG omitted {context.dropped_evidence_count} lower-ranked document evidence item(s) due to context limits."
            )
        if condition_prompt_omitted:
            warnings.append(
                "Some structured condition details were omitted from the LLM prompt due to context limits; full matcher results remain in the Explore response."
            )

        return ExploreResponse(
            question=request.question,
            answer=answer,
            document_evidence=doc_evidence,
            experimental_evidence=exp_evidence,
            rag_context=context,
            condition_match=match_result,
            requested_conditions=dict(conditions),
            normalized_conditions=normalized.values,
            requested_condition_units=dict(condition_units or {}),
            routing=routing.to_dict() if routing is not None else None,
            condition_units=normalized.units,
            condition_availability=availability,
            condition_match_summary=_match_summary(conditions, match_result, availability),
            evidence_mode=evidence_mode,
            status=answer_status,
            max_serialized_condition_matches=self.config.max_condition_evidence,
            warnings=tuple(dict.fromkeys(warnings)),
        )
