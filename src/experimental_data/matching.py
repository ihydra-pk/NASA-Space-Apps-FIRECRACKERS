"""Transparent categorical and numeric condition matching."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .models import ExperimentRecord
from .normalization import normalize_categorical, normalize_fuel

MatchType = Literal["exact_match", "close_match", "partial_match", "no_match", "parameter_unavailable"]

# Only fields actually present as per-test fields in the local FLEX test matrices.
CATEGORICAL_FIELDS = {"fuel", "suppressant", "diluent", "fiber_support"}
NUMERIC_FIELDS = {
    "oxygen_concentration": "mole_fraction",
    "pressure": "atm",
    "flow_velocity": "cm/s",
    "suppressant_concentration": "mole_fraction",
}


@dataclass(frozen=True, slots=True)
class ParameterMatch:
    parameter: str
    match_type: MatchType
    requested_value: object
    experimental_value: object
    unit: str | None
    difference: float | None = None
    tolerance: float | None = None
    message: str | None = None


@dataclass(frozen=True, slots=True)
class ExperimentMatch:
    record: ExperimentRecord
    match_type: MatchType
    parameters: tuple[ParameterMatch, ...]


@dataclass(frozen=True, slots=True)
class MatchResult:
    requested_conditions: dict[str, object]
    tolerances: dict[str, float]
    match_type: MatchType
    matches: tuple[ExperimentMatch, ...]


def _categorical_equal(field: str, requested: object, actual: object) -> bool:
    if field == "fuel":
        requested_value = normalize_fuel(str(requested))
        actual_value = normalize_fuel(str(actual)) if actual is not None else None
        wanted, found = normalize_categorical(requested_value), normalize_categorical(actual_value)
        if wanted == found:
            return True
        # Source material descriptions such as "100 micron PMMA film 2 cm wide"
        # contain the requested fuel as whole words.
        return bool(wanted and found and re.search(rf"(?<!\w){re.escape(wanted)}(?!\w)", found))
    return normalize_categorical(str(requested)) == normalize_categorical(str(actual))


def match_experiments(
    records: list[ExperimentRecord],
    requested_conditions: dict[str, object],
    *,
    absolute_tolerances: dict[str, float] | None = None,
) -> MatchResult:
    """Match only observed fields in the structured dataset; never imply prediction."""

    tolerances = dict(absolute_tolerances or {})
    parameter_results: list[ExperimentMatch] = []
    for record in records:
        if record.kind != "test_condition":
            continue
        details: list[ParameterMatch] = []
        for field, requested in requested_conditions.items():
            if field not in CATEGORICAL_FIELDS and field not in NUMERIC_FIELDS:
                details.append(ParameterMatch(
                    field, "parameter_unavailable", requested, None, None,
                    message="Parameter is not available in the local per-test dataset.",
                ))
                continue
            actual = getattr(record, field)
            actual_range = (
                record.original_values.get(f"{field}_range")
                if field == "oxygen_concentration" and actual is None
                else None
            )
            if actual_range and field in NUMERIC_FIELDS:
                if not isinstance(requested, (float, int)) or isinstance(requested, bool):
                    details.append(ParameterMatch(field, "parameter_unavailable", requested, actual_range,
                                                  record.units.get(field), message="Requested numeric parameter must be a number."))
                    continue
                low, high = (float(v) for v in actual_range)
                value = float(requested)
                inside = low <= value <= high
                tolerance = tolerances.get(field, 0.0)
                distance = 0.0 if inside else min(abs(value - low), abs(value - high))
                status = "close_match" if inside or (tolerance > 0 and distance <= tolerance) else "no_match"
                details.append(ParameterMatch(
                    field, status, requested, actual_range, record.units.get(field), distance, tolerance,
                    "Request falls within source-reported range." if inside else None,
                ))
                continue
            if actual is None:
                details.append(ParameterMatch(
                    field, "parameter_unavailable", requested, None,
                    record.units.get(field), message="Source record has no value for this parameter.",
                ))
            elif field in CATEGORICAL_FIELDS:
                exact = _categorical_equal(field, requested, actual)
                details.append(ParameterMatch(
                    field, "exact_match" if exact else "no_match", requested, actual,
                    None, message=None if exact else "Categorical values differ.",
                ))
            else:
                if isinstance(requested, bool) or not isinstance(requested, (float, int)):
                    details.append(ParameterMatch(
                        field, "parameter_unavailable", requested, actual,
                        record.units.get(field), message="Requested numeric parameter must be a number.",
                    ))
                    continue
                tolerance = tolerances.get(field, 0.0)
                difference = abs(float(actual) - float(requested))
                status: MatchType = "exact_match" if difference == 0 else (
                    "close_match" if tolerance > 0 and difference <= tolerance else "no_match"
                )
                details.append(ParameterMatch(
                    field, status, requested, actual, record.units.get(field), difference, tolerance,
                    None if status != "no_match" else "Difference exceeds configured absolute tolerance.",
                ))

        statuses = {item.match_type for item in details}
        comparable = statuses - {"parameter_unavailable"}
        if not details or not comparable:
            overall: MatchType = "parameter_unavailable"
        elif "no_match" in comparable:
            overall = "partial_match" if any(s in comparable for s in ("exact_match", "close_match")) else "no_match"
        elif "parameter_unavailable" in statuses:
            overall = "partial_match"
        elif "close_match" in statuses:
            overall = "close_match"
        else:
            overall = "exact_match"
        parameter_results.append(ExperimentMatch(record, overall, tuple(details)))

    rank = {"exact_match": 0, "close_match": 1, "partial_match": 2, "parameter_unavailable": 3, "no_match": 4}
    # Within a match type, measured outcomes outrank proposed conditions.
    status_rank = {"observed": 0, "interpreted": 1, "planned": 2}
    parameter_results.sort(
        key=lambda result: (
            rank[result.match_type],
            status_rank.get(result.record.evidence_status, 3),
            result.record.record_id,
        )
    )
    overall = parameter_results[0].match_type if parameter_results else "no_match"
    return MatchResult(dict(requested_conditions), tolerances, overall, tuple(parameter_results))
