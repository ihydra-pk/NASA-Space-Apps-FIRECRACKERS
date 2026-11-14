"""Validation without filling missing source fields or changing evidence status."""

from __future__ import annotations

import math

from .models import ExperimentRecord


def validate_record(record: ExperimentRecord) -> tuple[str, ...]:
    issues: list[str] = []
    for name in ("record_id", "source_id", "source_file", "source_locator", "content_sha256"):
        if not getattr(record, name):
            issues.append(f"missing required provenance field: {name}")
    if record.kind == "test_condition" and not record.experiment_id:
        issues.append("missing experiment_id (test number)")
    for name in ("oxygen_concentration", "pressure", "flow_velocity", "suppressant_concentration"):
        value = getattr(record, name)
        if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)):
            issues.append(f"invalid numeric field: {name}")
    if record.evidence_status == "observed" and record.kind == "test_condition" and not (
        record.observed_result or record.measurement
    ):
        issues.append("observed record has no observation/result")
    return tuple(issues)
