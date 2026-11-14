"""Structured records reflecting fields actually present in FLEX project data."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

EvidenceStatus = Literal["planned", "observed", "interpreted", "metadata", "unknown"]
RecordKind = Literal["test_condition", "source_metadata"]


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    """One explicitly represented test-plan condition or source metadata record.

    Values remain in the units and representation indicated by the source. The
    absent result fields are intentional: the local sources contain no FLEX
    test-by-test observation/result dataset.
    """

    record_id: str
    kind: RecordKind
    experiment: str | None
    experiment_id: str | None
    evidence_status: EvidenceStatus
    source_id: str
    source_file: str
    source_locator: str
    content_sha256: str
    source_url: str | None = None
    table_number: int | None = None
    fuel: str | None = None
    oxygen_concentration: float | None = None
    pressure: float | None = None
    flow_velocity: float | None = None
    suppressant: str | None = None
    suppressant_concentration: float | None = None
    diluent: str | None = None
    fiber_support: str | None = None
    observed_result: str | None = None
    measurement: dict[str, object] | None = None
    units: dict[str, str | None] = field(default_factory=dict)
    original_values: dict[str, object] = field(default_factory=dict)
    source_metadata: dict[str, object] = field(default_factory=dict)
    issues: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-serializable representation."""

        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "ExperimentRecord":
        """Rebuild a record loaded from persistent JSON storage."""

        normalized = dict(value)
        normalized["units"] = dict(normalized.get("units") or {})
        normalized["original_values"] = dict(normalized.get("original_values") or {})
        normalized["source_metadata"] = dict(normalized.get("source_metadata") or {})
        normalized["issues"] = tuple(normalized.get("issues") or ())
        return cls(**normalized)  # type: ignore[arg-type]
