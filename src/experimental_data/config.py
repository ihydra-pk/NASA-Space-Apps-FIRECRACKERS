"""Configuration for structured source ingestion and matching."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ExperimentalDataConfig:
    project_root: Path = Path(".")
    documents_dir: Path = Path("data/documents")
    structured_dir: Path = Path("data/structured")
    psi_download_dir: Path = Path("data/psi-download")
    store_path: Path = Path("data/experimental_data/experiments.sqlite3")
    # These are software defaults, not uncertainties stated by the source.
    absolute_tolerances: dict[str, float] = field(
        default_factory=lambda: {
            "oxygen_concentration": 0.005,
            "pressure": 0.01,
            "flow_velocity": 0.1,
            "suppressant_concentration": 0.005,
        }
    )

    def __post_init__(self) -> None:
        for parameter, tolerance in self.absolute_tolerances.items():
            if not math.isfinite(tolerance) or tolerance < 0:
                raise ValueError(f"Tolerance for {parameter!r} must be finite and non-negative")

    def resolve(self, path: Path) -> Path:
        return path if path.is_absolute() else self.project_root / path
