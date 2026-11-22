"""Parsers for NASA PSI experiment tables (CSV) into ExperimentRecords.

Each parser is keyed by the PSI accession in the file name (``PSI-69_...csv``).
Values are kept as the source states them: unparseable or ranged values are
left null (and preserved in ``original_values``) rather than guessed.
"""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import Callable

from .loaders import LoadDiagnostics, sha256_file
from .models import EvidenceStatus, ExperimentRecord

PSI_PAGE = "https://psi.nasa.gov/physci/repo/data/investigations/"
MMHG_PER_ATM = 760.0
_NUMBER_RE = re.compile(r"^\s*(~)?\s*([+-]?\d+(?:\.\d+)?)\s*(?:[A-Za-z/%]+)?\s*$")


def read_csv_rows(path: Path) -> list[list[str]]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
    return [[cell.strip() for cell in row] for row in csv.reader(io.StringIO(text))]


def parse_number(value: str | None) -> tuple[float | None, bool]:
    """Return (number, approximate). Ranges, notes, and 'n/a' give (None, False)."""

    if value is None:
        return None, False
    match = _NUMBER_RE.match(value)
    if not match:
        return None, False
    return float(match.group(2)), bool(match.group(1))


def _fraction(value: float | None) -> float | None:
    if value is None:
        return None
    return value / 100 if value > 1 else value


def _clean(row: dict[str, str]) -> dict[str, object]:
    return {key: value for key, value in row.items() if key and value}


def _record(
    *,
    accession: str,
    experiment: str,
    experiment_id: str,
    line_number: int,
    status: EvidenceStatus,
    source_file: str,
    digest: str,
    original: dict[str, object],
    measurement: dict[str, object] | None = None,
    observed_result: str | None = None,
    issues: tuple[str, ...] = (),
    **conditions: object,
) -> ExperimentRecord:
    units = {
        "oxygen_concentration": "mole_fraction",
        "pressure": "atm",
        "flow_velocity": "cm/s",
    }
    return ExperimentRecord(
        record_id=f"nasa-{digest}:row-{line_number}",
        kind="test_condition",
        experiment=experiment,
        experiment_id=experiment_id,
        evidence_status=status,
        source_id=f"nasa-{digest}",
        source_file=source_file,
        source_locator=f"row:{line_number}",
        content_sha256=digest,
        source_url=PSI_PAGE + accession,
        measurement=measurement or None,
        observed_result=observed_result,
        units={key: value for key, value in units.items() if conditions.get(key) is not None},
        original_values=original,
        issues=issues,
        **conditions,  # type: ignore[arg-type]
    )


Parser = Callable[[list[list[str]], str, str], tuple[list[ExperimentRecord], LoadDiagnostics]]


def parse_flex(rows, source_file, digest):
    """PSI-69: one observed droplet-combustion test per row."""

    diagnostics = LoadDiagnostics()
    records = []
    header = rows[0]
    for line_number, cells in enumerate(rows[1:], start=2):
        diagnostics.discovered += 1
        row = dict(zip(header, cells))
        test_id = cells[0] if cells else ""
        if not test_id.isdigit() or len(cells) != len(header):
            diagnostics.rejected.append(f"row {line_number}: not a FLEX test row")
            continue
        pressure_mmhg, _ = parse_number(cells[5])
        oxygen, _ = parse_number(cells[6])
        co2, _ = parse_number(cells[8])
        helium, _ = parse_number(cells[9])
        suppressants = [(name, v) for name, v in (("CO2", co2), ("He", helium)) if v]
        issues: list[str] = []
        suppressant = concentration = None
        if len(suppressants) == 1:
            suppressant, concentration = suppressants[0]
        elif len(suppressants) > 1:
            suppressant = "+".join(name for name, _ in suppressants)
            issues.append("multiple suppressants; concentration left null")
        if pressure_mmhg is None or oxygen is None:
            diagnostics.missing_fields.append(f"row {line_number}: pressure/oxygen missing")
        measurement = {
            key: value
            for key, value in {
                "droplet_initial_diameter_mm": parse_number(cells[10])[0],
                "visible_flame_extinction_diameter_mm": parse_number(cells[11])[0],
                "burning_rate_mm": parse_number(cells[12])[0],
                "burn_time_s": parse_number(cells[13])[0],
            }.items()
            if value is not None
        }
        records.append(
            _record(
                accession="PSI-69", experiment="FLEX", experiment_id=test_id,
                line_number=line_number,
                status="observed", source_file=source_file, digest=digest,
                original={**_clean(row), "pressure_unit": "mmHg (converted to atm)"},
                measurement=measurement, observed_result=cells[14] or None,
                issues=tuple(issues),
                fuel=cells[4].lower() or None,
                oxygen_concentration=oxygen,
                pressure=None if pressure_mmhg is None else pressure_mmhg / MMHG_PER_ATM,
                diluent="N2",
                suppressant=suppressant,
                suppressant_concentration=concentration,
            )
        )
    return records, diagnostics


def _parse_bass_like(accession: str, experiment: str, id_column: str):
    def parse(rows, source_file, digest):
        diagnostics = LoadDiagnostics()
        records = []
        header = rows[0]
        index = {name.strip().lower(): i for i, name in enumerate(header)}
        id_index = next((i for name, i in index.items() if name == id_column), None)
        material_index = next((i for name, i in index.items() if name.startswith("fuel sample")), None)
        oxygen_index = next((i for name, i in index.items() if "initial o2" in name), None)
        if id_index is None or material_index is None:
            raise ValueError(f"{source_file}: expected columns not found in header {header}")
        for line_number, cells in enumerate(rows[1:], start=2):
            diagnostics.discovered += 1
            if len(cells) != len(header) or not cells[id_index]:
                diagnostics.rejected.append(f"row {line_number}: malformed")
                continue
            row = dict(zip(header, cells))
            oxygen = None
            if oxygen_index is not None:
                oxygen = _fraction(parse_number(cells[oxygen_index])[0])
            measurement = {}
            for name, i in index.items():
                if any(key in name for key in ("initial co", "final", "total frames")) and cells[i]:
                    number, _ = parse_number(cells[i])
                    measurement[name.strip()] = number if number is not None else cells[i]
            records.append(
                _record(
                    accession=accession, experiment=experiment, experiment_id=cells[id_index],
                    line_number=line_number,
                    status="observed" if measurement else "metadata",
                    source_file=source_file, digest=digest, original=_clean(row),
                    measurement=measurement,
                    issues=() if measurement else ("test log only; no outcome values in source table",),
                    fuel=cells[material_index] or None,
                    oxygen_concentration=oxygen,
                )
            )
        return records, diagnostics

    return parse


def parse_flex2(rows, source_file, digest):
    """PSI-68 FLEX-2 design matrix; rows are planned ranges, not test results."""

    diagnostics = LoadDiagnostics()
    records = []
    if not rows:
        return records, diagnostics
    header = rows[0]
    for line_number, cells in enumerate(rows[1:], start=2):
        diagnostics.discovered += 1
        if len(cells) != len(header):
            diagnostics.rejected.append(f"row {line_number}: malformed FLEX-2 matrix row")
            continue
        row = _clean(dict(zip(header, cells)))
        fuel = cells[0].strip() if cells else ""
        if not fuel:
            diagnostics.rejected.append(f"row {line_number}: missing sample/fuel type")
            continue
        records.append(_record(
            accession="PSI-68", experiment="FLEX-2", experiment_id=fuel,
            line_number=line_number, status="planned", source_file=source_file,
            digest=digest, original=row,
            measurement={"design_conditions": {key: value for key, value in row.items() if key != header[0]}},
            fuel=fuel,
            issues=("FLEX-2 design matrix gives ranges/alternatives, not individual tests or observed results",),
        ))
    return records, diagnostics


def _parse_saffire(accession: str, experiment: str):
    def parse(rows, source_file, digest):
        diagnostics = LoadDiagnostics()
        records = []
        header = rows[0]
        names = [name.strip().lower() for name in header]

        def column(*keys):
            return next((i for i, name in enumerate(names) if all(k in name for k in keys)), None)

        material_i, flow_i, oxygen_i = column("material"), column("air flow"), column("percent o2")
        for line_number, cells in enumerate(rows[1:], start=2):
            sample = cells[0] if cells else ""
            if not re.match(r"^(S\d+|\d+-\d+)$", sample):
                continue  # camera/flow-visualization and note rows
            diagnostics.discovered += 1
            row = dict(zip(header, cells + [""] * (len(header) - len(cells))))
            issues: list[str] = []
            oxygen, approximate = parse_number(cells[oxygen_i]) if oxygen_i is not None else (None, False)
            oxygen_range = None
            if oxygen is None and oxygen_i is not None and cells[oxygen_i]:
                bounds = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*", cells[oxygen_i])
                if bounds:
                    low, high = float(bounds.group(1)), float(bounds.group(2))
                    oxygen_range = [min(low, high) / 100, max(low, high) / 100]
            if oxygen is None and oxygen_i is not None and cells[oxygen_i]:
                if oxygen_range:
                    issues.append(f"oxygen value {cells[oxygen_i]!r} preserved as a range")
                else:
                    issues.append(f"oxygen value {cells[oxygen_i]!r} is a range/note; left null")
            elif approximate:
                issues.append("oxygen concentration is approximate (~) in source")
            flow = parse_number(cells[flow_i])[0] if flow_i is not None else None
            measurement = {
                header[i].strip(): cells[i]
                for i, name in enumerate(names)
                if i < len(cells) and cells[i] and ("burn" in name or "spread" in name or "direction" in name)
            }
            has_outcome = any("direction" not in key.lower() for key in measurement)
            material = cells[material_i] if material_i is not None and cells[material_i] else None
            if material is None:
                issues.append("material blank in source table")
            records.append(
                _record(
                    accession=accession, experiment=experiment, experiment_id=sample,
                    line_number=line_number,
                    status="observed" if has_outcome else "metadata",
                    source_file=source_file, digest=digest,
                    original={**_clean(row), **({"oxygen_concentration_range": oxygen_range} if oxygen_range else {})},
                    measurement=measurement,
                    issues=tuple(issues) + (() if has_outcome else ("sample list only; no outcome values",)),
                    fuel=material,
                    oxygen_concentration=_fraction(oxygen),
                    flow_velocity=flow,
                )
            )
        return records, diagnostics

    return parse


PARSERS: dict[str, Parser] = {
    "PSI-68": parse_flex2,
    "PSI-69": parse_flex,
    "PSI-25": _parse_bass_like("PSI-25", "BASS-II", "test #"),
    "PSI-26": _parse_bass_like("PSI-26", "BASS", "test #"),
    "PSI-98": _parse_saffire("PSI-98", "SAFFIRE-I"),
    "PSI-99": _parse_saffire("PSI-99", "SAFFIRE-II"),
    "PSI-100": _parse_saffire("PSI-100", "SAFFIRE-III"),
}


def accession_of(path: Path) -> str | None:
    match = re.match(r"^(PSI-\d+)_", path.name)
    return match.group(1) if match else None


def load_table(path: Path, *, source_file: str) -> tuple[list[ExperimentRecord], LoadDiagnostics]:
    accession = accession_of(path)
    parser = PARSERS.get(accession or "")
    if parser is None:
        raise ValueError(f"no table parser registered for {path.name!r}")
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError("empty table")
    return parser(rows, source_file, sha256_file(path))


def discover_tables(structured_dir: Path) -> list[Path]:
    if not structured_dir.exists():
        return []
    return sorted(structured_dir.rglob("*.csv"), key=lambda p: p.as_posix().casefold())
