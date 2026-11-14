"""Load the locally available FLEX PDF test plan and DOCX metadata description."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from .models import ExperimentRecord
from .normalization import normalize_fuel, units_for_test_plan

TABLE_RE = re.compile(r"\bTable\s+(\d+)\s*:", re.IGNORECASE)
FUEL_RE = re.compile(r"Fuel\s*:\s*(C7H16|CH3OH)", re.IGNORECASE)
ROW_RE = re.compile(r"^\s*(\d+)\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+))(?:\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)))?\s*$")
_NUMBER = r"[+-]?(?:\d+(?:\s*\.\s*\d*)?|\.\s*\d+)"
FLOW_RE = re.compile(rf"Flow\s*:\s*({_NUMBER})\s*cm\s*/\s*s", re.IGNORECASE)
SUPPRESSANT_RE = re.compile(r"Suppressant\s*:\s*([A-Za-z0-9]+)", re.IGNORECASE)
NO_FIBER_RE = re.compile(r"No\s+(?:fi|\ufb01)ber", re.IGNORECASE)
FIBER_RE = re.compile(r"\b(SiC)\s+(?:fi|\ufb01)ber", re.IGNORECASE)


@dataclass(slots=True)
class LoadDiagnostics:
    discovered: int = 0
    rejected: list[str] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_test_plan_pages(
    pages: list[tuple[int, str]],
    *,
    source_file: str,
    content_sha256: str,
) -> tuple[list[ExperimentRecord], LoadDiagnostics]:
    """Parse only numbered rows in the explicitly labeled FLEX Tables 6–12."""

    diagnostics = LoadDiagnostics()
    source_id = f"nasa-{content_sha256}"
    records: list[ExperimentRecord] = []
    table_number: int | None = None
    fuel: str | None = None
    flow: float | None = None
    suppressant: str | None = None
    fiber_support: str | None = None
    fiber_support_source: str | None = None

    for page_number, text in pages:
        for line_number, line in enumerate(text.splitlines(), start=1):
            # PDF text extraction sometimes separates the decimal point from
            # adjacent digits (e.g. ``0 .50``); this repairs layout whitespace
            # only and leaves the numeric token/value unchanged.
            line = re.sub(r"(?<=\d)\s*\.\s*(?=\d)", ".", line)
            table_match = TABLE_RE.search(line)
            if table_match:
                table_number = int(table_match.group(1))
                fuel = flow = suppressant = fiber_support = fiber_support_source = None

            if table_number not in range(6, 13):
                continue

            fuel_match = FUEL_RE.search(line)
            if fuel_match:
                fuel = fuel_match.group(1)
            flow_match = FLOW_RE.search(line)
            if flow_match:
                flow = float(re.sub(r"\s+", "", flow_match.group(1)))
            suppressant_match = SUPPRESSANT_RE.search(line)
            if suppressant_match:
                suppressant = suppressant_match.group(1).upper()
            if NO_FIBER_RE.search(line):
                fiber_support = "none"
                fiber_support_source = "No fiber support"
            else:
                fiber_match = FIBER_RE.search(line)
                if fiber_match:
                    fiber_support = fiber_match.group(1).upper()
                    fiber_support_source = line.strip()

            if not re.match(r"^\s*\d+\b", line):
                continue
            row = ROW_RE.match(line)
            if not row:
                # Page numbers/captions are not rows. A numeric-leading line
                # with multiple values is likely a damaged/unparseable matrix row.
                if len(re.findall(r"\d+(?:\.\d+)?", line)) >= 3:
                    diagnostics.discovered += 1
                    diagnostics.rejected.append(
                        f"page {page_number} line {line_number}: malformed test-matrix row: {line.strip()}"
                    )
                continue

            diagnostics.discovered += 1
            test_number, pressure_raw, oxygen_raw, suppressant_raw = row.groups()
            has_suppressant_concentration = suppressant_raw is not None
            missing_context: list[str] = []
            expected_context = ["fuel", "flow_velocity"]
            if table_number in (7,):
                expected_context.append("fiber_support")
            if table_number in (8, 9, 10, 11, 12):
                expected_context.extend(("suppressant", "fiber_support"))
            for field_name, value in (
                ("fuel", fuel),
                ("flow_velocity", flow),
                ("suppressant", suppressant),
                ("fiber_support", fiber_support),
            ):
                if field_name in expected_context and value is None:
                    missing_context.append(field_name)
                    diagnostics.missing_fields.append(
                        f"page {page_number} line {line_number}: {field_name}"
                    )
            record_id = f"{source_id}:table-{table_number}:test-{test_number}"
            row_flow = flow
            original = {
                "test_number": test_number,
                "fuel": fuel,
                "pressure": pressure_raw,
                "oxygen_concentration": oxygen_raw,
                "suppressant": suppressant,
                "suppressant_concentration": suppressant_raw,
                "flow_velocity": None if row_flow is None else str(row_flow),
                "fiber_support": fiber_support_source,
                "diluent": "N2" if table_number in (6, 7, 8, 9, 10, 11, 12) else None,
            }
            record = ExperimentRecord(
                record_id=record_id,
                kind="test_condition",
                experiment="FLEX",
                experiment_id=test_number,
                evidence_status="planned",
                source_id=source_id,
                source_file=source_file,
                source_locator=f"page:{page_number}:table:{table_number}:line:{line_number}",
                content_sha256=content_sha256,
                table_number=table_number,
                fuel=normalize_fuel(fuel),
                oxygen_concentration=float(oxygen_raw),
                pressure=float(pressure_raw),
                flow_velocity=row_flow,
                suppressant=suppressant,
                suppressant_concentration=(
                    float(suppressant_raw) if suppressant_raw is not None else None
                ),
                diluent="N2" if table_number in (6, 7, 8, 9, 10, 11, 12) else None,
                fiber_support=fiber_support,
                units=units_for_test_plan(
                    has_suppressant_concentration=has_suppressant_concentration
                ),
                original_values=original,
                issues=(
                    "source is a proposed/test-plan matrix; no outcome is reported",
                    *(f"missing source condition: {name}" for name in missing_context),
                ),
            )
            records.append(record)

    return records, diagnostics


def load_pdf(path: Path, *, source_file: str) -> tuple[list[ExperimentRecord], LoadDiagnostics]:
    """Extract source PDF pages with the project's declared pypdf dependency."""

    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - depends on environment setup
        raise RuntimeError("PDF ingestion requires pypdf; install requirements.txt") from exc

    digest = sha256_file(path)
    reader = PdfReader(str(path), strict=False)
    if reader.is_encrypted:
        raise ValueError(f"PDF is encrypted: {path}")
    pages = [(index, page.extract_text() or "") for index, page in enumerate(reader.pages, 1)]
    return parse_test_plan_pages(pages, source_file=source_file, content_sha256=digest)


def load_docx_metadata(path: Path, *, source_file: str) -> ExperimentRecord:
    """Represent the DOCX as metadata only; it contains no measurement rows."""

    try:
        from docx import Document
    except ImportError as exc:  # pragma: no cover - depends on environment setup
        raise RuntimeError("DOCX ingestion requires python-docx; install requirements.txt") from exc

    digest = sha256_file(path)
    document = Document(str(path))
    text = "\n".join(paragraph.text for paragraph in document.paragraphs if paragraph.text.strip())
    properties = document.core_properties
    metadata = {
        key: value
        for key, value in {
            "title": properties.title or None,
            "author": properties.author or None,
            "created": properties.created.isoformat() if properties.created else None,
            "modified": properties.modified.isoformat() if properties.modified else None,
        }.items()
        if value is not None
    }
    return ExperimentRecord(
        record_id=f"nasa-{digest}:document-metadata",
        kind="source_metadata",
        experiment="FLEX",
        experiment_id=None,
        evidence_status="metadata",
        source_id=f"nasa-{digest}",
        source_file=source_file,
        source_locator="document",
        content_sha256=digest,
        source_metadata={
            **metadata,
            "summary": text,
            "described_data_record_labels": [
                "FLEX – Calibration",
                "FLEX - LCTF Calibration",
                "FLEX - Radiometer Calibration",
                "FLEX – HEP",
                "FLEX – MeOH",
            ],
            "described_data_files": ["raw/analyzed .txt", "raw videos", "images/.zip"],
        },
        issues=("metadata describes external FLEX data records; no measurement values included",),
    )


def discover_sources(documents_dir: Path) -> list[Path]:
    """Discover original supported sources, excluding generated/processed files."""

    if not documents_dir.exists():
        return []
    return sorted(
        (path for path in documents_dir.rglob("*") if path.is_file() and path.suffix.casefold() in {".pdf", ".docx"}),
        key=lambda path: path.as_posix().casefold(),
    )
