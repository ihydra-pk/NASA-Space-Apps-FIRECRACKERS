"""Discovery, metadata assignment, normalization, and output serialization."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from .extractors import extract_docx, extract_pdf, extract_pdf_metadata
from .models import DocumentType, EvidenceStatus, IngestionResult, NormalizedDocument

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".md"}
VALID_DOCUMENT_TYPES = {
    "experiment_metadata",
    "test_plan",
    "science_report",
    "publication",
    "data_description",
    "engineering",
    "unknown",
}
VALID_EVIDENCE_STATUSES = {"planned", "observed", "interpreted", "metadata"}


@dataclass(frozen=True, slots=True)
class MetadataOverride:
    """Explicit source labels; missing labels are not inferred from content."""

    document_type: DocumentType = "unknown"
    evidence_status: EvidenceStatus | None = None
    experiment: str | None = None
    source_url: str | None = None
    source_accession: str | None = None


@dataclass(frozen=True, slots=True)
class IngestionConfig:
    """Options for repeatable local document ingestion."""

    documents_dir: Path = Path("data/documents")
    output_dir: Path = Path("data/processed")
    experiment_folders: tuple[str, ...] = ("flex", "bass", "saffire")
    metadata_overrides: Mapping[str, MetadataOverride] = field(default_factory=dict)
    min_pdf_text_chars: int = 40


def load_metadata_overrides(path: Path) -> dict[str, MetadataOverride]:
    """Load source labels from a JSON object keyed by path under documents/."""

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Metadata configuration must be a JSON object")

    overrides: dict[str, MetadataOverride] = {}
    for source_path, values in data.items():
        if not isinstance(values, dict):
            raise ValueError(f"Metadata override for {source_path!r} must be an object")
        document_type = values.get("document_type", "unknown")
        evidence_status = values.get("evidence_status")
        experiment = values.get("experiment")
        source_url = values.get("source_url")
        source_accession = values.get("source_accession")
        if document_type not in VALID_DOCUMENT_TYPES:
            raise ValueError(f"Invalid document_type for {source_path!r}: {document_type!r}")
        if evidence_status is not None and evidence_status not in VALID_EVIDENCE_STATUSES:
            raise ValueError(f"Invalid evidence_status for {source_path!r}: {evidence_status!r}")
        if experiment is not None and (not isinstance(experiment, str) or not experiment.strip()):
            raise ValueError(f"experiment for {source_path!r} must be a non-empty string")
        if source_url is not None and (not isinstance(source_url, str) or not source_url.strip()):
            raise ValueError(f"source_url for {source_path!r} must be a non-empty string or null")
        if source_accession is not None and (
            not isinstance(source_accession, str) or not source_accession.strip()
        ):
            raise ValueError(
                f"source_accession for {source_path!r} must be a non-empty string or null"
            )

        normalized_path = Path(source_path).as_posix().lstrip("./")
        overrides[normalized_path] = MetadataOverride(
            document_type=document_type,
            evidence_status=evidence_status,
            experiment=experiment,
            source_url=source_url,
            source_accession=source_accession,
        )
    return overrides


def _content_sha256(path: Path) -> str:
    """Calculate a stable SHA-256 identity directly from source bytes."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _get_override(config: IngestionConfig, source_path: str) -> MetadataOverride:
    return config.metadata_overrides.get(source_path, MetadataOverride())


def ingest_file(path: Path, config: IngestionConfig) -> list[NormalizedDocument]:
    """Extract and normalize one supported source file without writing output."""

    path = path.resolve()
    documents_dir = config.documents_dir.resolve()
    try:
        relative_path = path.relative_to(documents_dir).as_posix()
    except ValueError as exc:
        raise ValueError(f"Source file is outside documents directory: {path}") from exc

    parts = Path(relative_path).parts
    if not parts or parts[0].casefold() not in {name.casefold() for name in config.experiment_folders}:
        raise ValueError(f"Source must be under one of {config.experiment_folders}: {relative_path}")

    suffix = path.suffix.casefold()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported document format: {path.suffix}")

    relative_path = Path(relative_path).as_posix()
    override = _get_override(config, relative_path)
    experiment = override.experiment or parts[0].upper()
    content_sha256 = _content_sha256(path)
    common = {
        "source_id": f"nasa-{content_sha256}",
        "content_sha256": content_sha256,
        "experiment": experiment,
        "filename": path.name,
        "file_type": suffix[1:],
        "source_path": relative_path,
        "document_type": override.document_type,
        "evidence_status": override.evidence_status,
        "source_url": override.source_url,
        "source_accession": override.source_accession,
    }

    if suffix == ".pdf":
        pages = extract_pdf(path, min_text_chars=config.min_pdf_text_chars)
        from pypdf import PdfReader

        document_metadata = extract_pdf_metadata(PdfReader(str(path), strict=False))
        return [
            NormalizedDocument(
                **common,
                text=page.text,
                source_locator=f"page:{page.page_number}",
                page_number=page.page_number,
                document_metadata=document_metadata,
                extraction_warnings=list(page.warnings),
            )
            for page in pages
        ]

    if suffix == ".md":
        return [
            NormalizedDocument(
                **common,
                text=path.read_text(encoding="utf-8").strip(),
                source_locator="document",
            )
        ]

    document = extract_docx(path)
    return [
        NormalizedDocument(
            **common,
            text=document.text,
            source_locator="document",
            document_metadata=document.metadata,
        )
    ]


def discover_documents(config: IngestionConfig) -> list[Path]:
    """Recursively discover PDFs and DOCX files under supported experiment folders."""

    documents_dir = config.documents_dir.resolve()
    candidates: list[Path] = []
    for experiment in config.experiment_folders:
        experiment_dir = documents_dir / experiment
        if not experiment_dir.is_dir():
            logger.info("Source directory does not exist yet; skipping: %s", experiment_dir)
            continue
        candidates.extend(
            path
            for path in experiment_dir.rglob("*")
            if path.is_file() and path.suffix.casefold() in SUPPORTED_SUFFIXES
        )
    return sorted(candidates, key=lambda item: item.relative_to(documents_dir).as_posix().casefold())


def _write_records(output_path: Path, records: list[NormalizedDocument]) -> None:
    """Atomically write one UTF-8 JSONL file per original source document."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            for record in records:
                temporary.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def ingest_documents(config: IngestionConfig | None = None) -> IngestionResult:
    """Discover, extract, and save supported documents below the output directory.

    Existing NASA source files are only read. Each normalized output is written
    atomically to the matching relative path under ``output_dir`` with a
    ``.jsonl`` suffix. Failed documents are logged and reported while remaining
    sources continue to be processed.
    """

    config = config or IngestionConfig()
    if config.min_pdf_text_chars < 0:
        raise ValueError("min_pdf_text_chars must be zero or greater")

    source_paths = discover_documents(config)
    all_records: list[NormalizedDocument] = []
    processed_paths: list[str] = []
    errors: list[str] = []

    for source in source_paths:
        relative_path = source.relative_to(config.documents_dir.resolve()).as_posix()
        try:
            records = ingest_file(source, config)
            output_path = config.output_dir / Path(relative_path).with_suffix(".jsonl")
            _write_records(output_path, records)
            all_records.extend(records)
            processed_paths.append(relative_path)
            logger.info("Ingested %s into %s (%d record(s))", relative_path, output_path, len(records))
        except Exception as exc:
            message = f"{relative_path}: {type(exc).__name__}: {exc}"
            errors.append(message)
            logger.exception("Could not ingest source document %s", source)

    return IngestionResult(
        source_documents=len(processed_paths),
        records=tuple(all_records),
        processed_paths=tuple(processed_paths),
        errors=tuple(errors),
    )
