"""End-to-end deterministic ingestion service and diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import ExperimentalDataConfig
from .loaders import discover_sources, load_docx_metadata, load_pdf
from .matching import MatchResult, match_experiments
from .models import ExperimentRecord
from .store import ExperimentStore
from .validation import validate_record


@dataclass(frozen=True, slots=True)
class IngestionReport:
    records_discovered: int
    records_normalized: int
    records_rejected: int
    source_files_discovered: int
    source_files_processed: tuple[str, ...]
    rejected: tuple[str, ...] = ()
    missing_or_invalid_fields: tuple[str, ...] = ()
    source_errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ExperimentalDataService:
    def __init__(self, config: ExperimentalDataConfig | None = None):
        self.config = config or ExperimentalDataConfig()

    def ingest(self) -> IngestionReport:
        documents_dir = self.config.resolve(self.config.documents_dir).resolve()
        store_path = self.config.resolve(self.config.store_path).resolve()
        sources = discover_sources(documents_dir)
        valid: list[ExperimentRecord] = []
        processed: list[str] = []
        rejected: list[str] = []
        issues: list[str] = []
        errors: list[str] = []
        discovered_count = 0

        for source in sources:
            relative = source.relative_to(self.config.project_root.resolve()).as_posix()
            try:
                if source.suffix.casefold() == ".pdf":
                    records, diagnostics = load_pdf(source, source_file=relative)
                    discovered_count += diagnostics.discovered
                    rejected.extend(diagnostics.rejected)
                    issues.extend(diagnostics.missing_fields)
                else:
                    records = [load_docx_metadata(source, source_file=relative)]
                    discovered_count += 1

                accepted_for_source: list[ExperimentRecord] = []
                for record in records:
                    if record.issues:
                        issues.extend(
                            f"{relative} [{record.source_locator}]: {issue}"
                            for issue in record.issues
                        )
                    validation_issues = validate_record(record)
                    if validation_issues:
                        rejected.append(f"{relative} [{record.source_locator}]: {'; '.join(validation_issues)}")
                        issues.extend(validation_issues)
                    else:
                        accepted_for_source.append(record)
                valid.extend(accepted_for_source)
                processed.append(relative)
            except Exception as exc:
                errors.append(f"{relative}: {type(exc).__name__}: {exc}")

        # A failed source is not deleted from a prior successful store run.
        if processed:
            ExperimentStore(store_path).replace_sources(set(processed), valid)
        return IngestionReport(
            records_discovered=discovered_count,
            records_normalized=len(valid),
            records_rejected=len(rejected),
            source_files_discovered=len(sources),
            source_files_processed=tuple(processed),
            rejected=tuple(rejected),
            missing_or_invalid_fields=tuple(issues),
            source_errors=tuple(errors),
        )

    def records(self) -> list[ExperimentRecord]:
        path = self.config.resolve(self.config.store_path)
        return ExperimentStore(path).all_records()

    def match_conditions(self, conditions: dict[str, object]) -> MatchResult:
        """Compare requested conditions against stored plan rows transparently."""

        return match_experiments(
            self.records(),
            conditions,
            absolute_tolerances=self.config.absolute_tolerances,
        )
