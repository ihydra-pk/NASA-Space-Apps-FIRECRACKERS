"""Command line entry point: python -m src.experimental_data [--json]."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .config import ExperimentalDataConfig
from .service import ExperimentalDataService


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest local NASA FLEX experimental source files.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable report JSON.")
    parser.add_argument(
        "--database",
        help="SQLite output path (default: data/experimental_data/experiments.sqlite3).",
    )
    args = parser.parse_args()
    config = ExperimentalDataConfig()
    if args.database:
        config = replace(config, store_path=Path(args.database))
    service = ExperimentalDataService(config)
    report = service.ingest()
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(f"Sources processed: {len(report.source_files_processed)}/{report.source_files_discovered}")
        print(f"Records discovered/normalized/rejected: {report.records_discovered}/{report.records_normalized}/{report.records_rejected}")
        for error in report.source_errors:
            print(f"ERROR: {error}")
        for issue in report.missing_or_invalid_fields:
            print(f"FIELD: {issue}")
        for record in service.records():
            if record.kind == "test_condition":
                continue
            print(f"METADATA: {record.source_file} [{record.source_locator}]")


if __name__ == "__main__":
    main()
