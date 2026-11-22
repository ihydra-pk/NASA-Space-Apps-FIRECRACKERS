"""SQLite persistence for normalized experiment records."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .models import ExperimentRecord


class ExperimentStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS experiment_records ("
                "record_id TEXT PRIMARY KEY, source_id TEXT NOT NULL, "
                "source_file TEXT NOT NULL, evidence_status TEXT NOT NULL, "
                "payload_json TEXT NOT NULL)"
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_experiment_source ON experiment_records(source_id)")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def replace_sources(self, source_files: set[str], records: list[ExperimentRecord]) -> None:
        """Replace only successfully processed sources; retries are idempotent."""

        with self._connect() as connection:
            if source_files:
                placeholders = ",".join("?" for _ in source_files)
                connection.execute(
                    f"DELETE FROM experiment_records WHERE source_file IN ({placeholders})",
                    tuple(sorted(source_files)),
                )
            connection.executemany(
                "INSERT OR REPLACE INTO experiment_records "
                "(record_id, source_id, source_file, evidence_status, payload_json) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        record.record_id,
                        record.source_id,
                        record.source_file,
                        record.evidence_status,
                        json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True),
                    )
                    for record in records
                ],
            )

    def all_records(self) -> list[ExperimentRecord]:
        with self._connect() as connection:
            rows = connection.execute("SELECT payload_json FROM experiment_records ORDER BY record_id").fetchall()
        return [ExperimentRecord.from_dict(json.loads(row["payload_json"])) for row in rows]
