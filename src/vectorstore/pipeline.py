"""Indexing pipeline for provider/model-specific embedding JSONL files."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

from .config import VectorStoreConfig
from .models import IndexResult
from .sqlite_store import SQLiteVectorStore
from .store import VectorStore

logger = logging.getLogger(__name__)

REQUIRED_EMBEDDING_FIELDS = (
    "chunk_id",
    "source_id",
    "content_sha256",
    "experiment",
    "filename",
    "file_type",
    "source_path",
    "document_type",
    "evidence_status",
    "source_locator",
    "page_number",
    "text",
    "chunk_index",
    "chunk_count",
    "estimated_token_count",
    "document_metadata",
    "embedding",
    "embedding_model",
    "embedding_provider",
    "embedding_dimension",
)


def validate_embedding_record(
    record: dict[str, Any], config: VectorStoreConfig
) -> tuple[float, ...]:
    missing = [field for field in REQUIRED_EMBEDDING_FIELDS if field not in record]
    if missing:
        raise ValueError(f"missing required embedding fields: {', '.join(missing)}")
    if not isinstance(record["chunk_id"], str) or not record["chunk_id"].strip():
        raise ValueError("chunk_id must be a non-empty string")
    if not isinstance(record["text"], str):
        raise ValueError("text must be a string")
    if not isinstance(record["document_metadata"], dict):
        raise ValueError("document_metadata must be an object")
    if record["embedding_provider"] != config.provider:
        raise ValueError(
            f"embedding provider {record['embedding_provider']!r} does not match "
            f"collection provider {config.provider!r}"
        )
    if record["embedding_model"] != config.model_name:
        raise ValueError(
            f"embedding model {record['embedding_model']!r} does not match "
            f"collection model {config.model_name!r}"
        )
    if record["embedding_dimension"] != config.dimension:
        raise ValueError(
            f"embedding record dimension {record['embedding_dimension']!r} does not match "
            f"collection dimension {config.dimension}"
        )
    raw_vector = record["embedding"]
    if not isinstance(raw_vector, list):
        raise ValueError("embedding must be a numeric array")
    if len(raw_vector) != config.dimension:
        raise ValueError(
            f"embedding has dimension {len(raw_vector)}; expected {config.dimension}"
        )
    vector: list[float] = []
    for index, value in enumerate(raw_vector):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"embedding value at index {index} is not numeric")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"embedding value at index {index} is not finite")
        vector.append(number)
    return tuple(vector)


def _read_records(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                yield line_number, None, f"invalid JSON: {exc}"
                continue
            if not isinstance(value, dict):
                yield line_number, None, "JSONL record must be an object"
                continue
            yield line_number, value, None


def index_embeddings(
    input_dir: Path,
    config: VectorStoreConfig,
    store: VectorStore | None = None,
) -> IndexResult:
    """Index embedding JSONL only; repeated identical records are idempotent.

    A duplicate chunk ID in a single indexing operation is rejected and
    reported, rather than silently choosing between conflicting records.
    Re-indexing a chunk in a later call upserts that deterministic ID.
    """

    input_dir = input_dir.resolve()
    files = sorted(input_dir.rglob("*.jsonl")) if input_dir.is_dir() else []
    owned_store = store is None
    store = store or SQLiteVectorStore(config)
    input_count = indexed = upserted = duplicates = rejected = 0
    failures: list[str] = []
    seen_ids: set[str] = set()
    try:
        for path in files:
            for line_number, record, parse_error in _read_records(path):
                input_count += 1
                location = f"{path.as_posix()}:{line_number}"
                if parse_error:
                    rejected += 1
                    failures.append(f"{location}: {parse_error}")
                    continue
                assert record is not None
                try:
                    vector = validate_embedding_record(record, config)
                    chunk_id = record["chunk_id"]
                    if chunk_id in seen_ids:
                        duplicates += 1
                        failures.append(f"{location}: duplicate chunk_id {chunk_id!r} in input")
                        continue
                    seen_ids.add(chunk_id)
                    is_new = store.upsert(record, vector)
                    if is_new:
                        indexed += 1
                    else:
                        upserted += 1
                except (ValueError, TypeError, OverflowError) as exc:
                    rejected += 1
                    failures.append(f"{location}: {exc}")
        if failures:
            for failure in failures:
                logger.error("Indexing record rejected: %s", failure)
        return IndexResult(input_count, indexed, upserted, duplicates, rejected, tuple(failures))
    finally:
        if owned_store:
            store.close()
