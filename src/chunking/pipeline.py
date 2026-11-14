"""Chunk normalized JSONL records and write provenance-preserving JSONL."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import ChunkRecord
from .text import estimated_token_count, partition_text

logger = logging.getLogger(__name__)

REQUIRED_SOURCE_FIELDS = (
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
)


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Chunk limits, expressed in estimated tokens rather than model tokens."""

    input_dir: Path = Path("data/processed")
    output_dir: Path = Path("data/chunks")
    max_chunk_tokens: int = 800
    overlap_tokens: int = 120

    def __post_init__(self) -> None:
        if self.max_chunk_tokens <= 0:
            raise ValueError("max_chunk_tokens must be greater than zero")
        if self.overlap_tokens < 0 or self.overlap_tokens >= self.max_chunk_tokens:
            raise ValueError("overlap_tokens must be non-negative and less than max_chunk_tokens")


@dataclass(frozen=True, slots=True)
class ChunkingResult:
    input_records: int
    chunks: tuple[ChunkRecord, ...]
    processed_paths: tuple[str, ...]
    zero_chunk_records: tuple[str, ...]
    errors: tuple[str, ...] = ()


def _chunk_id(
    record: dict[str, Any],
    *,
    chunk_index: int,
    text: str,
    config: ChunkingConfig,
) -> str:
    identity = {
        "version": 1,
        "source_id": record["source_id"],
        "content_sha256": record["content_sha256"],
        "source_locator": record["source_locator"],
        "page_number": record["page_number"],
        "chunk_index": chunk_index,
        "max_chunk_tokens": config.max_chunk_tokens,
        "overlap_tokens": config.overlap_tokens,
        "chunk_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"chunk-{hashlib.sha256(encoded).hexdigest()}"


def chunk_record(record: dict[str, Any], config: ChunkingConfig | None = None) -> list[ChunkRecord]:
    """Chunk one normalized record while retaining all of its original fields."""

    config = config or ChunkingConfig()
    missing = [field for field in REQUIRED_SOURCE_FIELDS if field not in record]
    if missing:
        raise ValueError(f"Normalized record is missing required fields: {', '.join(missing)}")
    if not isinstance(record["text"], str):
        raise ValueError("Normalized record text must be a string")

    texts = partition_text(
        record["text"],
        max_tokens=config.max_chunk_tokens,
        overlap_tokens=config.overlap_tokens,
    )
    chunk_count = len(texts)
    return [
        ChunkRecord(
            source_record=dict(record),
            chunk_id=_chunk_id(record, chunk_index=index, text=text, config=config),
            chunk_index=index,
            chunk_count=chunk_count,
            text=text,
            estimated_token_count=estimated_token_count(text),
        )
        for index, text in enumerate(texts, start=1)
    ]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"line {line_number} must contain a JSON object")
            records.append(record)
    return records


def _write_jsonl(path: Path, chunks: list[ChunkRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            for chunk in chunks:
                temporary.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def run_chunking(config: ChunkingConfig | None = None) -> ChunkingResult:
    """Chunk all normalized JSONL under ``input_dir`` into ``output_dir``."""

    config = config or ChunkingConfig()
    input_dir = config.input_dir.resolve()
    output_dir = config.output_dir.resolve()
    if input_dir == output_dir or input_dir in output_dir.parents:
        raise ValueError("output_dir must not be the input directory or a child of it")

    source_files = sorted(input_dir.rglob("*.jsonl")) if input_dir.is_dir() else []
    all_chunks: list[ChunkRecord] = []
    processed_paths: list[str] = []
    zero_chunk_records: list[str] = []
    errors: list[str] = []
    input_records = 0

    for source_file in source_files:
        relative_path = source_file.relative_to(input_dir)
        try:
            records = _read_jsonl(source_file)
            file_chunks: list[ChunkRecord] = []
            for record_index, record in enumerate(records, start=1):
                input_records += 1
                try:
                    chunks = chunk_record(record, config)
                except Exception as exc:
                    raise ValueError(f"record {record_index}: {exc}") from exc
                if not chunks:
                    zero_chunk_records.append(
                        f"{relative_path.as_posix()} record {record_index} "
                        f"({record.get('source_path', 'unknown')} {record.get('source_locator', '')})"
                    )
                file_chunks.extend(chunks)

            # Preserve relative folders and source stem; spaces in the output
            # basename are changed to underscores for convenient CLI use.
            output_relative = relative_path.with_name(
                f"{relative_path.stem.replace(' ', '_')}.jsonl"
            )
            output_path = output_dir / output_relative
            _write_jsonl(output_path, file_chunks)
            all_chunks.extend(file_chunks)
            processed_paths.append(relative_path.as_posix())
            logger.info(
                "Chunked %s: %d source record(s) -> %d chunk(s)",
                relative_path,
                len(records),
                len(file_chunks),
            )
        except Exception as exc:
            message = f"{relative_path.as_posix()}: {type(exc).__name__}: {exc}"
            errors.append(message)
            logger.exception("Could not chunk normalized file %s", source_file)

    return ChunkingResult(
        input_records=input_records,
        chunks=tuple(all_chunks),
        processed_paths=tuple(processed_paths),
        zero_chunk_records=tuple(zero_chunk_records),
        errors=tuple(errors),
    )
