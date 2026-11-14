"""Batch embedding pipeline for JSONL chunk records."""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

from .config import EmbeddingConfig
from .models import EmbeddingFailure, EmbeddingRecord, EmbeddingResult
from .providers import EmbeddingProvider, create_embedding_provider

logger = logging.getLogger(__name__)

REQUIRED_CHUNK_FIELDS = (
    "chunk_id",
    "source_id",
    "content_sha256",
    "experiment",
    "filename",
    "source_path",
    "document_type",
    "evidence_status",
    "source_locator",
    "page_number",
    "text",
    "chunk_index",
    "chunk_count",
    "estimated_token_count",
)


def _validate_chunk(record: dict[str, Any]) -> str:
    missing = [field for field in REQUIRED_CHUNK_FIELDS if field not in record]
    if missing:
        raise ValueError(f"missing required chunk fields: {', '.join(missing)}")
    text = record["text"]
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    if not text.strip():
        raise ValueError("text is empty")
    return text


def _validate_vectors(
    vectors: Sequence[Sequence[float]], expected_count: int, config: EmbeddingConfig
) -> list[tuple[float, ...]]:
    if len(vectors) != expected_count:
        raise ValueError(
            f"provider returned {len(vectors)} vectors for {expected_count} input texts"
        )
    validated: list[tuple[float, ...]] = []
    for vector_index, vector in enumerate(vectors, start=1):
        values = tuple(float(value) for value in vector)
        if len(values) != config.dimension:
            raise ValueError(
                f"provider vector {vector_index} has dimension {len(values)}; "
                f"expected {config.dimension}"
            )
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"provider vector {vector_index} contains non-finite values")
        validated.append(values)
    return validated


def embed_record(
    record: dict[str, Any],
    config: EmbeddingConfig | None = None,
    provider: EmbeddingProvider | None = None,
) -> EmbeddingRecord:
    """Embed a single chunk through the configured provider interface."""

    config = config or EmbeddingConfig()
    provider = provider or create_embedding_provider(config)
    text = _validate_chunk(record)
    vectors = _validate_vectors(provider.embed_batch([text], config), 1, config)
    return EmbeddingRecord(
        chunk_record=dict(record),
        embedding=vectors[0],
        embedding_model=config.model_name,
        embedding_provider=config.provider,
        embedding_dimension=config.dimension,
    )


def _read_jsonl(path: Path) -> tuple[list[tuple[int, dict[str, Any]]], list[EmbeddingFailure]]:
    records: list[tuple[int, dict[str, Any]]] = []
    failures: list[EmbeddingFailure] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                failures.append(
                    EmbeddingFailure(
                        input_file=path.as_posix(),
                        line_number=line_number,
                        error=f"invalid JSON: {exc}",
                    )
                )
                continue
            if not isinstance(record, dict):
                failures.append(
                    EmbeddingFailure(
                        input_file=path.as_posix(),
                        line_number=line_number,
                        error="chunk JSONL line must contain an object",
                        chunk_record={"value": record},
                    )
                )
                continue
            records.append((line_number, record))
    return records, failures


def _atomic_write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
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
            for record in records:
                temporary.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _embed_batch(
    batch: list[tuple[int, dict[str, Any]]],
    *,
    source_path: Path,
    config: EmbeddingConfig,
    provider: EmbeddingProvider,
) -> tuple[list[EmbeddingRecord], list[EmbeddingFailure]]:
    texts = [record["text"] for _, record in batch]
    try:
        vectors = _validate_vectors(provider.embed_batch(texts, config), len(batch), config)
    except Exception as exc:
        # A batch-level provider error is made explicit for each affected chunk.
        message = f"provider batch failed: {type(exc).__name__}: {exc}"
        logger.exception("Embedding batch failed for %s", source_path)
        return [], [
            EmbeddingFailure(source_path.as_posix(), line_number, message, record)
            for line_number, record in batch
        ]

    records = [
        EmbeddingRecord(
            chunk_record=dict(record),
            embedding=vector,
            embedding_model=config.model_name,
            embedding_provider=config.provider,
            embedding_dimension=config.dimension,
        )
        for (_, record), vector in zip(batch, vectors)
    ]
    return records, []


def _provider_output_dir(output_dir: Path, provider_name: str, config: EmbeddingConfig) -> Path:
    if provider_name == "mock":
        return output_dir
    safe_model_name = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in config.model_name
    ).strip("._") or "model"
    return output_dir / provider_name / safe_model_name


def run_embedding(
    config: EmbeddingConfig | None = None,
    provider: EmbeddingProvider | None = None,
    *,
    skip_existing: bool = False,
) -> EmbeddingResult:
    """Embed chunk JSONL in batches and write separate embedding JSONL files.

    Failed records are preserved in a sibling ``.errors.jsonl`` file and in
    the returned result. The default hash provider creates deterministic mock
    vectors, not semantic embeddings. Non-mock providers must be explicitly
    supplied through the provider interface.
    """

    config = config or EmbeddingConfig()
    provider = provider or create_embedding_provider(config)
    provider_name = getattr(provider, "provider_name", None)
    if provider_name != config.provider:
        raise ValueError(
            f"Configured provider {config.provider!r} does not match implementation "
            f"{provider_name!r}; refusing to mislabel embedding records."
        )

    input_dir = config.input_dir.resolve()
    output_dir = config.output_dir.resolve()
    if input_dir == output_dir or input_dir in output_dir.parents:
        raise ValueError("output_dir must not be the input directory or a child of it")

    source_files = sorted(input_dir.rglob("*.jsonl")) if input_dir.is_dir() else []
    input_chunks = 0
    all_records: list[EmbeddingRecord] = []
    all_failures: list[EmbeddingFailure] = []
    processed_paths: list[str] = []
    output_paths: list[str] = []

    for source_file in source_files:
        relative_path = source_file.relative_to(input_dir)
        source_records, failures = _read_jsonl(source_file)
        input_chunks += len(source_records) + len(failures)
        file_records: list[EmbeddingRecord] = []
        file_failures = list(failures)

        valid_records: list[tuple[int, dict[str, Any]]] = []
        for line_number, record in source_records:
            try:
                _validate_chunk(record)
            except Exception as exc:
                file_failures.append(
                    EmbeddingFailure(
                        source_file.as_posix(),
                        line_number,
                        f"invalid chunk: {type(exc).__name__}: {exc}",
                        record,
                    )
                )
            else:
                valid_records.append((line_number, record))

        if skip_existing and valid_records and not file_failures:
            existing_path = _provider_output_dir(output_dir, provider_name, config) / relative_path
            if existing_path.is_file():
                existing_ids = {
                    rec.get("chunk_id")
                    for _, rec in _read_jsonl(existing_path)[0]
                    if isinstance(rec.get("embedding"), list)
                }
                if {rec["chunk_id"] for _, rec in valid_records} <= existing_ids:
                    logger.info("Skipping %s: all chunks already embedded", relative_path)
                    processed_paths.append(relative_path.as_posix())
                    continue

        for offset in range(0, len(valid_records), config.batch_size):
            batch = valid_records[offset : offset + config.batch_size]
            embedded, batch_failures = _embed_batch(
                batch,
                source_path=source_file,
                config=config,
                provider=provider,
            )
            file_records.extend(embedded)
            file_failures.extend(batch_failures)

        # The relative tree and input basename are retained, including spaces.
        # Keep real provider/model output separate from existing mock vectors.
        provider_output_dir = _provider_output_dir(output_dir, provider_name, config)
        output_path = provider_output_dir / relative_path
        _atomic_write_jsonl(output_path, [record.to_dict() for record in file_records])
        output_paths.append(output_path.as_posix())
        if file_failures:
            error_path = output_path.with_name(f"{output_path.stem}.errors.jsonl")
            _atomic_write_jsonl(error_path, [failure.to_dict() for failure in file_failures])
            output_paths.append(error_path.as_posix())
        all_records.extend(file_records)
        all_failures.extend(file_failures)
        processed_paths.append(relative_path.as_posix())
        logger.info(
            "Embedded %s: %d chunk(s) succeeded, %d failed",
            relative_path,
            len(file_records),
            len(file_failures),
        )

    return EmbeddingResult(
        input_chunks=input_chunks,
        records=tuple(all_records),
        failures=tuple(all_failures),
        processed_paths=tuple(processed_paths),
        output_paths=tuple(output_paths),
    )
