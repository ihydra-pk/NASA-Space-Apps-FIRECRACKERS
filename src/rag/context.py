"""Validation, deduplication, and size-bounded RAG context construction."""

from __future__ import annotations

import math
import re
from typing import Any, Iterable

from src.vectorstore.models import RetrievalResult, SearchResult

from .config import RAGContextConfig
from .formatting import format_context
from .models import EvidenceItem, EvidenceStatus, RAGContext

_EVIDENCE_STATUSES = {"planned", "observed", "interpreted", "metadata", "unknown"}
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_REQUIRED_FIELDS = (
    "rank",
    "similarity_score",
    "chunk_id",
    "source_id",
    "content_sha256",
    "experiment",
    "filename",
    "source_locator",
    "document_type",
    "evidence_status",
    "text",
)


def _estimated_tokens(text: str) -> int:
    """Count a stable approximate lexical-token measure (not model tokens)."""

    return len(_TOKEN_RE.findall(text))


def _get(result: Any, key: str, *, default: Any = ...):
    if isinstance(result, dict):
        value = result.get(key, default)
    else:
        value = getattr(result, key, default)
    if value is ...:
        raise ValueError(f"retrieval result is missing required field {key!r}")
    return value


def _validate_result(result: Any) -> EvidenceItem:
    values = {field: _get(result, field) for field in _REQUIRED_FIELDS}
    for field in ("experiment", "filename", "document_type", "source_id", "chunk_id", "content_sha256", "source_locator"):
        if not isinstance(values[field], str) or not values[field].strip():
            raise ValueError(f"retrieval result field {field!r} must be a non-empty string")
    text = values["text"]
    if not isinstance(text, str):
        raise ValueError("retrieval result field 'text' must be a string")
    rank = values["rank"]
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("retrieval result rank must be a positive integer")
    score = values["similarity_score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        raise ValueError("retrieval result similarity_score must be a finite number")

    file_type = _get(result, "file_type", default=None)
    source_path = _get(result, "source_path", default=None)
    metadata = _get(result, "metadata", default={})
    if isinstance(metadata, dict):
        if source_path is None:
            source_path = metadata.get("source_path")
        file_type = file_type or metadata.get("file_type")
    if not isinstance(source_path, str) or not source_path.strip():
        raise ValueError("retrieval result field 'source_path' must be available for traceability")
    if not isinstance(file_type, str) or not file_type.strip():
        raise ValueError("retrieval result field 'file_type' must be available for traceability")

    page_number = _get(result, "page_number", default=None)
    if page_number is not None and (
        isinstance(page_number, bool) or not isinstance(page_number, int) or page_number <= 0
    ):
        raise ValueError("retrieval result page_number must be a positive integer or null")

    raw_status = values["evidence_status"]
    # A null/unclassified source is explicitly represented as unknown; known
    # evidence statuses are copied unchanged and never inferred from document type.
    status = "unknown" if raw_status is None else raw_status
    if not isinstance(status, str) or status not in _EVIDENCE_STATUSES:
        raise ValueError(f"unsupported evidence_status {raw_status!r}")

    return EvidenceItem(
        rank=rank,
        similarity_score=float(score),
        chunk_id=values["chunk_id"],
        source_id=values["source_id"],
        content_sha256=values["content_sha256"],
        experiment=values["experiment"],
        filename=values["filename"],
        file_type=file_type,
        source_path=source_path,
        document_type=values["document_type"],
        evidence_status=status,  # type: ignore[arg-type]
        source_locator=values["source_locator"],
        page_number=page_number,
        text=text,
    )


def _retrieval_configuration(retrieval: Any, config: RAGContextConfig) -> dict[str, Any]:
    return {
        "top_k": config.top_k,
        "provider": _get(retrieval, "provider", default=None),
        "model_name": _get(retrieval, "model_name", default=None),
        "dimension": _get(retrieval, "dimension", default=None),
        "similarity_metric": _get(retrieval, "similarity_metric", default=None),
    }


def _within_limits(text: str, config: RAGContextConfig) -> bool:
    if config.max_context_chars is not None and len(text) > config.max_context_chars:
        return False
    if config.max_estimated_tokens is not None and _estimated_tokens(text) > config.max_estimated_tokens:
        return False
    return True


def build_context(
    query: str,
    retrieval: SearchResult | Iterable[RetrievalResult] | Any,
    config: RAGContextConfig | None = None,
) -> RAGContext:
    """Build an LLM-ready evidence package from existing retrieval results.

    No source files or databases are read here. If size limits are exceeded,
    lower-ranked unique items are omitted in full and reported; source text is
    never truncated.
    """

    config = config or RAGContextConfig()
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must not be empty")
    retrieval_query = _get(retrieval, "query", default=query)
    if retrieval_query != query:
        raise ValueError("query does not match retrieval result query")
    raw_results = _get(retrieval, "matches", default=retrieval)
    if isinstance(raw_results, (str, bytes)):
        raise ValueError("retrieval results must be an iterable of result records")
    try:
        results = list(raw_results)
    except TypeError as exc:
        raise ValueError("retrieval results must be iterable") from exc

    validated = [_validate_result(result) for result in results]
    validated.sort(key=lambda item: (item.rank, item.chunk_id))

    unique: list[EvidenceItem] = []
    seen_chunk_ids: set[str] = set()
    for item in validated:
        if item.chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(item.chunk_id)
        if len(unique) == config.top_k:
            break
        unique.append(item)

    retrieval_configuration = _retrieval_configuration(retrieval, config)
    kept = list(unique)
    dropped: list[EvidenceItem] = []
    while True:
        dropped_ids = [item.chunk_id for item in dropped]
        context_text = format_context(
            query,
            kept,
            include_similarity_scores=config.include_similarity_scores,
            include_full_provenance=config.include_full_provenance,
            dropped_chunk_ids=dropped_ids,
        )
        if _within_limits(context_text, config):
            break
        if not kept:
            raise ValueError(
                "query and context-limit notice exceed configured context size; "
                "increase max_context_chars or max_estimated_tokens"
            )
        dropped.insert(0, kept.pop())

    return RAGContext(
        query=query,
        retrieval_count=len(kept),
        evidence_items=tuple(kept),
        retrieval_configuration=retrieval_configuration,
        context_text=context_text,
        context_characters=len(context_text),
        estimated_context_tokens=_estimated_tokens(context_text),
        truncation_occurred=bool(dropped),
        dropped_evidence_count=len(dropped),
        dropped_chunk_ids=tuple(item.chunk_id for item in dropped),
    )
