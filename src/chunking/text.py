"""Boundary-aware text partitioning with an explicitly estimated token count."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

# This estimate counts lexical terms/punctuation, splitting unusually long
# terms into approximate four-character token pieces. It is deliberately named
# estimated_token_count: it is not tied to a particular model tokenizer.
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_SENTENCE_OR_PARAGRAPH_BOUNDARY = re.compile(
    r"(?<=[.!?])(?=\s+(?:[A-Z0-9\(\[\"“]))|(?<=\n)(?=\s*\n)"
)
_WORD_RE = re.compile(r"\S+")


@dataclass(frozen=True, slots=True)
class TextSpan:
    """Half-open character offsets into the original text."""

    start: int
    end: int


def estimated_token_count(text: str) -> int:
    """Return a stable rough token estimate without external tokenizer data."""

    count = 0
    for match in _TOKEN_RE.finditer(text):
        token = match.group()
        count += max(1, math.ceil(len(token) / 4)) if token[0].isalnum() or token[0] == "_" else 1
    return count


def _split_oversized_span(
    text: str, span: TextSpan, max_tokens: int, overlap_tokens: int
) -> list[TextSpan]:
    """Split a long paragraph/sentence at words, keeping overlap possible."""

    segment = text[span.start : span.end]
    words = list(_WORD_RE.finditer(segment))
    if not words:
        return [span]

    # Include any leading whitespace in the first atom. Each following atom
    # starts at a word and ends immediately before the next word, preserving
    # every original character and avoiding cuts inside numeric/unit tokens.
    starts = [span.start]
    starts.extend(span.start + word.start() for word in words[1:])
    ends = starts[1:] + [span.end]

    # Bound sub-spans to the configured overlap size when overlap is enabled.
    # This allows later chunks to preserve meaningful overlap even within a
    # single long paragraph/table, without cutting a word or changing text.
    span_limit = min(max_tokens, overlap_tokens) if overlap_tokens else max_tokens
    result: list[TextSpan] = []
    current_start = starts[0]
    current_end = current_start
    for atom_start, atom_end in zip(starts, ends):
        atom_tokens = estimated_token_count(text[atom_start:atom_end])
        current_tokens = estimated_token_count(text[current_start:current_end]) if current_end > current_start else 0
        if current_end > current_start and current_tokens + atom_tokens > span_limit:
            result.append(TextSpan(current_start, current_end))
            current_start = atom_start
        current_end = atom_end
    if current_end > current_start:
        result.append(TextSpan(current_start, current_end))
    return result or [span]


def _semantic_spans(text: str, max_tokens: int, overlap_tokens: int) -> list[TextSpan]:
    """Find paragraph/sentence spans, splitting only oversized spans at words."""

    if not text:
        return []

    boundaries = [0]
    boundaries.extend(
        match.start()
        for match in _SENTENCE_OR_PARAGRAPH_BOUNDARY.finditer(text)
        if match.start() > 0
    )
    boundaries.append(len(text))
    spans = [TextSpan(start, end) for start, end in zip(boundaries, boundaries[1:]) if start < end]

    result: list[TextSpan] = []
    for span in spans:
        if estimated_token_count(text[span.start : span.end]) <= max_tokens:
            result.append(span)
        else:
            result.extend(_split_oversized_span(text, span, max_tokens, overlap_tokens))
    return result


def _overlap_start(
    text: str,
    spans: list[TextSpan],
    chunk_start: int,
    previous_end: int,
    overlap_tokens: int,
) -> int:
    """Choose the earliest preceding span that fits the overlap budget."""

    if overlap_tokens <= 0:
        return previous_end

    start = previous_end
    token_count = 0
    for candidate in range(previous_end - 1, chunk_start - 1, -1):
        candidate_count = estimated_token_count(text[spans[candidate].start : spans[candidate].end])
        if token_count + candidate_count > overlap_tokens:
            break
        token_count += candidate_count
        start = candidate
    return start


def partition_text(text: str, *, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Partition text near natural boundaries with overlapping source spans.

    Chunk text is always an unchanged contiguous substring of the source. PDF
    pages are passed independently by the caller, so this function never joins
    page records. Empty or whitespace-only inputs yield no chunks.
    """

    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero")
    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be non-negative and less than max_tokens")
    if not text or not text.strip():
        return []

    spans = _semantic_spans(text, max_tokens, overlap_tokens)
    chunks: list[str] = []
    start = 0
    while start < len(spans):
        end = start
        while end < len(spans):
            candidate_start = spans[start].start
            candidate_end = spans[end].end
            if end > start and estimated_token_count(text[candidate_start:candidate_end]) > max_tokens:
                break
            end += 1

        # _semantic_spans has already divided any overlong individual span;
        # therefore at least one span is consumed each iteration.
        if end == start:
            end += 1

        chunk_text = text[spans[start].start : spans[end - 1].end]
        if chunk_text.strip():
            chunks.append(chunk_text)
        if end >= len(spans):
            break

        next_start = _overlap_start(text, spans, start, end, overlap_tokens)
        # Ensure progress even when a single semantic span exceeds the overlap
        # budget or an unusual whitespace-heavy input is encountered.
        start = max(start + 1, next_start)

    return chunks
