"""Optional CLI for answering with an already-built RAGContext JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.rag.models import EvidenceItem, RAGContext

from .config import LLMConfig
from .service import LLMService


def _load_context(path: Path) -> RAGContext:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return RAGContext(
        query=raw["query"],
        retrieval_count=raw["retrieval_count"],
        evidence_items=tuple(EvidenceItem(**item) for item in raw["evidence_items"]),
        retrieval_configuration=raw["retrieval_configuration"],
        context_text=raw["context_text"],
        context_characters=raw["context_characters"],
        estimated_context_tokens=raw["estimated_context_tokens"],
        truncation_occurred=raw.get("truncation_occurred", False),
        dropped_evidence_count=raw.get("dropped_evidence_count", 0),
        dropped_chunk_ids=tuple(raw.get("dropped_chunk_ids", [])),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate an evidence-grounded answer from RAGContext JSON.")
    parser.add_argument("context_json", type=Path, help="Existing serialized RAGContext")
    parser.add_argument("--question", help="Question; defaults to RAGContext.query")
    args = parser.parse_args()
    context = _load_context(args.context_json)
    answer = LLMService(LLMConfig.from_env()).answer(args.question or context.query, context)
    print(json.dumps(answer.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
