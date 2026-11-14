"""Command-line entry point: ``python -m src.embeddings``."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from .config import EmbeddingConfig
from .pipeline import run_embedding
from .providers import OpenAICompatibleEmbeddingProvider


def main() -> int:
    parser = argparse.ArgumentParser(description="Embed NASA FireLens chunk JSONL records.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/chunks"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/embeddings"))
    parser.add_argument(
        "--provider", default=os.environ.get("FIRELENS_EMBEDDING_PROVIDER", "mock")
    )
    parser.add_argument(
        "--base-url", default=os.environ.get("FIRELENS_EMBEDDING_BASE_URL")
    )
    parser.add_argument("--model", default=os.environ.get("FIRELENS_EMBEDDING_MODEL"))
    parser.add_argument("--dimension", type=int, default=os.environ.get("FIRELENS_EMBEDDING_DIMENSION"))
    parser.add_argument(
        "--batch-size", type=int, default=int(os.environ.get("FIRELENS_EMBEDDING_BATCH_SIZE", "16"))
    )
    args = parser.parse_args()

    default_model, default_dimension = {
        "mock": ("firelens-hash-v1", 384),
        "openai-compatible": ("text-embedding-3-small", 1536),
    }.get(args.provider, ("", 0))
    if not default_model:
        parser.error(f"Unsupported provider {args.provider!r}; choose mock or openai-compatible.")
    model_name = args.model or default_model
    dimension = args.dimension if args.dimension is not None else default_dimension
    provider = None
    if args.provider == "openai-compatible":
        provider = OpenAICompatibleEmbeddingProvider(base_url=args.base_url)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        result = run_embedding(
            EmbeddingConfig(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                provider=args.provider,
                model_name=model_name,
                dimension=dimension,
                batch_size=args.batch_size,
            ),
            provider=provider,
        )
    except ValueError as exc:
        parser.error(str(exc))

    print(
        f"Input chunks: {result.input_chunks}; embedded: {result.successfully_embedded}; "
        f"failed: {result.failed_chunks}; provider={args.provider}; model={model_name}; "
        f"dimension={dimension}; batch_size={args.batch_size}."
    )
    print("Output files:")
    for path in result.output_paths:
        print(f"  - {path}")
    if result.failures:
        print("Failed records:")
        for failure in result.failures:
            print(f"  - {failure.input_file}:{failure.line_number}: {failure.error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
