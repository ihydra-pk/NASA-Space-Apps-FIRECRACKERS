"""Command-line interface for indexing and semantic search."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

from src.embeddings.config import EmbeddingConfig
from src.embeddings.providers import (
    OpenAICompatibleEmbeddingProvider,
    create_embedding_provider,
)

from .config import VectorStoreConfig
from .pipeline import index_embeddings
from .search import search
from .sqlite_store import SQLiteVectorStore


def _config(args) -> VectorStoreConfig:
    return VectorStoreConfig(
        database_path=args.database,
        collection=args.collection,
        provider=args.provider,
        model_name=args.model,
        dimension=args.dimension,
        top_k=getattr(args, "top_k", 5),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Index/search NASA FireLens embedding records.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    index_parser = subparsers.add_parser("index", help="Index embedding JSONL")
    index_parser.add_argument("--input-dir", type=Path, required=True)
    search_parser = subparsers.add_parser("search", help="Search indexed embeddings")
    search_parser.add_argument("query")
    for subparser in (index_parser, search_parser):
        subparser.add_argument("--database", type=Path, default=Path("data/vectorstore/firelens.sqlite3"))
        subparser.add_argument("--collection", default="flex-qwen3-embedding-4b")
        subparser.add_argument("--provider", required=True)
        subparser.add_argument("--model", required=True)
        subparser.add_argument("--dimension", type=int, required=True)
        subparser.add_argument("--base-url", default=None)
    search_parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = _config(args)

    if args.command == "index":
        result = index_embeddings(args.input_dir, config)
        print(
            f"Input records: {result.input_records}; indexed: {result.indexed}; "
            f"upserted: {result.upserted}; duplicates: {result.duplicates}; "
            f"rejected: {result.rejected}; unique chunk IDs: {result.unique_chunk_ids}."
        )
        print(f"SQLite store: {config.database_path} (collection={config.collection})")
        for failure in result.failures:
            print(f"  - {failure}")
        return 1 if result.rejected or result.duplicates else 0

    embedding_config = EmbeddingConfig(
            provider=config.provider,
            model_name=config.model_name,
            dimension=config.dimension,
            batch_size=1,
        )
    provider = (
        OpenAICompatibleEmbeddingProvider(
            base_url=args.base_url,
            use_gateway_credential=args.base_url is not None,
        )
        if config.provider == "openai-compatible"
        else create_embedding_provider(embedding_config)
    )
    with SQLiteVectorStore(config) as store:
        result = search(args.query, store, config, provider=provider, top_k=args.top_k)
    for match in result.matches:
        print(json.dumps(asdict(match), ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
