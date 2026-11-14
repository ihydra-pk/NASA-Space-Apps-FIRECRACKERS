"""CLI for one unified Explore request using existing project services."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from src.embeddings.config import EmbeddingConfig
from src.embeddings.providers import OpenAICompatibleEmbeddingProvider, create_embedding_provider
from src.experimental_data.config import ExperimentalDataConfig
from src.experimental_data.service import ExperimentalDataService
from src.llm.config import LLMConfig
from src.llm.service import LLMService
from src.rag.config import RAGContextConfig
from src.vectorstore.config import VectorStoreConfig
from src.vectorstore.search import search
from src.vectorstore.sqlite_store import SQLiteVectorStore

from .config import ExploreConfig
from .models import ExploreRequest
from .router import QuestionRouter
from .service import ExploreService


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a unified FireLens Explore query.")
    parser.add_argument("question")
    parser.add_argument("--conditions", default="{}", help="JSON object of selected conditions")
    parser.add_argument("--condition-units", default="{}", help="JSON object of explicit condition units")
    parser.add_argument("--vector-database", type=Path, default=Path("data/vectorstore/flex-qwen3-embedding-4b.sqlite3"))
    parser.add_argument("--collection", default="flex-qwen3-embedding-4b")
    parser.add_argument("--embedding-provider", default="openai-compatible")
    parser.add_argument("--embedding-model", default="gwdg.qwen3-embedding-4b")
    parser.add_argument("--embedding-dimension", type=int, default=2560)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--llm-provider", default="openai-compatible")
    parser.add_argument("--llm-model", default="openai.gpt-6-luna")
    parser.add_argument(
        "--search-mode", choices=("semantic", "hybrid"), default="hybrid",
        help="hybrid (default) ranks by meaning and by keyword; semantic uses meaning only",
    )
    parser.add_argument(
        "--no-router", action="store_true",
        help="do not read test conditions from the question; use --conditions only",
    )
    parser.add_argument("--experimental-database", type=Path, default=Path("data/experimental_data/experiments.sqlite3"))
    args = parser.parse_args()

    try:
        conditions = json.loads(args.conditions)
        units = json.loads(args.condition_units)
    except json.JSONDecodeError as exc:
        parser.error(f"Invalid JSON input: {exc}")
    if not isinstance(conditions, dict) or not isinstance(units, dict):
        parser.error("--conditions and --condition-units must each be JSON objects")

    vector_config = VectorStoreConfig(
        database_path=args.vector_database,
        collection=args.collection,
        provider=args.embedding_provider,
        model_name=args.embedding_model,
        dimension=args.embedding_dimension,
        top_k=args.top_k,
    )
    embedding_config = EmbeddingConfig(
        provider=vector_config.provider,
        model_name=vector_config.model_name,
        dimension=vector_config.dimension,
        batch_size=1,
    )
    embedding_provider = (
        OpenAICompatibleEmbeddingProvider()
        if vector_config.provider == "openai-compatible"
        else create_embedding_provider(embedding_config)
    )

    def retrieve(query: str, *, top_k: int):
        with SQLiteVectorStore(vector_config) as store:
            return search(
                query, store, vector_config,
                provider=embedding_provider, top_k=top_k, mode=args.search_mode,
            )

    rag_config = RAGContextConfig(top_k=args.top_k)
    llm_config = replace(
        LLMConfig.from_env(provider=args.llm_provider),
        model_name=args.llm_model,
    )
    experimental_config = replace(
        ExperimentalDataConfig(),
        store_path=args.experimental_database,
    )
    llm_service = LLMService(config=llm_config)
    service = ExploreService(
        retrieve,
        ExperimentalDataService(experimental_config),
        llm_service,
        config=ExploreConfig(rag=rag_config),
        router=None if args.no_router else QuestionRouter(llm_service.provider, llm_config),
    )
    response = service.explore(ExploreRequest(args.question, conditions, units))
    print(json.dumps(response.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
