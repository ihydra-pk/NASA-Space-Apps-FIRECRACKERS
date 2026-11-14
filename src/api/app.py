"""FastAPI application wrapping the existing ExploreService."""

from __future__ import annotations

import logging
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from src.api.conditions import extract_conditions
from src.embeddings.config import EmbeddingConfig
from src.embeddings.providers import OpenAICompatibleEmbeddingProvider, create_embedding_provider
from src.experimental_data.config import ExperimentalDataConfig
from src.experimental_data.service import ExperimentalDataService
from src.explore.config import ExploreConfig
from src.explore.models import ExploreRequest
from src.explore.router import QuestionRouter
from src.explore.service import ExploreInputError, ExploreService
from src.llm.config import LLMConfig
from src.llm.service import LLMService
from src.rag.config import RAGContextConfig
from src.vectorstore.config import VectorStoreConfig
from src.vectorstore.search import search
from src.vectorstore.sqlite_store import SQLiteVectorStore

logger = logging.getLogger("firelens.api")


def _env(name: str, default: str) -> str:
    return os.environ.get(f"FIRELENS_API_{name}", default)


ALL_EXPERIMENTS_INDEX = (Path("data/vectorstore/firelens.sqlite3"), "firelens-qwen3-embedding-4b")
FLEX_INDEX = (Path("data/vectorstore/flex-qwen3-embedding-4b.sqlite3"), "flex-qwen3-embedding-4b")


def default_index() -> tuple[Path, str]:
    """The all-experiment index when it has been built locally, else the committed FLEX one."""

    return ALL_EXPERIMENTS_INDEX if ALL_EXPERIMENTS_INDEX[0].is_file() else FLEX_INDEX


def build_explore_service() -> ExploreService:
    """Build the service with the same search and routing as the `src.explore` CLI.

    Overridable via FIRELENS_API_* environment variables.
    """

    default_database, default_collection = default_index()
    search_mode = _env("SEARCH_MODE", "hybrid")
    vector_config = VectorStoreConfig(
        database_path=Path(_env("VECTOR_DATABASE", str(default_database))),
        collection=_env("COLLECTION", default_collection),
        provider=_env("EMBEDDING_PROVIDER", "openai-compatible"),
        model_name=_env("EMBEDDING_MODEL", "gwdg.qwen3-embedding-4b"),
        dimension=int(_env("EMBEDDING_DIMENSION", "2560")),
        top_k=int(_env("TOP_K", "5")),
        min_chunk_words=int(_env("MIN_CHUNK_WORDS", "25")),
    )
    if vector_config.provider == "openai-compatible":
        embedding_provider = OpenAICompatibleEmbeddingProvider()
    else:
        embedding_provider = create_embedding_provider(
            EmbeddingConfig(
                provider=vector_config.provider,
                model_name=vector_config.model_name,
                dimension=vector_config.dimension,
                batch_size=1,
            )
        )

    def retrieve(query: str, *, top_k: int):
        with SQLiteVectorStore(vector_config) as store:
            return search(query, store, vector_config, provider=embedding_provider, top_k=top_k, mode=search_mode)

    llm_config = replace(
        LLMConfig.from_env(provider=_env("LLM_PROVIDER", "openai-compatible")),
        model_name=_env("LLM_MODEL", "openai.gpt-6-luna"),
    )
    experimental_config = replace(
        ExperimentalDataConfig(),
        store_path=Path(_env("EXPERIMENTAL_DATABASE", "data/experimental_data/experiments.sqlite3")),
    )
    llm_service = LLMService(config=llm_config)
    use_router = _env("ROUTER", "on").lower() not in {"off", "0", "false", "no"}
    return ExploreService(
        retrieve,
        ExperimentalDataService(experimental_config),
        llm_service,
        config=ExploreConfig(rag=RAGContextConfig(top_k=vector_config.top_k)),
        router=QuestionRouter(llm_service.provider, llm_config) if use_router else None,
    )


def create_app(service: ExploreService | None = None) -> FastAPI:
    """Create the API. Pass `service` to inject a preconfigured/fake service."""

    app = FastAPI(title="FireLens API", version="0.1.0")
    origins = [o.strip() for o in _env("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["*"])
    holder: dict[str, ExploreService | None] = {"service": service}

    def get_service() -> ExploreService:
        if holder["service"] is None:
            try:
                holder["service"] = build_explore_service()
            except Exception as exc:
                logger.exception("Failed to build Explore service")
                raise HTTPException(503, f"Explore service is not configured: {exc}") from exc
        return holder["service"]  # type: ignore[return-value]

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        page = Path(__file__).resolve().parents[2] / "index.html"
        if not page.is_file():
            raise HTTPException(404, "index.html not found")
        return FileResponse(page, media_type="text/html")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/experiments")
    def experiments(limit: int = 100, offset: int = 0) -> dict[str, Any]:
        """List stored structured records; each carries its own evidence_status (planned/observed/metadata)."""

        if limit < 1 or limit > 1000 or offset < 0:
            raise HTTPException(422, "limit must be 1-1000 and offset >= 0")
        records = get_service().experimental_data.records()  # type: ignore[union-attr]
        page = records[offset : offset + limit]
        return {"total": len(records), "limit": limit, "offset": offset, "records": [r.to_dict() for r in page]}

    @app.post("/api/explore")
    def explore(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        try:
            request = ExploreRequest.from_dict(payload)
            if not request.conditions and isinstance(request.question, str):
                inferred = extract_conditions(request.question)
                if inferred:
                    request = replace(request, conditions=inferred)
            return get_service().explore(request).to_dict()
        except (ExploreInputError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Explore request failed")
            raise HTTPException(502, f"Explore pipeline failed: {type(exc).__name__}: {exc}") from exc

    return app


app = create_app()
