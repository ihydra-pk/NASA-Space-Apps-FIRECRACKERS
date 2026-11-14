# Render deployment

This repo is configured as a single Render web service. `render.yaml` installs
`requirements.txt`, runs `python -m src.api`, and serves both `index.html` and
the `/api/explore` backend from the same origin. The API binds to `0.0.0.0` and
uses Render's `PORT` environment variable.

## Deploy

1. Push or select the `prod` branch in Render and create a Blueprint using
   `render.yaml` (or create a Python web service with the same build/start
   commands).
2. In Render's service environment, set `UPB_AI_GATEWAY_KEY` to a valid key.
   It is intentionally not stored in this repository.
3. Deploy. Check `/health`, then open `/` and try Ask NASA or **Ask RAG about
   this scenario**.

The checked-in local vector index and structured SQLite database are used by
default. If the vector database is moved to a persistent disk, set
`FIRELENS_API_VECTOR_DATABASE` to its path. The service needs outbound access to
the configured embeddings and LLM gateway. Free instances may sleep and have
limited memory; the vector index is about 24 MB.

## Configuration

- `UPB_AI_GATEWAY_KEY`: required for the configured UPB embeddings/LLM gateway.
- `FIRELENS_LLM_MODEL`: defaults to `openai.gpt-6-luna`.
- `FIRELENS_LLM_MAX_OUTPUT_TOKENS`: defaults to `3000`.
- `FIRELENS_EMBEDDING_BASE_URL`: set by the Blueprint to the UPB gateway.
- `FIRELENS_API_*`: optional API overrides (see `src/api/app.py`), including
  vector database, collection, embedding profile, top-k and CORS origins.

All chat and scenario requests go through the existing grounded Explore
pipeline. Citation validation remains server-side; a model response is not
treated as grounded unless it passes that validator.
