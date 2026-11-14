# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

FireLens ("Flame in Freefall", NASA Space Apps): a retrieval-and-evidence backend over NASA microgravity combustion data from the Physical Sciences Informatics (PSI) repository. It answers questions from NASA documents with validated citations and matches user-selected test conditions against structured experiment records. `index.html` is a single-file front-end served by the FastAPI app in `src/api`; `Instructions.txt` is a pasted chat transcript containing a fictional-data React prototype, not project instructions.

## Commands

Run everything from the repository root; packages are imported as `src.<name>` and there is no install step for the project itself.

```bash
python -m pip install -r requirements.txt   # pypdf, python-docx, fastapi, uvicorn, httpx, pytest

python -m pytest -q                                          # all tests
python -m pytest tests/test_chunking.py -q                   # one file
python -m pytest tests/test_chunking.py::test_name -q        # one test
python -m pytest src/explore/tests src/experimental_data/tests -q
```

Tests live in two places: `tests/` (ingestion, chunking, embeddings, vectorstore, rag, llm) and `src/<package>/tests/` (explore, experimental_data). Tests make no network calls; the one real-LLM smoke test only runs with `FIRELENS_RUN_LLM_SMOKE=1`.

Pipeline stages, in order (each is `python -m src.<stage>`, see `--help`):

```bash
python psi_download.py --dry-run             # fetch source documents from NASA PSI
python -m src.ingestion                      # data/documents -> data/processed
python -m src.chunking                       # data/processed -> data/chunks
python -m src.embeddings --provider openai-compatible --model gwdg.qwen3-embedding-4b --dimension 2560
python -m src.vectorstore index --input-dir <embeddings dir> --database <sqlite> --collection <name> ...
python -m src.vectorstore search "question" --database <sqlite> --collection <name> ...
python -m src.experimental_data              # documents + data/structured tables -> data/experimental_data/experiments.sqlite3
python -m src.explore "question" --conditions '{"fuel":"Methanol","oxygen_concentration":21}'
python -m src.api                            # front-end and JSON API on http://127.0.0.1:8000
```

The module READMEs give examples in PowerShell syntax (backtick continuations, `$env:`); translate for bash/zsh.

## Architecture

A linear pipeline of independent packages. Each stage reads only the previous stage's output files and never reaches back to earlier ones; keep that boundary when changing code.

```
psi_download.py -> <out>/<PSI-ID>/<category>/   (staged into data/documents/<experiment>/ by scripts/stage_psi_documents.py)
  ingestion   -> data/processed/   JSONL, one record per PDF page / one per DOCX
  chunking    -> data/chunks/      JSONL (800 estimated tokens, 120 overlap)
  embeddings  -> data/embeddings/[<provider>/<model>/]  JSONL
  vectorstore -> data/vectorstore/*.sqlite3  (SQLite, cosine search)
  rag         -> RAGContext        (in memory; packages ranked results)
  llm         -> AnswerResponse    (grounded answer + citation validation)

experimental_data -> experiments.sqlite3 (structured rows + condition matching), separate branch
explore           -> composes retriever + experimental_data + llm into one response
api               -> FastAPI: GET / (index.html), GET /api/experiments, POST /api/explore
```

Search has two modes. `semantic` ranks by cosine similarity alone. `hybrid` also scores chunk text by keyword (BM25 in `src/vectorstore/keyword.py`) and orders by reciprocal rank fusion of the two rankings, which helps with test numbers, acronyms and formulae; `similarity_score` is still the cosine value. `src.explore` defaults to hybrid, `src.vectorstore search` to semantic (`--mode hybrid`).

Three search-time rules keep general questions from retrieving noise; none touch stored embeddings. Queries to a Qwen3 embedding model are sent as `Instruct: <task>\nQuery: <question>` (`query_text_for_embedding`, text in `VectorStoreConfig.query_instruction`); without it a plain question scores highest against slide titles of a few words. `min_chunk_words` skips chunks shorter than that when ranking (25 in `src.api` and `src.explore`, 0 in `src.vectorstore search` and the library default). Results with identical text, which arise because NASA files the same paper under several names, take one slot.

`src/explore/router.py` reads a plain question with one LLM call, labels it `documents`, `test_data` or `both`, and extracts the test conditions it states so the matcher can run without the caller passing `conditions`. Its output is validated: unknown fields, unsupported units and numbers not written in the question are dropped and listed under `routing.ignored`. Conditions passed by the caller bypass it, a router failure falls back to document search with a warning, and document retrieval runs on every route because answers are cited against retrieved chunks. Disable with `--no-router`.

`src/explore` is the integration point intended for a frontend. `ExploreService` takes an injected retriever callable, an `ExperimentalDataService` and an `LLMService`, and returns a JSON contract with separate `document_evidence`, `experimental_evidence`, `condition_match_summary`, `answer`, `citations`, `status` and `evidence_mode`.

`src/api/app.py` builds that service once per process. It uses the all-experiment index `data/vectorstore/firelens.sqlite3` when the file exists and the committed FLEX index otherwise, with hybrid search and the router on; `FIRELENS_API_*` variables (`VECTOR_DATABASE`, `COLLECTION`, `SEARCH_MODE`, `ROUTER`, `LLM_MODEL`, `EXPERIMENTAL_DATABASE`, `CORS_ORIGINS`, ...) override. Before calling Explore it extracts fuel, oxygen, pressure and flow written in the question with regexes (`src/api/conditions.py`), so the LLM router only runs when those find nothing.

### Rules that span modules

These are deliberate design constraints, enforced by tests, and easy to break by accident:

- **Evidence status is carried, never inferred.** Every record has `planned`, `observed`, `interpreted`, `metadata` or `unknown`. Labels come from `data/document_metadata.json` (keyed by path relative to `data/documents/`); unlisted files are `unknown`. Nothing classifies documents from their content.
- **No fabricated measurements.** Structured records come from two places: the FLEX test plan's proposed matrix (Tables 6-12, `planned`) and NASA's as-run test tables under `data/structured/` (`src/experimental_data/tables.py`), whose rows are `observed` where the table reports an outcome and `metadata` or `planned` otherwise. Fields absent from the source (temperature, per-test gravity, outcomes in a plan) stay unavailable and are reported as `parameter_unavailable` / `UNAVAILABLE` instead of being filled in.
- **Similarity is not confidence.** Document `similarity_score` and structured `match_type`/difference/tolerance are different scales and are never merged into one score.
- **Citations are validated, not trusted.** `src/llm` rejects any citation whose chunk ID, source ID, locator, page or status does not match the supplied `RAGContext`; provenance is copied from the context, not from model output. With no retrieved evidence the model is not called and the answer is `insufficient_evidence`.
- **Source text is never rewritten or truncated.** When context limits are hit, `rag` drops whole lowest-ranked items and records the dropped chunk IDs.
- **Identity is content-based.** `source_id` derives from the SHA-256 of the original file bytes, so the same file at a different path has the same identity.
- **A vector collection is bound to one provider, model, dimension and metric.** Use a new collection for a different embedding profile. The `mock` embedding provider (`firelens-hash-v1`) produces hash vectors for tests only; it is the CLI default for `src.embeddings` and is rejected when querying a real collection.

### Units

Explore requests take oxygen in percent, pressure in atm and flow in cm/s by default, with overrides via `condition_units`. The structured store and matcher use mole fraction for oxygen and suppressant (0.21, not 21); `explore` converts. Matching tolerances are software defaults, not NASA uncertainty bounds.

## Configuration

Models are reached through an OpenAI-compatible gateway (`https://ai-gateway.uni-paderborn.de/v1`), configured only by environment variables:

- Key: `UPB_AI_GATEWAY_KEY` (also `FIRELENS_EMBEDDING_API_KEY` / `FIRELENS_LLM_API_KEY` / `OPENAI_API_KEY`).
- Embeddings: `FIRELENS_EMBEDDING_BASE_URL`, `_PROVIDER`, `_MODEL`, `_DIMENSION`, `_BATCH_SIZE`. The indexed collection uses `gwdg.qwen3-embedding-4b` at 2560 dimensions; set the base URL explicitly so the gateway key is selected.
- Chat: `FIRELENS_LLM_BASE_URL`, `_MODEL` (default `openai.gpt-6-luna`), `_TIMEOUT`, `_TEMPERATURE`, `_MAX_OUTPUT_TOKENS`.

`opencode.json` configures a different coding agent against the same gateway and is not read by this code.

## Data state

- 16 experiments are ingested (122 PDF and DOCX files under `data/documents/<experiment>/`), with derived artifacts committed under `data/processed`, `data/chunks` (4,103 chunks) and `data/embeddings/openai-compatible/gwdg.qwen3-embedding-4b/`. PSI-181 (a solder study) is left out. Every document except the two FLEX files has document type and evidence status `unknown` until someone labels it in `data/document_metadata.json`.
- The all-experiment index `data/vectorstore/firelens.sqlite3` (collection `firelens-qwen3-embedding-4b`) is git-ignored because it is over GitHub's 100 MB file limit. Build it from the committed embeddings, which needs no network: `python -m src.vectorstore index --input-dir data/embeddings/openai-compatible/gwdg.qwen3-embedding-4b --database data/vectorstore/firelens.sqlite3 --collection firelens-qwen3-embedding-4b --provider openai-compatible --model gwdg.qwen3-embedding-4b --dimension 2560`.
- The `explore` CLI still defaults to the committed FLEX-only index `flex-qwen3-embedding-4b.sqlite3`; pass `--vector-database data/vectorstore/firelens.sqlite3 --collection firelens-qwen3-embedding-4b` for all experiments.
- `data/experimental_data/experiments.sqlite3` is committed and `python -m src.experimental_data` rebuilds it: 836 records (414 `observed`, 298 `planned`, 124 `metadata`) from the FLEX test plan and DOCX, the tables in `data/structured/{flex,bass,saffire}/` and the PSI-68 FLEX-2 table under `data/psi-download/`.
- `data/psi-download/` is a second, smaller downloader cache (PSI-25, 26, 68, 69, 98, 99, 100) that `src/ingestion/psi_sync.py` copies into `data/documents` and `data/structured`. Its document copies were not merged into `data/documents`, because the same files are already there under their `PSI-<id>_...` names; running `psi_sync` would add them again under other names and duplicate content in search. The experimental-table CSVs for the other experiments under `data/PSI-*/` are still read by no code.
- Ingestion handles `.pdf`, `.docx` and `.md`, with no OCR. Legacy `.doc`/`.ppt` files from PSI are not extracted.

## psi_download.py

Standard-library-only script that pulls text-rich files from PSI and writes `manifest.csv`, plus per-experiment `catalogue.json`, `metadata.json` and `summary.md`. It writes to `<out>/<PSI-ID>/<category>/`, which is not the `data/documents/<experiment>/` layout ingestion expects. `python scripts/stage_psi_documents.py` links the PDFs and DOCX files into that layout (deduplicated by content) and adds an entry per file to `data/document_metadata.json` with the experiment, accession and source URL; document type and evidence status stay unset until someone labels them. Ingestion reads only `flex`, `bass` and `saffire` unless run with `--all-experiments`.

- Endpoints: search `https://psi.nasa.gov/geode-py/ws/repo/search?term=...&source=psi&type=investigation`, catalogue `.../ws/repo/investigations/<ID>`, file `.../ws/studies/<ID>/download?source=datamanager&file=<name>`.
- Keeps Reports, Science Documents, Presentations, Reference Data and the experimental table; skips Raw Data (over a terabyte across the combustion set), archives, media and files above `--max-mb`. `--include-analyzed` and `--include-engineering` widen the filter.
- `--experimental-table-only` fetches just the PSI-69 FLEX run table and refuses if the name or size NASA lists has changed.
- PSI is slow and drops connections mid-transfer: catalogues of several MB can fail with `http.client.IncompleteRead`. `with_retries` retries everything in `NETWORK_ERRORS` (including `http.client.HTTPException`) but gives up at once on 403/404, because some listed files are permanently missing.
- The "combustion" search returns 26 experiments, including PSI-181, a solder-joint materials study unrelated to fire.
