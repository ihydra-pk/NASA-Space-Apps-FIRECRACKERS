# Running FireLens

How to start the server that serves the web page and answers questions from the NASA documents. Commands are for macOS and Linux, run from the repository root.

## 1. One-time setup

```bash
git clone https://github.com/ihydra-pk/NASA-Space-Apps-FIRECRACKERS.git
cd NASA-Space-Apps-FIRECRACKERS
git checkout SI_Dashboard_Ravish

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 2. Build the search index (once)

The documents, chunks and embeddings for all 16 experiments are in the repository. The search index built from them is 441 MB, too large for GitHub, so build it locally. This takes about 15 seconds and needs no key or network:

```bash
python -m src.vectorstore index \
  --input-dir data/embeddings/openai-compatible/gwdg.qwen3-embedding-4b \
  --database data/vectorstore/firelens.sqlite3 \
  --collection firelens-qwen3-embedding-4b \
  --provider openai-compatible --model gwdg.qwen3-embedding-4b --dimension 2560
```

It should end with `Input records: 4103`. Run it again after pulling new embeddings.

## 3. Set your gateway key

Every question is embedded and answered by models on the Paderborn AI gateway, so the server needs your own key. Do not commit it.

```bash
export UPB_AI_GATEWAY_KEY=your-key-here
```

Add that line to `~/.zshrc` to avoid typing it in every new terminal.

## 4. Start the server

```bash
source .venv/bin/activate      # if not already active
python -m src.api
```

Open http://127.0.0.1:8000 and ask a question in the page.

| Option           | Use                                                                |
| ---------------- | ------------------------------------------------------------------ |
| `--port 8080`    | another port                                                       |
| `--host 0.0.0.0` | let other devices on the same network open `http://<your-ip>:8000` |

Stop the server with Ctrl+C.

## 5. Check that it works

In a second terminal:

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok"}

curl -X POST http://127.0.0.1:8000/api/explore \
  -H 'Content-Type: application/json' \
  -d '{"question":"What happens to a flame in microgravity when the forced flow velocity changes?"}'
```

The second command returns JSON with `"status": "grounded"`, an answer and its citations. Expect roughly 30 seconds per question, longer when several are asked at once.

## Troubleshooting

| What you see                                                               | Cause and fix                                                                                                                                                |
| -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Page says "Backend unavailable ... showing matching NASA passages instead" | The server is stopped, or the question failed inside it. Check the terminal running the server; most often `UPB_AI_GATEWAY_KEY` is not set in that terminal. |
| Answers only ever cite FLEX documents                                      | `data/vectorstore/firelens.sqlite3` is missing, so the server fell back to the small FLEX-only index. Do step 2 and restart.                                 |
| `address already in use`                                                   | Another server is on port 8000. Stop it, or start with `--port 8080`.                                                                                        |
| `No module named fastapi`                                                  | The virtual environment is not active, or step 1 was skipped.                                                                                                |
| "Insufficient evidence" for a question the documents should cover          | Check you have pulled the latest `SI_Dashboard_Ravish` and restarted the server.                                                                             |

## Optional settings

Set these before starting the server to change its defaults:

| Variable                                                  | Default              | Meaning                                                           |
| --------------------------------------------------------- | -------------------- | ----------------------------------------------------------------- |
| `FIRELENS_API_LLM_MODEL`                                  | `openai.gpt-6-luna`  | chat model that writes the answer                                 |
| `FIRELENS_API_TOP_K`                                      | `5`                  | passages given to the model per question                          |
| `FIRELENS_API_MIN_CHUNK_WORDS`                            | `25`                 | passages shorter than this (slide titles) are ignored             |
| `FIRELENS_API_SEARCH_MODE`                                | `hybrid`             | `hybrid` (meaning plus keywords) or `semantic`                    |
| `FIRELENS_API_VECTOR_DATABASE`, `FIRELENS_API_COLLECTION` | all-experiment index | use a different search index                                      |
| `FIRELENS_API_ROUTER`                                     | `on`                 | `off` skips the step that reads test conditions from the question |
