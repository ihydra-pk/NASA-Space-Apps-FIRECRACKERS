# NASA-Space-Apps-FIRECRACKERS
Flame in Freefall: SI-Powered Fire Safety Insights from Microgravity Combustion Data

## Using the backend (for teammates)

Reviewers who want to download and run the Qwen3 embedding model locally can
follow [QWEN_LOCAL_GUIDE.md](QWEN_LOCAL_GUIDE.md).

Everything slow to produce is already in this branch (`SI_Dashboard_Ravish`): the NASA PSI downloads, the extracted text, the chunks and the embeddings for all 16 experiments. You do not need to re-download or re-embed anything.

```bash
git clone https://github.com/ihydra-pk/NASA-Space-Apps-FIRECRACKERS.git
cd NASA-Space-Apps-FIRECRACKERS
git checkout SI_Dashboard_Ravish
python -m pip install -r requirements.txt
```

One file is not in the repository: the all-experiment search index `data/vectorstore/firelens.sqlite3` (441 MB, above GitHub's 100 MB file limit). Build it once from the committed embeddings. It takes about 15 seconds and needs no network or API key:

```bash
python -m src.vectorstore index \
  --input-dir data/embeddings/openai-compatible/gwdg.qwen3-embedding-4b \
  --database data/vectorstore/firelens.sqlite3 \
  --collection firelens-qwen3-embedding-4b \
  --provider openai-compatible --model gwdg.qwen3-embedding-4b --dimension 2560
```

Asking a question needs the gateway key, because the question is embedded and the answer is written by the chat model:

```bash
export UPB_AI_GATEWAY_KEY=...
export FIRELENS_EMBEDDING_BASE_URL=https://ai-gateway.uni-paderborn.de/v1
export FIRELENS_LLM_BASE_URL=https://ai-gateway.uni-paderborn.de/v1

python -m src.explore "What causes cool flame extinction in microgravity?" \
  --vector-database data/vectorstore/firelens.sqlite3 \
  --collection firelens-qwen3-embedding-4b
```

To use the web front-end instead, follow [Run.md](Run.md): it covers setup, starting the server and troubleshooting. `POST /api/explore` with `{"question": "..."}` is the endpoint the page calls; `GET /api/experiments` lists the stored NASA test records.

Without the two `--vector-database` / `--collection` options, the `src.explore` command line searches only the FLEX documents. The response is JSON with the answer, validated citations and the matching test conditions; `src/explore/README.md` describes the contract for a frontend. `CLAUDE.md` covers the architecture and the state of the data.
