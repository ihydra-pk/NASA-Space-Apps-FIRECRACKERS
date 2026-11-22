# Reviewer guide: download and run Qwen3 Embedding locally

This guide shows how to download the **Qwen3-Embedding-4B** model, run it on
your own machine, and connect it to FireLens for local query/document
embeddings. Qwen is an **embedding model**, not a chat model: it turns a
question or NASA passage into a vector. FireLens compares those vectors to
retrieve NASA passages; a separate LLM is needed to write a synthesized answer.

The model is published at [Qwen/Qwen3-Embedding-4B on Hugging Face](https://huggingface.co/Qwen/Qwen3-Embedding-4B)
under the Apache-2.0 license. It supports 32K-token inputs and up to 2,560
embedding dimensions. FireLens's current Qwen index uses **2,560 dimensions**.

## What you need

- Linux is recommended. On Windows, use **WSL2 with Ubuntu** and an NVIDIA GPU
  supported by the installed CUDA/PyTorch/vLLM stack.
- The 4B checkpoint is several gigabytes (allow at least 10 GB free for the
  model files and cache). A GPU with around 16 GB or more VRAM is a practical
  target for running the full-precision model with room for inference. CPU
  execution may work with compatible libraries, but can be very slow and needs
  ample system RAM.
- A network connection is needed for the initial download. Afterward, the
  model files can be used offline.
- Install the FireLens repository dependencies in one environment, and install
  vLLM in a separate Linux environment as described below. vLLM's platform and
  CUDA support depend on your hardware; consult its current installation guide
  if installation fails.

## 1. Get FireLens and install its dependencies

From a terminal (use the WSL/Linux shell if you are on Windows):

```bash
git clone https://github.com/ihydra-pk/NASA-Space-Apps-FIRECRACKERS.git
cd NASA-Space-Apps-FIRECRACKERS
git checkout SI_Dashboard_Ravish

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Download the Qwen model files

Install the Hugging Face download utility and download the official checkpoint
to a local folder. The first download is large and can take a while.

```bash
python -m pip install -U huggingface_hub
hf download Qwen/Qwen3-Embedding-4B \
  --local-dir ./.local/models/Qwen3-Embedding-4B
```

The model directory is local and should not be committed to Git. Keep it on a
disk with enough free space. For later offline starts, set Hugging Face's
offline flags before launching the server:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Only set these after the download has completed successfully.

## 3. Start a local OpenAI-compatible embeddings server

FireLens's existing real embedding adapter calls an OpenAI-compatible
`POST /v1/embeddings` endpoint. vLLM provides that endpoint for embedding
models. Install vLLM in a **separate** environment (vLLM is primarily supported
on Linux; use its official installation instructions for your CUDA/PyTorch
combination):

```bash
python3 -m venv .venv-vllm
source .venv-vllm/bin/activate
python -m pip install --upgrade pip
python -m pip install vllm
```

Start Qwen from the downloaded local path, on the default local-only address:

```bash
vllm serve ./.local/models/Qwen3-Embedding-4B \
  --task embed \
  --trust-remote-code \
  --served-model-name gwdg.qwen3-embedding-4b \
  --port 8001 \
  --dtype auto
```

Leave this terminal running. The first start loads the local checkpoint and can
take a few minutes. The `--served-model-name` value intentionally matches the
model identifier recorded by FireLens's existing Qwen index. Do not expose this
server to the public internet without adding appropriate authentication and
network protections.

In a second terminal, test the server. The response should contain an
embedding with **2,560 numbers**:

```bash
curl http://127.0.0.1:8001/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"gwdg.qwen3-embedding-4b","input":"A NASA flame experiment in microgravity."}'
```

If your server requires an API key, configure the same key in FireLens below.
Do not paste any real secret into source files or commit it.

## 4. Use the local model to generate FireLens document embeddings

This step sends the repository's 4,103 chunks to the local vLLM server. Run it
from the repository root in the FireLens `.venv` (not `.venv-vllm`):

```bash
source .venv/bin/activate
export FIRELENS_EMBEDDING_BASE_URL=http://127.0.0.1:8001/v1
export FIRELENS_EMBEDDING_API_KEY=local

python -m src.embeddings \
  --input-dir data/chunks \
  --output-dir data/local-embeddings \
  --provider openai-compatible \
  --model gwdg.qwen3-embedding-4b \
  --dimension 2560 \
  --batch-size 16
```

`local` is a placeholder for a server with no authentication; it is not a
provider credential. If you enabled authentication in vLLM, set the actual
local server key in `FIRELENS_EMBEDDING_API_KEY` instead. The new embedding
files are written under this ignored local-only path, so the committed Qwen
vectors are not overwritten:

```text
data/local-embeddings/openai-compatible/gwdg.qwen3-embedding-4b/
```

This does not alter the source chunks. If the command is interrupted, resume
without re-requesting already completed files by adding `--skip-existing`.
Review the printed embedded/failed counts and any `*.errors.jsonl` files before
building the index; do not index an incomplete embedding run for a full-corpus
demo.

> **Already have the committed Qwen embeddings?** You can skip this step and
> index those files instead, as in the next section. To ensure maximum
> reproducibility, use the same Qwen checkpoint and embedding configuration
> that generated the document vectors.

## 5. Build the Qwen vector index

From the repository root, index the local embeddings. This builds a SQLite
database; it does not make any API calls:

```bash
python -m src.vectorstore index \
  --input-dir data/local-embeddings/openai-compatible/gwdg.qwen3-embedding-4b \
  --database data/vectorstore/firelens.sqlite3 \
  --collection firelens-qwen3-embedding-4b \
  --provider openai-compatible \
  --model gwdg.qwen3-embedding-4b \
  --dimension 2560 \
  --allow-duplicates
```

For the full repository corpus, the indexer should report **4,103 input
records**. If it reports fewer, check the input directory and complete the
embedding run before continuing.

## 6. Point FireLens query embedding at local Qwen

The local server must remain running while FireLens is running. In the terminal
where you start FireLens, set:

```bash
source .venv/bin/activate
export FIRELENS_EMBEDDING_BASE_URL=http://127.0.0.1:8001/v1
export FIRELENS_EMBEDDING_API_KEY=local
export FIRELENS_API_EMBEDDING_PROVIDER=openai-compatible
export FIRELENS_API_EMBEDDING_MODEL=gwdg.qwen3-embedding-4b
export FIRELENS_API_EMBEDDING_DIMENSION=2560
export FIRELENS_API_VECTOR_DATABASE=data/vectorstore/firelens.sqlite3
export FIRELENS_API_COLLECTION=firelens-qwen3-embedding-4b
```

FireLens adds its Qwen retrieval instruction to each **query** and sends it to
the local embedding server. Indexed **documents** are embedded as plain text.
Do not change the model or dimension on only one side; query and document
vectors must come from a compatible Qwen model/configuration.

Start the FireLens API:

```bash
python -m src.api
```

Open <http://127.0.0.1:8000> in a browser. vLLM was assigned port 8001 above,
leaving port 8000 for the FireLens API.

Local Qwen handles **search embeddings only**. The final natural-language answer
still needs a chat/LLM backend configured for FireLens. If that backend is
unavailable, FireLens may show retrieved NASA passages as a fallback instead
of a synthesized answer. The existing Render demo is separate and continues to
use its configured hosted services; a model running on a reviewer’s laptop is
not automatically accessible to Render.

## Troubleshooting checklist

- **Out of memory / CUDA out of memory:** close other GPU workloads, reduce
  model serving batch/concurrency settings, or try a smaller Qwen embedding
  variant. A different model or dimension requires regenerating all document
  embeddings and rebuilding the matching index.
- **`Connection refused`:** make sure vLLM is still running and the URL/port in
  `FIRELENS_EMBEDDING_BASE_URL` matches it.
- **401/403 from local endpoint:** use the same local vLLM API key in
  `FIRELENS_EMBEDDING_API_KEY`; never put it in the repository.
- **Dimension/profile mismatch:** confirm the server returns 2,560 values and
  that index and API use provider `openai-compatible`, model
  `gwdg.qwen3-embedding-4b`, dimension `2560`, and the same database and
  collection.
- **NASA passages appear without a synthesized answer:** local Qwen retrieval
  may be working, while the separate answer-generation backend is unavailable.

## References

- [Official Qwen3-Embedding-4B model card](https://huggingface.co/Qwen/Qwen3-Embedding-4B)
- [Qwen3-Embedding project and usage examples](https://github.com/QwenLM/Qwen3-Embedding)
- [vLLM online serving documentation](https://docs.vllm.ai/en/stable/serving/online_serving/openai_compatible_server/)
- [FireLens embedding documentation](src/embeddings/README.md)
