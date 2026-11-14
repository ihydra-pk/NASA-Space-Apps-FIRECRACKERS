# Chunk embeddings

The embedding stage reads only JSONL chunks from `data/chunks/` and writes
separate JSONL records beneath `data/embeddings/`:

```powershell
python -m src.embeddings
python -m src.embeddings --dimension 768 --batch-size 32
```

The default `mock` provider (`firelens-hash-v1`) creates repeatable hash-derived
vectors for pipeline tests. These vectors are **not semantic embeddings** and
must not be used as if they represented scientific text similarity.

`EmbeddingProvider` defines the batch interface for a future real provider.
The `openai-compatible` provider makes batched `POST /embeddings` requests to
an OpenAI-compatible API. Select it explicitly:

```powershell
$env:FIRELENS_EMBEDDING_API_KEY = "<your key>"
$env:FIRELENS_EMBEDDING_BASE_URL = "https://api.openai.com/v1" # optional
python -m src.embeddings --provider openai-compatible `
  --model text-embedding-3-small --dimension 1536 --batch-size 16
```

Provider, model, dimension, and batch size can alternatively be configured
with `FIRELENS_EMBEDDING_PROVIDER`, `FIRELENS_EMBEDDING_MODEL`,
`FIRELENS_EMBEDDING_DIMENSION`, and `FIRELENS_EMBEDDING_BATCH_SIZE`.

The key may also be supplied as `OPENAI_API_KEY`; the existing `UPB_AI_GATEWAY_KEY`
is accepted for compatible gateways. `text-embedding-3-small` is a configurable
example model, not a guarantee that every compatible gateway provides it. The
endpoint must offer the OpenAI embeddings API schema and support the selected
model. `text-embedding-3` models receive the requested `dimensions` parameter;
other models are checked against the configured native dimension. Missing
credentials, API errors, and incompatible response vectors fail explicitly;
there is no mock fallback.

The configured UPB gateway was validated with the real
`gwdg.qwen3-embedding-4b` model at its native 2560 dimensions. Its compatible
gateway credential can be supplied as `UPB_AI_GATEWAY_KEY`.

Other providers can implement `EmbeddingProvider` and be passed to
`run_embedding(config, provider=...)`. Provider secrets belong in environment
variables or external provider configuration, not source code or chunk records.
Real-provider/model outputs are isolated under
`data/embeddings/<provider>/<model>/` so they do not mix with the existing mock
outputs in `data/embeddings/`.

All original chunk fields are retained in each successful embedding record.
Malformed chunks, empty text, and failed provider batches are reported in the
returned result and written to a sibling `*.errors.jsonl` file; they are not
silently dropped. The input chunk files are read-only to this stage.
