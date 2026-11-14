# Local vector store and semantic retrieval

The vector-store stage reads **only embedding JSONL**. It does not read
original documents, normalized records, or chunk files. The current real FLEX
embeddings are under
`data/embeddings/openai-compatible/gwdg.qwen3-embedding-4b/flex/`.

## Technology and identity

The MVP uses Python's built-in SQLite for persistent local storage. Vectors are
stored without quantization as JSON float arrays; the full embedding record is
stored alongside each vector. `chunk_id` is the primary key within a named
collection. Re-indexing updates the existing row (idempotent upsert); repeated
IDs within one indexing input are reported as duplicate errors. A collection
is bound to one provider, model, dimension, and similarity metric. Create a
new collection for a different embedding profile.

Search uses cosine similarity and returns citation-ready chunk provenance.
Scores are ranking signals, not statements of scientific correctness.

## Index real FLEX vectors

```powershell
python -m src.vectorstore index `
  --input-dir data/embeddings/openai-compatible/gwdg.qwen3-embedding-4b/flex `
  --database data/vectorstore/flex-qwen3-embedding-4b.sqlite3 `
  --collection flex-qwen3-embedding-4b `
  --provider openai-compatible `
  --model gwdg.qwen3-embedding-4b `
  --dimension 2560
```

## Search

Use the same real model that produced the indexed vectors. Set the API key in
`FIRELENS_EMBEDDING_API_KEY`, `OPENAI_API_KEY`, or (for the configured UPB
gateway) `UPB_AI_GATEWAY_KEY`. To use a compatible gateway, configure its base
URL explicitly so the matching gateway credential is selected:

```powershell
$env:FIRELENS_EMBEDDING_BASE_URL = "https://ai-gateway.uni-paderborn.de/v1"
python -m src.vectorstore search "How is limiting oxygen index determined?" `
  --database data/vectorstore/flex-qwen3-embedding-4b.sqlite3 `
  --collection flex-qwen3-embedding-4b `
  --provider openai-compatible `
  --model gwdg.qwen3-embedding-4b `
  --dimension 2560 --top-k 3
```

Query-provider errors and dimension/profile mismatches fail explicitly. The
mock provider remains available for deterministic unit tests but is rejected
when querying the real Qwen collection. This layer stops at ranked evidence; it
does not build RAG context or generate answers.
