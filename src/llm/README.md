# Grounded LLM answers

The answer service accepts an existing structured `RAGContext`. It does not
retrieve, read documents, access embeddings/vector storage, or generate a
combustion prediction.

```text
User question
     ↓
Retrieval
     ↓
RAGContext
     ↓
Evidence Policy
     ↓
LLM
     ↓
Structured Answer
     ↓
Citation Validation
```

## Configuration

The OpenAI-compatible chat provider uses environment configuration. The UPB
gateway defaults match the existing OpenCode configuration:

```powershell
$env:UPB_AI_GATEWAY_KEY = "<your key>"
$env:FIRELENS_LLM_BASE_URL = "https://ai-gateway.uni-paderborn.de/v1" # optional
$env:FIRELENS_LLM_MODEL = "openai.gpt-6-luna" # optional
$env:FIRELENS_LLM_TIMEOUT = "90"
$env:FIRELENS_LLM_TEMPERATURE = "0"
$env:FIRELENS_LLM_MAX_OUTPUT_TOKENS = "1200"
```

`FIRELENS_LLM_API_KEY` and `OPENAI_API_KEY` are also supported. The provider
uses the OpenAI-compatible `/chat/completions` endpoint and requests JSON
output. No credentials belong in source files.

## Evidence policy and citations

The system prompt instructs the model to use only supplied evidence and
preserve `planned`, `observed`, `interpreted`, `metadata`, and `unknown`
statuses. Empty evidence is handled locally as `insufficient_evidence`; the
model is not called. For non-empty context, output status and citation IDs are
validated. Citations must reference supplied chunk IDs and match the exact
source ID, locator, page, and evidence status; returned citation provenance is
copied from `RAGContext`, never trusted as model-authored metadata. Invalid or
invented citations are rejected.

Similarity remains a retrieval/ranking signal, never confidence. This component
generates explanations from retrieved evidence; it is not a scientific
simulation, a predictive combustion model, or a source of experimental results.
No response is proof of scientific correctness; every factual claim should be
checked against its cited evidence.

Use `MockLLMProvider` for deterministic tests. An optional real-provider smoke
test is guarded by `FIRELENS_RUN_LLM_SMOKE=1`; normal tests make no API calls.
