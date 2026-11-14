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
$env:FIRELENS_LLM_MAX_OUTPUT_TOKENS = "4000" # default
```

`FIRELENS_LLM_API_KEY` and `OPENAI_API_KEY` are also supported. The provider
uses the OpenAI-compatible `/chat/completions` endpoint and requests JSON
output. No credentials belong in source files.

## Answer contract

The model returns one JSON object:

```json
{"status": "grounded | partial | insufficient_evidence",
 "answer": "direct answer; NASA-based statements end with [chunk-id]",
 "findings": [{"statement": "...", "support": "direct | inference", "chunk_ids": ["chunk-..."]}],
 "scientific_context": "general combustion science, no chunk ids",
 "limitations": "what the evidence does not establish"}
```

- `findings` are the claims the NASA evidence supports. `direct` means a cited
  chunk states the claim; `inference` means it is reasoned from the cited chunks
  together. Every finding cites at least one supplied chunk.
- `scientific_context` explains the findings from general science. It carries
  no citations and is never attributed to NASA; the page shows it in a separate
  box.
- `partial` answers part of the question and must say in `limitations` what is
  not covered. `insufficient_evidence` is for evidence that is not relevant at
  all, and the answer must say "insufficient".

## Evidence policy and citations

The system prompt asks the model to reason across every supplied chunk rather
than refuse when no single chunk answers the question, and keeps
`planned`, `observed`, `interpreted`, `metadata` and `unknown` distinct.
`unknown` means a document has not been labelled yet, not that it is weak; the
prompt also explains NASA's filename categories (`Reports`, `Science Documents`
written before flight, ...) and that a bibliography entry is not a finding.
Empty evidence is handled locally as `insufficient_evidence`; the model is not
called.

The model writes chunk IDs only. `citations` are built by the service from the
IDs in `findings`, in `[chunk-id]` markers of `answer`, and in `limitations`,
with source ID, locator, page and evidence status copied from `RAGContext`, so
model output can never put an invented source, page or status into a citation.
A reply is rejected when it:

- cites a chunk ID that was not supplied, anywhere in the reply;
- claims NASA or the experiments observed, measured or found something while
  citing a `planned` or `metadata` chunk (checked per finding and per sentence;
  `unknown` chunks may support such claims);
- puts chunk IDs in `scientific_context`, gives a grounded or partial answer
  without findings, or a partial one without limitations;
- uses the word "confidence".

A rejected reply gets one repair attempt: the model is asked again with the
reason it was rejected. If that also fails, `LLMResponseError` is raised, which
the API returns as HTTP 502.

Similarity remains a retrieval/ranking signal, never confidence. This component
generates explanations from retrieved evidence; it is not a scientific
simulation, a predictive combustion model, or a source of experimental results.
No response is proof of scientific correctness; every factual claim should be
checked against its cited evidence.

Use `MockLLMProvider` for deterministic tests. An optional real-provider smoke
test is guarded by `FIRELENS_RUN_LLM_SMOKE=1`; normal tests make no API calls.
