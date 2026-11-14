# RAG evidence context

This module packages existing ranked retrieval results into a structured,
traceable `RAGContext`. It does not access source files, JSONL corpora, vectors,
or SQLite directly and contains no LLM or answer-generation calls.

The next optional stage is a separate grounded answer service:

```text
User question → retrieval → RAGContext → Evidence Policy → LLM
             → structured answer → citation validation
```

That service lives in `src/llm/`; this package remains responsible only for
retrieval-result-to-context construction.

```python
from src.rag import RAGContextConfig, retrieve_context
from src.vectorstore.search import search

context = retrieve_context(
    "How is limiting oxygen index defined?",
    lambda query, top_k: search(query, store, vector_config, provider=provider, top_k=top_k),
    RAGContextConfig(top_k=5, max_context_chars=20_000),
)
```

Evidence items preserve rank, similarity score, exact retrieved text, source
IDs/hashes, document and page locator, and the original evidence status.
Null/unclassified evidence status is represented as `unknown`; statuses such
as `planned` and `metadata` remain unchanged. Similarity is never interpreted
as confidence.

When a configured character or estimated-token limit is exceeded, the builder
omits the lowest-ranked evidence items in full, adds a deterministic omission
notice, and records dropped chunk IDs. It never truncates scientific source
text. `estimated_context_tokens` is a rough lexical estimate, not a model
tokenizer measurement.
