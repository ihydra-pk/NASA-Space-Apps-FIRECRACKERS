# Explore pipeline validation report

## Scenarios exercised

Offline end-to-end tests compose the real Explore orchestration, RAG context
builder, existing condition matcher, and grounded LLM citation validator using
deterministic retrieval/LLM fixtures. Structured test rows are parsed from a
fixture shaped like actual FLEX Table 6 rows. They remain `planned`; no result
or measurement is created.

| Scenario | Result / evidence mode |
| --- | --- |
| Exact methanol, 21% O2, 1 atm, quiescent flow plus relevant page-6 text | Exact planned row (test 24), retrieved document; `combined` |
| Temperature and gravity requested | Both explicitly `UNAVAILABLE`; other parameters report `AVAILABLE` |
| 20.8% oxygen near planned 21% row | `close_match`, preserving requested/source values, 0.002 mole-fraction difference (0.2 percentage points), and 0.005 tolerance |
| No matching condition, relevant document still retrieved | No structured evidence emitted; document remains; `document_only` |
| Document-only question without conditions | Matcher skipped; `document_only` |
| Structured exact condition but no retrieved document chunks | Planned record remains in response; LLM answer remains insufficient and uncited; `experimental_only` |
| Low semantic score plus condition record | Similarity remains 0.12 and separate from match details; `combined` |
| Neither relevant document nor condition record | No evidence, no citations; `insufficient` |
| Planned-only conditions queried as observations | `experimental_only`, evidence remains `planned`, answer insufficient |
| Partial and unsupported-parameter cases | Partial and `parameter_unavailable` remain distinct in per-field details |

Additional checks cover valid citation provenance and rejection of invented
chunk/source/page/status values by the existing grounded LLM validator; raw
source text; source IDs, paths, locators, page and hash preservation; planned
status; bounded frontend JSON; and condition/unit availability.

## Integrity and behavior

- Condition evidence comes from planned FLEX matrix rows and observed NASA PSI
  table rows (FLEX, BASS-II, SAFFIRE-I/II). Each keeps its own source status; a
  planned row is never presented as observed, and no outcome is invented.
- RAG source text passes through unchanged as evidence. Similarity is retained
  only on document evidence; condition type/differences/tolerance are retained
  only on structured evidence.
- Citations continue to be generated/validated by the existing LLM service
  against retrieved RAG chunk IDs and exact provenance. Structured record IDs
  are returned as evidence provenance, not forged RAG citations.
- `AVAILABLE`, `UNAVAILABLE`, and `NO_DATA` distinguish parameter support from
  a value matching. Match summaries distinguish exact, close, partial,
  no-match, and parameter-unavailable counts/details.
- `evidence_mode` reflects evidence presence; `status` reports the existing
  grounded-answer result. These are not confidence/quality scores.

## Limitations and readiness

Structured data covers FLEX planned conditions plus observed tables for FLEX,
BASS-II and SAFFIRE-I/II. Original BASS and SAFFIRE-III are metadata only.
Temperature, time series and per-test gravity are not in the structured rows,
and answers say so instead of inventing them. When a question asks for
observed/measured results, observed records are ordered ahead of planned ones
before the bounded prompt is built (order only; nothing is dropped or
relabelled). The prompt and the response are still capped, so lists can be
incomplete and the answer says when records were omitted.

Live runs against the real index and gateway LLM (SAFFIRE-I, SAFFIRE-II,
BASS-II, FLEX) returned grounded answers whose values matched the source CSVs,
with citations accepted by the validator. Semantic retrieval still depends on
the configured embedding credentials and the existing vector index (the UPB
gateway is the default when only `UPB_AI_GATEWAY_KEY` is set). Unit tests use
deterministic adapters and make no network calls. The current system
provides NASA document evidence, structured condition matching, and grounded
explanation. **It is not a scientific prediction model and does not predict
flame behavior from physics.**

The separated JSON boundary (`question`, conditions, typed evidence arrays,
condition summary, answer/status, citations, warnings) is stable enough for
frontend integration. Before deployment, install project dependencies, ingest
the experimental source, configure credentials/index profile, and run the full
pytest suite against the target environment.
