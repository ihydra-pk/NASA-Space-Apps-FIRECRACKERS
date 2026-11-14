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

- Condition evidence comes from planned FLEX matrix rows only; it is never
  presented as observed and contains no fabricated outcomes or measurements.
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

Only the local FLEX test-plan conditions are structured. No verified observed
measurements are currently available; temperature and per-test gravity are
not in the structured rows. Semantic retrieval still depends on configured
embedding credentials/profile and existing vector index. End-to-end tests use
deterministic adapters and do not make network calls. The current system
provides NASA document evidence, structured condition matching, and grounded
explanation. **It is not a scientific prediction model and does not predict
flame behavior from physics.**

The separated JSON boundary (`question`, conditions, typed evidence arrays,
condition summary, answer/status, citations, warnings) is stable enough for
frontend integration. Before deployment, install project dependencies, ingest
the experimental source, configure credentials/index profile, and run the full
pytest suite against the target environment. No live LLM/API test was run for
this report.
