# Unified FireLens Explore pipeline

`src.explore` composes existing interfaces. It does not replace any document
ingestion, chunking, embedding, vector-store, RAG, structured-data, or grounded
LLM implementation.

```text
ExploreRequest
  ├─ question → existing retriever → retrieve_context → RAGContext ─┐
  └─ conditions → ExperimentalDataService.match_conditions ─────────┤
                                                                    ↓
                                              unified evidence + grounded LLM
```

The Explore response retains the original `RAGContext`, structured
`MatchResult`, typed evidence arrays, and validated `AnswerResponse`. Its JSON
contract separates `document_evidence`, `experimental_evidence`,
`condition_match_summary`, `answer`, `citations`, `status`, and `evidence_mode`.
A
document item's semantic `similarity_score` stays on `DocumentEvidence`. A
condition item's `match_type`, requested/actual values, difference, unit, and
tolerance stay on `ExperimentalEvidence`. These are separate scales and are
never combined into a confidence score.

Each requested condition also has a frontend-friendly availability entry:
`AVAILABLE` means source records contain values for that parameter,
`UNAVAILABLE` means the FLEX structured schema does not contain it (e.g.
temperature or per-test gravity), and `NO_DATA` means the parameter is part of
the supported schema but there were no structured rows loaded. Availability
is distinct from whether the requested value exactly/closely matches a record.
The serialized match summary includes the total candidate count, match-status
counts, and a bounded relevant-record sample; the full `MatchResult` remains
available on the in-process response object. No-match candidates are counted
but omitted from the default evidence sample to avoid returning hundreds of
unhelpful records to the dashboard.

`evidence_mode` reports which evidence classes were actually returned:
`document_only`, `experimental_only`, `combined`, or `insufficient`. This is
about evidence presence, not a quality or certainty rating. `status` is the
answer validator's `grounded`, `partial` (part of the question is answered;
`answer.limitations` says what is missing) or `insufficient_evidence`. An experimental-only evidence set can therefore accompany an
`insufficient_evidence` answer when there are no document citations available.

## Evidence and status

Every unified evidence item carries its evidence type, experiment, source ID,
source file, source locator, page or record identifier, evidence status,
original source text/data, and provenance. Document provenance includes chunk
ID, content hash, filename, source path, and document type. Structured
provenance includes source file, record/test ID, table/page locator, and source
content hash. Source evidence status is copied unchanged. FLEX rows are marked
`planned`; they are proposed test conditions, not observations/results.

The grounded LLM's existing citation validator remains based on the original
RAGContext chunk evidence. Structured matches are supplied as a separate
explicitly labeled section of the LLM context and are returned as first-class
Explore evidence; they are not misrepresented as document chunks or citations.
If no document chunks are retrieved, the existing LLM service returns its
insufficient-evidence response rather than asking the model to answer from
memory. Structured matching results remain available in the Explore response.
`answer` carries the LLM service's contract (see `src/llm/README.md`): the
direct `answer`, `findings` marked `direct` or `inference` with their chunk IDs,
`scientific_context` (general science, never cited or attributed to NASA) and
`limitations`. Use the top-level unified `evidence` list for both evidence types
and provenance.

## Request fields and units

```python
ExploreRequest(
    question="What can NASA evidence tell us about these conditions?",
    conditions={
        "fuel": "Methanol",
        "oxygen_concentration": 21,  # public request convention: percent
        "pressure": 1,               # atm
        "temperature": 300,          # K; unsupported by FLEX row data
        "gravity": "microgravity",  # unsupported by FLEX row data
        "flow_velocity": 0,          # cm/s
    },
)
```

Defaults for known request parameters: oxygen concentration is percent,
pressure is atm, and flow is cm/s. The service converts oxygen percent to
mole-fraction for the existing matcher. Explicit unit declarations are
available via `condition_units`, e.g. `{"oxygen_concentration":
"mole_fraction", "pressure": "kPa", "flow_velocity": "m/s"}`. Supported
pressure units are atm, kPa, Pa; flow units cm/s and m/s; oxygen/suppressant
concentration units percent or mole fraction. No unit is guessed for arbitrary
fields. Temperature is retained with its stated unit and the matcher reports
it unavailable (temperature is not present per test in the current structured
dataset). Gravity is similarly unavailable as a test-level field.

Document search always uses the `question` string, plus the router's search
queries when the router is on (see below). Structured conditions are matched
separately; request fields are not merged into semantic text or converted into
another document-search query.

Unrecognized condition names are preserved through to the matcher, which marks
them `parameter_unavailable`; they are not silently dropped. Units for unknown
or categorical fields are rejected rather than ignored.

## Use as a library

Pass the already configured retrieval callable, existing
`ExperimentalDataService`, and existing `LLMService`:

```python
from src.explore import ExploreRequest, ExploreService
from src.experimental_data.service import ExperimentalDataService
from src.llm.service import LLMService
from src.rag.config import RAGContextConfig
from src.vectorstore.config import VectorStoreConfig
from src.vectorstore.search import search_many
from src.vectorstore.sqlite_store import SQLiteVectorStore

vector_config = VectorStoreConfig(...)  # same collection/profile as indexed vectors

def retrieve(query, *, top_k, extra_queries=()):
    with SQLiteVectorStore(vector_config) as store:
        return search_many([query, *extra_queries], store, vector_config, top_k=top_k)

explore = ExploreService(
    retriever=retrieve,
    experimental_data=ExperimentalDataService(),
    llm=LLMService(),
)
response = explore.explore(ExploreRequest("Question?", {"fuel": "methanol"}))
print(response.to_dict())
```

Configure embeddings and LLM credentials exactly as the existing components
require. No endpoint, database, provider, or credential is silently changed by
the library adapter.

## CLI

```powershell
python -m src.explore "What can NASA evidence tell us about these conditions?" `
  --conditions '{"fuel":"Methanol","oxygen_concentration":21,"pressure":1,"temperature":300,"gravity":"microgravity","flow_velocity":0}'
```

The CLI uses the project's existing vector and structured SQLite defaults. The
experimental database must first be populated by `python -m
src.experimental_data`. See `python -m src.explore --help` for provider/profile
overrides. Configure API credentials using the existing embedding and LLM
environment variables.

## Tests

```powershell
python -m pytest src/explore/tests -q
```

The end-to-end suite uses realistic FLEX record combinations and fake retrieval
and LLM adapters; it does not require network access or assert invented NASA
observations. Install the project's declared dependencies, then run the suite
with the repository-wide tests before frontend integration.
See [EVALUATION.md](EVALUATION.md) for scenario outcomes and known limitations.

## Question routing and hybrid search

`python -m src.explore "question"` no longer needs `--conditions`. A router
(`router.py`) reads the question with one LLM call, labels it `documents`,
`test_data` or `both`, extracts the conditions it states, and writes 2 to 4
search queries for the aspects the question's own wording may miss (mechanism,
conditions, experiment names, acronyms):

```text
"Which FLEX tests are planned for methanol at 21% oxygen and 1 atm?"
  -> route: test_data
     conditions: {"fuel": "methanol", "oxygen_concentration": 21, "pressure": 1}
     condition_units: {"oxygen_concentration": "%", "pressure": "atm"}
     search_queries: ["FLEX methanol droplet test matrix", ...]
```

The decision is returned as `routing` in the response. The router's output is
checked before use: fields the test data does not have, unsupported units and
numbers that are not written in the question are dropped and listed in
`routing.ignored`. The router runs on every question. Conditions passed by
the caller (or read from the question by the API's regexes) take precedence
over the router's; `routing.conditions_applied` says whether the router's were
used. Its search queries are searched together with the question
(`search_many`), at most four of at most 200 characters each. If the router
cannot be reached or returns something unusable, the request continues as a
plain document question and a warning is added. Document retrieval runs on
every route, because answers are cited against retrieved chunks. Use
`--no-router` to turn it off.

Retrieval defaults to `--search-mode hybrid`: chunks are ranked by cosine
similarity and by keyword match, and the two rankings are fused by rank. This
helps questions that name a test number, acronym or formula. The reported
`similarity_score` is unchanged; it is always the cosine similarity. Use
`--search-mode semantic` for the previous behaviour.


Search also ranks reference lists after findings and adds the next page of a
report after each of the top 3 hits (`--keep-reference-lists`, `--neighbours`);
see `src/vectorstore/README.md`. The CLI gives the answer model 10 passages by
default (`--top-k`).
