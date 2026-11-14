# Structured experimental-data ingestion and matching

This package is deliberately separate from document ingestion, chunking,
embeddings, vector search, RAG, and grounded LLM code. It reads the original
files under `data/documents/`, writes a local SQLite database, and never edits
source documents.

## Inventory and schema basis

The current original sources are:

| Source | Format / actual contents | Structured use | Evidence |
| --- | --- | --- | --- |
| `data/documents/flex/FLEX_Test_Plan.pdf` | PDF pages; FLEX proposed test matrix Tables 6–12 and descriptive text | Numbered matrix rows: test number, fuel formula, pressure, oxygen mole fraction; suppressant and its mole fraction for Tables 8–12; flow specified in the table preamble; fiber support when stated | `planned` only |
| `data/documents/flex/FLEX DoD.docx` | Word document; description of FLEX data-record organization and file types | One `source_metadata` record, not fabricated test rows | `metadata` |

The project also contains generated `data/processed/`, `data/chunks/`, and
`data/embeddings/` JSONL artifacts derived from those two sources. They preserve
document text/chunks/vectors rather than independently reported test-result
records, so this component intentionally reads the original PDF/DOCX only and
does not re-ingest derived copies.

The matching experiment conditions are the fields explicitly represented in
individual test-matrix rows or their associated table preambles: fuel, oxygen
mole fraction, pressure, flow velocity, suppressant, diluent, and fiber
support where available. Pressure is
shown as atm; flow as cm/s; oxygen and suppressant concentrations are `X`
mole-fraction values (for example, `0.21`, not `21`). Fuel formulae `C7H16`
and `CH3OH` are normalized to `heptane` and `methanol`, while the exact source
spelling remains in `original_values`. `N2` is preserved as the table's stated
diluent, and categorical condition values are matched using case-insensitive
equality after whitespace normalization. Fiber support retains the source
phrase/value (`none` or the stated support type) rather than being inferred for
tables where it is omitted.

The project has no separate observed experiment-level measurement file. The
test plan describes measurement requirements (including ambient temperature
and pressure), outcomes such as extinction, and proposed test procedures, but
does not supply per-test observed temperature, measurement series, or results.
Accordingly, no `observed_result`/measurement values are populated and no plan
row is labeled observed. The DOCX describes external raw/analyzed TXT records,
videos, and images (and names record groups), but those assets are not present
in this project's data directories. It also supplies no row-level experiment
conditions. NASA URL/accession is null in the project metadata, and the planned
test matrix is stated to be subject to change during the experiment.

Fields not currently available per test and therefore not treated as
matchable: gravity, temperature, observed outcome, actual test date/time,
uncertainties, and calibrated measurements. Gravity/environment appears in
document prose, not as a per-row condition. The plan specifies temperature
measurement accuracy in prose, not actual temperature values. These omissions
remain null/unavailable; the component does not synthesize values from prose.

## Record and provenance behavior

`ExperimentRecord` stores only the dataset-backed conditions above, evidence
status, `source_file`, SHA-256-derived `source_id`, original byte-level
`content_sha256`, and a page/table/line source locator. Original test numbers
are retained as `experiment_id`, and original condition strings/formulae are
preserved under `original_values`. Unknown units are null rather than guessed.
PDF matrix values have explicit units from their table labels/preambles; source
metadata records do not have experimental units.

The loader recognizes rows only in Tables 6–12 and only parses fully numeric
table rows. The condition number is not promoted to an observed result. It
does not parse prose references as measurements. Data format dependencies
(`pypdf` and `python-docx`) are already declared in the root `requirements.txt`.

## Ingest

Run from the repository root:

```powershell
python -m src.experimental_data
python -m src.experimental_data --json
# Optional isolated output for dry runs:
python -m src.experimental_data --database data/experimental_data/dry-run.sqlite3
```

Output is persisted at `data/experimental_data/experiments.sqlite3` by default.
Repeated ingestion is deterministic and replaces rows for successfully
processed source paths; an unreadable source does not delete its prior stored
records. Reports include source files discovered/processed, candidate records
discovered, normalized/rejected counts, source errors, and missing/invalid
fields. Records are immutable snapshots of the values and provenance of the
input files at ingestion time.

## Condition matching

Use `match_experiments(records, conditions, absolute_tolerances=...)` or
`ExperimentalDataService.match_conditions(conditions)` to match persisted
records. Numeric matching reports requested value, source value, difference, unit, and the
configured absolute tolerance. Defaults are software matching settings, not
NASA uncertainty bounds, and can be replaced by the caller. A zero difference
is exact; a nonzero difference within a positive configured tolerance is
close—not exact. Oxygen tolerance is in mole-fraction units (0.005 equals 0.5
percentage points). Unsupported parameters (including gravity and temperature)
are explicitly `parameter_unavailable`. A record-level `partial_match` means
some requested parameters matched but others did not or were unavailable.
No-match records remain in results so the caller can inspect every comparison.

There is no physics model or inference: matching identifies existing planned
conditions only. It cannot answer what a real experiment observed.

## Tests

```powershell
python -m pytest src/experimental_data/tests -q
```
