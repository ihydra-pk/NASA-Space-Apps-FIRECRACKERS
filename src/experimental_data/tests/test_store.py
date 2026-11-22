from src.experimental_data.models import ExperimentRecord
from src.experimental_data.store import ExperimentStore


def test_store_replacement_is_repeatable_and_round_trips(tmp_path):
    store = ExperimentStore(tmp_path / "records.sqlite3")
    record = ExperimentRecord(
        record_id="one",
        kind="test_condition",
        experiment="FLEX",
        experiment_id="1",
        evidence_status="planned",
        source_id="nasa-hash",
        source_file="plan.pdf",
        source_locator="page:6:table:6:line:1",
        content_sha256="hash",
        units={"pressure": "atm"},
    )
    store.replace_sources({"plan.pdf"}, [record])
    store.replace_sources({"plan.pdf"}, [record])
    loaded = store.all_records()
    assert loaded == [record]
