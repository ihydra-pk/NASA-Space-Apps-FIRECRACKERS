from src.experimental_data.matching import match_experiments
from src.experimental_data.models import ExperimentRecord


def record(**overrides):
    values = {
        "record_id": "record-1",
        "kind": "test_condition",
        "experiment": "FLEX",
        "experiment_id": "10",
        "evidence_status": "planned",
        "source_id": "nasa-hash",
        "source_file": "data/documents/flex/plan.pdf",
        "source_locator": "page:6:table:6:line:10",
        "content_sha256": "hash",
        "fuel": "heptane",
        "oxygen_concentration": 0.21,
        "pressure": 1.0,
        "flow_velocity": 0.0,
        "units": {"oxygen_concentration": "mole_fraction", "pressure": "atm", "flow_velocity": "cm/s"},
    }
    values.update(overrides)
    return ExperimentRecord(**values)


def test_exact_and_close_are_distinguished_and_report_tolerance():
    result = match_experiments(
        [record()],
        {"fuel": "C7H16", "oxygen_concentration": 0.208},
        absolute_tolerances={"oxygen_concentration": 0.005},
    )
    match = result.matches[0]
    assert match.match_type == "close_match"
    oxygen = next(item for item in match.parameters if item.parameter == "oxygen_concentration")
    assert oxygen.requested_value == 0.208
    assert oxygen.experimental_value == 0.21
    assert round(oxygen.difference, 3) == 0.002
    assert oxygen.tolerance == 0.005
    assert next(item for item in match.parameters if item.parameter == "fuel").match_type == "exact_match"


def test_unavailable_parameter_is_explicit_and_produces_partial_match():
    result = match_experiments([record()], {"fuel": "heptane", "gravity": "microgravity"})
    match = result.matches[0]
    assert match.match_type == "partial_match"
    gravity = next(item for item in match.parameters if item.parameter == "gravity")
    assert gravity.match_type == "parameter_unavailable"
    assert gravity.experimental_value is None


def test_nonmatching_conditions_are_not_promoted_to_close_match():
    result = match_experiments(
        [record()], {"pressure": 1.2}, absolute_tolerances={"pressure": 0.01}
    )
    detail = result.matches[0].parameters[0]
    assert result.matches[0].match_type == "no_match"
    assert detail.match_type == "no_match"
    assert detail.difference == 0.2
    assert detail.tolerance == 0.01


def test_empty_candidate_set_is_explicitly_no_match():
    result = match_experiments([], {"fuel": "methanol"})
    assert result.match_type == "no_match"
    assert result.matches == ()
