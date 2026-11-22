from src.api.conditions import extract_conditions


def test_extracts_only_explicit_values():
    q = ("Find observed FLEX tests for Methanol with approximately 21% oxygen and around 760 mmHg "
         "ambient pressure.")
    assert extract_conditions(q) == {"fuel": "methanol", "oxygen_concentration": 21.0, "pressure": 1.0}


def test_flow_and_fuel():
    assert extract_conditions("SAFFIRE-I SIBAL fabric at 20 cm/s air flow") == {
        "fuel": "sibal", "flow_velocity": 20.0}


def test_nothing_guessed_from_plain_question():
    assert extract_conditions("How does smoke behave in microgravity?") == {}
