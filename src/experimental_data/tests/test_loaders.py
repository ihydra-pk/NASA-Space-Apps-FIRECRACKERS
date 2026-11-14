from src.experimental_data.loaders import parse_test_plan_pages


def test_parses_test_matrix_as_planned_and_preserves_source_values():
    pages = [
        (
            8,
            "Table 8: CO2 suppressant efficacy test matrix.\n"
            "Fuel: C7H16, Diluent: N2, Suppressant: CO2, No ﬁber support,\n"
            "Flow: 0 .0 cm/s\n"
            "Test No. P (atm) XO2 XCO2\n"
            "51 0.70 0.25 0.15\n",
        )
    ]
    records, diagnostics = parse_test_plan_pages(
        pages,
        source_file="data/documents/flex/FLEX_Test_Plan.pdf",
        content_sha256="abc123",
    )
    assert diagnostics.discovered == 1
    assert not diagnostics.rejected
    record = records[0]
    assert record.evidence_status == "planned"
    assert record.experiment_id == "51"
    assert record.fuel == "heptane"
    assert record.pressure == 0.7
    assert record.oxygen_concentration == 0.25
    assert record.suppressant == "CO2"
    assert record.suppressant_concentration == 0.15
    assert record.flow_velocity == 0.0
    assert record.observed_result is None
    assert record.measurement is None
    assert record.units["pressure"] == "atm"
    assert record.units["oxygen_concentration"] == "mole_fraction"
    assert record.source_locator == "page:8:table:8:line:5"
    assert record.original_values["fuel"] == "C7H16"


def test_does_not_treat_narrative_numbers_outside_test_tables_as_records():
    pages = [(1, "Table 2: historic results\n1 0.50 0.20\n")]
    records, diagnostics = parse_test_plan_pages(
        pages, source_file="plan.pdf", content_sha256="digest"
    )
    assert records == []
    assert diagnostics.discovered == 0
