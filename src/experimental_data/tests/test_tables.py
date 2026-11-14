import pytest
from pathlib import Path

from src.experimental_data.matching import match_experiments
from src.experimental_data.config import ExperimentalDataConfig
from src.experimental_data.service import ExperimentalDataService
from src.experimental_data.tables import load_table, parse_number


def write(tmp_path, name, text, encoding="utf-8"):
    path = tmp_path / name
    path.write_bytes(text.encode(encoding))
    return path


FLEX_HEADER = (
    "FLEX Test #,FLEX Identifier,Test Date,Test GMT,Fuel,Ambient pressure; mmHg,"
    "O initial ambient composition; mole fraction,N initial ambient composition; mole fraction,"
    "CO initial ambient composition; mole fraction,He initial ambient composition; mole fraction,"
    "Droplet initial diameter; mm,Visible flame extinction diameter; mm,Burning rate; mm,Burn time; s,Test end\n"
)


def test_flex_rows_are_observed_with_converted_pressure_and_measurements(tmp_path):
    path = write(
        tmp_path,
        "PSI-69_FLEX.csv",
        FLEX_HEADER
        + "1,20CAL08,3/5/2009,17:22:03,Methanol,760,0.21,0.79,0,0,1.69,0.89,0.52,4.3,Disruption\n"
        + "2,X,3/5/2009,1:00:00,Heptane,380,0.3,0.55,0.15,0,2,,0.6,5,Extinction\n"
        + "bad,row\n",
    )
    records, diagnostics = load_table(path, source_file="structured/flex/PSI-69_FLEX.csv")
    assert len(records) == 2 and len(diagnostics.rejected) == 1
    first, second = records
    assert first.evidence_status == "observed"
    assert first.fuel == "methanol" and first.pressure == pytest.approx(1.0)
    assert first.observed_result == "Disruption"
    assert first.measurement["burn_time_s"] == 4.3
    assert second.pressure == pytest.approx(0.5)
    assert second.suppressant == "CO2" and second.suppressant_concentration == 0.15
    assert "visible_flame_extinction_diameter_mm" not in second.measurement
    assert first.source_url.endswith("PSI-69")


def test_cp1252_encoding_and_matching_against_observed_rows(tmp_path):
    path = write(
        tmp_path,
        "PSI-69_FLEX.csv",
        FLEX_HEADER + "1,A,3/5/2009,1:00,Methanol,760,0.21,0.79,0,0,1.7,0.9,0.5,4,Extinction \u2013 ok\n",
        encoding="cp1252",
    )
    records, _ = load_table(path, source_file="f.csv")
    result = match_experiments(records, {"fuel": "Methanol", "pressure": 1, "oxygen_concentration": 0.21})
    assert result.match_type == "exact_match"


def test_saffire_skips_camera_rows_and_leaves_ranges_null(tmp_path):
    path = write(
        tmp_path,
        "PSI-98_SAFFIRE-I.csv",
        "Sample Number ,Material,Air Flow (cm/s),Percent O2,Burn Time (s)\n"
        "S1,SIBAL Fabric,20,21.5 - 21.7,420\n"
        "S2,SIBAL Fabric,20,~ 21.5,70\n"
        "F1C1/F1C2,Pre-test flow data,,,\n"
        "Note 1 something,,,,\n",
    )
    records, diagnostics = load_table(path, source_file="s.csv")
    assert [r.experiment_id for r in records] == ["S1", "S2"]
    assert records[0].oxygen_concentration is None
    assert records[0].original_values["oxygen_concentration_range"] == [0.215, 0.217]
    assert records[1].oxygen_concentration == pytest.approx(0.215)
    assert records[0].flow_velocity == 20
    assert records[0].measurement == {"Burn Time (s)": "420"}


def test_bass_log_rows_without_outcomes_are_metadata_not_observed(tmp_path):
    path = write(
        tmp_path,
        "PSI-26_BASS.csv",
        "Date,GMT,As Run Test #,Test #,Sample #,Fuel Sample Material\n"
        "3/30/2012,90,1,1,4,SIBAL 2 cm\n"
        "3/30/2012,90,2\n",
    )
    records, diagnostics = load_table(path, source_file="b.csv")
    assert len(records) == 1 and len(diagnostics.rejected) == 1
    assert records[0].evidence_status == "metadata"
    assert records[0].fuel == "SIBAL 2 cm"


def test_unknown_table_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_table(write(tmp_path, "PSI-999_x.csv", "a\n1\n"), source_file="x")


def test_parse_number():
    assert parse_number("~ 21.5") == (21.5, True)
    assert parse_number("20 cm/s") == (20.0, False)
    assert parse_number("n/a") == (None, False)
    assert parse_number("21.5 - 21.7") == (None, False)


def test_saffire_oxygen_range_matches_without_becoming_a_point_value(tmp_path):
    path = write(
        tmp_path,
        "PSI-98_SAFFIRE-I.csv",
        "Sample Number,Material,Air Flow (cm/s),Percent O2,Burn Time (s)\n"
        "S1,SIBAL Fabric,20,21.5 - 21.7,420\n",
    )
    records, _ = load_table(path, source_file="s.csv")
    result = match_experiments(records, {"oxygen_concentration": 0.216})
    detail = result.matches[0].parameters[0]
    assert records[0].oxygen_concentration is None
    assert detail.match_type == "close_match"
    assert detail.experimental_value == [0.215, 0.217]
    outside = match_experiments(records, {"oxygen_concentration": 0.22})
    assert outside.matches[0].parameters[0].match_type == "no_match"


def test_flex2_matrix_rows_are_planned_ranges(tmp_path):
    path = write(
        tmp_path,
        "PSI-68_Experimental table_Flex2.csv",
        "Sample (Fuel Type),Initial Droplet Diameter (mm),Ambient Gas Mixture ,Oxygen Mole Fraction,"
        "Ambient Pressure (atm), Ambient Temperature (C),Flow Velocity (cm/s)\n"
        "IsoOctane (pure),1.5 - 5,O2/N2 or O2/He,0.17; 0.21;0.25; 0.3,"
        "0.5; 1.0; 1.5; 2.0; or 3.0 ,25,0.1 - 5\n",
    )
    records, diagnostics = load_table(path, source_file="data/psi-download/PSI-68/table.csv")
    record = records[0]
    assert diagnostics.discovered == 1
    assert record.experiment == "FLEX-2"
    assert record.evidence_status == "planned"
    assert record.experiment_id == "IsoOctane (pure)"
    assert record.oxygen_concentration is None
    assert record.measurement["design_conditions"]["Oxygen Mole Fraction"] == "0.17; 0.21;0.25; 0.3"
    assert record.observed_result is None


def test_ingest_discovers_local_psi68_design_table(tmp_path):
    download = tmp_path / "data/psi-download/PSI-68/Experimental table"
    download.mkdir(parents=True)
    (download / "PSI-68_Experimental table_Flex2.csv").write_text(
        "Sample (Fuel Type),Initial Droplet Diameter (mm),Ambient Gas Mixture,Oxygen Mole Fraction,"
        "Ambient Pressure (atm),Ambient Temperature (C),Flow Velocity (cm/s)\n"
        "IsoOctane (pure),1.5 - 5,O2/N2 or O2/He,0.17; 0.21,0.5; 1.0,25,0.1 - 5\n",
        encoding="utf-8",
    )
    config = ExperimentalDataConfig(project_root=tmp_path, documents_dir=tmp_path / "no-docs",
                                    structured_dir=tmp_path / "no-structured",
                                    psi_download_dir=Path("data/psi-download"),
                                    store_path=Path("data/test-experiments.sqlite3"))
    service = ExperimentalDataService(config)
    report = service.ingest()
    assert any("PSI-68" in path for path in report.source_files_processed)
    assert len([r for r in service.records() if r.experiment == "FLEX-2"]) == 1


def test_saffire_direction_is_kept_as_measurement_context(tmp_path):
    path = write(
        tmp_path,
        "PSI-98_SAFFIRE-I.csv",
        "Sample Number ,Material,Air Flow (cm/s),Percent O2,Burn Time (s),Flow Direction \n"
        "S1,SIBAL Fabric,20,21.5,420,Concurrent\n",
    )
    records, _ = load_table(path, source_file="s.csv")
    assert records[0].measurement == {"Burn Time (s)": "420", "Flow Direction": "Concurrent"}


def test_bass_ii_keeps_initial_co_and_co2(tmp_path):
    path = write(
        tmp_path,
        "PSI-25_BASS-II.csv",
        "Test #,Fuel Sample Material,Calibrated  initial O2 % by vol ,Calibrated final O2 % by vol,"
        "Initial CO2 % by vol,Final CO2 % by vol,Initial CO (ppm),Final CO (ppm)\n"
        "B1,100 micron PMMA film   2 cm wide,22.2,22.2,0.44,0.44,0,6\n",
    )
    records, _ = load_table(path, source_file="b.csv")
    m = records[0].measurement
    assert m["initial co2 % by vol"] == 0.44
    assert m["initial co (ppm)"] == 0
    assert m["final co (ppm)"] == 6


def test_saffire_direction_alone_does_not_make_a_record_observed(tmp_path):
    path = write(
        tmp_path,
        "PSI-100_SAFFIRE-III.csv",
        "Sample Number ,Material,Air Flow ,Direction\nS1,SIBAL Fabric,30 cm/s,Concurrent\n",
    )
    records, _ = load_table(path, source_file="s.csv")
    assert records[0].evidence_status == "metadata"
