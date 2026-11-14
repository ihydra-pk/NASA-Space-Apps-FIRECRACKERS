from __future__ import annotations

import json
import hashlib
import logging
from pathlib import Path

import pytest
from docx import Document as WordDocument

from src.ingestion import IngestionConfig, MetadataOverride, ingest_documents, ingest_file
from src.ingestion.pipeline import load_metadata_overrides


def _make_pdf(path: Path, page_texts: list[str]) -> None:
    """Create a minimal standards-compliant text PDF without another dependency."""

    def literal_string(text: str) -> bytes:
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        return escaped.encode("ascii")

    objects: list[bytes] = []
    page_refs = " ".join(f"{4 + 2 * index} 0 R" for index in range(len(page_texts)))
    objects.extend(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            f"<< /Type /Pages /Kids [{page_refs}] /Count {len(page_texts)} >>".encode("ascii"),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        ]
    )
    for index, text in enumerate(page_texts):
        page_object_number = 4 + 2 * index
        stream_object_number = page_object_number + 1
        stream = b"BT /F1 12 Tf 72 720 Td (" + literal_string(text) + b") Tj ET"
        objects.extend(
            [
                (
                    f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                    f"/Resources << /Font << /F1 3 0 R >> >> "
                    f"/Contents {stream_object_number} 0 R >>"
                ).encode("ascii"),
                b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n"
                + stream
                + b"\nendstream",
            ]
        )

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{number} 0 obj\n".encode("ascii"))
        pdf.extend(body)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(offsets)}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        (
            f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    path.write_bytes(pdf)


def _make_docx(path: Path, *, include_table: bool = True) -> None:
    document = WordDocument()
    document.core_properties.title = "FLEX fixture"
    document.core_properties.author = "NASA test team"
    document.add_paragraph("Experiment description.")
    if include_table:
        table = document.add_table(rows=2, cols=2)
        table.style = "Table Grid"
        table.cell(0, 0).text = "Fuel"
        table.cell(0, 1).text = "Pressure"
        table.cell(1, 0).text = "Methanol"
        table.cell(1, 1).text = "1 atm"
    document.add_paragraph("Closing note.")
    document.save(path)


def test_pdf_extraction_preserves_page_numbers_and_warns_for_low_text(tmp_path, caplog):
    source = tmp_path / "documents" / "flex" / "test_plan.pdf"
    source.parent.mkdir(parents=True)
    _make_pdf(source, ["A planned FLEX experiment with adequate readable test description.", ""])
    config = IngestionConfig(
        documents_dir=source.parents[1],
        output_dir=tmp_path / "processed",
        min_pdf_text_chars=20,
    )

    with caplog.at_level(logging.WARNING):
        records = ingest_file(source, config)

    assert [record.page_number for record in records] == [1, 2]
    assert records[0].text.startswith("A planned FLEX experiment")
    assert records[0].file_type == "pdf"
    assert records[1].text == ""
    assert "page 2" in records[1].extraction_warnings[0]
    assert "very little or no extractable text" in caplog.text


def test_pdf_page_extraction_failure_is_logged_and_other_pages_are_retained(
    tmp_path, monkeypatch, caplog
):
    source = tmp_path / "documents" / "flex" / "broken_page.pdf"
    source.parent.mkdir(parents=True)
    _make_pdf(source, ["Page one contains enough readable words to pass the threshold.", "Page two"])

    from src.ingestion import extractors

    original_reader = extractors.PdfReader

    class BrokenSecondPageReader:
        def __init__(self, *args, **kwargs):
            self.reader = original_reader(*args, **kwargs)
            self.is_encrypted = self.reader.is_encrypted
            self.pages = list(self.reader.pages)
            self.pages[1] = BrokenPage()

    class BrokenPage:
        def extract_text(self):
            raise ValueError("damaged page content")

    monkeypatch.setattr(extractors, "PdfReader", BrokenSecondPageReader)
    with caplog.at_level(logging.WARNING):
        records = ingest_file(
            source,
            IngestionConfig(documents_dir=source.parents[1], min_pdf_text_chars=15),
        )

    assert len(records) == 2
    assert records[1].page_number == 2
    assert any("text extraction failed" in warning for warning in records[1].extraction_warnings)
    assert "damaged page content" in caplog.text


def test_docx_extraction_preserves_paragraph_table_order_and_core_metadata(tmp_path):
    source = tmp_path / "documents" / "flex" / "metadata.docx"
    source.parent.mkdir(parents=True)
    _make_docx(source)
    config = IngestionConfig(
        documents_dir=source.parents[1],
        metadata_overrides={
            "flex/metadata.docx": MetadataOverride(
                document_type="experiment_metadata",
                evidence_status="metadata",
            ),
        },
    )

    [record] = ingest_file(source, config)

    assert record.page_number is None
    assert record.document_type == "experiment_metadata"
    assert record.evidence_status == "metadata"
    assert "author" in record.document_metadata
    assert record.document_metadata["author"] == "NASA test team"
    assert record.document_metadata["title"] == "FLEX fixture"
    assert record.text.index("Experiment description.") < record.text.index("[Table 1]")
    assert record.text.index("Methanol") < record.text.index("Closing note.")
    assert "Fuel\tPressure" in record.text
    assert "Methanol\t1 atm" in record.text


def test_recursive_discovery_writes_normalized_jsonl_with_explicit_metadata(tmp_path):
    documents_dir = tmp_path / "documents"
    output_dir = tmp_path / "processed"
    nested = documents_dir / "flex" / "nested"
    nested.mkdir(parents=True)
    _make_pdf(nested / "plan.pdf", ["A readable planned test description for one PDF page."])
    _make_docx(documents_dir / "flex" / "metadata.docx", include_table=False)
    (documents_dir / "bass").mkdir()
    _make_docx(documents_dir / "bass" / "unknown.docx", include_table=False)
    (documents_dir / "flex" / "notes.txt").write_text("not supported", encoding="utf-8")

    config_path = tmp_path / "metadata.json"
    config_path.write_text(
        json.dumps(
            {
                "flex/nested/plan.pdf": {
                    "document_type": "test_plan",
                    "evidence_status": "planned",
                    "source_url": None,
                    "source_accession": None,
                }
            }
        ),
        encoding="utf-8",
    )
    config = IngestionConfig(
        documents_dir=documents_dir,
        output_dir=output_dir,
        metadata_overrides=load_metadata_overrides(config_path),
    )

    result = ingest_documents(config)

    assert result.source_documents == 3
    assert result.pdf_pages == 1
    assert result.docx_documents == 2
    assert len(result.records) == 3
    plan = next(record for record in result.records if record.filename == "plan.pdf")
    unknown = next(record for record in result.records if record.filename == "unknown.docx")
    assert plan.document_type == "test_plan"
    assert plan.evidence_status == "planned"
    assert plan.source_url is None
    assert plan.source_accession is None
    assert unknown.document_type == "unknown"
    assert unknown.evidence_status is None
    assert unknown.experiment == "BASS"
    output_path = output_dir / "flex" / "nested" / "plan.jsonl"
    saved_records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert saved_records[0]["page_number"] == 1
    assert saved_records[0]["source_id"] == plan.source_id
    assert saved_records[0]["content_sha256"] == plan.content_sha256


def test_content_identity_is_deterministic_across_different_directories(tmp_path):
    documents_dir = tmp_path / "documents"
    first_dir = documents_dir / "flex" / "first"
    second_dir = documents_dir / "flex" / "second"
    first_dir.mkdir(parents=True)
    second_dir.mkdir(parents=True)
    content = ["Same source bytes produce the same stable identity across source locations."]
    first_path = first_dir / "source.pdf"
    second_path = second_dir / "source.pdf"
    _make_pdf(first_path, content)
    second_path.write_bytes(first_path.read_bytes())

    config = IngestionConfig(documents_dir=documents_dir)
    first = ingest_file(first_path, config)[0]
    second = ingest_file(second_path, config)[0]

    expected_sha256 = hashlib.sha256(first_path.read_bytes()).hexdigest()
    assert first.content_sha256 == second.content_sha256 == expected_sha256
    assert first.source_id == second.source_id == f"nasa-{expected_sha256}"
    assert first.source_path != second.source_path


def test_source_hash_matches_file_bytes_and_output_has_provenance_fields():
    """Regression check against the two tracked FLEX sources and their output."""

    repository_root = Path(__file__).resolve().parents[1]
    metadata = load_metadata_overrides(repository_root / "data" / "document_metadata.json")
    config = IngestionConfig(
        documents_dir=repository_root / "data" / "documents",
        output_dir=repository_root / "data" / "processed",
        metadata_overrides=metadata,
    )
    pdf_path = repository_root / "data" / "documents" / "flex" / "FLEX_Test_Plan.pdf"
    docx_path = repository_root / "data" / "documents" / "flex" / "FLEX DoD.docx"
    pdf_bytes_before = pdf_path.read_bytes()
    docx_bytes_before = docx_path.read_bytes()

    pdf_records = ingest_file(pdf_path, config)
    [docx_record] = ingest_file(docx_path, config)

    assert [record.page_number for record in pdf_records] == list(range(1, 16))
    assert [record.source_locator for record in pdf_records] == [
        f"page:{page_number}" for page_number in range(1, 16)
    ]
    assert {record.document_type for record in pdf_records} == {"test_plan"}
    assert {record.evidence_status for record in pdf_records} == {"planned"}
    assert {record.source_id for record in pdf_records} == {
        f"nasa-{hashlib.sha256(pdf_bytes_before).hexdigest()}"
    }
    assert all(
        record.content_sha256 == hashlib.sha256(pdf_bytes_before).hexdigest()
        for record in pdf_records
    )
    assert pdf_records[0].document_metadata["producer"]
    assert "title" not in pdf_records[0].document_metadata
    assert all(record.source_url is None and record.source_accession is None for record in pdf_records)

    assert docx_record.source_locator == "document"
    assert docx_record.document_type == "experiment_metadata"
    assert docx_record.evidence_status == "metadata"
    assert docx_record.content_sha256 == hashlib.sha256(docx_bytes_before).hexdigest()
    assert docx_record.source_id == f"nasa-{docx_record.content_sha256}"
    assert docx_record.document_metadata["author"]
    assert docx_record.document_metadata["created"]
    assert docx_record.source_url is None
    assert docx_record.source_accession is None

    assert pdf_path.read_bytes() == pdf_bytes_before
    assert docx_path.read_bytes() == docx_bytes_before


def test_metadata_override_rejects_unknown_classification(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps({"flex/plan.pdf": {"document_type": "guessed_from_content"}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid document_type"):
        load_metadata_overrides(metadata_path)


def test_metadata_override_accepts_canonical_source_fields(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "flex/source.pdf": {
                    "source_url": "https://example.invalid/source",
                    "source_accession": "PSI-EXAMPLE",
                }
            }
        ),
        encoding="utf-8",
    )

    [override] = load_metadata_overrides(metadata_path).values()

    assert override.source_url == "https://example.invalid/source"
    assert override.source_accession == "PSI-EXAMPLE"


def test_pdf_whitespace_normalization_preserves_lines_and_scientific_text():
    from src.ingestion.extractors import normalize_pdf_whitespace

    text = "Fuel:\t\n  CH3OH   pressure 1.00 atm\n\n\nO₂  =  21%"

    assert normalize_pdf_whitespace(text) == "Fuel: CH3OH pressure 1.00 atm\n\nO₂ = 21%"
