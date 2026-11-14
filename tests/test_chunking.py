from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.chunking import ChunkingConfig, chunk_record, run_chunking
from src.chunking.text import estimated_token_count


def _record(text: str, **overrides) -> dict:
    record = {
        "source_id": "nasa-test-source",
        "content_sha256": hashlib.sha256(b"original source bytes").hexdigest(),
        "experiment": "FLEX",
        "filename": "example.pdf",
        "file_type": "pdf",
        "source_path": "flex/example.pdf",
        "document_type": "test_plan",
        "evidence_status": "planned",
        "source_locator": "page:3",
        "page_number": 3,
        "document_metadata": {"author": "NASA"},
        "extraction_warnings": [],
        "source_url": None,
        "source_accession": None,
        "text": text,
    }
    record.update(overrides)
    return record


def test_chunk_ids_and_serialized_chunks_are_deterministic():
    text = " ".join(f"Condition{i} is stable at 21% oxygen.\n\n" for i in range(80))
    record = _record(text)
    config = ChunkingConfig(max_chunk_tokens=100, overlap_tokens=20)

    first = [chunk.to_dict() for chunk in chunk_record(record, config)]
    second = [chunk.to_dict() for chunk in chunk_record(record, config)]

    assert first == second
    assert len(first) > 1
    assert len({chunk["chunk_id"] for chunk in first}) == len(first)
    assert all(chunk["chunk_count"] == len(first) for chunk in first)


def test_chunks_preserve_provenance_and_add_estimated_size():
    record = _record(" ".join(f"measurement{i} value" for i in range(150)))
    chunks = [chunk.to_dict() for chunk in chunk_record(record, ChunkingConfig( max_chunk_tokens=50, overlap_tokens=10))]

    assert chunks
    for chunk in chunks:
        for field in (
            "source_id",
            "content_sha256",
            "experiment",
            "filename",
            "file_type",
            "source_path",
            "document_type",
            "evidence_status",
            "source_locator",
            "page_number",
            "document_metadata",
            "extraction_warnings",
            "source_url",
            "source_accession",
        ):
            assert chunk[field] == record[field]
        assert chunk["estimated_token_count"] == estimated_token_count(chunk["text"])


def test_adjacent_chunks_overlap_using_exact_source_text():
    text = " ".join(f"Condition{i} is stable at 21% oxygen." for i in range(50))
    record = _record(text)
    chunks = chunk_record(record, ChunkingConfig(max_chunk_tokens=60, overlap_tokens=15))

    assert len(chunks) > 1
    for previous, current in zip(chunks, chunks[1:]):
        shared_suffix = next(
            (
                suffix
                for suffix_length in range(len(previous.text), 0, -1)
                if (suffix := previous.text[-suffix_length:]).strip()
                and current.text.startswith(suffix)
            ),
            None,
        )
        assert shared_suffix is not None
        assert estimated_token_count(shared_suffix) <= 15
        assert estimated_token_count(shared_suffix) >= 1
        assert previous.text in text
        assert current.text in text


def test_source_text_is_not_rewritten_and_short_or_empty_records_are_safe():
    original = "\nFuel condition: C7H16 at 1.00 atm; O₂ = 21%.\n\n"
    [chunk] = chunk_record(_record(original), ChunkingConfig(max_chunk_tokens=100, overlap_tokens=10))
    assert chunk.text == original

    assert chunk_record(_record("")) == []
    assert chunk_record(_record(" \n\t ")) == []
    [short] = chunk_record(_record("LOI = 0.21."))
    assert short.text == "LOI = 0.21."


def test_pdf_and_docx_locators_and_evidence_labels_are_retained():
    pdf = chunk_record(_record("planned flame test", source_locator="page:7", page_number=7))[0]
    docx_record = _record(
        "FLEX data catalog metadata",
        filename="FLEX DoD.docx",
        file_type="docx",
        source_path="flex/FLEX DoD.docx",
        document_type="experiment_metadata",
        evidence_status="metadata",
        source_locator="document",
        page_number=None,
    )
    docx = chunk_record(docx_record)[0]

    assert pdf.source_record["source_locator"] == "page:7"
    assert pdf.source_record["page_number"] == 7
    assert pdf.source_record["evidence_status"] == "planned"
    assert docx.source_record["source_locator"] == "document"
    assert docx.source_record["page_number"] is None
    assert docx.source_record["document_type"] == "experiment_metadata"
    assert docx.source_record["evidence_status"] == "metadata"


def test_pipeline_chunks_normalized_jsonl_without_changing_input(tmp_path):
    input_dir = tmp_path / "processed"
    output_dir = tmp_path / "chunks"
    input_path = input_dir / "flex" / "FLEX DoD.jsonl"
    input_path.parent.mkdir(parents=True)
    record = _record("NASA FLEX document metadata.", filename="FLEX DoD.docx", file_type="docx", source_locator="document", page_number=None)
    input_bytes = (json.dumps(record) + "\n").encode("utf-8")
    input_path.write_bytes(input_bytes)

    result = run_chunking(ChunkingConfig(input_dir=input_dir, output_dir=output_dir))

    assert result.input_records == 1
    assert len(result.chunks) == 1
    assert not result.errors
    output_path = output_dir / "flex" / "FLEX_DoD.jsonl"
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["source_locator"] == "document"
    assert saved["source_id"] == record["source_id"]
    assert input_path.read_bytes() == input_bytes


def test_real_flex_processed_records_keep_all_provenance():
    repository_root = Path(__file__).resolve().parents[1]
    processed_dir = repository_root / "data" / "processed" / "flex"
    if not processed_dir.exists():
        return

    inputs = [
        json.loads(line)
        for path in sorted(processed_dir.glob("*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    chunks = [chunk.to_dict() for record in inputs for chunk in chunk_record(record)]

    assert len(inputs) == 16
    assert chunks
    assert all(chunk["source_id"] and chunk["content_sha256"] for chunk in chunks)
    assert all("evidence_status" in chunk and "source_locator" in chunk for chunk in chunks)
    assert {chunk["page_number"] for chunk in chunks if chunk["file_type"] == "pdf"} == set(range(1, 16))
    assert {chunk["source_locator"] for chunk in chunks if chunk["file_type"] == "docx"} == {"document"}
