"""Select the useful PSI downloads and place them where the pipeline reads them.

``psi_download.py`` caches NASA files in ``data/psi-download``. This module copies
only the curated subset below into ``data/documents/<experiment>`` (text for
retrieval) and ``data/structured/<experiment>`` (test tables for matching),
removes byte-identical duplicates, and records provenance in
``data/document_metadata.json``. Nothing is downloaded here.

    python -m src.ingestion.psi_sync
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

PSI_PAGE = "https://psi.nasa.gov/physci/repo/data/investigations/"


@dataclass(frozen=True, slots=True)
class Selection:
    accession: str
    cached: str  # path relative to the PSI-xx/ folder
    folder: str  # flex | bass | saffire
    target: str  # file name in the project
    document_type: str
    evidence_status: str
    experiment: str


def _summary(accession, folder, experiment):
    return Selection(accession, "summary.md", folder, f"{accession}_{experiment}_summary.md",
                     "experiment_metadata", "metadata", experiment)


# Chosen for question answering: experiment overviews, science requirements and
# result presentations. Skipped: crew-training slides (procedures), duplicate
# slide decks, image-directory file lists, and anything not text.
DOCUMENTS = (
    _summary("PSI-69", "flex", "FLEX"),
    _summary("PSI-68", "flex", "FLEX-2"),
    _summary("PSI-25", "bass", "BASS-II"),
    _summary("PSI-26", "bass", "BASS"),
    _summary("PSI-98", "saffire", "SAFFIRE-I"),
    _summary("PSI-99", "saffire", "SAFFIRE-II"),
    _summary("PSI-100", "saffire", "SAFFIRE-III"),
    Selection("PSI-25", "Science Documents/PSI-25_Science Documents_BASS-II_SRD.pdf", "bass",
              "BASS-II_Science_Requirements.pdf", "test_plan", "planned", "BASS-II"),
    Selection("PSI-25", "Science Documents/PSI-25_Science Documents_BASS-II_ReadMe.docx", "bass",
              "BASS-II_ReadMe.docx", "data_description", "metadata", "BASS-II"),
    Selection("PSI-26", "Science Documents/PSI-26_Science Documents_SRD_BASS.pdf", "bass",
              "BASS_Science_Requirements.pdf", "test_plan", "planned", "BASS"),
    Selection("PSI-26", "Science Documents/PSI-26_Science Documents_DLD_BASS.pdf", "bass",
              "BASS_Data_Layout.pdf", "data_description", "metadata", "BASS"),
    Selection("PSI-26", "Presentations/PSI-26_Presentations_BASS_Overview of Recent ISS Results.pdf", "bass",
              "BASS_Overview_of_ISS_Results.pdf", "publication", "interpreted", "BASS"),
    Selection("PSI-26", "Presentations/PSI-26_Presentations_Flammability Aspects of Fabric_USNationalCombustion2013.pdf",
              "bass", "BASS_Flammability_Aspects_of_Fabric_2013.pdf", "publication", "interpreted", "BASS"),
    Selection("PSI-26", "Presentations/PSI-26_Presentations_FlammabilityAspectsofCottonFiberglassFabric_ASGSR2012.pdf",
              "bass", "BASS_Cotton_Fiberglass_Fabric_ASGSR2012.pdf", "publication", "interpreted", "BASS"),
    Selection("PSI-98", "Science Documents/PSI-98_Science Documents_SAFFIRE Objectives, Development and Status.pdf",
              "saffire", "SAFFIRE_Objectives_Development_Status.pdf", "publication", "interpreted", "SAFFIRE"),
    Selection("PSI-98", "Science Documents/PSI-98_Science Documents_SAFFIRE-I and SAFFIRE-III Readme.docx",
              "saffire", "SAFFIRE-I_and_III_Readme.docx", "data_description", "metadata", "SAFFIRE"),
)

# (accession, cached file, folder, project file name)
TABLES = (
    ("PSI-69", "Experimental table/PSI-69_Experimental table_FLEX.csv", "flex", "PSI-69_FLEX.csv"),
    ("PSI-25", "Experimental table/PSI-25_Experimental table_BASS-II.csv", "bass", "PSI-25_BASS-II.csv"),
    ("PSI-26", "Experimental table/PSI-26_Experimental table_BASS.csv", "bass", "PSI-26_BASS.csv"),
    ("PSI-98", "Experimental table/PSI-98_Experimental table_SAFFIRE-1.csv", "saffire", "PSI-98_SAFFIRE-I.csv"),
    ("PSI-99", "Experimental table/PSI-99_Experimental table_SAFFIRE-2.csv", "saffire", "PSI-99_SAFFIRE-II.csv"),
    ("PSI-100", "Experimental table/PSI-100_Experimental table_SAFFIRE-3.csv", "saffire", "PSI-100_SAFFIRE-III.csv"),
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sync(cache: Path, documents: Path, structured: Path, metadata_path: Path) -> dict[str, list[str]]:
    report: dict[str, list[str]] = {"copied": [], "missing": [], "duplicate": []}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    seen: dict[str, str] = {}

    for item in DOCUMENTS:
        source = cache / item.accession / item.cached
        if not source.is_file():
            report["missing"].append(f"{item.accession}/{item.cached}")
            continue
        digest = _sha(source)
        if digest in seen:
            report["duplicate"].append(f"{item.target} (same as {seen[digest]})")
            continue
        seen[digest] = item.target
        target = documents / item.folder / item.target
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        metadata[f"{item.folder}/{item.target}"] = {
            "document_type": item.document_type,
            "evidence_status": item.evidence_status,
            "experiment": item.experiment,
            "source_url": PSI_PAGE + item.accession,
            "source_accession": item.accession,
        }
        report["copied"].append(f"documents/{item.folder}/{item.target}")

    for accession, cached, folder, name in TABLES:
        source = cache / accession / cached
        if not source.is_file():
            report["missing"].append(f"{accession}/{cached}")
            continue
        target = structured / folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        report["copied"].append(f"structured/{folder}/{name}")

    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Copy curated PSI downloads into the data layout.")
    parser.add_argument("--cache", type=Path, default=Path("data/psi-download"))
    parser.add_argument("--documents-dir", type=Path, default=Path("data/documents"))
    parser.add_argument("--structured-dir", type=Path, default=Path("data/structured"))
    parser.add_argument("--metadata", type=Path, default=Path("data/document_metadata.json"))
    args = parser.parse_args()
    report = sync(args.cache, args.documents_dir, args.structured_dir, args.metadata)
    for key in ("copied", "duplicate", "missing"):
        print(f"{key}: {len(report[key])}")
        for line in report[key]:
            print(f"  - {line}")
    return 1 if report["missing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
