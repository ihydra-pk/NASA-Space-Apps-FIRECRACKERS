#!/usr/bin/env python3
"""Place files fetched by psi_download.py where ingestion expects them.

psi_download.py writes ``<psi-dir>/<PSI-ID>/<category>/<file>``. Ingestion reads
``<documents-dir>/<experiment>/<file>`` and takes its labels from
``document_metadata.json``. This script links (or copies) every PDF and DOCX
into one folder per experiment and records where each file came from.

    python scripts/stage_psi_documents.py --dry-run
    python scripts/stage_psi_documents.py

Files are matched by content, so a document already present under the
documents folder, or listed twice by NASA, is staged once. Labels that need a
human decision (document type, evidence status) are left unset: nothing is
inferred from a file's name, category or contents.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

SUPPORTED_SUFFIXES = {".pdf", ".docx"}
INVESTIGATION_PAGE = "https://psi.nasa.gov/physci/repo/data/investigations/"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def experiment_folder(accession: str, acronym: str | None) -> str:
    """``FLEX`` -> ``flex``; experiments without an acronym use their PSI id."""

    slug = re.sub(r"[^a-z0-9]+", "-", (acronym or "").casefold()).strip("-")
    return slug or accession.casefold()


def supported_files(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.casefold() in SUPPORTED_SUFFIXES
    )


def place(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)  # same disk: no extra space used
    except OSError:
        shutil.copy2(source, target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--psi-dir", type=Path, default=Path("data"),
                        help="folder holding the PSI-* download folders (default: data)")
    parser.add_argument("--documents-dir", type=Path, default=Path("data/documents"))
    parser.add_argument("--metadata", type=Path, default=Path("data/document_metadata.json"))
    parser.add_argument("--exclude", nargs="*", default=[], metavar="PSI-N",
                        help="experiments to leave out, e.g. --exclude PSI-181")
    parser.add_argument("--dry-run", action="store_true", help="report only; change nothing")
    args = parser.parse_args()

    labels = json.loads(args.metadata.read_text(encoding="utf-8")) if args.metadata.is_file() else {}
    staged_hashes = (
        {sha256(path): path for path in supported_files(args.documents_dir)}
        if args.documents_dir.is_dir()
        else {}
    )

    def accession_number(folder: Path) -> int:
        return int(re.sub(r"\D", "", folder.name) or 0)

    excluded = {name.upper() for name in args.exclude}
    staged = duplicates = 0
    for psi_folder in sorted(args.psi_dir.glob("PSI-*"), key=accession_number):
        details_path = psi_folder / "metadata.json"
        if not psi_folder.is_dir() or not details_path.is_file():
            continue
        if psi_folder.name.upper() in excluded:
            print(f"{psi_folder.name:8s} -> excluded")
            continue
        accession = psi_folder.name
        details = json.loads(details_path.read_text(encoding="utf-8"))
        experiment = experiment_folder(accession, details.get("acronym"))
        count = 0
        for source in supported_files(psi_folder):
            digest = sha256(source)
            if digest in staged_hashes:
                duplicates += 1
                continue
            target = args.documents_dir / experiment / source.name
            if target.exists():  # same name, different content: keep both
                target = target.with_name(f"{target.stem}-{digest[:8]}{target.suffix}")
            staged_hashes[digest] = target
            key = target.relative_to(args.documents_dir).as_posix()
            labels.setdefault(key, {
                "document_type": "unknown",
                "evidence_status": None,
                "experiment": (details.get("acronym") or "").strip() or accession,
                "source_url": INVESTIGATION_PAGE + accession,
                "source_accession": accession,
            })
            if not args.dry_run:
                place(source, target)
            count += 1
        staged += count
        print(f"{accession:8s} -> {experiment:24s} {count:3d} file(s)")

    if not args.dry_run:
        args.metadata.write_text(json.dumps(labels, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    verb = "Would stage" if args.dry_run else "Staged"
    print(f"{verb} {staged} file(s); skipped {duplicates} duplicate(s) by content.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
