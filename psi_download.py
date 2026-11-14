#!/usr/bin/env python3
"""Download the text-rich files from NASA PSI experiments (for chunking and embedding).

NASA's Physical Sciences Informatics (PSI) repository holds more than a terabyte
of raw sensor data, images and video per topic. This script fetches only the
parts that explain the science: reports, science documents, presentations,
reference documents and the table of test runs. Everything else is listed in a
manifest with a link back to NASA, but not downloaded.

Needs Python 3.8 or newer. No extra packages.

    python3 psi_download.py --dry-run          # show what would be downloaded
    python3 psi_download.py --ids PSI-47       # one small experiment, as a test
    python3 psi_download.py                    # all combustion experiments

Output (in ./data by default):
    data/manifest.csv              one row per file NASA lists, with the decision made
    data/<ID>/summary.md           the experiment described in prose, ready to embed
    data/<ID>/metadata.json        the same information as structured fields
    data/<ID>/catalogue.json       NASA's full record, unmodified
    data/<ID>/<category>/<file>    the downloaded documents

Safe to re-run: files already downloaded are skipped.
"""

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = "https://psi.nasa.gov/geode-py/ws"
PAGE = "https://psi.nasa.gov/physci/repo/data/investigations/"
USER_AGENT = "psi-download-script/1.0 (python-urllib)"

TEXT_TYPES = {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".txt"}
TABLE_TYPES = {".csv", ".xls", ".xlsx", ".dat"}

# Categories whose documents explain the experiment and its findings.
TEXT_CATEGORIES = {"Reports", "Science Documents", "Presentations", "Reference Data"}


# ---------------------------------------------------------------- networking

def open_url(url, timeout):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(request, timeout=timeout)


def with_retries(action, attempts, label):
    """Run action(); on a network error wait and try again."""
    for attempt in range(1, attempts + 1):
        try:
            return action()
        except (urllib.error.URLError, OSError, ValueError) as error:
            if attempt == attempts:
                raise
            wait = 5 * attempt
            print(f"    retry {attempt}/{attempts - 1} for {label} in {wait}s ({error})")
            time.sleep(wait)


def fetch_json(url, timeout, attempts, label):
    def action():
        with open_url(url, timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    return with_retries(action, attempts, label)


# ------------------------------------------------------------------ catalogue

def find_ids(term, timeout, attempts):
    query = urllib.parse.urlencode({
        "term": term, "source": "psi", "type": "investigation",
        "sort": "Public Release Date", "order": "desc", "size": 500,
    }, quote_via=urllib.parse.quote)
    result = fetch_json(f"{BASE}/repo/search?{query}", timeout, attempts, "search")
    hits = result.get("hits", {})
    ids = [h["_source"]["Accession"] for h in hits.get("hits", [])]
    total = int(hits.get("total") or len(ids))
    if total > len(ids):
        print(f"WARNING: NASA reports {total} matches but returned only {len(ids)}.")
    return sorted(ids, key=id_number)


def id_number(accession):
    digits = "".join(ch for ch in accession if ch.isdigit())
    return int(digits) if digits else 0


def load_catalogue(accession, folder, refresh, timeout, attempts):
    path = os.path.join(folder, "catalogue.json")
    if os.path.exists(path) and not refresh:
        try:
            with open(path, encoding="utf-8") as handle:
                return json.load(handle)
        except ValueError:
            pass  # damaged copy from an interrupted run: fetch again
    catalogue = fetch_json(f"{BASE}/repo/investigations/{accession}",
                           timeout, attempts, f"{accession} catalogue")
    os.makedirs(folder, exist_ok=True)
    temporary = path + ".part"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(catalogue, handle)
    os.replace(temporary, path)
    return catalogue


def clean_metadata(accession, catalogue):
    people = []
    for person in catalogue.get("contacts") or []:
        name = " ".join(p for p in (person.get("firstName"), person.get("lastName")) if p)
        roles = ", ".join(r.get("annotationValue", "") for r in person.get("roles") or [])
        if name:
            people.append({"name": name, "role": roles,
                           "affiliation": person.get("affiliation", "")})
    papers = [{"title": p.get("title", ""), "authors": p.get("authorList", ""),
               "doi": p.get("doi", "")} for p in catalogue.get("publications") or []]

    def text(key):
        return (catalogue.get(key) or "").strip()

    return {
        "id": accession,
        "source_page": PAGE + accession,
        "title": text("title"),
        "acronym": text("investigationAcronym"),
        "research_area": text("researchArea"),
        "sub_research_area": text("subResearchArea"),
        "project_type": text("projectType"),
        "platform": text("flightProgram"),
        "nasa_center": text("managingNasaCenter"),
        "start_date": text("investigationStartDate"),
        "end_date": text("investigationEndDate"),
        "objective": text("objective"),
        "hypothesis": text("hypothesis"),
        "approach": text("approach"),
        "research_impacts": text("researchImpacts"),
        "hardware": text("experimentHardware"),
        "dataset_doi": text("doi"),
        "license": text("licenseName"),
        "people": people,
        "publications": papers,
    }


def summary_markdown(meta):
    heading = meta["title"] or meta["id"]
    if meta["acronym"]:
        heading += f" ({meta['acronym']})"
    lines = [f"# {heading}", "", f"NASA PSI record: {meta['id']} - {meta['source_page']}", ""]
    facts = [("Research area", meta["research_area"]),
             ("Sub-area", meta["sub_research_area"]),
             ("Project type", meta["project_type"]),
             ("Platform", meta["platform"]),
             ("NASA center", meta["nasa_center"]),
             ("Dates", " to ".join(d for d in (meta["start_date"], meta["end_date"]) if d)),
             ("Licence", meta["license"])]
    lines += [f"- {label}: {value}" for label, value in facts if value]
    for label, key in (("Objective", "objective"), ("Hypothesis", "hypothesis"),
                       ("Approach", "approach"), ("Research impacts", "research_impacts"),
                       ("Hardware", "hardware")):
        if meta[key]:
            lines += ["", f"## {label}", "", meta[key]]
    if meta["people"]:
        lines += ["", "## People", ""]
        lines += [f"- {p['name']}" + (f", {p['role']}" if p["role"] else "")
                  + (f" ({p['affiliation']})" if p["affiliation"] else "")
                  for p in meta["people"]]
    if meta["publications"]:
        lines += ["", "## Publications", ""]
        for paper in meta["publications"]:
            entry = f"- {paper['title']}"
            if paper["authors"]:
                entry += f". {paper['authors']}"
            if paper["doi"]:
                entry += f". https://doi.org/{paper['doi']}"
            lines.append(entry)
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ filtering

def list_files(catalogue):
    """Every file NASA lists for one experiment, without duplicates."""
    seen, files = set(), []
    for entry in catalogue.get("experimentalTable") or []:
        files.append((entry, "Experimental table"))
    for entry in catalogue.get("added") or []:
        parts = (entry.get("s3_path") or "").split("/")
        files.append((entry, parts[2] if len(parts) > 3 else "Uncategorised"))
    for entry, category in files:
        name = entry.get("file_name") or ""
        if not name or name in seen:
            continue
        seen.add(name)
        try:
            size = int(float(entry.get("file_size") or 0))
        except ValueError:
            size = 0
        yield {"name": name, "category": category, "size": size,
               "restricted": str(entry.get("restricted")) == "True",
               "visible": str(entry.get("visible", "True")) != "False"}


def decide(item, options):
    """Return (keep, reason)."""
    extension = os.path.splitext(item["name"])[1].lower()
    category = item["category"]
    if item["restricted"]:
        return False, "restricted by NASA"
    if not item["visible"]:
        return False, "hidden by NASA"
    if category == "Experimental table":
        wanted = extension in TABLE_TYPES
        why = "table of test runs" if wanted else "not a table file"
    elif category in TEXT_CATEGORIES:
        wanted = extension in TEXT_TYPES
        why = "explanatory document" if wanted else f"{extension or 'unknown'} type is not text"
    elif category == "Analyzed Data":
        if extension in TEXT_TYPES:
            wanted, why = True, "explanatory document"
        elif extension in TABLE_TYPES and options.include_analyzed:
            wanted, why = True, "analysed results table"
        elif extension in TABLE_TYPES:
            wanted, why = False, "analysed table (use --include-analyzed)"
        else:
            wanted, why = False, f"{extension or 'unknown'} type is not text"
    elif category == "Engineering Documents":
        if options.include_engineering and extension in TEXT_TYPES:
            wanted, why = True, "engineering document"
        else:
            wanted, why = False, "engineering document (use --include-engineering)"
    else:
        wanted, why = False, f"{category} is not downloaded"
    if wanted and item["size"] > options.max_mb * 1024 * 1024:
        return False, f"larger than {options.max_mb} MB"
    return wanted, why


# ---------------------------------------------------------------- downloading

def safe_name(name):
    name = name.replace("/", "_").replace("\\", "_").strip()
    return name.lstrip(".") or "unnamed"


def download_url(accession, name):
    query = urllib.parse.urlencode({"source": "datamanager", "file": name},
                                   quote_via=urllib.parse.quote)
    return f"{BASE}/studies/{accession}/download?{query}"


def download(accession, item, destination, options):
    """Return a status string for the manifest."""
    expected = item["size"]
    if os.path.exists(destination):
        if not expected or os.path.getsize(destination) == expected:
            return "already present"

    def action():
        temporary = destination + ".part"
        with open_url(download_url(accession, item["name"]), options.timeout) as response:
            kind = response.headers.get("Content-Type", "")
            if "text/html" in kind and not item["name"].lower().endswith((".htm", ".html")):
                raise ValueError("server returned a web page instead of the file")
            with open(temporary, "wb") as handle:
                while True:
                    block = response.read(1024 * 256)
                    if not block:
                        break
                    handle.write(block)
        os.replace(temporary, destination)

    os.makedirs(os.path.dirname(destination), exist_ok=True)
    try:
        with_retries(action, options.attempts, item["name"])
    except (urllib.error.URLError, OSError, ValueError) as error:
        return f"failed: {error}"
    actual = os.path.getsize(destination)
    if expected and actual != expected:
        return f"downloaded (size differs: got {actual}, NASA lists {expected})"
    return "downloaded"


# ----------------------------------------------------------------------- main

def megabytes(size):
    return f"{size / 1024 / 1024:.1f} MB"


def main():
    parser = argparse.ArgumentParser(
        description="Download the text-rich files from NASA PSI experiments.")
    parser.add_argument("--term", default="combustion",
                        help="search word used to find experiments (default: combustion)")
    parser.add_argument("--ids", nargs="+", metavar="PSI-N",
                        help="download only these experiments, e.g. --ids PSI-101 PSI-47")
    parser.add_argument("--out", default="data", help="output folder (default: data)")
    parser.add_argument("--dry-run", action="store_true",
                        help="list what would be downloaded, without downloading files")
    parser.add_argument("--include-analyzed", action="store_true",
                        help="also download analysed result tables (.xls .xlsx .csv .dat)")
    parser.add_argument("--include-engineering", action="store_true",
                        help="also download engineering documents (drawings, hardware specs)")
    parser.add_argument("--max-mb", type=int, default=100,
                        help="skip files larger than this (default: 100)")
    parser.add_argument("--workers", type=int, default=4,
                        help="files downloaded at the same time (default: 4)")
    parser.add_argument("--timeout", type=int, default=120,
                        help="seconds to wait on a stalled connection (default: 120)")
    parser.add_argument("--attempts", type=int, default=3,
                        help="tries per file before giving up (default: 3)")
    parser.add_argument("--refresh", action="store_true",
                        help="fetch NASA's catalogues again instead of using saved copies")
    options = parser.parse_args()

    if options.ids:
        ids = [i.upper() for i in options.ids]
    else:
        print(f"Searching NASA PSI for '{options.term}' experiments...")
        ids = find_ids(options.term, options.timeout, options.attempts)
    print(f"{len(ids)} experiments: {' '.join(ids)}\n")
    if not ids:
        return 1

    os.makedirs(options.out, exist_ok=True)
    rows, jobs, problems = [], [], []

    for accession in ids:
        folder = os.path.join(options.out, accession)
        try:
            catalogue = load_catalogue(accession, folder, options.refresh,
                                       options.timeout, options.attempts)
        except (urllib.error.URLError, OSError, ValueError) as error:
            print(f"{accession}: could not read catalogue ({error})")
            problems.append(accession)
            continue
        meta = clean_metadata(accession, catalogue)
        with open(os.path.join(folder, "metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2, ensure_ascii=False)
        with open(os.path.join(folder, "summary.md"), "w", encoding="utf-8") as handle:
            handle.write(summary_markdown(meta))

        kept = kept_size = total = 0
        for item in list_files(catalogue):
            total += 1
            keep, reason = decide(item, options)
            local = os.path.join(folder, safe_name(item["category"]), safe_name(item["name"]))
            row = {"id": accession, "category": item["category"], "file_name": item["name"],
                   "size_bytes": item["size"], "decision": "download" if keep else "skip",
                   "reason": reason, "status": "" if keep else "not downloaded",
                   "nasa_page": PAGE + accession, "local_path": local if keep else ""}
            rows.append(row)
            if keep:
                kept += 1
                kept_size += item["size"]
                jobs.append((accession, item, local, row))
        label = meta["acronym"] or meta["title"][:40]
        print(f"{accession:8s} {label:42s} keep {kept:3d} of {total:4d} files ({megabytes(kept_size)})")

    wanted_size = sum(job[1]["size"] for job in jobs)
    print(f"\n{len(jobs)} files to download, {megabytes(wanted_size)} in total.")

    if options.dry_run:
        for row in rows:
            if row["decision"] == "download":
                row["status"] = "dry run"
        print("Dry run: no files downloaded.")
    else:
        done = 0
        with ThreadPoolExecutor(max_workers=max(1, options.workers)) as pool:
            futures = {pool.submit(download, a, item, local, options): (item, row)
                       for a, item, local, row in jobs}
            for future in as_completed(futures):
                item, row = futures[future]
                row["status"] = future.result()
                done += 1
                print(f"  [{done}/{len(jobs)}] {row['status']:16.16s} {row['id']} {item['name']}")

    manifest = os.path.join(options.out, "manifest.csv")
    with open(manifest, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else
                                ["id", "category", "file_name", "size_bytes", "decision",
                                 "reason", "status", "nasa_page", "local_path"])
        writer.writeheader()
        writer.writerows(rows)

    wanted = [r for r in rows if r["decision"] == "download"]
    failed = [r for r in wanted if r["status"].startswith("failed")]
    print(f"\nManifest: {manifest}")
    if not options.dry_run:
        print(f"Downloaded or already present: {len(wanted) - len(failed)}")
        print(f"Failed: {len(failed)}")
        for row in failed[:20]:
            print(f"  {row['id']} {row['file_name']}: {row['status']}")
    print(f"Skipped: {len(rows) - len(wanted)}")
    if problems:
        print(f"Catalogues that could not be read: {' '.join(problems)} (run again to retry)")
    return 1 if failed or problems else 0


if __name__ == "__main__":
    sys.exit(main())
