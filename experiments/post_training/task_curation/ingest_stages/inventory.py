# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Record historical ingest scripts without importing or executing them."""

import argparse
import csv
import hashlib
import os
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

SCRIPT_SUFFIXES = frozenset({".py", ".sh", ".bash", ".zsh"})
COPY_DIRECTORIES = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "site-packages",
        "node_modules",
        "upstream",
        "candidate",
        "baseline",
        "extracted-production",
    }
)
CURATION_PACKAGE = "experiments.post_training.task_curation"

# These are reuse destinations, not assertions that the unified graph has adopted
# every historical script's behavior. Unmatched original scripts need review.
REUSE_MODULES = (
    "license_check",
    "license_resolution",
    "pipeline",
    "intake_population",
    "native_population",
    "published_census",
    "census_binding",
    "daytona_census",
    "within_source_census",
    "source_fingerprints",
    "source_population_cache",
    "source_duplicate_ci",
    "source_duplicates",
    "backend_capabilities",
    "backend_live_qualification",
    "oracle_filter",
    "lexical_scan",
    "launch_sample_qc",
    "sample_qc",
)


class Disposition(StrEnum):
    HISTORICAL_COPY = "historical-copy"
    REUSE_REVIEW = "reuse-review"
    ORIGINAL_REVIEW = "original-review"
    FRICTION_REVIEW = "friction-review"
    MAINTAINED = "maintained"
    SYMLINK_REVIEW = "symlink-review"


@dataclass(frozen=True)
class InventoryRow:
    path: str
    disposition: Disposition
    reuse_modules: tuple[str, ...]
    sha256: str


def classify_script(path: Path, relative: Path) -> InventoryRow:
    """Separate copied source trees from original launch and measurement code."""
    if path.is_symlink():
        return InventoryRow(str(relative), Disposition.SYMLINK_REVIEW, (), "")
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if any(part in COPY_DIRECTORIES for part in relative.parts):
        return InventoryRow(str(relative), Disposition.HISTORICAL_COPY, (), digest)
    text = payload.decode("utf-8", errors="replace")
    modules = tuple(f"{CURATION_PACKAGE}.{name}" for name in REUSE_MODULES if name in text)
    disposition = Disposition.REUSE_REVIEW if modules else Disposition.ORIGINAL_REVIEW
    return InventoryRow(str(relative), disposition, modules, digest)


def inventory_rows(root: Path, checkout: Path) -> list[InventoryRow]:
    """Inventory each lane script, friction record, and maintained ingest helper."""
    rows = []
    for lane in sorted(root.glob("agent-*")):
        if not lane.is_dir() or lane.is_symlink():
            raise ValueError(f"Expected a real lane directory: {lane.name}")
        for directory, subdirectories, filenames in os.walk(lane, followlinks=False):
            subdirectories.sort()
            for filename in sorted(filenames):
                path = Path(directory) / filename
                if path.suffix in SCRIPT_SUFFIXES:
                    rows.append(classify_script(path, path.relative_to(root)))
    for path in sorted(root.glob("FRICTION-*.md")):
        if path.is_symlink():
            raise ValueError(f"Friction record must be a regular file: {path.name}")
        rows.append(
            InventoryRow(path.name, Disposition.FRICTION_REVIEW, (), hashlib.sha256(path.read_bytes()).hexdigest())
        )
    package = checkout / "experiments/post_training/task_curation"
    for name in REUSE_MODULES:
        path = package / f"{name}.py"
        rows.append(
            InventoryRow(
                str(path.relative_to(checkout)),
                Disposition.MAINTAINED,
                (f"{CURATION_PACKAGE}.{name}",),
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    return rows


def write_inventory(rows: list[InventoryRow], output: Path) -> None:
    """Write the complete machine-readable inventory and a bounded review summary."""
    output.mkdir(parents=True, exist_ok=True)
    ledger = output / "inventory.tsv"
    with ledger.open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(("path", "disposition", "reuse_modules", "sha256"))
        writer.writerows((row.path, row.disposition.value, ",".join(row.reuse_modules), row.sha256) for row in rows)
    counts = Counter(row.disposition for row in rows)
    digest = hashlib.sha256(ledger.read_bytes()).hexdigest()
    summary = [
        "# Atlas ingest script inventory",
        "",
        f"Complete inventory: [inventory.tsv](inventory.tsv), SHA-256 `{digest}`.",
        "",
        "Every Python and shell script under each agent directory is listed, including",
        "copied source trees. The inventory reads files without importing or executing",
        "them. Original scripts and friction records require semantic review before",
        "their behavior can be adopted or retired. A reuse candidate is a textual",
        "match to a maintained helper, not proof that behavior has been folded in.",
        "Historical copies remain evidence and are never pipeline entry points.",
        "",
        "| Disposition | Files |",
        "| --- | ---: |",
        *(f"| {kind.value} | {counts[kind]} |" for kind in Disposition),
        "",
        "## Maintained reuse destinations",
        "",
        *(f"- `{CURATION_PACKAGE}.{name}`" for name in REUSE_MODULES),
        "",
        "This inventory does not establish completion of the ingestion command,",
        "retirement review, source admission, or a regional run.",
        "",
    ]
    (output / "INVENTORY.md").write_text("\n".join(summary))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkout = Path(__file__).resolve().parents[4]
    write_inventory(inventory_rows(args.root, checkout), args.output)


if __name__ == "__main__":
    main()
