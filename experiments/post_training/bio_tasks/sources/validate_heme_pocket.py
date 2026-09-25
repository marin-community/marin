# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade a fresh native heme analysis and discriminating artifact corruptions."""

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path

from tasktrove_verify.grade import Status

from experiments.post_training.bio_tasks.contract import grade_files
from experiments.post_training.bio_tasks.generators.real_heme_pocket import heme_pocket_contract


def write_rows(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> tuple[list[dict], list[str]]:
    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
        return rows, list(reader.fieldnames or ())


def validate(
    reference: Path, inputs: Path, native: Path, output: Path, lock: Path, revision: str, native_seconds: float
) -> None:
    output.mkdir(parents=True, exist_ok=False)
    contract = heme_pocket_contract(json.loads(gzip.decompress(reference.read_bytes())))
    contract_path = output / "reference.json"
    contract_path.write_text(contract.model_dump_json())
    started = time.monotonic()
    checks = {}

    def grade(name: str, path: Path, expected: int) -> None:
        reward = grade_files(contract_path, path / "answer.json")
        checks[name] = asdict(reward)
        if reward.status != Status.SCORED or reward.reward != expected:
            (output / "failed-controls.json").write_text(json.dumps(checks, indent=2) + "\n")
            raise ValueError(f"Unexpected verifier outcome for {name}: {reward.status}, reward={reward.reward}")

    grade("native", native, 1)
    shuffled = output / "reordered"
    shutil.copytree(native, shuffled)
    for name in contract.tables:
        rows, columns = read_rows(shuffled / name)
        write_rows(shuffled / name, list(reversed(rows)), list(reversed(columns)))
    grade("reordered_rows_and_columns", shuffled, 1)
    corruptions = [
        ("changed_contact_distance", "contacts.tsv", "distance"),
        ("changed_residue_distance", "residues.tsv", "minimum_distance"),
        ("changed_residue_burial", "residues.tsv", "buried_area"),
        ("changed_heme_atom_exposure", "heme_atoms.tsv", "sasa_complex"),
        ("changed_author_chain", "residues.tsv", "chain"),
        ("changed_iron_neighbor", "answer.json", "nearest_iron_protein_atom"),
        ("halved_total_burial", "answer.json", "total_buried_area"),
        ("reversed_protein_burial", "answer.json", "protein_buried_area"),
        ("pairs_as_residue_count", "answer.json", "pocket_residues"),
    ]
    for name, filename, field in corruptions:
        destination = output / name
        shutil.copytree(native, destination)
        path = destination / filename
        if filename == "answer.json":
            rows = json.loads(path.read_text())
            if name == "changed_iron_neighbor":
                rows[0][field] = "protein:wrong:0:.:UNK:CA"
            elif name == "halved_total_burial":
                rows[0][field] /= 2
            elif name == "reversed_protein_burial":
                rows[0][field] *= -1
            else:
                rows[0][field] = rows[0]["contact_pairs"]
            path.write_text(json.dumps(rows) + "\n")
        else:
            rows, columns = read_rows(path)
            rows[0][field] = "wrong_chain" if field == "chain" else str(float(rows[0][field]) + 0.1)
            write_rows(path, rows, columns)
        grade(name, destination, 0)
    for filename in contract.tables:
        for damage in ("missing_artifact", "missing_row", "duplicate_row"):
            name = f"{damage}:{filename}"
            destination = output / name.replace(":", "-")
            shutil.copytree(native, destination)
            path = destination / filename
            if damage == "missing_artifact":
                path.unlink()
            else:
                rows, columns = read_rows(path)
                write_rows(path, rows[1:] if damage == "missing_row" else [*rows, rows[0]], columns)
            grade(name, destination, 0)
    destination = output / "missing_noncontact_residue"
    shutil.copytree(native, destination)
    rows, columns = read_rows(destination / "residues.tsv")
    dropped = next(index for index, row in enumerate(rows) if row["in_pocket"] == "0")
    del rows[dropped]
    write_rows(destination / "residues.tsv", rows, columns)
    grade("missing_noncontact_residue", destination, 0)
    artifact_hashes = {
        name: hashlib.sha256((native / name).read_bytes()).hexdigest() for name in ("answer.json", *contract.tables)
    }
    packages = json.loads(lock.read_text())
    evidence = {
        "schema_version": 1,
        "source_revision": revision,
        "scope": (
            "Complete observed 4HHB; independent paired-PDB arithmetic reference, fresh native Biopython mmCIF/SASA "
            "oracle and complete artifact controls. No model calls."
        ),
        "checks": [
            {
                "repository_index": 19,
                "repository": "Biopython",
                "recipe": "real-heme-pocket-burial",
                "status": "passed",
                "execution": "completed",
                "environment": {"resolved_packages": packages},
                "cases": [
                    {
                        "case_id": "PDB4HHB-four-hemes-complete-globin-complex",
                        "recipe": "real-heme-pocket-burial",
                        "execution": "completed",
                        "runtime_seconds": native_seconds,
                        "verification": checks["native"],
                        "input_sha256": {
                            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(inputs.iterdir())
                        },
                        "contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
                        "artifact_sha256": artifact_hashes,
                        "artifact_rows": {name: len(table.expected) for name, table in contract.tables.items()},
                        "positive_controls": {name: 1 for name in list(checks)[:2]},
                        "negative_controls": {name: 0 for name in list(checks)[2:]},
                        "grading_seconds": time.monotonic() - started,
                    }
                ],
            }
        ],
    }
    (output / "controls.json").write_text(json.dumps(checks, indent=2) + "\n")
    (output / "checks.json").write_text(json.dumps(evidence, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--native-seconds", type=float, required=True)
    args = parser.parse_args()
    validate(args.reference, args.inputs, args.native, args.output, args.lock, args.source_revision, args.native_seconds)


if __name__ == "__main__":
    main()
