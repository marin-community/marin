# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate a fresh native BAM oracle against the independent source reference."""

import argparse
import gzip
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import pysam

from experiments.post_training.bio_tasks.contract import grade_files
from experiments.post_training.bio_tasks.generators.real_bam import bam_contract
from experiments.post_training.bio_tasks.solvers.real_bam import solve_bam


def _mutation_copy(source: Path, destination: Path) -> None:
    destination.mkdir()
    for name in (
        "answer.json",
        "reads.bam",
        "reads.bam.bai",
        "primary_reads.tsv",
        "read_lengths.tsv",
        "flag_counts.tsv",
        "eligible_reads.tsv",
    ):
        shutil.copyfile(source / name, destination / name)


def _change_tsv(path: Path, column: str, value: str | None) -> None:
    lines = path.read_text().splitlines()
    fields = lines[0].split("\t")
    values = lines[1].split("\t")
    index = fields.index(column)
    values[index] = value if value is not None else str(int(values[index]) + 1)
    lines[1] = "\t".join(values)
    path.write_text("\n".join(lines) + "\n")


def _changed_bam(path: Path) -> None:
    changed = path.with_name("changed.bam")
    updated = False
    with pysam.AlignmentFile(path, "rb") as source, pysam.AlignmentFile(changed, "wb", header=source.header) as output:
        for record in source.fetch(until_eof=True):
            if (
                not updated
                and not record.is_secondary
                and not record.is_supplementary
                and not record.is_unmapped
                and record.is_proper_pair
                and 30 <= record.mapping_quality < 255
            ):
                record.mapping_quality = 29
                updated = True
            output.write(record)
    if not updated:
        raise ValueError("No eligible native BAM record to mutate")
    pysam.index(str(changed))
    shutil.copyfile(changed, path)
    shutil.copyfile(changed.with_name(changed.name + ".bai"), path.with_name(path.name + ".bai"))


def validate(inputs: Path, prepared: Path, output: Path) -> dict:
    """Require positive full-artifact reward and independent negative controls."""
    output.mkdir(parents=True, exist_ok=False)
    with gzip.open(prepared / "private-bam-reference.json.gz", "rt") as handle:
        reference = json.load(handle)
    contract = bam_contract(reference)
    solved = solve_bam(inputs, output)
    (output / "answer.json").write_text(json.dumps(solved, indent=2) + "\n")
    reference_path = output / "reference.json"
    reference_path.write_text(contract.model_dump_json())
    positive = grade_files(reference_path, output / "answer.json")
    if positive.reward != 1:
        raise ValueError(f"Fresh BAM oracle disagrees with source reference: {positive.detail}")
    mutations = {}
    with TemporaryDirectory(prefix="bam-artifact-controls-") as directory:
        root = Path(directory)
        for name in (
            "wrong_mapq",
            "wrong_mate",
            "wrong_length",
            "missing_eligible",
            "changed_bam",
            "truncated_bam",
            "bad_bai",
        ):
            destination = root / name
            _mutation_copy(output, destination)
            if name == "wrong_mapq":
                _change_tsv(destination / "primary_reads.tsv", "mapq", None)
            elif name == "wrong_mate":
                _change_tsv(destination / "primary_reads.tsv", "mate", "3")
            elif name == "wrong_length":
                _change_tsv(destination / "read_lengths.tsv", "reads", None)
            elif name == "missing_eligible":
                path = destination / "eligible_reads.tsv"
                path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
            elif name == "changed_bam":
                _changed_bam(destination / "reads.bam")
            elif name == "truncated_bam":
                path = destination / "reads.bam"
                with path.open("r+b") as handle:
                    handle.truncate(max(0, path.stat().st_size // 2))
            else:
                (destination / "reads.bam.bai").write_bytes(b"invalid-index")
            verdict = grade_files(reference_path, destination / "answer.json")
            if verdict.reward != 0:
                raise ValueError(f"Invalid BAM artifact passed: {name}")
            mutations[name] = {
                "reward": verdict.reward,
                "bam": verdict.detail.get("artifact_checks", {}).get("reads.bam"),
                "index": verdict.detail.get("artifact_checks", {}).get("reads.bam.bai"),
            }
    return {"positive_reward": positive.reward, "negative_controls": mutations}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.inputs, args.prepared, args.output)
    (args.output / "validation-report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
