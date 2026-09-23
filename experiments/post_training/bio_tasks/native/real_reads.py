# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reference operations on observed FASTQ reads using installed bioinformatics tools."""

import csv
import json
from pathlib import Path

from experiments.post_training.bio_tasks.native.commands import execute
from experiments.post_training.bio_tasks.solvers.formats import fastq


def solve_fastp(inputs: Path, work: Path) -> list[dict]:
    query = json.loads((inputs / "query.json").read_text())
    execute(
        [
            "fastp",
            "--in1",
            str(inputs / "reads_R1.fastq"),
            "--in2",
            str(inputs / "reads_R2.fastq"),
            "--out1",
            str(work / "filtered_R1.fastq"),
            "--out2",
            str(work / "filtered_R2.fastq"),
            "--thread",
            "1",
            "--disable_adapter_trimming",
            "--disable_trim_poly_g",
            "--dont_eval_duplication",
            "--qualified_quality_phred",
            str(query["qualified_phred"]),
            "--unqualified_percent_limit",
            str(query["maximum_unqualified_percent"]),
            "--n_base_limit",
            str(query["maximum_ns"]),
            "--length_required",
            str(query["minimum_length"]),
            "--json",
            str(work / "fastp.json"),
            "--html",
            str(work / "fastp.html"),
        ],
        work,
        "fastp.stdout",
    )
    first = [name for name, _, _ in fastq(work / "filtered_R1.fastq")]
    second = [name for name, _, _ in fastq(work / "filtered_R2.fastq")]
    if first != second or len(first) != len(set(first)):
        raise ValueError("Native fastp outputs have mismatched or duplicate pairs")
    retained = set(first)
    return [{"id": name, "keep": int(name in retained)} for name, _, _ in fastq(inputs / "reads_R1.fastq")]


def solve_picard(inputs: Path, work: Path) -> list[dict]:
    alignment = work / "unmapped.bam"
    execute(
        [
            "picard",
            "-Xmx1g",
            "FastqToSam",
            f"FASTQ={inputs / 'reads_R1.fastq'}",
            f"FASTQ2={inputs / 'reads_R2.fastq'}",
            f"OUTPUT={alignment}",
            "SAMPLE_NAME=PhiX",
            "READ_GROUP_NAME=observed",
            "QUALITY_FORMAT=Standard",
        ],
        work,
        "fastq-to-sam.stdout",
    )
    metrics = work / "quality-yield.txt"
    execute(
        [
            "picard",
            "-Xmx1g",
            "CollectQualityYieldMetrics",
            f"INPUT={alignment}",
            f"OUTPUT={metrics}",
            "USE_ORIGINAL_QUALITIES=false",
        ],
        work,
        "quality-yield.stdout",
    )
    lines = [line for line in metrics.read_text().splitlines() if line and not line.startswith("#")]
    row = next(csv.DictReader(lines, delimiter="\t"))
    return [
        {
            "id": "library",
            "reads": int(row["TOTAL_READS"]),
            "bases": int(row["TOTAL_BASES"]),
            "q20": int(row["Q20_BASES"]),
            "q30": int(row["Q30_BASES"]),
        }
    ]
