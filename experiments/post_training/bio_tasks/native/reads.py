# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Native alignment-depth, genotype-QC, and adapter-trimming oracles."""

import csv
from pathlib import Path

from experiments.post_training.bio_tasks.native.commands import execute
from experiments.post_training.bio_tasks.solvers.formats import tab_rows


def solve_samtools(inputs: Path, work: Path) -> list[dict]:
    aligned = work / "sorted.bam"
    execute(["samtools", "sort", "-@", "0", "-o", str(aligned), str(inputs / "reads.sam")], work, "sort.log")
    # Replace defaults with exactly unmapped/secondary/supplementary exclusions.
    depth = execute(
        ["samtools", "depth", "-a", "-q", "0", "-Q", "0", "-g", "1796", "-G", "2308", str(aligned)], work, "depth.tsv"
    )
    values = {(chrom, int(position) - 1): int(count) for chrom, position, count in tab_rows(depth)}
    answer = []
    for chrom, start, end, name in tab_rows(inputs / "windows.bed"):
        counts = [values.get((chrom, position), 0) for position in range(int(start), int(end))]
        answer.append({"id": name, "depth_sum": sum(counts), "covered_bases": sum(count > 0 for count in counts)})
    return answer


def fastq_answer(path: Path) -> list[dict]:
    rows = path.read_text().splitlines()
    assert len(rows) % 4 == 0
    return [
        {"id": rows[i][1:].split()[0], "sequence": rows[i + 1], "quality": rows[i + 3], "length": len(rows[i + 1])}
        for i in range(0, len(rows), 4)
    ]


def solve_cutadapt(inputs: Path, work: Path) -> list[dict]:
    destination = work / "trimmed.fastq"
    execute(
        [
            "cutadapt",
            "-j",
            "1",
            "-a",
            "AGATCGGAAGAGC",
            "-O",
            "6",
            "-e",
            "0",
            "--no-indels",
            "-o",
            str(destination),
            str(inputs / "reads.fastq"),
        ],
        work,
        "cutadapt.log",
    )
    return fastq_answer(destination)


def solve_plink(inputs: Path, work: Path) -> list[dict]:
    prefix = work / "missing"
    execute(
        [
            "plink2",
            "--vcf",
            str(inputs / "variants.vcf"),
            "--vcf-half-call",
            "missing",
            "--missing",
            "sample-only",
            "--threads",
            "1",
            "--memory",
            "512",
            "--out",
            str(prefix),
        ],
        work,
        "plink.log",
    )
    with prefix.with_suffix(".smiss").open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [
            {
                "id": row["IID"],
                "missing": int(row["MISSING_CT"]),
                "rate": int(row["MISSING_CT"]) / int(row["OBS_CT"]),
                "keep": int(int(row["MISSING_CT"]) / int(row["OBS_CT"]) <= 0.25),
            }
            for row in reader
        ]


def solve_vcftools(inputs: Path, work: Path) -> list[dict]:
    prefix = work / "missing"
    execute(
        ["vcftools", "--vcf", str(inputs / "variants.vcf"), "--missing-indv", "--out", str(prefix)], work, "vcftools.log"
    )
    lines = prefix.with_suffix(".imiss").read_text().splitlines()
    header = lines[0].split()
    answer = []
    for line in lines[1:]:
        row = dict(zip(header, line.split(), strict=True))
        missing, total = int(row["N_MISS"]), int(row["N_DATA"])
        answer.append(
            {"id": row["INDV"], "missing": missing, "rate": missing / total, "keep": int(missing / total <= 0.25)}
        )
    return answer
