# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sequencing-read solutions using flags, CIGARs, and native qualities."""

import re
from collections import Counter, defaultdict
from pathlib import Path

from experiments.post_training.bio_tasks.solvers.formats import tab_rows, table


def sam_rows(inputs: Path) -> list[list[str]]:
    return [line.split("\t") for line in (inputs / "reads.sam").read_text().splitlines() if not line.startswith("@")]


def sam_eligible(row: list[str]) -> bool:
    return not (int(row[1]) & (4 | 256 | 512 | 1024 | 2048)) and 30 <= int(row[4]) < 255


def solve_inclusion(inputs: Path) -> list[dict]:
    return [{"id": row[0], "keep": int(sam_eligible(row))} for row in sam_rows(inputs)]


def solve_fragments(inputs: Path) -> list[dict]:
    mates = defaultdict(set)
    for row in sam_rows(inputs):
        names = mates[row[0]]
        if sam_eligible(row):
            names.add(int(row[1]) & (64 | 128))
    return [{"id": name, "keep": int(bits == {64, 128}), "passing_mates": len(bits)} for name, bits in mates.items()]


def aligned_calls(row: list[str]) -> list[tuple[int, str, int]]:
    ref, query = int(row[3]) - 1, 0
    calls = []
    for size, operation in re.findall(r"(\d+)([MIDNSHP=X])", row[5]):
        n = int(size)
        if operation in "M=X":
            calls.extend((ref + i, row[9][query + i], ord(row[10][query + i]) - 33) for i in range(n))
        if operation in "MDN=X":
            ref += n
        if operation in "MIS=X":
            query += n
    return calls


def solve_pileup(inputs: Path) -> list[dict]:
    counts = defaultdict(Counter)
    for row in sam_rows(inputs):
        if sam_eligible(row):
            for position, base, quality in aligned_calls(row):
                if quality >= 25 and base in "ACGT":
                    counts[row[2], position][base] += 1
    return [
        {"id": name, **{base: counts[chrom, int(start)][base] for base in "ACGT"}}
        for chrom, start, _, name in tab_rows(inputs / "loci.bed")
    ]


def solve_coverage(inputs: Path) -> list[dict]:
    depths = Counter()
    for row in sam_rows(inputs):
        if not int(row[1]) & (4 | 256 | 2048):
            for position, _, _ in aligned_calls(row):
                depths[row[2], position] += 1
    answer = []
    for chrom, start, end, name in tab_rows(inputs / "windows.bed"):
        values = [depths[chrom, p] for p in range(int(start), int(end))]
        answer.append({"id": name, "depth_sum": sum(values), "covered_bases": sum(v > 0 for v in values)})
    return answer


def solve_junctions(inputs: Path) -> list[dict]:
    support = defaultdict(set)
    for row in sam_rows(inputs):
        if int(row[1]) & (4 | 256 | 2048):
            continue
        pos = int(row[3]) - 1
        for size, operation in re.findall(r"(\d+)([MIDNSHP=X])", row[5]):
            n = int(size)
            if operation == "N":
                support[f"{row[2]}/{pos}/{pos+n}"].add(row[0])
            if operation in "MDN=X":
                pos += n
    return [{"id": name, "reads": len(names)} for name, names in support.items()]


def fastq_records(inputs: Path) -> list[tuple[str, str, str]]:
    lines = (inputs / "reads.fastq").read_text().splitlines()
    return [(lines[i][1:].split()[0], lines[i + 1], lines[i + 3]) for i in range(0, len(lines), 4)]


def solve_adapter(inputs: Path) -> list[dict]:
    adapter = "AGATCGGAAGAGC"
    answer = []
    for name, sequence, quality in fastq_records(inputs):
        cut = sequence.find(adapter)
        if cut < 0:
            cut = len(sequence)
            for size in range(min(len(adapter) - 1, len(sequence)), 5, -1):
                if sequence.endswith(adapter[:size]):
                    cut = len(sequence) - size
                    break
        answer.append({"id": name, "sequence": sequence[:cut], "quality": quality[:cut], "length": cut})
    return answer


def solve_quality_trim(inputs: Path) -> list[dict]:
    answer = []
    for name, sequence, quality in fastq_records(inputs):
        cut = len(sequence)
        while cut and ord(quality[cut - 1]) - 33 < 20:
            cut -= 1
        answer.append({"id": name, "sequence": sequence[:cut], "quality": quality[:cut], "length": cut})
    return answer


def solve_umis(inputs: Path) -> list[dict]:
    cells = (inputs / "cells.txt").read_text().splitlines()
    genes = (inputs / "genes.txt").read_text().splitlines()
    molecules = defaultdict(set)
    for row in table(inputs / "alignments.csv"):
        if row["cell"] in cells and row["gene"] in genes:
            molecules[row["cell"], row["gene"]].add(row["umi"])
    return [{"id": f"{cell}/{gene}", "molecules": len(molecules[cell, gene])} for cell in cells for gene in genes]


SOLVERS = {
    "sam-inclusion": solve_inclusion,
    "sam-fragment-counts": solve_fragments,
    "sam-allele-pileup": solve_pileup,
    "sam-cigar-coverage": solve_coverage,
    "sam-junction-support": solve_junctions,
    "fastq-adapter-trimming": solve_adapter,
    "fastq-quality-trimming": solve_quality_trim,
    "umi-deduplication": solve_umis,
}
