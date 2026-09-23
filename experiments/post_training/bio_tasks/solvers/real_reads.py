# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming read analyses independent of the array-based task references."""

import json
import math
from contextlib import ExitStack
from functools import partial
from pathlib import Path

from experiments.post_training.bio_tasks.solvers.formats import fastq


def solve_real_reads(inputs: Path, output: Path, operation: str) -> list[dict]:
    query = json.loads((inputs / "query.json").read_text())
    paths = [inputs / f"reads_R{mate}.fastq" for mate in (1, 2)]
    if operation in ("real-fastq-pair-filter", "real-fastq-fixed-trim"):
        action = "filtered" if operation == "real-fastq-pair-filter" else "trimmed"
        counts = [[0, 0], [0, 0]]
        answer = []
        with ExitStack() as stack:
            handles = [stack.enter_context((output / f"{action}_R{mate}.fastq").open("w")) for mate in (1, 2)]
            for pair in zip(*(fastq(path) for path in paths), strict=True):
                if pair[0][0] != pair[1][0]:
                    raise ValueError("Mismatched read pair")
                if action == "filtered":
                    keep = all(
                        sum(ord(q) - 33 < query["qualified_phred"] for q in qualities) * 100
                        <= len(sequence) * query["maximum_unqualified_percent"]
                        and sequence.count("N") <= query["maximum_ns"]
                        and len(sequence) >= query["minimum_length"]
                        for _, sequence, qualities in pair
                    )
                    answer.append({"id": pair[0][0], "keep": int(keep)})
                    if not keep:
                        continue
                for index, (name, sequence, qualities) in enumerate(pair):
                    if action == "trimmed":
                        start, end = query["trim_front"], len(sequence) - query["trim_tail"]
                        sequence, qualities = sequence[start:end], qualities[start:end]
                    handles[index].write(f"@{name}\n{sequence}\n+\n{qualities}\n")
                    counts[index][0] += 1
                    counts[index][1] += len(sequence)
        if action == "trimmed":
            return [
                {"id": f"R{index + 1}", "reads": reads, "bases": bases} for index, (reads, bases) in enumerate(counts)
            ]
        return answer
    if operation == "real-fastq-expected-errors":
        answer = []
        for first, second in zip(*(fastq(path) for path in paths), strict=True):
            if first[0] != second[0]:
                raise ValueError("Mismatched read pair")
            errors = math.fsum(10 ** (-(ord(character) - 33) / 10) for character in first[2] + second[2])
            answer.append({"id": first[0], "errors": errors})
        return answer
    if operation == "real-fastq-cycle-quality":
        answer = []
        for mate, path in enumerate(paths, 1):
            cycles = {}
            for _, _sequence, quality in fastq(path):
                for cycle, character in enumerate(quality, 1):
                    value = ord(character) - 33
                    row = cycles.setdefault(cycle, [0, 0, 0, 0])
                    row[0] += value
                    row[1] += value >= 20
                    row[2] += value >= 30
                    row[3] += 1
            answer.extend(
                {"id": f"R{mate}:{cycle}", "mean_phred": total / n, "q20": q20, "q30": q30, "bases": n}
                for cycle, (total, q20, q30, n) in cycles.items()
            )
        return answer
    if operation == "real-fastq-quality-yield":
        totals = {"id": "library", "reads": 0, "bases": 0, "q20": 0, "q30": 0}
        for path in paths:
            for _, sequence, qualities in fastq(path):
                totals["reads"] += 1
                totals["bases"] += len(sequence)
                totals["q20"] += sum(ord(q) >= 53 for q in qualities)
                totals["q30"] += sum(ord(q) >= 63 for q in qualities)
        return [totals]
    raise ValueError(operation)


OUTPUT_SOLVERS = {
    name: partial(solve_real_reads, operation=name)
    for name in (
        "real-fastq-pair-filter",
        "real-fastq-fixed-trim",
        "real-fastq-cycle-quality",
        "real-fastq-expected-errors",
        "real-fastq-quality-yield",
    )
}
