# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Independent, standard-library calculations over supplied RNA-seq observations."""

import csv
import json
import math
import statistics
from functools import partial
from pathlib import Path


def solve_real_expression(inputs: Path, operation: str) -> list[dict]:
    query = json.loads((inputs / "query.json").read_text())
    with (inputs / "samples.tsv").open() as stream:
        metadata = list(csv.DictReader(stream, delimiter="\t"))
    selected = [row for row in metadata if query["population"] in ("all", row["population"])]
    if operation.endswith("normalized-contrast"):
        selected = [row for row in selected if row["stage"] in (query["baseline"], query["treatment"])]
    names = [row["sample"] for row in selected]
    with (inputs / "counts.tsv").open() as stream:
        genes = {
            row["EntrezGeneID"]: [int(row[name]) for name in names] for row in csv.DictReader(stream, delimiter="\t")
        }
    totals = [sum(column) for column in zip(*genes.values(), strict=True)]
    if operation.endswith("library-qc"):
        return [
            {"id": name, "total_counts": totals[j], "detected_genes": sum(row[j] > 0 for row in genes.values())}
            for j, name in enumerate(names)
        ]
    if operation.endswith("cpm-filter"):
        answer = []
        for gene in (inputs / "panel.txt").read_text().splitlines():
            hits = sum(
                value * 1_000_000 / total >= query["minimum_cpm"]
                for value, total in zip(genes[gene], totals, strict=True)
            )
            answer.append({"id": gene, "n_samples": hits, "keep": int(hits >= 2)})
        return answer
    ratios = [[] for _ in names]
    for values in genes.values():
        if 0 in values:
            continue
        geometric_mean = math.exp(math.fsum(math.log(value) for value in values) / len(values))
        for column, value in zip(ratios, values, strict=True):
            column.append(math.log(value) - math.log(geometric_mean))
    factors = [math.exp(statistics.median(column)) for column in ratios]
    if operation.endswith("size-factors"):
        return [
            {"id": name, "size_factor": factors[j], "eligible_genes": len(ratios[j])} for j, name in enumerate(names)
        ]
    baseline = [j for j, row in enumerate(selected) if row["stage"] == query["baseline"]]
    treatment = [j for j, row in enumerate(selected) if row["stage"] == query["treatment"]]
    answer = []
    for gene in (inputs / "panel.txt").read_text().splitlines():
        values = genes[gene]
        before = statistics.mean(values[j] / factors[j] for j in baseline)
        after = statistics.mean(values[j] / factors[j] for j in treatment)
        answer.append(
            {
                "id": gene,
                "log2_ratio": math.log2((after + 1) / (before + 1)),
                "baseline_replicates": len(baseline),
                "treatment_replicates": len(treatment),
            }
        )
    return answer


SOLVERS = {
    name: partial(solve_real_expression, operation=name)
    for name in (
        "real-rnaseq-library-qc",
        "real-rnaseq-cpm-filter",
        "real-rnaseq-size-factors",
        "real-rnaseq-normalized-contrast",
    )
}
