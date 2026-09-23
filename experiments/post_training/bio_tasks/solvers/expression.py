# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Expression calculations from sparse coordinates and tabular counts."""

import math
from functools import partial
from pathlib import Path
from statistics import median

from experiments.post_training.bio_tasks.solvers.formats import tab_rows, table


def solve_expression(inputs: Path, operation: str) -> list[dict]:
    answer = []
    if operation.startswith("matrixmarket"):
        features = tab_rows(inputs / "features.tsv")
        barcodes = (inputs / "barcodes.tsv").read_text().splitlines()
        lines = [line for line in (inputs / "matrix.mtx").read_text().splitlines() if not line.startswith("%")]
        counts = {}
        for line in lines[1:]:
            row, column, value = map(int, line.split())
            counts[row - 1, column - 1] = value
        genes = [i for i, row in enumerate(features) if row[2] == "Gene Expression"]
        for j, barcode in enumerate(barcodes):
            total = sum(counts.get((i, j), 0) for i in genes)
            if operation == "matrixmarket-cell-qc":
                mito = sum(counts.get((i, j), 0) for i in genes if features[i][1].startswith("MT-"))
                answer.append(
                    {
                        "id": barcode,
                        "counts": total,
                        "features": sum(counts.get((i, j), 0) > 0 for i in genes),
                        "mt_fraction": mito / total if total else None,
                    }
                )
            elif operation == "matrixmarket-log-normalization":
                for i in genes:
                    value = math.log1p(counts.get((i, j), 0) / total * 10000) if total else 0.0
                    answer.append({"id": features[i][0] + ":" + barcode, "log_count": value})
        if operation == "matrixmarket-feature-filtering":
            for i in genes:
                n = sum(counts.get((i, j), 0) >= 2 for j in range(len(barcodes)))
                answer.append({"id": features[i][0], "cells_ge_two": n, "keep": int(n >= 2)})
    elif operation == "splice-psi":
        for row in table(inputs / "junctions.csv"):
            left, right, skip = [int(row[key]) for key in ["junction_upstream", "junction_downstream", "junction_skip"]]
            denominator = left + right + 2 * skip
            answer.append(
                {
                    "id": row["event"],
                    "psi": (left + right) / denominator if denominator else None,
                    "effective_support": denominator / 2,
                }
            )
    elif operation == "bulk-size-factors":
        rows = table(inputs / "counts.csv")
        samples = [key for key in rows[0] if key != "gene"]
        eligible = [row for row in rows if all(int(row[sample]) > 0 for sample in samples)]
        means = [math.exp(sum(math.log(int(row[sample])) for sample in samples) / len(samples)) for row in eligible]
        for sample in samples:
            answer.append(
                {
                    "id": sample,
                    "size_factor": median(int(row[sample]) / gm for row, gm in zip(eligible, means, strict=True)),
                    "eligible_genes": len(eligible),
                }
            )
    elif operation == "bulk-cpm-filter":
        rows = table(inputs / "counts.csv")
        samples = [key for key in rows[0] if key != "gene"]
        totals = {sample: sum(int(row[sample]) for row in rows) for sample in samples}
        threshold = float((inputs / "threshold.txt").read_text())
        for row in rows:
            n = sum(int(row[sample]) * 1000000 >= threshold * totals[sample] for sample in samples)
            answer.append({"id": row["gene"], "n_samples": n, "keep": int(n >= 2)})
    else:
        rows = table(inputs / "results.csv")
        tested = sorted([row for row in rows if row["pvalue"] != ""], key=lambda row: float(row["pvalue"]))
        adjusted = {}
        bound = 1.0
        for index in range(len(tested) - 1, -1, -1):
            row = tested[index]
            bound = min(bound, float(row["pvalue"]) * len(tested) / (index + 1))
            adjusted[row["gene"]] = bound
        for row in rows:
            q = adjusted.get(row["gene"])
            answer.append(
                {
                    "id": row["gene"],
                    "padj": q,
                    "upregulated": int(q is not None and q <= 0.05 and float(row["log2fc"]) >= 1),
                }
            )
    return answer


NAMES = (
    "matrixmarket-cell-qc",
    "matrixmarket-log-normalization",
    "matrixmarket-feature-filtering",
    "splice-psi",
    "bulk-size-factors",
    "bulk-cpm-filter",
    "differential-expression-bh",
)
SOLVERS = {name: partial(solve_expression, operation=name) for name in NAMES}
