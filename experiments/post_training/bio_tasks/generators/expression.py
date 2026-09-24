# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sparse expression matrices and bulk or splicing analysis recipes."""

import math
import random
from functools import partial

from experiments.post_training.bio_tasks.contract import Column, Contract
from experiments.post_training.bio_tasks.recipe_types import Instance, Recipe, csv_text


def generate_expression(seed: int, operation: str) -> Instance:
    rng = random.Random(seed)
    inputs, expected = {}, {}
    extra_mutations = {}
    if operation.startswith("matrixmarket"):
        a, b, c = [rng.randint(3, 9) for _ in range(3)]
        dense = [[a, 0, 1, 0], [b, 2, 0, 0], [0, c, 2, 0], [9, 9, 9, 0], [0, 0, 0, 0]]
        if operation == "matrixmarket-feature-filtering":
            dense[0][2] = rng.randint(1, 4)
            dense[1][1] = rng.randint(1, 4)
            dense[2][2] = rng.randint(1, 4)
        triples = [(i + 1, j + 1, value) for i, row in enumerate(dense) for j, value in enumerate(row) if value]
        rng.shuffle(triples)
        inputs = {
            "matrix.mtx": (
                "%%MatrixMarket matrix coordinate integer general\n% features by barcodes; 1-based indices\n5 4 "
                + str(len(triples))
                + "\n"
                + "".join(f"{i} {j} {value}\n" for i, j, value in triples)
            ),
            "features.tsv": (
                "g0\tMT-CO1\tGene Expression\n"
                "g1\tGeneX\tGene Expression\n"
                "g2\tGeneX\tGene Expression\n"
                "ab0\tCD3\tAntibody Capture\n"
                "g3\tSilent\tGene Expression\n"
            ),
            "barcodes.tsv": "c0\nc1\nc2\nc3\n",
        }
        totals = [a + b, c + 2, 3, 0]
        if operation == "matrixmarket-cell-qc":
            for j, total in enumerate(totals):
                expected[f"c{j}"] = {
                    "counts": total,
                    "features": 2 if j < 3 else 0,
                    "mt_fraction": [a / (a + b), 0, 1 / 3, None][j],
                }
            columns = {
                "counts": Column(kind="integer", unit="UMIs", description="Gene Expression count sum"),
                "features": Column(
                    kind="integer", unit="genes", description="Gene Expression features with nonzero count"
                ),
                "mt_fraction": Column(
                    kind="number",
                    unit="fraction",
                    description="MT- symbol counts / total; null for zero total",
                    nullable=True,
                    atol=1e-10,
                    rtol=1e-8,
                ),
            }
            prompt = (
                "Compute per-barcode QC from the 10x-style Matrix Market bundle. Matrix rows index "
                "features.tsv and columns index barcodes.tsv (both 1-based in the matrix). Use only Gene "
                "Expression features; mitochondrial symbols start MT-. Keep zero-count cells, with null "
                "mitochondrial fraction. Distinct feature IDs with the same symbol remain distinct. Use "
                "barcode id."
            )
            wrong = [{"id": k, **v, "counts": v["counts"] + 9} for k, v in expected.items()]
            reason = "included_antibody_capture_counts"
        elif operation == "matrixmarket-log-normalization":
            for i, gene in [(0, "g0"), (1, "g1"), (2, "g2"), (4, "g3")]:
                for j, total in enumerate(totals):
                    expected[f"{gene}:c{j}"] = {"log_count": math.log1p(dense[i][j] * 10000 / total) if total else 0.0}
            columns = {
                "log_count": Column(
                    kind="number",
                    unit="natural log normalized counts",
                    description="log1p(count / cell Gene Expression total * 10000)",
                    atol=1e-8,
                    rtol=1e-8,
                )
            }
            prompt = (
                "Library-normalize Gene Expression entries of the 10x-style Matrix Market bundle to 10,000 "
                "counts per barcode, then apply natural log1p. Exclude Antibody Capture features from both "
                "output and denominators. Return every Gene Expression feature-ID:barcode pair including "
                "implicit zeros; zero-library cells stay all zero. Do not combine duplicate symbols."
            )
            wrong = [{"id": k, "log_count": math.expm1(v["log_count"])} for k, v in expected.items()]
            reason = "omitted_log1p"
        else:
            for name, row_index in [("g0", 0), ("g1", 1), ("g2", 2), ("g3", 4)]:
                detected = sum(value >= 2 for value in dense[row_index])
                expected[name] = {"cells_ge_two": detected, "keep": int(detected >= 2)}
            columns = {
                "cells_ge_two": Column(kind="integer", unit="barcodes", description="cells with count >= 2"),
                "keep": Column(kind="integer", unit="decision", description="1 when count >=2 in at least two cells"),
            }
            prompt = (
                "Filter features in the 10x-style Matrix Market bundle: a Gene Expression feature passes "
                "when its count is at least 2 in at least two barcodes. Report the qualifying-cell count "
                "and keep=0/1 for every Gene Expression feature ID, including all-zero rows. Ignore "
                "antibody features and keep duplicate symbols separate."
            )
            wrong = [{"id": k, **v, "keep": 1} for k, v in expected.items()]
            reason = "retained_all_zero_features"
    elif operation == "splice-psi":
        rows = []
        for i in range(4):
            inclusion, skipping = (0, 0) if i == 3 else (rng.randint(4, 10), rng.randint(2, 8))
            imbalance = 0 if i == 3 else rng.randint(1, 3)
            rows.append(
                {
                    "event": f"e{i}",
                    "junction_upstream": inclusion + imbalance,
                    "junction_downstream": inclusion - imbalance,
                    "junction_skip": skipping,
                }
            )
            expected[f"e{i}"] = {
                "psi": inclusion / (inclusion + skipping) if inclusion + skipping else None,
                "effective_support": float(inclusion + skipping),
            }
        inputs = {"junctions.csv": csv_text(rows)}
        columns = {
            "psi": Column(
                kind="number",
                unit="fraction",
                description="inclusion / (inclusion + skipping), null if unobserved",
                nullable=True,
                atol=1e-10,
                rtol=1e-8,
            ),
            "effective_support": Column(
                kind="number",
                unit="junction support",
                description="mean inclusion-junction count plus skipping count",
                atol=1e-10,
                rtol=1e-8,
            ),
        }
        prompt = (
            "Compute cassette-exon PSI from junctions.csv. Define inclusion support as the arithmetic "
            "mean of the upstream and downstream inclusion junction counts, skipping support as the "
            "skip junction count, and PSI=I/(I+S). Return null PSI for zero total support. Report "
            "effective support I+S and use event id."
        )
        wrong = [
            {"id": k, **v, "psi": 2 * v["psi"] / (1 + v["psi"]) if v["psi"] is not None else None}
            for k, v in expected.items()
        ]
        reason = "double_counted_inclusion_junctions"
    elif operation == "bulk-size-factors":
        ratio = rng.choice([2, 3, 4])
        factors = rng.sample([1, ratio, ratio**2], 3)
        base = [rng.randint(2, 8) for _ in range(5)]
        rows = [
            {"gene": f"g{i}", **{f"s{j}": count * scale for j, scale in enumerate(factors)}}
            for i, count in enumerate(base)
        ]
        # Five central genes determine the median despite two asymmetric outliers.
        outliers = [
            {"gene": "outlier_a", **{f"s{j}": 4 * scale * (64 if j == 0 else 1) for j, scale in enumerate(factors)}},
            {"gene": "outlier_b", **{f"s{j}": 6 * scale * (8 if j == 1 else 1) for j, scale in enumerate(factors)}},
        ]
        rows = [*outliers, *rows]
        rows.append({"gene": "partial", "s0": 0, "s1": 99, "s2": 99})
        inputs = {"counts.csv": csv_text(rows)}
        expected = {
            f"s{i}": {"size_factor": scale / ratio, "eligible_genes": len(base) + 2} for i, scale in enumerate(factors)
        }
        columns = {
            "size_factor": Column(
                kind="number",
                unit="relative scale",
                description="median count / across-sample gene geometric mean",
                atol=1e-8,
                rtol=1e-8,
            ),
            "eligible_genes": Column(kind="integer", unit="genes", description="genes positive in every sample"),
        }
        prompt = (
            "Estimate bulk RNA-seq median-of-ratios size factors from counts.csv. Restrict to genes "
            "with strictly positive counts in every sample; compute each eligible gene geometric mean "
            "across samples, then each sample median of count/geometric-mean ratios. No extra "
            "rescaling. Use sample column name as id."
        )
        wrong = [{"id": k, **v, "eligible_genes": v["eligible_genes"] + 1} for k, v in expected.items()]
        reason = "included_zero_geometric_mean_gene"
        outlier_ratios = [(16, 0.5), (0.25, 4), (0.25, 0.5)]
        extra_mutations["used_mean_instead_of_median"] = [
            {"id": f"s{i}", "eligible_genes": 7, "size_factor": scale / ratio * (5 + sum(outlier_ratios[i])) / 7}
            for i, scale in enumerate(factors)
        ]
        extra_mutations["used_first_gene_instead_of_median"] = [
            {"id": f"s{i}", "eligible_genes": 7, "size_factor": scale / ratio * outlier_ratios[i][0]}
            for i, scale in enumerate(factors)
        ]
    elif operation == "bulk-cpm-filter":
        n = rng.randint(4, 8)
        positive_a = set(rng.sample(range(3), rng.choice([2, 3])))
        positive_b = set(rng.sample(range(3), rng.randrange(4)))
        counts = {
            "g0": [n * int(i in positive_a) for i in range(3)],
            "g2": [0, 0, n],
            "g3": [n * int(i in positive_b) for i in range(3)],
        }
        counts["g1"] = [10 * n - sum(values[i] for values in counts.values()) for i in range(3)]
        rows = [{"gene": name, **{f"s{i}": value for i, value in enumerate(values)}} for name, values in counts.items()]
        inputs = {"counts.csv": csv_text(rows), "threshold.txt": "100000\n"}
        expected = {
            name: {"n_samples": number, "keep": int(number >= 2)}
            for name, number in [("g0", len(positive_a)), ("g1", 3), ("g2", 1), ("g3", len(positive_b))]
        }
        columns = {
            "n_samples": Column(kind="integer", unit="samples", description="samples with CPM >= threshold"),
            "keep": Column(kind="integer", unit="decision", description="1 if at least two samples pass"),
        }
        prompt = (
            "Filter bulk genes by CPM >= threshold.txt in at least two samples. Library sizes are sums "
            "over all original rows of counts.csv, before filtering. Use inclusive threshold comparison"
            " and 1,000,000 scaling. Report every gene, count of passing samples, and keep=0/1."
        )
        wrong = [{"id": k, **v, "keep": 0 if k != "g1" else 1} for k, v in expected.items()]
        reason = "exclusive_cpm_boundary"
    else:
        assert operation == "differential-expression-bh"
        scale = rng.choice([0.5, 1.0, 1.5])
        pvalues = [0.001, 0.01, 0.03, 0.03, 0.4, None]
        adjusted = [0.005, 0.025, 0.0375, 0.0375, 0.4, None]
        effects = [2, -2, 0.5, 1, 3, 4]
        rows = []
        for i, (p, q, effect) in enumerate(zip(pvalues, adjusted, effects, strict=True)):
            rows.append({"gene": f"g{i}", "pvalue": "" if p is None else p * scale, "log2fc": effect})
            expected[f"g{i}"] = {
                "padj": None if q is None else q * scale,
                "upregulated": int(q is not None and q * scale <= 0.05 and effect >= 1),
            }
        rng.shuffle(rows)
        inputs = {"results.csv": csv_text(rows)}
        columns = {
            "padj": Column(
                kind="number",
                unit="probability",
                description="BH-adjusted p-value, null if untested",
                nullable=True,
                atol=1e-10,
                rtol=1e-8,
            ),
            "upregulated": Column(kind="integer", unit="decision", description="padj <=0.05 and log2fc >=1"),
        }
        prompt = (
            "Apply Benjamini-Hochberg correction to all nonmissing p-values in results.csv. Missing "
            "p-values remain null and are excluded from the number of tests. Mark upregulated only when"
            " adjusted p <=0.05 and signed log2fc >=1. Include every gene; handle tied p-values "
            "consistently."
        )
        wrong = [{"id": k, **v, "upregulated": 1 if k == "g1" else v["upregulated"]} for k, v in expected.items()]
        reason = "used_absolute_effect_for_upregulation"
    return Instance(
        prompt + " Inputs are in /app/inputs.",
        inputs,
        Contract(columns=columns, expected=expected),
        {reason: wrong, **extra_mutations},
    )


SKILLS = {
    "matrixmarket-cell-qc": ("sparse-matrix", "feature-types", "mitochondrial-fraction"),
    "matrixmarket-log-normalization": ("library-normalization", "implicit-zeros", "feature-identifiers"),
    "matrixmarket-feature-filtering": ("detection-threshold", "matrix-orientation", "duplicate-symbols"),
    "splice-psi": ("splice-junctions", "isoform-proportion", "zero-support"),
    "bulk-size-factors": ("geometric-means", "median-ratios", "zero-counts"),
    "bulk-cpm-filter": ("library-size", "cpm", "inclusive-threshold"),
    "differential-expression-bh": ("multiple-testing", "effect-direction", "missing-pvalues"),
}
RECIPES = tuple(
    Recipe(
        name,
        "2" if name in ["bulk-cpm-filter", "bulk-size-factors"] else "1",
        skills,
        (
            ("matrix-market-coordinate-integer", "10x-features-tsv", "10x-barcodes-tsv")
            if name.startswith("matrixmarket")
            else ("csv-header",)
        ),
        (
            ("https://math.nist.gov/MatrixMarket/formats.html",)
            if name.startswith("matrixmarket")
            else ("https://bioconductor.org/packages/release/bioc/html/DESeq2.html",)
        ),
        partial(generate_expression, operation=name),
    )
    for name, skills in SKILLS.items()
)
