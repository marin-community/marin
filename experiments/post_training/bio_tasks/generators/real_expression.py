# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""RNA-seq tasks on unchanged observations from twelve GSE60450 libraries."""

import csv
import io
import json
import random
from functools import cache, partial

import numpy as np

from experiments.post_training.bio_tasks.contract import Column, Contract
from experiments.post_training.bio_tasks.real_data import mammary_samples, source_text, tsv_text
from experiments.post_training.bio_tasks.recipe_types import DataOrigin, Instance, Recipe

SOURCE_URL = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE60450"
NAMES = (
    "real-rnaseq-library-qc",
    "real-rnaseq-cpm-filter",
    "real-rnaseq-size-factors",
    "real-rnaseq-normalized-contrast",
)


@cache
def observations() -> tuple[tuple[str, ...], tuple[str, ...], np.ndarray]:
    reader = csv.reader(io.StringIO(source_text("GSE60450", "gse60450-counts.txt.gz")), delimiter="\t")
    columns = next(reader)[2:]
    genes, counts = [], []
    for gene, _length, *values in reader:
        genes.append(gene)
        counts.append([int(value) for value in values])
    return tuple(columns), tuple(genes), np.asarray(counts, dtype=np.int64)


def integer(description: str, unit: str) -> Column:
    return Column(kind="integer", description=description, unit=unit)


def number(description: str, unit: str) -> Column:
    return Column(kind="number", description=description, unit=unit, atol=1e-9, rtol=1e-8)


def generate_real_expression(seed: int, operation: str) -> Instance:
    rng = random.Random(seed)
    samples, genes, counts = observations()
    metadata = mammary_samples()
    population = rng.choice(["basal", "luminal", "all"])
    selected = [row["sample"] for row in metadata if population == "all" or row["population"] == population]
    query = {"population": population}
    prompt = (
        "Analyze the real GSE60450 mouse mammary-gland RNA-seq counts in counts.tsv. "
        "These are 27,179 Entrez gene rows and twelve sequenced libraries. Length is a gene-length "
        "annotation, not an expression column. Join samples.tsv to count columns by sample ID; "
        "their row orders differ. Preserve the observed counts. Select the population in query.json "
        "('all' means both). All inputs are in /app/inputs. "
    )
    if operation.endswith("normalized-contrast"):
        population = rng.choice(["basal", "luminal"])
        baseline, treatment = rng.sample(["virgin", "18.5 dP", "2 dL"], 2)
        query = {"population": population, "baseline": baseline, "treatment": treatment}
        selected = [
            r["sample"] for r in metadata if r["population"] == population and r["stage"] in (baseline, treatment)
        ]
    matrix = counts[:, [samples.index(sample) for sample in selected]]
    panel = rng.sample(range(len(genes)), 256)
    inputs = {"counts.tsv": source_text("GSE60450", "gse60450-counts.txt.gz"), "samples.tsv": tsv_text(metadata)}
    expected = {}
    if operation.endswith("library-qc"):
        for j, name in enumerate(selected):
            expected[name] = {"total_counts": int(matrix[:, j].sum()), "detected_genes": int((matrix[:, j] > 0).sum())}
        columns = {
            "total_counts": integer("sum across every gene row", "counts"),
            "detected_genes": integer("genes with a positive count", "genes"),
        }
        prompt += "Report total counts and detected genes for every selected library, using its count-column ID."
        wrong = [{"id": name, **row, "total_counts": row["detected_genes"]} for name, row in expected.items()]
        reason = "confused_library_size_and_detected_genes"
    elif operation.endswith("cpm-filter"):
        query["minimum_cpm"] = rng.choice([1, 2, 5])
        cpm = matrix / matrix.sum(axis=0) * 1_000_000
        hits = (cpm >= query["minimum_cpm"]).sum(axis=1)
        for i in panel:
            expected[genes[i]] = {"n_samples": int(hits[i]), "keep": int(hits[i] >= 2)}
        columns = {
            "n_samples": integer("selected libraries meeting the inclusive CPM cutoff", "libraries"),
            "keep": integer("1 when at least two libraries meet the cutoff", "decision"),
        }
        prompt += (
            "For every Entrez ID in panel.txt, compute CPM using the full gene matrix's column totals, "
            "count selected libraries with CPM >= minimum_cpm, and keep a gene when that count is >=2. "
            "Do not calculate library sizes from the requested panel."
        )
        wrong_cpm = matrix[panel] / matrix[panel].sum(axis=0) * 1_000_000
        wrong = [
            {"id": genes[i], "n_samples": int(n), "keep": int(n >= 2)}
            for i, n in zip(panel, (wrong_cpm >= query["minimum_cpm"]).sum(axis=1), strict=True)
        ]
        reason = "panel_only_library_size"
    else:
        positive = (matrix > 0).all(axis=1)
        log_counts = np.log(matrix[positive])
        log_ratios = log_counts - log_counts.mean(axis=1)[:, None]
        factors = np.exp(np.median(log_ratios, axis=0))
        prompt += (
            "Estimate median-ratio size factors using every gene positive in every selected library: "
            "divide each count by that gene's geometric mean across the selected libraries, then take "
            "the exponential of the median log-ratio for each library (DESeq2 convention). "
            "Do not rescale the resulting factors. "
        )
        if operation.endswith("size-factors"):
            expected = {
                name: {"size_factor": float(factors[j]), "eligible_genes": int(positive.sum())}
                for j, name in enumerate(selected)
            }
            columns = {
                "size_factor": number("exp(median(log(count) - mean(log(count)))) across eligible genes", "ratio"),
                "eligible_genes": integer("genes positive in every selected library", "genes"),
            }
            prompt += "Return one record per selected count-column ID."
            wrong = [
                {"id": name, **row, "size_factor": float(matrix[:, j].sum() / matrix.sum(axis=0).mean())}
                for j, (name, row) in enumerate(expected.items())
            ]
            reason = "total_count_normalization"
        else:
            stage_by_sample = {r["sample"]: r["stage"] for r in metadata}
            a = [j for j, s in enumerate(selected) if stage_by_sample[s] == query["baseline"]]
            b = [j for j, s in enumerate(selected) if stage_by_sample[s] == query["treatment"]]
            normalized = matrix / factors
            effects = np.log2((normalized[:, b].mean(axis=1) + 1) / (normalized[:, a].mean(axis=1) + 1))
            expected = {
                genes[i]: {
                    "log2_ratio": float(effects[i]),
                    "baseline_replicates": len(a),
                    "treatment_replicates": len(b),
                }
                for i in panel
            }
            columns = {
                "log2_ratio": number(
                    "log2((mean normalized treatment count + 1)/(mean normalized baseline count + 1))", "log2 ratio"
                ),
                "baseline_replicates": integer("selected baseline biological samples", "samples"),
                "treatment_replicates": integer("selected treatment biological samples", "samples"),
            }
            prompt += (
                "Within the population, retain only the baseline and treatment stages in query.json before "
                "estimating factors. Divide each count by its library factor, then average biological "
                "replicates within each stage. For every panel.txt Entrez ID, report "
                "log2((treatment mean + 1)/(baseline mean + 1)) and both replicate counts. "
                "This is a descriptive normalized contrast; do not infer a p-value or call significance."
            )
            wrong = [{"id": name, **row, "log2_ratio": -row["log2_ratio"]} for name, row in expected.items()]
            reason = "reversed_contrast"
    query["analysis"] = operation
    inputs["query.json"] = json.dumps(query) + "\n"
    if operation.endswith(("cpm-filter", "normalized-contrast")):
        inputs["panel.txt"] = "\n".join(genes[i] for i in panel) + "\n"
    return Instance(
        prompt,
        inputs,
        Contract(columns=columns, expected=expected),
        {reason: wrong},
        data_origin=DataOrigin.REAL,
        source_ids=("GSE60450",),
        derivation=(
            "All counts unchanged. GEO sample metadata joined through supplementary filenames. "
            "Query selects biological populations/stages; a seed-selected 256-gene reporting panel "
            "does not subset normalization inputs."
        ),
    )


RECIPES = tuple(
    Recipe(
        name,
        "1",
        ("sample-identifiers", "biological-replicates", "raw-counts", "library-normalization"),
        ("gene-count-tsv", "sample-metadata-tsv"),
        (SOURCE_URL,),
        partial(generate_real_expression, operation=name),
    )
    for name in NAMES
)
