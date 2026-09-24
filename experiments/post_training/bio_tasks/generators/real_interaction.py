# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Distinguish within-population responses from population-by-stage interactions."""

import csv
import io
import json
import math

from experiments.post_training.bio_tasks.contract import Contract, TableContract
from experiments.post_training.bio_tasks.generators.real_expression import integer, observations
from experiments.post_training.bio_tasks.generators.real_rnaseq import log_probability, quantity
from experiments.post_training.bio_tasks.real_data import mammary_samples, source_text, tsv_text
from experiments.post_training.bio_tasks.recipe_types import DataOrigin, Instance, OracleRuntime, Recipe, WorkflowScope

NAME = "real-rnaseq-population-interaction"
DESIGN_COLUMNS = ("intercept", "luminal", "pregnant", "lactating", "luminal_pregnant", "luminal_lactating")
EFFECT_COLUMNS = {
    "base_mean": "baseMean",
    "log2_fold_change": "log2FoldChange",
    "standard_error": "lfcSE",
    "wald_statistic": "stat",
    "neg_log10_p": "pvalue",
    "neg_log10_padj": "padj",
}


def generate_interaction(_seed: int) -> Instance:
    query = {"analysis": NAME, "minimum_total_count": 10, "maximum_fdr": 0.05, "minimum_effect": 1.0}
    metadata = sorted(mammary_samples(), key=lambda row: row["sample"])
    columns, genes, counts = observations()
    selected = [row["sample"] for row in metadata]
    matrix = counts[:, [columns.index(name) for name in selected]]
    retained = [
        gene for gene, total in zip(genes, matrix.sum(axis=1), strict=True) if total >= query["minimum_total_count"]
    ]
    factors = {
        row["id"]: float(row["size_factor"])
        for row in csv.DictReader(
            io.StringIO(source_text("GSE60450", "gse60450-population-interaction-size-factors.tsv.gz")), delimiter="\t"
        )
    }
    qc = {
        sample: {
            "library_counts": int(matrix[:, i].sum()),
            "detected_genes": int((matrix[:, i] > 0).sum()),
            "size_factor": factors[sample],
        }
        for i, sample in enumerate(selected)
    }
    design = {}
    for row in metadata:
        luminal, pregnant, lactating = (
            int(row["population"] == "luminal"),
            int(row["stage"] == "18.5 dP"),
            int(row["stage"] == "2 dL"),
        )
        design[row["sample"]] = dict(
            zip(DESIGN_COLUMNS, (1, luminal, pregnant, lactating, luminal * pregnant, luminal * lactating), strict=True)
        )
    weight_rows = {"basal": (0, 0, 1, 0, 0, 0), "luminal": (0, 0, 1, 0, 1, 0), "interaction": (0, 0, 0, 0, 1, 0)}
    weights = {key: dict(zip(DESIGN_COLUMNS, row, strict=True)) for key, row in weight_rows.items()}
    design_columns = {
        name: integer("coefficient in the declared model or contrast", "coefficient") for name in DESIGN_COLUMNS
    }
    tables = {
        "sample_qc.tsv": TableContract(
            columns={
                "library_counts": integer("all observed counts before gene filtering", "counts"),
                "detected_genes": integer("positive-count genes before filtering", "genes"),
                "size_factor": quantity("DESeq2 ratio size factor on all filtered libraries", "factor"),
            },
            expected=qc,
            max_bytes=16 * 1024,
        ),
        "design.tsv": TableContract(columns=design_columns, expected=design, max_bytes=16 * 1024),
        "contrast_weights.tsv": TableContract(columns=design_columns, expected=weights, max_bytes=16 * 1024),
    }
    fitted = {}
    for contrast in weights:
        filename = f"gse60450-population-interaction-{contrast}-results.tsv.gz"
        fitted[contrast] = {
            row["id"]: {name: None if row[name] == "NA" else float(row[name]) for name in EFFECT_COLUMNS.values()}
            for row in csv.DictReader(io.StringIO(source_text("GSE60450", filename)), delimiter="\t")
        }
        assert set(fitted[contrast]) == set(retained)
        tables[f"{contrast}_results.tsv"] = TableContract(
            columns={
                name: quantity(
                    f"{contrast} {raw_name}; unshrunk effect or capped -log10 probability",
                    "log probability" if name.startswith("neg_log10") else "model quantity",
                    nullable=True,
                )
                for name, raw_name in EFFECT_COLUMNS.items()
            },
            expected={
                gene: {
                    name: log_probability(row[raw_name]) if name.startswith("neg_log10") else row[raw_name]
                    for name, raw_name in EFFECT_COLUMNS.items()
                }
                for gene, row in fitted[contrast].items()
            },
            max_bytes=16 * 1024 * 1024,
        )
    calls = {}
    for gene in retained:
        directions = {}
        all_testable = True
        for contrast in weights:
            row = fitted[contrast][gene]
            valid = all(
                row[name] is not None and math.isfinite(row[name]) for name in ("pvalue", "padj", "log2FoldChange")
            )
            all_testable &= valid
            effect = row["log2FoldChange"]
            detected = valid and row["padj"] <= query["maximum_fdr"] and abs(effect) >= query["minimum_effect"]
            directions[contrast] = (1 if effect > 0 else -1) if detected else 0
        one_only = (directions["basal"] != 0) != (directions["luminal"] != 0)
        calls[gene] = {
            **{name + "_direction": value for name, value in directions.items()},
            "all_testable": int(all_testable),
            "one_population_only": int(one_only),
            "one_population_only_without_interaction": int(one_only and directions["interaction"] == 0),
            "both_same_direction": int(directions["basal"] != 0 and directions["basal"] == directions["luminal"]),
        }
    tables["gene_calls.tsv"] = TableContract(
        columns={
            name: integer(
                (
                    "-1/0/+1 thresholded direction"
                    if name in {"basal_direction", "luminal_direction", "interaction_direction"}
                    else "0/1 indicator"
                ),
                "call",
            )
            for name in next(iter(calls.values()))
        },
        expected=calls,
        max_bytes=4 * 1024 * 1024,
    )
    summary = {
        "genes_modeled": len(retained),
        "genes_testable": sum(row["all_testable"] for row in calls.values()),
        "interaction_genes": sum(row["interaction_direction"] != 0 for row in calls.values()),
        **{
            name: sum(row[name] for row in calls.values())
            for name in ("one_population_only", "one_population_only_without_interaction", "both_same_direction")
        },
    }
    return Instance(
        "Does the pregnancy-associated transcriptional response differ between basal and luminal mammary "
        "populations in GSE60450? Use all twelve observed libraries, two biological replicates per "
        "population and developmental stage. The supplied metadata does not establish donor pairing; "
        "do not infer it from replicate labels. Join counts.tsv to samples.tsv by sample ID, "
        "ignore the Length annotation, and preserve the counts. Inputs are in /app/inputs. "
        "Use DESeq2 1.50.2 with design ~population*stage, basal as the reference population, and "
        "stage levels virgin, '18.5 dP' (pregnant), '2 dL' (lactating). Filter total count >=10 across "
        "all twelve libraries. Use ratio size factors, parametric dispersion, unshrunk Wald effects, "
        "betaPrior=FALSE, no count replacement, no Cook's or independent filtering, and BH within "
        "each contrast over retained genes. Keep lactating libraries in the fit. Compare pregnant "
        "versus virgin in each population and test (luminal pregnancy response)-(basal pregnancy response). "
        "Write full sample QC, the design matrix and all three contrast vectors. Matrix columns are "
        "intercept, luminal, pregnant, lactating, luminal_pregnant, luminal_lactating, in that order. "
        "Write basal_results.tsv, luminal_results.tsv and interaction_results.tsv for every retained "
        "Entrez gene, including NA results. Report probabilities as -log10(max(p,1e-300)), preserving NA. "
        "QC counts and detected genes use the unfiltered matrix; size factors use the filtered model. "
        "For each contrast call direction -1/+1 only when BH<=maximum_fdr and absolute unshrunk "
        "effect>=minimum_effect in query.json; otherwise call 0, including missing results. Write "
        "gene_calls.tsv with the three directions, all_testable (finite raw p, BH and effect for all "
        "contrasts), one_population_only (exactly one within-population direction nonzero), "
        "one_population_only_without_interaction (that flag and interaction direction zero), and "
        "both_same_direction (both within-population directions nonzero and equal). Summarize counts "
        "as id=comparison. One-population significance does not establish different responses; a "
        "non-significant interaction or matching directions does not establish equivalence. These "
        "post-test effect cutoffs are not formal tests against a nonzero effect threshold. Stage "
        "associations in this small observational study do not establish causal or drug-safety effects.",
        {
            "counts.tsv": source_text("GSE60450", "gse60450-counts.txt.gz"),
            "samples.tsv": tsv_text(metadata),
            "query.json": json.dumps(query, indent=2) + "\n",
        },
        Contract(
            columns={name: integer("number of genes satisfying the declared rule", "genes") for name in summary},
            expected={"comparison": summary},
            tables=tables,
        ),
        {
            "one_population_only_is_not_interaction": [
                {"id": "comparison", **summary, "interaction_genes": summary["one_population_only"]}
            ]
        },
        data_origin=DataOrigin.REAL,
        source_ids=("GSE60450",),
        workflow_scope=WorkflowScope.CONNECTED,
        derivation=(
            "Unchanged complete observed count matrix and deposited sample metadata. Native reference uses "
            "design-row contrasts; the fresh input-reading oracle constructs coefficient-name contrasts "
            "with the same pinned DESeq2 engine."
        ),
    )


RECIPES = (
    Recipe(
        NAME,
        "1",
        ("factorial count models", "interaction contrasts", "multiple testing", "response interpretation"),
        ("TSV count matrix", "TSV sample metadata", "design matrix", "contrast vectors"),
        ("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE60450", "https://bioconductor.org/packages/DESeq2"),
        generate_interaction,
        oracle_timeout=1800,
        oracle_runtime=OracleRuntime.NATIVE,
    ),
)
