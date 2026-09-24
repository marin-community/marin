# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Connected count-to-contrast and count-to-enrichment workflows on GSE60450."""

import csv
import io
import json
import math
import random
from functools import partial

import numpy as np
from scipy.stats import false_discovery_control, hypergeom

from experiments.post_training.bio_tasks.contract import Column, Contract, TableContract
from experiments.post_training.bio_tasks.generators.real_expression import integer, observations
from experiments.post_training.bio_tasks.real_data import mammary_samples, source_text, tsv_text
from experiments.post_training.bio_tasks.recipe_types import DataOrigin, Instance, OracleRuntime, Recipe, WorkflowScope

STAGES = ("virgin", "18.5 dP", "2 dL")
NAMES = ("real-rnaseq-differential-expression", "real-rnaseq-go-enrichment")
SOURCE_URL = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE60450"


def quantity(description: str, unit: str, nullable: bool = False) -> Column:
    return Column(kind="number", description=description, unit=unit, atol=1e-5, rtol=1e-6, nullable=nullable)


def log_probability(value: float | None) -> float | None:
    return None if value is None else -math.log10(max(value, 1e-300))


def generate_rnaseq(seed: int, operation: str) -> Instance:
    rng = random.Random(seed)
    population = rng.choice(("basal", "luminal"))
    baseline, treatment = rng.sample(STAGES, 2)
    query = {
        "analysis": operation,
        "population": population,
        "baseline": baseline,
        "treatment": treatment,
        "maximum_fdr": 0.05,
        "minimum_effect": 1.0,
    }
    metadata = mammary_samples()
    rng.shuffle(metadata)
    inputs = {
        "counts.tsv": source_text("GSE60450", "gse60450-counts.txt.gz"),
        "samples.tsv": tsv_text(metadata),
    }
    columns, genes, counts = observations()
    selected = sorted(row["sample"] for row in metadata if row["population"] == population)
    matrix = counts[:, [columns.index(name) for name in selected]]
    keep = matrix.sum(axis=1) >= 10
    fitted = {}
    filename = f"gse60450-{population}-{STAGES.index(baseline)+1}-{STAGES.index(treatment)+1}-results.tsv.gz"
    for row in csv.DictReader(io.StringIO(source_text("GSE60450", filename)), delimiter="\t"):
        fitted[row.pop("id")] = {name: None if value == "NA" else float(value) for name, value in row.items()}
    assert set(fitted) == {gene for gene, retained in zip(genes, keep, strict=True) if retained}
    factors = {
        row["sample"]: float(row["size_factor"])
        for row in csv.DictReader(
            io.StringIO(source_text("GSE60450", f"gse60450-{population}-size-factors.tsv.gz")), delimiter="\t"
        )
    }
    qc = {
        name: {
            "library_counts": int(matrix[:, i].sum()),
            "detected_genes": int((matrix[:, i] > 0).sum()),
            "size_factor": factors[name],
        }
        for i, name in enumerate(selected)
    }
    results = {
        gene: {
            "base_mean": row["baseMean"],
            "log2_fold_change": row["log2FoldChange"],
            "standard_error": row["lfcSE"],
            "wald_statistic": row["stat"],
            "neg_log10_p": log_probability(row["pvalue"]),
            "neg_log10_padj": log_probability(row["padj"]),
        }
        for gene, row in fitted.items()
    }
    tables = {
        "sample_qc.tsv": TableContract(
            columns={
                "library_counts": integer("all observed counts before gene filtering", "counts"),
                "detected_genes": integer("positive-count genes before filtering", "genes"),
                "size_factor": quantity("DESeq2 ratio size factor on the filtered model matrix", "factor"),
            },
            expected=qc,
            max_bytes=16 * 1024,
        ),
        "de_results.tsv": TableContract(
            columns={
                "base_mean": quantity("mean of normalized counts in all six modeled samples", "counts", True),
                "log2_fold_change": quantity("unshrunk treatment relative to baseline effect", "log2", True),
                "standard_error": quantity("unshrunk effect standard error", "log2", True),
                "wald_statistic": quantity("Wald coefficient divided by standard error", "statistic", True),
                "neg_log10_p": quantity("negative log10 raw probability, capped at 300", "log probability", True),
                "neg_log10_padj": quantity("negative log10 BH probability, capped at 300", "log probability", True),
            },
            expected=results,
            max_bytes=16 * 1024 * 1024,
        ),
    }
    finite = {gene for gene, row in fitted.items() if row["pvalue"] is not None and row["log2FoldChange"] is not None}
    up = {
        gene
        for gene in finite
        if fitted[gene]["padj"] is not None
        and fitted[gene]["padj"] <= query["maximum_fdr"]
        and fitted[gene]["log2FoldChange"] >= query["minimum_effect"]
    }
    down = {
        gene
        for gene in finite
        if fitted[gene]["padj"] is not None
        and fitted[gene]["padj"] <= query["maximum_fdr"]
        and fitted[gene]["log2FoldChange"] <= -query["minimum_effect"]
    }
    prompt = (
        "Analyze the observed GSE60450 mouse mammary-gland RNA-seq experiment. Join samples.tsv to "
        "counts.tsv by sample ID. The matrix has 27,179 Entrez genes and twelve libraries; Length is an "
        "annotation column. Select query.json's cell population and retain its six libraries, two biological "
        "replicates at each of virgin, 18.5 days pregnant and 2 days lactating. Preserve the counts. "
        "Filter genes to total count >=10 across all six selected libraries. Fit DESeq2 1.50.2 design ~stage "
        "with stage levels virgin, '18.5 dP', '2 dL', ratio size factors, parametric dispersion, Wald tests, "
        "betaPrior=FALSE and no count replacement. Keep all three stages in the fit; extract treatment "
        "relative to baseline from query.json. Disable Cook's filtering and independent filtering; use "
        "BH adjustment over all retained genes, with no effect shrinkage. Write sample_qc.tsv for every "
        "selected library and de_results.tsv for every retained gene, including missing test results. "
        "QC totals/detected genes use the full unfiltered matrix; size factors use the filtered model. "
        "Probability fields are -log10(max(p,1e-300)), preserving NA. Use the specified inclusive FDR "
        "and absolute log2-effect cutoffs to identify up/downregulated genes. All inputs are in /app/inputs. "
    )
    if operation == NAMES[0]:
        expected = {
            "contrast": {
                "genes_tested": len(finite),
                "upregulated": len(up),
                "downregulated": len(down),
                "strongest_neg_log10_padj": max(
                    row["neg_log10_padj"] for row in results.values() if row["neg_log10_padj"] is not None
                ),
            }
        }
        answer_columns = {
            "genes_tested": integer("genes with finite raw test probability and effect", "genes"),
            "upregulated": integer("genes meeting both cutoffs with positive effect", "genes"),
            "downregulated": integer("genes meeting both cutoffs with negative effect", "genes"),
            "strongest_neg_log10_padj": quantity(
                "largest negative log10 BH probability, capped at 300", "log probability"
            ),
        }
        prompt += (
            "Summarize the tested gene count, both directional discovery counts "
            "and strongest BH evidence as id=contrast. "
        )
        wrong = {"contrast": {**expected["contrast"], "genes_tested": len(genes)}}
        mutation = "used_unfiltered_gene_count"
        sources = ("GSE60450",)
    elif operation == NAMES[1]:
        query.update(direction=rng.choice(("up", "down")), report_terms=10)
        inputs["go_membership.tsv"] = source_text("GO:mouse-3.22.0", "mouse-go-bp-3.22.0.tsv.gz")
        inputs["go_terms.tsv"] = source_text("GO:mouse-3.22.0", "go-terms-3.22.0.tsv.gz")
        memberships = {}
        for row in csv.DictReader(io.StringIO(inputs["go_membership.tsv"]), delimiter="\t"):
            memberships.setdefault(row["term"], set()).add(row["gene"])
        annotated = set().union(*memberships.values())
        universe = finite & annotated
        selected_genes = (up if query["direction"] == "up" else down) & universe
        memberships = {term: members & universe for term, members in memberships.items()}
        memberships = {term: members for term, members in memberships.items() if 10 <= len(members) <= 500}
        identifiers = sorted(memberships)
        sizes = np.array([len(memberships[term]) for term in identifiers])
        overlap = np.array([len(memberships[term] & selected_genes) for term in identifiers])
        pvalues = hypergeom.sf(overlap - 1, len(universe), sizes, len(selected_genes))
        adjusted = false_discovery_control(pvalues, method="bh")
        enrichment = {
            term: {
                "overlap": int(overlap[i]),
                "term_size": int(sizes[i]),
                "selected_genes": len(selected_genes),
                "background_genes": len(universe),
                "neg_log10_p": log_probability(float(pvalues[i])),
                "neg_log10_padj": log_probability(float(adjusted[i])),
            }
            for i, term in enumerate(identifiers)
        }
        answer_columns = {
            "overlap": integer("selected genes in this term", "genes"),
            "term_size": integer("term genes in declared universe", "genes"),
            "selected_genes": integer("selected DE genes in declared universe", "genes"),
            "background_genes": integer("finite tested genes with BP annotation", "genes"),
            "neg_log10_p": quantity("upper-tail hypergeometric probability as capped -log10", "log probability"),
            "neg_log10_padj": quantity("BH probability across all eligible terms as capped -log10", "log probability"),
        }
        tables["enrichment.tsv"] = TableContract(columns=answer_columns, expected=enrichment, max_bytes=4 * 1024 * 1024)
        order = sorted(range(len(identifiers)), key=lambda i: (pvalues[i], identifiers[i]))[: query["report_terms"]]
        expected = {identifiers[i]: enrichment[identifiers[i]] for i in order}
        wrong = {key: {**row, "background_genes": len(genes)} for key, row in expected.items()}
        mutation = "whole_matrix_instead_of_tested_annotated_universe"
        prompt += (
            "Continue the fitted contrast into GO biological-process over-representation analysis using "
            "the frozen propagated memberships in go_membership.tsv; go_terms.tsv gives term names. "
            "The universe is finite-tested genes with any supplied BP annotation. Select the significant "
            "direction in query.json and intersect it with that universe. Restrict each term to the universe; "
            "test every term with 10..500 universe genes, including zero-overlap terms. Use the upper-tail "
            "hypergeometric probability P(X>=observed overlap), then BH across all eligible terms. Write "
            "enrichment.tsv for every tested term. Return the report_terms smallest raw probabilities, "
            "breaking ties lexicographically by GO ID, with the GO ID as id. Interpret this as association "
            "in a selected gene set, without claiming pathway activation or a causal effect. "
        )
        sources = ("GSE60450", "GO:mouse-3.22.0")
    else:
        raise ValueError(operation)
    inputs["query.json"] = json.dumps(query) + "\n"
    return Instance(
        prompt,
        inputs,
        Contract(columns=answer_columns, expected=expected, tables=tables),
        {mutation: [{"id": key, **row} for key, row in wrong.items()]},
        data_origin=DataOrigin.REAL,
        source_ids=sources,
        workflow_scope=WorkflowScope.CONNECTED,
        derivation=(
            "Unchanged deposited counts; biological population and contrast selected explicitly. "
            "Frozen GO membership where supplied. Private fit references use the DESeq wrapper; "
            "the input-reading oracle executes the estimation stages with the same pinned statistical engine."
        ),
    )


RECIPES = tuple(
    Recipe(
        name,
        "1",
        (
            "sample-identity",
            "biological-replication",
            "negative-binomial-model",
            "contrasts",
            "multiple-testing",
            *(() if name == NAMES[0] else ("tested-gene-universe", "enrichment")),
        ),
        ("TSV-count-matrix", "TSV-sample-metadata", "JSON-query"),
        (SOURCE_URL,),
        partial(generate_rnaseq, operation=name),
        oracle_timeout=180,
        oracle_runtime=OracleRuntime.R,
    )
    for name in NAMES
)
