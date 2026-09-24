# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit effect shrinkage and two enrichment specifications on observed RNA counts."""

import csv
import io
import json

from experiments.post_training.bio_tasks.contract import Contract, TableContract
from experiments.post_training.bio_tasks.generators.real_expression import integer, observations
from experiments.post_training.bio_tasks.generators.real_rnaseq import log_probability, quantity
from experiments.post_training.bio_tasks.real_data import mammary_samples, source_text, tsv_text
from experiments.post_training.bio_tasks.recipe_types import DataOrigin, Instance, OracleRuntime, Recipe, WorkflowScope

NAME = "real-rnaseq-shrinkage-enrichment-audit"
VIEWS = ("adjusted_tested", "raw_genome")
RESULT_COLUMNS = {
    "base_mean": "baseMean",
    "log2_fold_change": "log2FoldChange",
    "standard_error": "lfcSE",
    "wald_statistic": "stat",
    "neg_log10_p": "pvalue",
    "neg_log10_padj": "padj",
    "shrunken_log2_fold_change": "shrunken_log2FC",
    "shrunken_standard_error": "shrunken_lfcSE",
}


def reference_rows(name: str) -> list[dict[str, str]]:
    content = source_text("GSE60450", f"gse60450-luminal-shrinkage-{name}.tsv.gz")
    return list(csv.DictReader(io.StringIO(content), delimiter="\t"))


def generate_enrichment(_seed: int) -> Instance:
    metadata = sorted(mammary_samples(), key=lambda row: row["sample"])
    samples = [row["sample"] for row in metadata if row["population"] == "luminal"]
    names, genes, counts = observations()
    matrix = counts[:, [names.index(sample) for sample in samples]]
    totals = dict(zip(genes, map(int, matrix.sum(axis=1)), strict=True))
    fitted = {
        row["gene"]: {key: None if row[key] == "NA" else float(row[key]) for key in RESULT_COLUMNS.values()}
        for row in reference_rows("gene-results")
    }
    assert set(fitted) == {gene for gene, total in totals.items() if total >= 10}
    factors = {row["sample"]: float(row["size_factor"]) for row in reference_rows("size-factors")}
    raw = {gene for gene, row in fitted.items() if row["pvalue"] is not None and row["pvalue"] < 0.05}
    adjusted = {gene for gene, row in fitted.items() if row["padj"] is not None and row["padj"] < 0.05}
    large = {
        gene for gene, row in fitted.items() if row["shrunken_log2FC"] is not None and abs(row["shrunken_log2FC"]) >= 1
    }
    unshrunk_large = {
        gene for gene, row in fitted.items() if row["log2FoldChange"] is not None and abs(row["log2FoldChange"]) >= 1
    }
    tables = {
        "sample_qc.tsv": TableContract(
            columns={
                "library_counts": integer("counts before gene filtering", "counts"),
                "detected_genes": integer("genes with positive counts before filtering", "genes"),
                "size_factor": quantity("ratio size factor from filtered model", "factor"),
            },
            expected={
                sample: {
                    "library_counts": int(matrix[:, i].sum()),
                    "detected_genes": int((matrix[:, i] > 0).sum()),
                    "size_factor": factors[sample],
                }
                for i, sample in enumerate(samples)
            },
            max_bytes=16384,
        ),
        "gene_results.tsv": TableContract(
            columns={
                key: quantity(
                    "named coefficient " + value,
                    "log probability" if key.startswith("neg_log10") else "model quantity",
                    nullable=True,
                )
                for key, value in RESULT_COLUMNS.items()
            },
            expected={
                gene: {
                    key: log_probability(row[value]) if key.startswith("neg_log10") else row[value]
                    for key, value in RESULT_COLUMNS.items()
                }
                for gene, row in fitted.items()
            },
            max_bytes=16 * 1024 * 1024,
        ),
        "gene_decisions.tsv": TableContract(
            columns={
                "count_sum": integer("total over the six luminal libraries", "counts"),
                **{
                    name: integer("0/1 membership in the declared mask", "indicator")
                    for name in ("fitted", "raw_significant", "adjusted_significant", "large_shrunken_effect")
                },
            },
            expected={
                gene: {
                    "count_sum": total,
                    "fitted": int(gene in fitted),
                    "raw_significant": int(gene in raw),
                    "adjusted_significant": int(gene in adjusted),
                    "large_shrunken_effect": int(gene in large),
                }
                for gene, total in totals.items()
            },
            max_bytes=4 * 1024 * 1024,
        ),
    }
    backgrounds = {view: {row["gene"] for row in reference_rows(view + "-background")} for view in VIEWS}
    selections = {view: {row["gene"] for row in reference_rows(view + "-selected")} for view in VIEWS}
    tables["annotation_membership.tsv"] = TableContract(
        columns={
            view + "_" + kind: integer("0/1 membership after BP annotation intersection", "indicator")
            for view in VIEWS
            for kind in ("background", "selected")
        },
        expected={
            gene: (
                {view + "_background": int(gene in backgrounds[view]) for view in VIEWS}
                | {view + "_selected": int(gene in selections[view]) for view in VIEWS}
            )
            for gene in sorted(backgrounds["raw_genome"])
        },
        max_bytes=4 * 1024 * 1024,
    )
    significant = {}
    for view in VIEWS:
        rows = reference_rows(view + "-ora")
        significant[view] = {row["ID"] for row in rows if float(row["p.adjust"]) < 0.05}
        tables[view + "_enrichment.tsv"] = TableContract(
            columns={
                **{
                    key: integer(description, "genes")
                    for key, description in {
                        "overlap": "selected genes in the term",
                        "term_size": "term genes in the background",
                        "selected_genes": "annotated selected genes",
                        "background_genes": "annotated background genes",
                    }.items()
                },
                "neg_log10_p": quantity("upper-tail hypergeometric probability as capped -log10", "log probability"),
                "neg_log10_padj": quantity("native BH probability as capped -log10", "log probability"),
            },
            expected={
                row["ID"]: {
                    "overlap": int(row["Count"]),
                    "term_size": int(row["BgRatio"].split("/")[0]),
                    "selected_genes": len(selections[view]),
                    "background_genes": len(backgrounds[view]),
                    "neg_log10_p": log_probability(float(row["pvalue"])),
                    "neg_log10_padj": log_probability(float(row["p.adjust"])),
                }
                for row in rows
            },
            max_bytes=4 * 1024 * 1024,
        )
    summary = {
        "genes_fitted": len(fitted),
        "raw_significant": len(raw),
        "adjusted_significant": len(adjusted),
        "large_shrunken_effects": len(large),
        "changed_effect_threshold": len(large ^ unshrunk_large),
        "adjusted_tested_significant_terms": len(significant["adjusted_tested"]),
        "raw_genome_significant_terms": len(significant["raw_genome"]),
        "shared_significant_terms": len(significant["adjusted_tested"] & significant["raw_genome"]),
    }
    return Instance(
        "Audit the lactation-associated expression response in luminal mammary cells from GSE60450. "
        "The two requested enrichment specifications reflect different analysis choices. Determine their "
        "gene/term selections and quantify how coefficient shrinkage changes the absolute-effect threshold. "
        "Inputs are in /app/inputs. Join samples.tsv to counts.tsv by sample ID; Length is annotation. "
        "Use all six luminal libraries, with two biological replicates at each stage. Do not infer donor pairing. "
        "Retain genes with count sum >=10 across those libraries. Use DESeq2 1.50.2, design ~stage, "
        "levels virgin, '18.5 dP', '2 dL', ratio size factors, parametric dispersions, Wald tests, betaPrior=FALSE, "
        "and no count replacement. Keep pregnant libraries in the fit. Extract the named stage_2.dL_vs_virgin "
        "coefficient (not the character contrast interface, which resets genes zero in both compared stages), "
        "using alpha=.05, BH adjustment, independentFiltering=TRUE and cooksCutoff=TRUE. "
        "Apply apeglm 1.32.0 shrinkage to that coefficient with apeMethod=nbinomCR and the same Wald result. "
        "Preserve original p-values, BH values and Wald statistics; do not derive significance from shrunken effects. "
        "Write sample_qc.tsv using unfiltered library totals/detected genes and fitted size factors; "
        "gene_results.tsv for every fitted gene; and gene_decisions.tsv for all 27,179 input genes. "
        "Raw/adjusted significance is finite p/padj <.05; large_shrunken_effect means finite absolute shrunken "
        "log2FC >=1 regardless of significance. Excluded genes receive zero membership indicators. "
        "All probability fields are -log10(max(p,1e-300)); preserve NA model values. "
        "Use the installed org.Mm.eg.db 3.22.0 and GO.db 3.22.0, propagated GOALL biological-process memberships "
        "whose IDs are valid BP terms in GO.db. adjusted_tested selects adjusted-significant genes and uses "
        "finite-raw-p genes as background; raw_genome selects raw-significant genes and uses all genes in the "
        "annotation database with valid BP annotation as background. Intersect each selection/background with "
        "annotated genes before testing. Neither ORA selection includes an effect-size or directional cutoff. "
        "Write annotation_membership.tsv for every annotated gene, including genes absent from the count matrix. "
        "For each specification run clusterProfiler 4.18.4 enrichGO, keyType=ENTREZID, ont=BP, readable=FALSE, "
        "BH, minGSSize=10, maxGSSize=500, pvalueCutoff=qvalueCutoff=1; set seed 42 before each call. "
        "Export its complete @result hypothesis family to the corresponding enrichment table: size-eligible "
        "terms with nonzero selected overlap. Preserve the native BH denominator; do not append zero-overlap "
        "terms or apply Storey-qvalue filtering. Term significance is p.adjust <.05. "
        "Return an answer.json array containing one object with id=audit and all eight summary fields. "
        "changed_effect_threshold "
        "counts fitted genes whose absolute effect >=1 status differs between raw and shrunken coefficients. "
        "The two ORA specifications change selection and background jointly, so their difference cannot isolate "
        "one cause. Enrichment does not establish pathway activation or causal developmental effects.",
        {
            "counts.tsv": source_text("GSE60450", "gse60450-counts.txt.gz"),
            "samples.tsv": tsv_text(metadata),
            "query.json": (
                json.dumps({"analysis": NAME, "population": "luminal", "coefficient": "stage_2.dL_vs_virgin"}) + "\n"
            ),
        },
        Contract(
            columns={key: integer("count under the specified analysis rules", "genes or terms") for key in summary},
            expected={"audit": summary},
            tables=tables,
        ),
        {
            "raw_p_is_not_bh": [{"id": "audit", **summary, "adjusted_significant": len(raw)}],
            "unshrunk_is_not_shrunken": [{"id": "audit", **summary, "large_shrunken_effects": len(unshrunk_large)}],
        },
        data_origin=DataOrigin.REAL,
        source_ids=("GSE60450", "GO:mouse-3.22.0"),
        workflow_scope=WorkflowScope.CONNECTED,
        derivation=(
            "Complete deposited counts and sample identities; pinned annotation databases. Private native references "
            "have separate Python checks of count filters, normalized means, Wald tails, ORA sets and BH arithmetic."
        ),
    )


RECIPES = (
    Recipe(
        NAME,
        "1",
        (
            "replicate-aware count models",
            "effect shrinkage",
            "multiple testing",
            "annotation provenance",
            "enrichment universe sensitivity",
        ),
        ("TSV count matrix", "TSV sample metadata", "Bioconductor annotation databases"),
        (
            "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE60450",
            "https://bioconductor.org/packages/DESeq2",
            "https://bioconductor.org/packages/clusterProfiler",
        ),
        generate_enrichment,
        oracle_timeout=1800,
        oracle_runtime=OracleRuntime.NATIVE,
    ),
)
