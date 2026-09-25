# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Connected CD14 pseudobulk gene-span association on observed Parse PBMCs."""

import json

from experiments.post_training.bio_tasks.contract import Column, Contract, TableContract
from experiments.post_training.bio_tasks.real_data import source_catalog, source_text
from experiments.post_training.bio_tasks.recipe_types import (
    DataOrigin,
    InputFile,
    Instance,
    OracleRuntime,
    Recipe,
    WorkflowScope,
)

SOURCE = "Parse:WT-Mini-v3-PBMC"
REFERENCE = "parse-wt-mini-v3-cd14-reference.json.gz"
RECIPE = "real-cd14-gene-length-association"
QUERY = {
    "minimum_cd14_cells_per_donor": 25,
    "minimum_total_gene_count": 10,
    "minimum_coding_gene_overlap": 15000,
    "minimum_eligible_genes": 5000,
}


def integer(unit: str, description: str) -> Column:
    return Column(kind="integer", unit=unit, description=description)


def number(unit: str, description: str, atol: float = 1e-9, rtol: float = 1e-8) -> Column:
    return Column(kind="number", unit=unit, description=description, atol=atol, rtol=rtol)


def cd14_contract(reference: dict) -> Contract:
    """Grade every classified cell and joined coding gene, not only Pearson r."""
    summary = {
        "input_cells": integer("cells", "all author-filtered observed PBMC cells"),
        "selected_cells": integer("cells", "marker-defined CD14-like cells in all donors"),
        "donors": integer("donors", "independent donor pseudobulks"),
        "joined_coding_genes": integer("genes", "stable-ID matched canonical protein-coding genes"),
        "eligible_genes": integer("genes", "joined coding genes with total CD14 count at least ten"),
        "cpm_prevalence_qc": integer("genes", "eligible genes with CPM at least one in two donors; QC only"),
        "pearson_r": number("correlation", "Pearson r of raw gene span against mean donor CPM", 1e-10, 1e-8),
        "pearson_p": number("two-sided p-value", "SciPy Pearson two-sided p-value", 1e-300, 1e-5),
    }
    cell = {
        "donor": Column(kind="text", unit="donor", description="Parse sample donor"),
        "label": Column(kind="text", unit="marker-defined group", description="frozen marker assignment"),
        "reason": Column(kind="text", unit="assignment reason", description="frozen assignment or exclusion reason"),
        "lineage_score": number("score", "highest log-normalized lineage marker mean", 1e-6, 1e-7),
        "lineage_margin": number("score", "highest minus runner-up lineage score", 1e-6, 1e-7),
    }
    donor = {
        "cd14_cells": integer("cells", "CD14-like cells in this donor"),
        "cd4_cells": integer("cells", "CD4-like comparison cells"),
        "cd8_cells": integer("cells", "CD8-like comparison cells"),
        "b_cells": integer("cells", "B-like comparison cells"),
        "other_ambiguous_cells": integer("cells", "all other or ambiguous cells"),
        "cd14_total_umi": integer("UMIs", "all-gene CD14 donor pseudobulk library size"),
        "cd14_marker_cpm": number("CPM", "CD14 held-out marker CPM in CD14 pseudobulk"),
        "max_other_cd14_marker_cpm": number("CPM", "largest comparison-group CD14 CPM"),
        "held_out_enrichment": number("ratio", "CD14 CPM enrichment with 0.1 pseudocount"),
    }
    gene = {
        "gene_name": Column(kind="text", unit="symbol", description="Parse gene symbol"),
        "span_bp": integer("base pairs", "inclusive canonical release-111 GTF gene span"),
        "total_cd14_count": integer("UMIs", "sum across four donor pseudobulks"),
        **{f"count_Donor_{i}": integer("UMIs", f"observed CD14 donor {i} pseudobulk count") for i in range(1, 5)},
        **{f"cpm_Donor_{i}": number("CPM", f"donor {i} count divided by all-gene library total") for i in range(1, 5)},
        "cpm_prevalence": integer("donors", "donors with CPM at least one; QC only"),
        "mean_cpm": number("CPM", "untransformed arithmetic mean of four donor CPM values"),
        "eligible": integer("indicator", "one when total CD14 count is at least ten"),
    }
    return Contract(
        columns=summary,
        expected={"CD14": reference["summary"]},
        tables={
            "cell_assignments.tsv": TableContract(
                columns=cell, expected=reference["tables"]["cell_assignments.tsv"], max_bytes=2 * 1024 * 1024
            ),
            "donor_qc.tsv": TableContract(
                columns=donor, expected=reference["tables"]["donor_qc.tsv"], max_bytes=64 * 1024
            ),
            "genes.tsv": TableContract(
                columns=gene, expected=reference["tables"]["genes.tsv"], max_bytes=16 * 1024 * 1024
            ),
        },
    )


def generate_cd14_gene_length(_seed: int) -> Instance:
    reference = json.loads(source_text(SOURCE, REFERENCE))
    assets = source_catalog()[SOURCE]["file_inputs"]
    contract = cd14_contract(reference)
    changed_r = [{**row, "pearson_r": row["pearson_r"] + 0.05} for row in contract.answer()]
    return Instance(
        "Analyze the observed Parse Evercode WT Mini v3 human PBMCs from four healthy donors. The dataset is "
        "CC BY 4.0; the cells are author-filtered but have no supplied cell-type label. All inputs are in "
        "/app/inputs. matrix.mtx.gz is a gzip-compressed MatrixMarket integer coordinate matrix with 4,861 "
        "cells as rows and 62,710 genes as columns; genes.csv and cells.csv list matching rows in order. "
        "An explicit zero coordinate is legal and contributes nothing. annotations.tsv.gz is a compact "
        "release-111 Ensembl GTF gene-feature export with gene_id, chromosome, start, end and gene_biotype. "
        "First verify matrix row totals against cells.csv tscp_count. For marker assignment only, compute "
        "log1p(10000 * marker count / cell total) and average the marker values within each panel: T "
        "{CD3D,CD3E,TRAC,CD2}, B {MS4A1,CD79A,CD79B,BANK1}, monocyte "
        "{LYZ,FCN1,S100A8,S100A9,CTSS}, NK exclusion {NKG7,GNLY,KLRD1,FCGR3A}. Assign the highest "
        "lineage only at score >=0.55 and margin >=0.20, with a detected anchor: T one of CD3D/CD3E/TRAC; "
        "B one of MS4A1/CD79A; monocyte LYZ plus one of FCN1/S100A8/S100A9. Otherwise label ambiguous; "
        "highest NK becomes other. Within T cells, compare CD4 {IL7R,CCR7,LTB,MAL} and CD8 "
        "{CD8A,GZMK,CCL5,GZMA}; require subtype score >=0.45, margin >=0.15 and an anchor "
        "(CD4 IL7R/CCR7/MAL; CD8 CD8A/GZMK), else ambiguous_T. Exclude other/ambiguous cells from "
        "the CD14 association. CD14 is held out from classification. Preserve each cell's label and reason "
        "in cell_assignments.tsv and write donor_qc.tsv with group counts, all-gene CD14 library totals "
        "and CD14 held-out CPM enrichment versus the maximum other classified group; use 0.1 CPM "
        "pseudocount for the ratio. Require at least 25 CD14 cells per donor, at most 65% "
        "other/ambiguous cells overall, and CD14 enrichment >=1.5 in at least three donors. "
        "Sum raw UMI counts for CD14-like cells separately within each donor across all 62,710 genes. "
        "Normalize each donor gene count to CPM using its all-gene CD14 pseudobulk total. Join Parse gene "
        "IDs to GTF genes after stripping only Ensembl version suffixes; retain unique protein-coding genes "
        "on chromosomes 1 through 22, X and Y, and compute inclusive gene span=end-start+1 base pairs. "
        "A gene is eligible only when its total observed CD14 count across donors is >=10. For every "
        "joined coding gene, write genes.tsv with span, four donor counts and CPMs, total count, "
        "mean CPM, CPM-prevalence QC and eligibility. The CPM >=1 in at least two donors prevalence is "
        "reported only as QC; it does not filter the coefficient. Compute untransformed arithmetic mean "
        "donor CPM and Pearson r plus two-sided p-value against untransformed gene span over precisely "
        "the eligible genes. Report the summary in answer.json. Marker-defined single-cell pseudobulks "
        "are an adaptation to sorted-cell bulk data, and CPM normalization defines the expression unit; "
        "do not claim equivalence to the original study's numerical answer.",
        {"query.json": json.dumps(QUERY, sort_keys=True) + "\n"},
        contract,
        {"changed_pearson_r": changed_r},
        data_origin=DataOrigin.REAL,
        source_ids=(SOURCE,),
        derivation=(
            "Unchanged observed Parse single-cell UMI coordinates and donor metadata; release-111 GTF "
            "gene features exported without changing coordinates; frozen marker-defined CD14 donor "
            "pseudobulks and exact total-count filter."
        ),
        workflow_scope=WorkflowScope.CONNECTED,
        input_files={
            name: InputFile(assets[name]["sha256"], assets[name]["bytes"])
            for name in ("matrix.mtx.gz", "genes.csv", "cells.csv", "annotations.tsv.gz")
        },
    )


RECIPES = (
    Recipe(
        id=RECIPE,
        version="1",
        skills=(
            "marker-defined cell selection",
            "donor pseudobulk aggregation",
            "Ensembl gene-span join",
            "total-count eligibility",
            "Pearson association",
        ),
        formats=("MatrixMarket gzip", "CSV", "GTF-derived TSV gzip", "TSV", "JSON"),
        sources=(
            "https://www.parsebiosciences.com/datasets/performance-of-evercode-wt-mini-v3-in-human-pbmcs/",
            "https://ftp.ensembl.org/pub/release-111/gtf/homo_sapiens/",
        ),
        generate=generate_cd14_gene_length,
        oracle_timeout=900,
        oracle_runtime=OracleRuntime.NATIVE,
    ),
)
