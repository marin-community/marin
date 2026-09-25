# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Observed GSE81682 sparse normalization, HVG and PCA audit."""

import json
from pathlib import Path

from experiments.post_training.bio_tasks.contract import Column, Contract, PcaSignContract, TableContract
from experiments.post_training.bio_tasks.h5ad_contract import H5adNormalizationContract
from experiments.post_training.bio_tasks.real_data import source_catalog, source_text
from experiments.post_training.bio_tasks.recipe_types import (
    DataOrigin,
    InputFile,
    Instance,
    OracleRuntime,
    Recipe,
    WorkflowScope,
)

NAME = "real-singlecell-representation-audit"
SOURCE = "GSE81682"
INPUT_ASSET = "gse81682-representation-input.h5ad"
REFERENCE_ASSET = "gse81682-representation-reference.json.gz"
PROTOCOL = Path(__file__).parents[1] / "sources" / "prepare_singlecell_representation.json"
PCS = tuple(f"PC{i}" for i in range(1, 31))


def integer(description: str, unit: str, nullable: bool = False) -> Column:
    return Column(kind="integer", description=description, unit=unit, nullable=nullable)


def number(description: str, unit: str, nullable: bool = False, atol: float = 1e-7, rtol: float = 1e-6) -> Column:
    return Column(kind="number", description=description, unit=unit, nullable=nullable, atol=atol, rtol=rtol)


def text(description: str, unit: str) -> Column:
    return Column(kind="text", description=description, unit=unit)


def generate_representation(_seed: int) -> Instance:
    protocol = json.loads(PROTOCOL.read_text())
    reference = json.loads(source_text(SOURCE, REFERENCE_ASSET))
    source = source_catalog()[SOURCE]
    input_asset = source["file_inputs"][INPUT_ASSET]
    answer = reference["answer"]
    tables = {
        "cell_depth.tsv": TableContract(
            columns={
                "geo_accession": text("original GEO accession", "accession"),
                "sorting_gate": text("original broad sorting gate", "gate"),
                "original_endogenous_reads": integer("endogenous reads before prior gene filtering", "reads"),
                "input_filtered_reads": integer("reads across all provided post-QC genes", "reads"),
                "input_detected_genes": integer("positive genes across all provided post-QC genes", "genes"),
                "eligible_gene_reads": integer("reads among genes detected in at least ten cells", "reads"),
                "normalization_factor": number("eligible_gene_reads divided by 10000", "factor"),
                "log_original_depth": number("log1p of original endogenous reads", "log reads"),
            },
            expected=reference["tables"]["cell_depth.tsv"],
            max_bytes=1024 * 1024,
        ),
        "gene_hvg.tsv": TableContract(
            columns={
                "input_prevalence_cells": integer("positive cells across all provided cells", "cells"),
                "input_read_sum": integer("raw reads across all provided cells", "reads"),
                "eligible": integer("one when input prevalence is at least ten cells", "indicator"),
                "mean": number("native Scanpy Seurat mean on log-normalized expression", "expression", True),
                "dispersion": number("native Scanpy Seurat dispersion", "dispersion", True),
                "normalized_dispersion": number("native binned normalized dispersion", "dispersion", True),
                "rank": integer("normalized dispersion rank with feature-ID tie break", "rank", True),
                "selected": integer("one for the highest-ranked 2000 eligible genes", "indicator"),
            },
            expected=reference["tables"]["gene_hvg.tsv"],
            max_bytes=16 * 1024 * 1024,
        ),
        "pca_scores.tsv": TableContract(
            columns={
                "sorting_gate": text("original broad sorting gate", "gate"),
                "original_endogenous_reads": integer("pre-gene-filter endogenous library", "reads"),
                **{pc: number("centered unscaled PCA cell score", "score", atol=1e-5) for pc in PCS},
            },
            expected=reference["tables"]["pca_scores.tsv"],
            max_bytes=4 * 1024 * 1024,
        ),
        "pca_loadings.tsv": TableContract(
            columns={pc: number("centered unscaled PCA gene loading", "loading", atol=1e-6) for pc in PCS},
            expected=reference["tables"]["pca_loadings.tsv"],
            max_bytes=4 * 1024 * 1024,
        ),
        "pca_variance.tsv": TableContract(
            columns={
                "eigenvalue": number("sample covariance eigenvalue of the selected genes", "variance"),
                "variance_fraction": number("fraction of selected-gene total variance", "fraction"),
            },
            expected=reference["tables"]["pca_variance.tsv"],
            max_bytes=16 * 1024,
        ),
        "diagnostics.tsv": TableContract(
            columns={
                "top_abs_loading": number("largest absolute loading for this PC", "loading"),
                "top_loading_gene": text("feature ID attaining this PC's largest absolute loading", "feature ID"),
                "depth_correlation": number("Pearson correlation of PC score and log1p original depth", "correlation"),
                "abs_depth_correlation": number("absolute depth correlation", "correlation"),
                "n_cells": integer("cells used in this diagnostic", "cells"),
            },
            expected=reference["tables"]["diagnostics.tsv"],
            max_bytes=16 * 1024,
        ),
    }
    summary_columns = {
        "median_input_reads_per_cell": number("median raw input-filtered read depth", "reads", atol=0),
        "pct_positive_counts_equal_one": number("percent of positive input counts equal to one", "percent"),
        "median_hvg_prevalence": number("median positive-cell count among selected genes", "cells", atol=0),
        "pc1_top_abs_load": number(
            "maximum absolute PC1 loading, half-even rounded to three decimals", "loading", atol=1e-9
        ),
        "max_top5_depth_corr": number(
            "maximum absolute PC1-5 correlation with log1p original endogenous depth, rounded to three decimals",
            "correlation",
            atol=1e-9,
        ),
    }
    manifest = {
        "source": SOURCE,
        "source_lineage": source["lineage"],
        "source_citation": source["citation"],
        "input_h5ad_sha256": input_asset["sha256"],
        "qc_reference_sha256": protocol["qc_reference_sha256"],
        "protocol": protocol,
    }
    return Instance(
        "Audit a declared representation of the observed GSE81682 mouse hematopoietic progenitor Smart-seq2 "
        "read counts. Inputs are in /app/inputs. input.h5ad has 1,422 QC-retained cells by 40,312 "
        "endogenous genes in original order; X and layers['counts'] are identical sparse integer reads. "
        "Use the supplied cell IDs, GEO accessions and broad HSPC/LT-HSC/Prog sorting gates unchanged. "
        "The original_endogenous_reads obs field precedes the prior gene filter. Counts are reads, not UMIs. "
        "On all input genes, report each cell's raw depth and detected-gene count, the median input depth, "
        "and the percentage of positive sparse entries equal to one. Count input-gene prevalence across "
        "all 1,422 cells; keep genes detected in at least ten cells, preserving original feature order. "
        "The per-cell eligible-gene read sum is the only normalization denominator. Preserve it separately "
        "from original_endogenous_reads and input_filtered_reads. Write cell_depth.tsv for every cell and "
        "gene_hvg.tsv for every input gene, including ineligible genes with NA score/rank fields. "
        "Copy eligible raw integer reads to normalized.h5ad layers['counts']; apply Scanpy 1.12.4 "
        "normalize_total(target_sum=10000, exclude_highly_expressed=False) once to sparse X, then log1p "
        "once. Preserve cell/feature order, source metadata and sparse support. "
        "On this log-normalized eligible matrix run highly_variable_genes(flavor='seurat', n_top_genes=2000, "
        "n_bins=20, subset=False). Rank finite normalized dispersions descending, breaking score ties by "
        "ascending feature ID, and select exactly 2,000 genes. Center without variance scaling, fit "
        "Scanpy ARPACK PCA with random_state=0 and 31 components on only those genes in original feature "
        "order. Write complete PC1-PC30 cell scores, selected-gene loadings and variance values; PC31 "
        "checks the cutoff but is not an output column. "
        "For each PC1-PC5, report its signed and absolute Pearson correlation with log1p of the supplied "
        "original_endogenous_reads. From that same fit report the maximum absolute PC1 loading and maximum "
        "absolute correlation across PC1-PC5. Round only these two answer.json values to three decimal "
        "places using decimal half-even; keep full precision in tables. Return one answer record with id=study. "
        "Do not regress depth, scale genes, infer donors or mitochondrial fractions, run UMAP/clustering, "
        "or interpret a PC as a causal cell-state effect. manifest.json supplies the frozen parameters.",
        {"manifest.json": json.dumps(manifest, indent=2) + "\n"},
        Contract(
            columns=summary_columns,
            expected={"study": answer},
            tables=tables,
            h5ad={"normalized.h5ad": H5adNormalizationContract.model_validate(reference["h5ad"])},
            pca_signs=PcaSignContract.model_validate(reference["pca_signs"]),
        ),
        {
            "changed_input_depth_summary": [
                {
                    "id": "study",
                    **answer,
                    "median_input_reads_per_cell": float(answer["median_input_reads_per_cell"]) + 1,
                }
            ],
        },
        data_origin=DataOrigin.REAL,
        source_ids=(SOURCE,),
        workflow_scope=WorkflowScope.CONNECTED,
        derivation=(
            "Complete observed, QC-retained GSE81682 read counts; pinned Scanpy sparse normalization, "
            "Seurat-dispersion selection and PCA. Private reference stores full row-keyed artifacts and "
            "count-preserving H5AD checks. Broad gates are descriptive source labels."
        ),
        input_files={"input.h5ad": InputFile(input_asset["sha256"], input_asset["bytes"])},
    )


RECIPES = (
    Recipe(
        id=NAME,
        version="1",
        skills=("sparse count normalization", "highly variable genes", "PCA", "read-depth diagnostics"),
        formats=("H5AD", "TSV", "JSON"),
        sources=("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE81682",),
        generate=generate_representation,
        oracle_timeout=1800,
        oracle_runtime=OracleRuntime.NATIVE,
    ),
)
