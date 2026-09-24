# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Full cell-QC, feature-QC and filtered-count artifact contracts."""

import json

from experiments.post_training.bio_tasks.contract import Column, Contract, MatrixMarketContract, TableContract
from experiments.post_training.bio_tasks.real_data import source_catalog, source_text
from experiments.post_training.bio_tasks.recipe_types import DataOrigin, InputFile, Instance, Recipe, WorkflowScope

SOURCE = "GSE81682"


def generate_singlecell(_seed: int) -> Instance:
    reference = json.loads(source_text(SOURCE, "gse81682-qc-reference.json.gz"))
    contract = singlecell_contract(reference)
    return Instance(
        "Prepare an auditable filtered count matrix for the observed mouse hematopoietic stem/progenitor "
        "Smart-seq2 study GSE81682. All inputs are in /app/inputs. matrix.mtx.gz is a gzip-compressed "
        "Matrix Market coordinate integer general matrix with features as rows and cells as columns. "
        "features.tsv and cells.tsv list those identities in matrix order. Counts are aligned reads. "
        "Use feature_type to distinguish endogenous genes from ERCC spike-ins; the __ columns in cell "
        "metadata are HTSeq alignment summaries and are excluded from gene counts. "
        "Apply the explicit analysis policy in query.json to every supplied cell: require endogenous read "
        "counts >= minimum_endogenous_counts, detected endogenous genes >= minimum_detected_endogenous, "
        "and ERCC/all-count fraction <= ercc_fraction_numerator / ercc_fraction_denominator. "
        "All counts means endogenous plus ERCC counts. Detected means a strictly positive count. "
        "Compute QC on the original matrix before gene filtering. An empty cell has a null ERCC fraction "
        "and fails the positive count requirements. Write cell_qc.tsv for every original cell, preserving "
        "its GEO accession and broad sorting gate, with the stated metrics, keep indicator and output "
        "matrix_column. Excluded cells have matrix_column=0; retained cells receive contiguous 1-based "
        "indices in original order. After cell selection, count retained cells and read counts for every "
        "original feature. Retain endogenous genes observed in at least minimum_retained_cells_per_gene "
        "retained cells, and exclude ERCC rows from the final matrix. Write feature_qc.tsv for all original "
        "features, including ERCC, with feature_type, retained_cell_count, retained_read_count, keep and "
        "matrix_row. Excluded features have matrix_row=0; retained features receive contiguous 1-based "
        "indices in original order. Export filtered_counts.mtx.gz with all retained integer counts, "
        "features as rows and cells as columns, and report the study totals in answer.json. "
        "Use these descriptive QC criteria as specified; they are not the original authors' final cell "
        "selection or universal biological cutoffs. Broad sorting gates do not supply donor identities "
        "or fine cell annotations. Do not infer mitochondrial, donor or cell-type labels.",
        {"query.json": json.dumps(reference["query"], indent=2) + "\n"},
        contract,
        {"retained_every_cell": [{**contract.answer()[0], "retained_cells": reference["summary"]["input_cells"]}]},
        data_origin=DataOrigin.REAL,
        source_ids=(SOURCE,),
        derivation=(
            "All 1,920 original cells, 46,078 endogenous features and 92 ERCC rows; unchanged counts, "
            "original identities and ordering. Scanpy reference and independent streaming oracle agree "
            "on all cell/feature tables and 17,332,418 filtered matrix entries."
        ),
        workflow_scope=WorkflowScope.CONNECTED,
        input_files={
            name: InputFile(asset["sha256"], asset["bytes"])
            for name, asset in source_catalog()[SOURCE]["file_inputs"].items()
        },
    )


def singlecell_contract(reference: dict) -> Contract:
    """Check cell and feature identities together with every filtered matrix entry."""
    return Contract(
        columns={
            name: Column(kind="integer", unit=unit, description=description)
            for name, unit, description in (
                ("input_cells", "cells", "number of supplied cell columns"),
                ("retained_cells", "cells", "cells passing all stated filters"),
                ("input_features", "features", "supplied endogenous and ERCC rows"),
                ("retained_features", "genes", "endogenous genes passing the post-cell-QC filter"),
                ("output_nonzeros", "entries", "positive counts in the exported matrix"),
                ("output_read_counts", "read counts", "sum of exported integer counts"),
            )
        },
        expected={"study": reference["summary"]},
        tables={
            "cell_qc.tsv": TableContract(
                columns={
                    "geo_accession": Column(kind="text", unit="accession", description="source GEO sample"),
                    "sorting_gate": Column(kind="text", unit="label", description="supplied broad sorting gate"),
                    **{
                        name: Column(kind="integer", unit=unit, description=description)
                        for name, unit, description in (
                            ("all_counts", "read counts", "endogenous plus ERCC counts before filtering"),
                            ("ercc_counts", "read counts", "ERCC counts before filtering"),
                            ("endogenous_counts", "read counts", "endogenous counts before filtering"),
                            ("detected_endogenous", "genes", "endogenous genes with positive counts before filtering"),
                            ("keep", "indicator", "one if all cell criteria pass, zero otherwise"),
                            (
                                "matrix_column",
                                "1-based index",
                                "output column in original cell order, or zero if excluded",
                            ),
                        )
                    },
                    "ercc_fraction": Column(
                        kind="number",
                        unit="fraction",
                        description="ERCC divided by all counts, null for an empty cell",
                        atol=1e-12,
                        rtol=1e-9,
                        nullable=True,
                    ),
                },
                expected=reference["cell_qc"],
                max_bytes=2 * 1024 * 1024,
            ),
            "feature_qc.tsv": TableContract(
                columns={
                    "feature_type": Column(kind="text", unit="label", description="endogenous or ERCC"),
                    **{
                        name: Column(kind="integer", unit=unit, description=description)
                        for name, unit, description in (
                            ("retained_cell_count", "cells", "QC-passing cells with a positive count for this feature"),
                            ("retained_read_count", "read counts", "sum of this feature across QC-passing cells"),
                            ("keep", "indicator", "one if endogenous and present in enough retained cells"),
                            ("matrix_row", "1-based index", "output row in original feature order, or zero if excluded"),
                        )
                    },
                },
                expected=reference["feature_qc"],
                max_bytes=16 * 1024 * 1024,
            ),
        },
        matrices={
            "filtered_counts.mtx.gz": MatrixMarketContract(
                **reference["matrix"], max_bytes=128 * 1024 * 1024, max_decoded_bytes=512 * 1024 * 1024
            )
        },
    )


RECIPES = (
    Recipe(
        id="real-singlecell-read-qc",
        version="1",
        skills=(
            "single-cell read-count QC",
            "ERCC spike-ins",
            "ordered cell and gene filtering",
            "sparse matrix export",
        ),
        formats=("Matrix Market", "gzip", "TSV cell metadata", "TSV feature metadata", "JSON query"),
        sources=("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE81682",),
        generate=generate_singlecell,
        oracle_timeout=900,
    ),
)
