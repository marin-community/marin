# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Full cell-QC, feature-QC and filtered-count artifact contracts."""

from experiments.post_training.bio_tasks.contract import Column, Contract, MatrixMarketContract, TableContract


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
