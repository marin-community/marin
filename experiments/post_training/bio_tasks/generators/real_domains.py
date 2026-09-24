# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Complete protein-domain and proteome-coverage artifact contracts."""

from experiments.post_training.bio_tasks.contract import Column, Contract, TableContract


def domain_contract(reference: dict) -> Contract:
    """Check every domain sequence, boundary, score and searched protein identity."""
    return Contract(
        columns={
            name: Column(kind="integer", unit=unit, description=description)
            for name, unit, description in (
                ("searched_proteins", "proteins", "all supplied target proteins"),
                ("matched_proteins", "proteins", "distinct proteins with a passing domain for this model"),
                ("domains", "domains", "passing domain instances for this model"),
                ("covered_residues", "amino acids", "sum of per-protein aligned-position unions for this model"),
            )
        },
        expected=reference["summaries"],
        tables={
            "domains.tsv": TableContract(
                columns={
                    **{
                        name: Column(kind="text", unit=unit, description=description)
                        for name, unit, description in (
                            ("protein", "accession", "UniProt target accession"),
                            ("model", "accession.version", "complete Pfam model identity"),
                            ("sequence", "amino acids", "unchanged protein residues inside alignment boundaries"),
                        )
                    },
                    **{
                        name: Column(kind="integer", unit="1-based index", description=description)
                        for name, description in (
                            ("domain_index", "native domain number within the protein/model pair"),
                            ("alignment_start", "first aligned protein residue"),
                            ("alignment_end", "last aligned protein residue, included"),
                            ("envelope_start", "first envelope protein residue"),
                            ("envelope_end", "last envelope protein residue, included"),
                            ("model_start", "first aligned HMM position"),
                            ("model_end", "last aligned HMM position, included"),
                        )
                    },
                    **{
                        name: Column(kind="number", unit="bits", description=description, atol=0.05, rtol=0)
                        for name, description in (
                            ("protein_score", "native per-sequence bit score"),
                            ("domain_score", "native domain bit score"),
                        )
                    },
                    **{
                        name: Column(
                            kind="number", unit="expected false positives", description=description, atol=0, rtol=0.05
                        )
                        for name, description in (
                            ("independent_evalue", "native independent domain E-value"),
                            ("conditional_evalue", "native conditional domain E-value"),
                        )
                    },
                },
                expected=reference["domains"],
                max_bytes=2 * 1024 * 1024,
            ),
            "proteins.tsv": TableContract(
                columns={
                    name: Column(kind="integer", unit=unit, description=description)
                    for name, unit, description in (
                        ("length", "amino acids", "complete supplied protein length"),
                        ("sequence_version", "version", "supplied UniProt sequence version"),
                        ("domains", "domains", "all passing domain instances across supplied models"),
                        ("distinct_models", "models", "distinct matching models"),
                        ("covered_residues", "amino acids", "union of aligned protein positions across models"),
                    )
                },
                expected=reference["proteins"],
                max_bytes=2 * 1024 * 1024,
            ),
        },
    )
