# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Complete proteome clustering and representative-sequence contracts."""

import json

from experiments.post_training.bio_tasks.contract import Column, Contract, FastaContract, TableContract
from experiments.post_training.bio_tasks.real_data import source_text
from experiments.post_training.bio_tasks.recipe_types import DataOrigin, Instance, OracleRuntime, Recipe, WorkflowScope

PROTEOME_SOURCE = "UniProt:UP000000625-20260924"


def cluster_contract(reference: dict) -> Contract:
    """Check every member assignment, cluster quantity and representative residue."""
    return Contract(
        columns={
            name: Column(kind="integer", unit=unit, description=description)
            for name, unit, description in (
                ("proteins", "proteins", "all input proteins"),
                ("clusters", "clusters", "native representative groups"),
                ("singletons", "clusters", "groups containing only their representative"),
                ("multimember_clusters", "clusters", "groups with at least two proteins"),
                ("nonrepresentative_proteins", "proteins", "input proteins omitted from the representative set"),
                ("input_residues", "amino acids", "sum of complete input protein lengths"),
                ("representative_residues", "amino acids", "sum of complete representative lengths"),
            )
        },
        expected=reference["summaries"],
        fasta={"representatives.fa": FastaContract(sequences=reference["representatives"], max_bytes=4 * 1024 * 1024)},
        tables={
            "membership.tsv": TableContract(
                columns={
                    "representative": Column(kind="text", unit="accession", description="native cluster representative"),
                    "length": Column(kind="integer", unit="amino acids", description="complete input protein length"),
                    "sequence_version": Column(
                        kind="integer", unit="version", description="input UniProt sequence version"
                    ),
                },
                expected=reference["membership"],
                max_bytes=2 * 1024 * 1024,
            ),
            "clusters.tsv": TableContract(
                columns={
                    name: Column(kind="integer", unit=unit, description=description)
                    for name, unit, description in (
                        ("members", "proteins", "all members including the representative"),
                        ("distinct_sequences", "sequences", "number of distinct full residue strings among members"),
                        ("total_residues", "amino acids", "sum of all member lengths without deduplication"),
                        ("minimum_length", "amino acids", "shortest member length"),
                        ("maximum_length", "amino acids", "longest member length"),
                        ("representative_length", "amino acids", "length of the native representative"),
                    )
                },
                expected=reference["clusters"],
                max_bytes=2 * 1024 * 1024,
            ),
        },
    )


def generate_clusters(_seed: int) -> Instance:
    reference = json.loads(source_text(PROTEOME_SOURCE, "ecoli-k12-cluster-reference.json.gz"))
    contract = cluster_contract(reference)
    return Instance(
        "Build a sequence-similarity representative set for the complete supplied E. coli K-12 proteome. "
        "All inputs are in /app/inputs. proteins.fa contains unchanged UniProt sequences; proteins.tsv "
        "contains their accessions, sequence versions and lengths. Use the installed MMseqs2 Linclust "
        "implementation. Create a protein database with --dbtype 1 and --shuffle 0. Run linclust with "
        "--min-seq-id set to query.json's minimum_identity, -c set to coverage, and --kmer-per-seq "
        "set to kmer_per_sequence. Identity and coverage are fractions. Also set --cov-mode 0, "
        "--cluster-mode 2, --alignment-mode 3 and --threads 1. "
        "Export cluster membership with createtsv. Extract the native representative sequences with "
        "createsubdb and convert2fasta. Write membership.tsv for every input protein, keyed by its "
        "UniProt accession, including singletons and each representative's own membership row. "
        "Preserve the native representative assignment, input length and sequence version. Write "
        "clusters.tsv keyed by representative accession: member count, number of distinct complete "
        "sequence strings, total member residues, shortest and longest member lengths, and representative "
        "length. Include the representative in every cluster statistic. Write representatives.fa using "
        "UniProt accessions as first-token IDs and unchanged native representative residues. Report "
        "answer.json for id=proteome, accounting for all input proteins, clusters, singleton and "
        "multimember clusters, omitted nonrepresentatives, input residues and retained representative "
        "residues. These are heuristic similarity clusters under the specified method; they do not "
        "establish orthology, functional equivalence or exhaustive pairwise similarity.",
        {
            "query.json": json.dumps(reference["query"], indent=2) + "\n",
            "proteins.fa": source_text(PROTEOME_SOURCE, "ecoli-k12-proteins.fasta.gz"),
            "proteins.tsv": source_text(PROTEOME_SOURCE, "ecoli-k12-metadata.tsv.gz"),
        },
        contract,
        {"omitted_singletons": [{**row, "proteins": row["proteins"] - row["singletons"]} for row in contract.answer()]},
        data_origin=DataOrigin.REAL,
        source_ids=(PROTEOME_SOURCE,),
        derivation=(
            "The complete observed UP000000625 reviewed proteome and sequence-version metadata. "
            "MMseqs2 native clustering and representative extraction, measured separately using "
            "Biopython in the reference and standard-library parsing in the oracle."
        ),
        workflow_scope=WorkflowScope.CONNECTED,
    )


RECIPES = (
    Recipe(
        id="real-proteome-clustering",
        version="1",
        skills=("protein similarity clustering", "representative extraction", "metadata joins", "proteome accounting"),
        formats=("FASTA", "MMseqs2 cluster TSV", "TSV protein metadata", "JSON query"),
        sources=("https://www.uniprot.org/proteomes/UP000000625", "https://github.com/soedinglab/MMseqs2"),
        generate=generate_clusters,
        oracle_timeout=1800,
        oracle_runtime=OracleRuntime.NATIVE,
    ),
)
