# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare gene-tree inference methods on observed mitochondrial proteins."""

import json

from experiments.post_training.bio_tasks.contract import Column, Contract, TableContract, TreeContract
from experiments.post_training.bio_tasks.real_data import source_text
from experiments.post_training.bio_tasks.recipe_types import DataOrigin, Instance, OracleRuntime, Recipe, WorkflowScope

SOURCE = "UniProt:metazoan-cox1-20260924"


def generate_phylogeny(_seed: int) -> Instance:
    reference = json.loads(source_text(SOURCE, "cox1-tree-reference.json.gz"))
    methods = sorted(reference["trees"])
    counts = {
        "tips": Column(kind="integer", unit="representatives", description="number of unique-sequence leaves"),
        "edges": Column(kind="integer", unit="edges", description="number of edges after suppressing a degree-two root"),
    }
    contract = Contract(
        columns={
            **counts,
            "tree_length": Column(
                kind="number",
                unit="substitutions per site",
                description="sum of all unrooted edge lengths",
                atol=1e-5,
                rtol=1e-5,
            ),
            "treeness": Column(
                kind="number",
                unit="fraction",
                description="nonterminal edge length divided by total tree length",
                atol=1e-5,
                rtol=1e-5,
            ),
        },
        expected=reference["summaries"],
        trees={
            method + ".nwk": TreeContract(reference=tree, atol=1e-5, rtol=1e-5, max_bytes=128 * 1024)
            for method, tree in reference["trees"].items()
        },
        tables={
            "distances.tsv": TableContract(
                columns={
                    method: Column(
                        kind="number",
                        unit="substitutions per site",
                        description="sum of branch lengths between the pair",
                        atol=1e-5,
                        rtol=1e-5,
                    )
                    for method in methods
                },
                expected=reference["distances"],
                max_bytes=2 * 1024 * 1024,
            ),
            "comparisons.tsv": TableContract(
                columns={
                    "rf_distance": Column(
                        kind="integer",
                        unit="splits",
                        description="size of the symmetric difference of nontrivial splits",
                    ),
                    "normalized_rf": Column(
                        kind="number",
                        unit="fraction",
                        description="RF divided by the total nontrivial split counts in both trees",
                        atol=1e-5,
                        rtol=1e-5,
                    ),
                },
                expected=reference["comparisons"],
                max_bytes=16384,
            ),
        },
    )
    return Instance(
        "Assess how three inference methods change an unrooted mitochondrial COX1 gene tree. "
        "The supplied alignment.fa contains 89 unchanged protein sequences from 94 curated metazoan "
        "UniProt accessions, aligned with pinned MAFFT. Exact duplicate proteins were represented by the "
        "lexicographically first accession; accessions.tsv preserves all source identities and sequence "
        "versions. Use the supplied alignment without realigning or trimming it. "
        "Infer trees with IQ-TREE 3.1.3 (-st AA -m LG+G4 -nt 1), FastTree 2.2.0 (-lg -gamma) and "
        "standard RAxML from Conda package 8.2.13 (executable reports 8.2.12; "
        "raxmlHPC-SSE3 -m PROTGAMMALG). Use query.json's integer seed with "
        "IQ-TREE -seed and RAxML -p. Use each program's default tree search; do not add bootstraps, "
        "constraints or alternative starting trees. Write iqtree.nwk, fasttree.nwk and raxml.nwk. "
        "Report leaf count, unrooted edge count, total edge length and treeness for each method in "
        "answer.json (ids fasttree, iqtree, raxml). Internal edges have at least two leaves on both sides. "
        "Produce distances.tsv for every unordered pair of representative accessions, id=left:right "
        "with lexicographically ordered names, and one distance column per method. Produce comparisons.tsv "
        "for every unordered method pair in the same naming convention. Its RF distance is the symmetric "
        "difference of nontrivial unrooted bipartitions; normalize by the sum of the two nontrivial split "
        "counts. Suppress a degree-two root by adding its incident edge lengths. "
        "All input files are in /app/inputs. FastTree's LG+CAT search and gamma rescaling differ from the "
        "other methods' LG+G4 fits; do not rank their raw likelihoods as one common-model comparison. "
        "The accession sample is curated rather than representative of Metazoa. These single-gene "
        "estimates and their agreement do not establish a species tree or branch support.",
        {
            "alignment.fa": source_text(SOURCE, "cox1-alignment.fa.gz"),
            "accessions.tsv": source_text(SOURCE, "cox1-accessions.tsv.gz"),
            "query.json": json.dumps(reference["query"], indent=2) + "\n",
        },
        contract,
        {"counted_all_accessions_as_leaves": [{**row, "tips": 94} for row in contract.answer()]},
        data_origin=DataOrigin.REAL,
        source_ids=(SOURCE,),
        derivation=(
            "Reviewed UniProt proteins, unchanged residues; retain one accession per identical sequence and "
            "all source mappings. MAFFT alignment is supplied as a fixed analysis boundary. Reference trees "
            "come from actual pinned packages; Biopython measures reference artifacts independently of the "
            "input-reading oracle's Newick parser and weighted-split calculations."
        ),
        workflow_scope=WorkflowScope.CONNECTED,
    )


RECIPES = (
    Recipe(
        id="real-cox1-tree-comparison",
        version="1",
        skills=("gene-tree inference", "duplicate sequence identities", "unrooted weighted splits", "method comparison"),
        formats=("aligned FASTA", "TSV accession metadata", "Newick", "JSON query"),
        sources=("https://www.uniprot.org/help/license",),
        generate=generate_phylogeny,
        oracle_timeout=1700,
        oracle_runtime=OracleRuntime.NATIVE,
    ),
)
