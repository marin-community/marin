# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Curated protein alignment with a verifiable, non-unique output objective."""

import json
import math
import random
from itertools import combinations

from experiments.post_training.bio_tasks.contract import AlignmentContract, Column, Contract
from experiments.post_training.bio_tasks.real_data import source_text
from experiments.post_training.bio_tasks.recipe_types import DataOrigin, Difficulty, Instance, Recipe

SOURCE = "UniProt:globins-20260923"


def generate_alignment(seed: int) -> Instance:
    data = json.loads(source_text(SOURCE, "uniprot-globins.json.gz"))
    selected = sorted(random.Random(seed).sample(sorted(data["proteins"]), 6))
    sequences = {key: data["proteins"][key]["sequence"] for key in selected}
    upper = sum(data["optimal_pair_scores"][f"{a}:{b}"] for a, b in combinations(selected, 2))
    minimum = math.ceil(0.95 * upper)
    query = {
        "scoring": data["scoring"],
        "gap_open": data["gap_open"],
        "gap_extend": data["gap_extend"],
        "minimum_score": minimum,
        "score_units": "twice BLOSUM62; gap open 20, extension 1",
        "bound_method": "95% of summed independent optimal global pair scores; not necessarily jointly attainable",
    }
    contract = Contract(
        columns={"residues": Column(kind="integer", unit="amino acids", description="ungapped input sequence length")},
        expected={key: {"residues": len(sequence)} for key, sequence in sequences.items()},
        alignments={
            "alignment.fa": AlignmentContract(
                sequences=sequences,
                scoring=data["scoring"],
                gap_open=data["gap_open"],
                gap_extend=data["gap_extend"],
                minimum_score=minimum,
                max_columns=2 * max(map(len, sequences.values())),
                max_bytes=32 * 1024,
            )
        },
    )
    return Instance(
        "Align the six complete, unchanged curated vertebrate globin proteins in /app/inputs/proteins.fa. "
        "Create alignment.fa and report each input protein's residue count. Use query.json's scoring objective; "
        "an aligner or a custom implementation is acceptable. The output must meet the stated objective while "
        "preserving all sequences. Proteins include paralogous alpha/beta subunits: do not interpret their alignment "
        "as evidence for a species tree or orthology. Species and sequence-version metadata are in proteins.tsv.",
        {
            "proteins.fa": "".join(f">{key}\n{sequence}\n" for key, sequence in sequences.items()),
            "proteins.tsv": (
                "accession\torganism\tsequence_version\n"
                + "".join(
                    f"{key}\t{data['proteins'][key]['organism']}\t{data['proteins'][key]['sequence_version']}\n"
                    for key in selected
                )
            ),
            "query.json": json.dumps(query, indent=2) + "\n",
        },
        contract,
        {"counted_initiator_twice": [{**row, "residues": row["residues"] + 1} for row in contract.answer()]},
        data_origin=DataOrigin.REAL,
        source_ids=(SOURCE,),
        derivation=("Six full-length proteins selected from ten curated UniProt entries; sequence bytes unchanged. "
                    "Independent pairwise bounds from pinned Biopython. A separate center-star solver demonstrates "
                    "feasibility for every admitted instance."),
    )


RECIPES = (
    Recipe(
        id="real-protein-alignment",
        version="1",
        difficulty=Difficulty.MEDIUM,
        skills=("protein multiple-sequence alignment", "affine gap scoring", "paralog awareness"),
        formats=("FASTA", "TSV", "JSON"),
        sources=("https://www.uniprot.org/help/license", "https://mafft.cbrc.jp/alignment/software/"),
        generate=generate_alignment,
        repositories=("MAFFT", "MUSCLE"),
    ),
)
