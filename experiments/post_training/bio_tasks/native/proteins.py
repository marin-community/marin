# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Native alignment operations on unchanged curated protein sequences."""

from pathlib import Path

from experiments.post_training.bio_tasks.native.commands import execute
from experiments.post_training.bio_tasks.solvers.formats import fasta


def solve_mafft(inputs: Path, work: Path) -> list[dict]:
    execute(
        ["mafft", "--amino", "--globalpair", "--maxiterate", "1000", "--thread", "1", str(inputs / "proteins.fa")],
        work,
        "alignment.fa",
    )
    return [{"id": key, "residues": len(value.replace("-", ""))} for key, value in fasta(work / "alignment.fa").items()]


def solve_muscle(inputs: Path, work: Path) -> list[dict]:
    execute(
        ["muscle", "-align", str(inputs / "proteins.fa"), "-output", str(work / "alignment.fa"), "-threads", "1"],
        work,
        "muscle.stdout",
    )
    return [{"id": key, "residues": len(value.replace("-", ""))} for key, value in fasta(work / "alignment.fa").items()]
