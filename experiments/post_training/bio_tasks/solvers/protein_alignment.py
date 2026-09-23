# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Affine-gap global alignment and a deterministic center-star reference solution."""

import json
from pathlib import Path

from experiments.post_training.bio_tasks.solvers.formats import fasta


def pair_alignment(left: str, right: str, scores: dict[str, int], opening: int, extension: int) -> tuple[str, str]:
    """Align two sequences with a three-state dynamic program, including terminal gaps."""
    rows, columns = len(left) + 1, len(right) + 1
    unreachable = -(10**9)
    values = [[[unreachable] * columns for _ in range(rows)] for _ in range(3)]
    parents = [[[0] * columns for _ in range(rows)] for _ in range(3)]
    values[0][0][0] = 0
    for i in range(1, rows):
        values[1][i][0] = -opening - (i - 1) * extension
        parents[1][i][0] = 1 if i > 1 else 0
    for j in range(1, columns):
        values[2][0][j] = -opening - (j - 1) * extension
        parents[2][0][j] = 2 if j > 1 else 0
    for i in range(1, rows):
        for j in range(1, columns):
            options = [values[k][i - 1][j - 1] for k in range(3)]
            state = max(range(3), key=options.__getitem__)
            values[0][i][j] = options[state] + scores[left[i - 1] + right[j - 1]]
            parents[0][i][j] = state
            for gap, previous_i, previous_j in ((1, i - 1, j), (2, i, j - 1)):
                options = [values[k][previous_i][previous_j] - (extension if k == gap else opening) for k in range(3)]
                state = max(range(3), key=options.__getitem__)
                values[gap][i][j] = options[state]
                parents[gap][i][j] = state
    i, j = len(left), len(right)
    state = max(range(3), key=lambda k: values[k][i][j])
    a, b = [], []
    while i or j:
        previous = parents[state][i][j]
        a.append(left[i - 1] if state != 2 else "-")
        b.append(right[j - 1] if state != 1 else "-")
        i -= state != 2
        j -= state != 1
        state = previous
    return "".join(reversed(a)), "".join(reversed(b))


def center_alignment(sequences: dict[str, str], scores: dict[str, int], opening: int, extension: int) -> dict[str, str]:
    center = min(sequences)
    aligned = {center: sequences[center]}
    for name in sorted(set(sequences) - {center}):
        anchor, added = pair_alignment(sequences[center], sequences[name], scores, opening, extension)
        existing = aligned[center]
        merged = {key: [] for key in (*aligned, name)}
        i = j = 0
        while i < len(existing) or j < len(anchor):
            old = existing[i] if i < len(existing) else None
            new = anchor[j] if j < len(anchor) else None
            take_old = old == "-" or old == new
            take_new = new == "-" or old == new
            if not take_old and not take_new:
                raise ValueError("Cannot reconcile center-star alignment")
            for key, sequence in aligned.items():
                merged[key].append(sequence[i] if take_old else "-")
            merged[name].append(added[j] if take_new else "-")
            i += take_old
            j += take_new
        aligned = {key: "".join(value) for key, value in merged.items()}
    return aligned


def solve_alignment(inputs: Path, output: Path) -> list[dict]:
    sequences = fasta(inputs / "proteins.fa")
    query = json.loads((inputs / "query.json").read_text())
    aligned = center_alignment(sequences, query["scoring"], query["gap_open"], query["gap_extend"])
    (output / "alignment.fa").write_text("".join(f">{key}\n{value}\n" for key, value in aligned.items()))
    return [{"id": key, "residues": len(sequence)} for key, sequence in sequences.items()]


OUTPUT_SOLVERS = {"real-protein-alignment": solve_alignment}
