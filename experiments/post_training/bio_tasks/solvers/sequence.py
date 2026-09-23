# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sequence reference solvers reading the supplied files."""

from functools import partial
from itertools import combinations, pairwise
from pathlib import Path

from experiments.post_training.bio_tasks.solvers.formats import fasta, reverse_complement, table, translate


def solve_sequence(inputs: Path, operation: str) -> list[dict]:
    filename = (
        "proteins.fa"
        if operation.startswith("protein")
        else "linear.fa" if operation == "fasta-restriction-fragments" else "sequences.fa"
    )
    sequences = fasta(inputs / filename)
    answer = []
    for name, raw in sequences.items():
        sequence = raw.upper()
        if operation == "fasta-iupac-gc":
            called = sum(sequence.count(base) for base in "ACGT")
            answer.append(
                {
                    "id": name,
                    "called": called,
                    "ambiguous": len(sequence) - called,
                    "gc": (sequence.count("G") + sequence.count("C")) / called,
                }
            )
        elif operation == "fasta-six-frame-translation":
            for strand, text in [("+", sequence), ("-", reverse_complement(sequence))]:
                for offset in range(3):
                    answer.append({"id": f"{name}:{strand}{offset+1}", "protein": translate(text[offset:])})
        elif operation == "fasta-orf-selection":
            candidates = []
            for start in range(len(sequence) - 2):
                if sequence[start : start + 3] != "ATG":
                    continue
                for stop in range(start + 3, len(sequence) - 2, 3):
                    if sequence[stop : stop + 3] in {"TAA", "TAG", "TGA"}:
                        candidates.append((stop + 3 - start, -start, stop + 3))
                        break
            _, negative_start, end = max(candidates)
            start = -negative_start
            answer.append({"id": name, "start": start, "end": end, "protein": translate(sequence[start : end - 3])})
        elif operation == "fasta-restriction-fragments":
            boundaries = (
                [0] + [i + 1 for i in range(len(sequence) - 5) if sequence[i : i + 6] == "GAATTC"] + [len(sequence)]
            )
            answer.append(
                {
                    "id": name,
                    "cuts": len(boundaries) - 2,
                    "lengths": ",".join(str(b - a) for a, b in pairwise(boundaries)),
                }
            )
        elif operation == "fasta-motif-hits":
            motif = (inputs / "motif.txt").read_text().strip()
            for strand, word in [("+", motif), ("-", reverse_complement(motif))]:
                positions = []
                position = sequence.find(word)
                while position >= 0:
                    positions.append(position)
                    position = sequence.find(word, position + 1)
                answer.append(
                    {"id": name + ":" + strand, "starts": ",".join(map(str, positions)), "hits": len(positions)}
                )
        elif operation == "protein-molecular-mass":
            masses = {row["residue"]: float(row["mass"]) for row in table(inputs / "residue_masses.csv")}
            answer.append({"id": name, "neutral_mass": sum(masses[residue] for residue in sequence) + 18.010565})
        elif operation == "protein-tryptic-digest":
            start, index = 0, 1
            for position, residue in enumerate(sequence):
                if position == len(sequence) - 1 or (residue in "KR" and sequence[position + 1] != "P"):
                    answer.append(
                        {
                            "id": f"{name}:{index}",
                            "peptide": sequence[start : position + 1],
                            "start": start,
                            "end": position + 1,
                        }
                    )
                    start, index = position + 1, index + 1
        elif operation == "protein-charge":
            ph = float((inputs / "ph.txt").read_text())
            charge = 0.0
            for row in table(inputs / "ionization.csv"):
                count = 1 if row["group"] in {"N_term", "C_term"} else sequence.count(row["group"])
                pka = float(row["pka"])
                charge += count / (1 + 10 ** (ph - pka)) if row["kind"] == "basic" else -count / (1 + 10 ** (pka - ph))
            answer.append({"id": name, "charge": charge})
        elif operation == "protein-hydropathy-windows":
            scores = {row["residue"]: float(row["score"]) for row in table(inputs / "scale.csv")}
            size = int((inputs / "window.txt").read_text())
            values = [sum(scores[x] for x in sequence[i : i + size]) / size for i in range(len(sequence) - size + 1)]
            best = max(range(len(values)), key=lambda i: (round(values[i], 10), -i))
            answer.append({"id": name, "start": best, "end": best + size, "mean_hydropathy": values[best]})
    if operation == "fasta-kmer-jaccard":
        size = int((inputs / "k.txt").read_text())
        sets = {}
        for name, sequence in sequences.items():
            words = {sequence[i : i + size].upper() for i in range(len(sequence) - size + 1)}
            sets[name] = {min(word, reverse_complement(word)) for word in words if set(word) <= set("ACGT")}
        for left, right in combinations(sorted(sets), 2):
            intersection, union = len(sets[left] & sets[right]), len(sets[left] | sets[right])
            answer.append(
                {
                    "id": left + ":" + right,
                    "intersection": intersection,
                    "union": union,
                    "jaccard": intersection / union if union else 1.0,
                }
            )
    return answer


NAMES = (
    "fasta-iupac-gc",
    "fasta-six-frame-translation",
    "fasta-orf-selection",
    "fasta-restriction-fragments",
    "fasta-motif-hits",
    "fasta-kmer-jaccard",
    "protein-molecular-mass",
    "protein-tryptic-digest",
    "protein-charge",
    "protein-hydropathy-windows",
)
SOLVERS = {name: partial(solve_sequence, operation=name) for name in NAMES}
