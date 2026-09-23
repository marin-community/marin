# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sequence and protein tasks with planted motifs and explicit biochemical tables."""

import random
from functools import partial

from experiments.post_training.bio_tasks.contract import Column, Contract
from experiments.post_training.bio_tasks.recipe_types import Difficulty, Instance, Recipe, csv_text


def generate_sequence(seed: int, operation: str) -> Instance:
    rng = random.Random(seed)
    inputs, expected = {}, {}
    if operation == "fasta-iupac-gc":
        records = []
        for i in range(4):
            counts = [rng.randint(2, 8) for _ in range(4)]
            bases = list("".join(base * n for base, n in zip("ACGT", counts, strict=True)) + "NRYSWKMBDHV")
            rng.shuffle(bases)
            sequence = "".join(bases)
            records.append(f">s{i} description\n{sequence[:12].lower()}\n{sequence[12:]}\n")
            expected[f"s{i}"] = {"called": sum(counts), "ambiguous": 11, "gc": (counts[1] + counts[2]) / sum(counts)}
        inputs = {"sequences.fa": "".join(records)}
        columns = {
            "called": Column(kind="integer", unit="bases", description="A/C/G/T positions"),
            "ambiguous": Column(kind="integer", unit="bases", description="other IUPAC positions"),
            "gc": Column(kind="number", unit="fraction", description="GC among A/C/G/T only", atol=1e-10, rtol=1e-8),
        }
        prompt = (
            "Summarize each sequences.fa record case-insensitively. Count A/C/G/T as called, all other "
            "IUPAC letters as ambiguous, and compute GC fraction among called bases only. Use the first"
            " whitespace-delimited header token as id."
        )
        wrong = [
            {"id": k, **v, "gc": v["gc"] * v["called"] / (v["called"] + v["ambiguous"])} for k, v in expected.items()
        ]
        reason = "ambiguity_in_gc_denominator"
    elif operation == "fasta-six-frame-translation":
        records = []
        for i, (codon, aa) in enumerate(
            [("ATG", ("M", "*", "D", "H", "I", "S")), ("GCC", ("A", "P", "R", "G", "A", "R"))]
        ):
            n = rng.randint(5, 9)
            records.append(f">s{i}\n{codon*n}\n")
            for index, frame in enumerate(["+1", "+2", "+3", "-1", "-2", "-3"]):
                expected[f"s{i}:{frame}"] = {"protein": aa[index] * (n if index % 3 == 0 else n - 1)}
        inputs = {"sequences.fa": "".join(records)}
        columns = {
            "protein": Column(
                kind="text", unit="amino acids", description="standard-code translation including stop as *"
            )
        }
        prompt = (
            "Translate all six reading frames of sequences.fa using the standard genetic code. Negative"
            " frames use the reverse complement, then offsets 0,1,2. Ignore incomplete terminal codons."
            " Retain stops as *. Use sequence_id:frame as id, with frames +1,+2,+3,-1,-2,-3."
        )
        wrong = [{"id": k, **v, "protein": v["protein"] + "X"} for k, v in expected.items()]
        reason = "translated_incomplete_terminal_codon"
    elif operation == "fasta-orf-selection":
        records = []
        for i in range(3):
            n, m = rng.randint(5, 8), rng.randint(1, 3)
            long = "ATG" + "GCT" * n + "TAA"
            short = "ATG" + "GGT" * m + "TAG"
            prefix = "CCC" + short + "CCC" if i % 2 else "CCC"
            sequence = prefix + long + ("CCC" if i % 2 else "CCC" + short + "CCC")
            records.append(f">s{i}\n{sequence}\n")
            expected[f"s{i}"] = {"start": len(prefix), "end": len(prefix) + len(long), "protein": "M" + "A" * n}
        inputs = {"sequences.fa": "".join(records)}
        columns = {
            "start": Column(kind="integer", unit="0-based bases", description="inclusive start of ATG"),
            "end": Column(kind="integer", unit="0-based bases", description="exclusive end including stop codon"),
            "protein": Column(kind="text", unit="amino acids", description="translation excluding terminal stop"),
        }
        prompt = (
            "Find the longest complete ORF on the forward strand of each sequences.fa record, "
            "considering every ATG start in all three frames and ending at its first in-frame "
            "TAA/TAG/TGA. Rank by coding length including stop, then smallest start. Report 0-based "
            "half-open coordinates including stop and protein excluding stop. Use sequence id."
        )
        wrong = [{"id": k, **v, "end": v["end"] - 3} for k, v in expected.items()]
        reason = "excluded_stop_from_coordinates"
    elif operation == "fasta-restriction-fragments":
        records = []
        for i in range(3):
            a, b, c = [rng.randint(3, 9) for _ in range(3)]
            sequence = "C" * a + "GAATTC" + "C" * b + "GAATTC" + "C" * c
            records.append(f">s{i}\n{sequence}\n")
            expected[f"s{i}"] = {"lengths": ",".join(map(str, [a + 1, b + 6, c + 5])), "cuts": 2}
        inputs = {"linear.fa": "".join(records)}
        columns = {
            "lengths": Column(
                kind="text", unit="bases", description="fragment lengths in left-to-right order, comma-separated"
            ),
            "cuts": Column(kind="integer", unit="cuts", description="number of distinct cut boundaries"),
        }
        prompt = (
            "Digest the linear double-stranded sequences in linear.fa with EcoRI, recognizing GAATTC "
            "and cutting the displayed strand G^AATTC. Return fragment lengths along that strand from "
            "left to right, including both terminal fragments, and the number of cuts. Do not treat "
            "molecules as circular. Use sequence id."
        )
        wrong = [{"id": k, **v, "lengths": ",".join(v["lengths"].split(",")[:-1])} for k, v in expected.items()]
        reason = "dropped_terminal_fragment"
    elif operation == "fasta-motif-hits":
        sequences = {f"s{i}": "".join(rng.choice("ACGT") for _ in range(25)) + "ATATAT" for i in range(3)}
        inputs = {"sequences.fa": "".join(f">{k}\n{v}\n" for k, v in sequences.items()), "motif.txt": "ATA\n"}
        for name, sequence in sequences.items():
            for strand, word in [("+", "ATA"), ("-", "TAT")]:
                positions = [i for i in range(len(sequence) - 2) if tuple(sequence[i : i + 3]) == tuple(word)]
                expected[name + ":" + strand] = {"starts": ",".join(map(str, positions)), "hits": len(positions)}
        columns = {
            "starts": Column(
                kind="text",
                unit="0-based positions",
                description="ascending genomic starts, comma-separated; empty if none",
            ),
            "hits": Column(kind="integer", unit="matches", description="overlapping matches on requested strand"),
        }
        prompt = (
            "Find all exact motif.txt matches on both strands of sequences.fa, including overlapping "
            "hits. Report genomic 0-based starts of matched spans in ascending order for each strand; "
            "negative-strand matches are reverse complements of the motif. Return every sequence:+ and "
            "sequence:- id, using an empty starts string if absent."
        )
        wrong = [{"id": k, **v, "hits": v["hits"] - 1} for k, v in expected.items()]
        reason = "lost_overlapping_hits"
    elif operation == "fasta-kmer-jaccard":
        k = 3
        a = "".join(rng.choice("ACGT") for _ in range(20))
        b = a[:10] + "NN" + "".join(rng.choice("ACGT") for _ in range(14))
        sequences = {"a": a, "b": b, "c": "AAAANNNNTTTT"}
        inputs = {
            "sequences.fa": "".join(f">{name}\n{sequence}\n" for name, sequence in sequences.items()),
            "k.txt": str(k) + "\n",
        }

        def kmers(sequence: str) -> set:
            complement = {"A": "T", "C": "G", "G": "C", "T": "A"}
            result = set()
            for start in range(len(sequence) - k + 1):
                word = sequence[start : start + k]
                if set(word) <= set(complement):
                    result.add(min(word, "".join(complement[x] for x in word[::-1])))
            return result

        sets = {name: kmers(sequence) for name, sequence in sequences.items()}
        for left, right in [("a", "b"), ("a", "c"), ("b", "c")]:
            expected[f"{left}:{right}"] = {
                "intersection": len(sets[left] & sets[right]),
                "union": len(sets[left] | sets[right]),
                "jaccard": len(sets[left] & sets[right]) / len(sets[left] | sets[right]),
            }
        columns = {
            name: Column(kind="integer", unit="canonical k-mers", description=name + " set size")
            for name in ["intersection", "union"]
        }
        columns["jaccard"] = Column(
            kind="number", unit="fraction", description="set intersection divided by union", atol=1e-10, rtol=1e-8
        )
        prompt = (
            "Compute canonical DNA k-mer set Jaccard similarities for all unordered sequence pairs in "
            "sequences.fa, with k from k.txt. Skip windows containing non-ACGT; identify a k-mer with "
            "its reverse complement via lexical minimum and remove duplicates. Use lexically ordered "
            "left:right sequence IDs. An empty union has similarity 1."
        )
        wrong = [{"id": name, **row, "union": row["union"] + 1} for name, row in expected.items()]
        reason = "counted_ambiguous_kmer"
    else:
        counts = [rng.randint(1, 3) for _ in range(3)]
        sequences = {f"p{i}": "A" * n + "KPG" + "R" + "V" * i + "K" for i, n in enumerate(counts)}
        inputs = {"proteins.fa": "".join(f">{name}\n{sequence}\n" for name, sequence in sequences.items())}
        if operation == "protein-molecular-mass":
            mass = {"A": 71.037114, "K": 128.094963, "P": 97.052764, "G": 57.021464, "R": 156.101111, "V": 99.068414}
            inputs["residue_masses.csv"] = csv_text([{"residue": key, "mass": value} for key, value in mass.items()])
            for i, n in enumerate(counts):
                expected[f"p{i}"] = {
                    "neutral_mass": (
                        n * mass["A"] + 2 * mass["K"] + mass["P"] + mass["G"] + mass["R"] + i * mass["V"] + 18.010565
                    )
                }
            columns = {
                "neutral_mass": Column(
                    kind="number",
                    unit="Da",
                    description="unmodified neutral monoisotopic peptide mass",
                    atol=1e-6,
                    rtol=1e-8,
                )
            }
            prompt = (
                "Compute unmodified neutral monoisotopic mass of each proteins.fa sequence using the "
                "supplied residue_masses.csv (residue masses, not free amino acid masses). Add one water "
                "molecule, 18.010565 Da, per peptide. Use protein id."
            )
            wrong = [{"id": k, "neutral_mass": v["neutral_mass"] - 18.010565} for k, v in expected.items()]
            reason = "omitted_terminal_water"
        elif operation == "protein-tryptic-digest":
            for i, n in enumerate(counts):
                first = "A" * n + "KPGR"
                second = "V" * i + "K"
                expected[f"p{i}:1"] = {"peptide": first, "start": 0, "end": len(first)}
                expected[f"p{i}:2"] = {"peptide": second, "start": len(first), "end": len(first) + len(second)}
            columns = {
                "peptide": Column(kind="text", unit="amino acids", description="fully cleaved peptide"),
                "start": Column(kind="integer", unit="0-based residues", description="inclusive start"),
                "end": Column(kind="integer", unit="0-based residues", description="exclusive end"),
            }
            prompt = (
                "Perform complete in-silico trypsin digestion of proteins.fa with zero missed cleavages: "
                "cut after K or R unless the following residue is P. Include terminal peptides, no length "
                "filtering, no empty peptide after a terminal cut. Report genomic-style 0-based half-open "
                "protein coordinates; id=protein_id:1-based_peptide_index."
            )
            wrong = [{"id": k, **v, "start": v["start"] + 1} for k, v in expected.items()]
            reason = "one_based_peptide_coordinates"
        elif operation == "protein-charge":
            ph = rng.choice([6.0, 7.0, 8.0])
            inputs["ionization.csv"] = csv_text(
                [
                    {"group": "N_term", "pka": 9.0, "kind": "basic"},
                    {"group": "C_term", "pka": 2.0, "kind": "acidic"},
                    {"group": "K", "pka": 10.5, "kind": "basic"},
                    {"group": "R", "pka": 12.0, "kind": "basic"},
                ]
            )
            inputs["ph.txt"] = str(ph) + "\n"
            charge = (
                1 / (1 + 10 ** (ph - 9))
                + 2 / (1 + 10 ** (ph - 10.5))
                + 1 / (1 + 10 ** (ph - 12))
                - 1 / (1 + 10 ** (2 - ph))
            )
            expected = {name: {"charge": charge} for name in sequences}
            columns = {
                "charge": Column(
                    kind="number",
                    unit="elementary charges",
                    description="independent-site expected net charge",
                    atol=1e-10,
                    rtol=1e-8,
                )
            }
            prompt = (
                "Estimate peptide net charge at ph.txt using ionization.csv and the independent-site "
                "Henderson-Hasselbalch model. Basic groups contribute +1/(1+10^(pH-pKa)); acidic groups "
                "contribute -1/(1+10^(pKa-pH)). Include one N and C terminus per sequence plus every listed"
                " side chain. Unlisted residues are neutral. Use proteins.fa IDs."
            )
            wrong = [{"id": name, "charge": charge + 1 / (1 + 10 ** (2 - ph))} for name in sequences]
            reason = "omitted_c_terminus"
        else:
            assert operation == "protein-hydropathy-windows"
            scores = {"A": 1.8, "K": -3.9, "P": -1.6, "G": -0.4, "R": -4.5, "V": 4.2}
            inputs["scale.csv"] = csv_text([{"residue": k, "score": v} for k, v in scores.items()])
            inputs["window.txt"] = "3\n"
            for name, sequence in sequences.items():
                prefix = [0.0]
                for residue in sequence:
                    prefix.append(prefix[-1] + scores[residue])
                values = [(prefix[i + 3] - prefix[i]) / 3 for i in range(len(sequence) - 2)]
                best = max(range(len(values)), key=lambda i: (values[i], -i))
                expected[name] = {"start": best, "end": best + 3, "mean_hydropathy": values[best]}
            columns = {
                "start": Column(kind="integer", unit="0-based residues", description="inclusive best-window start"),
                "end": Column(kind="integer", unit="0-based residues", description="exclusive best-window end"),
                "mean_hydropathy": Column(
                    kind="number",
                    unit="scale units",
                    description="arithmetic mean over full window",
                    atol=1e-8,
                    rtol=1e-8,
                ),
            }
            prompt = (
                "Find the highest-mean hydropathy window in each proteins.fa sequence using scale.csv and "
                "window.txt. Consider only full-length windows, break exact ties by earliest start, and "
                "report 0-based half-open coordinates and mean. Use protein id."
            )
            wrong = [{"id": k, **v, "mean_hydropathy": v["mean_hydropathy"] * 3} for k, v in expected.items()]
            reason = "reported_window_sum"
    return Instance(
        prompt + " Inputs are in /app/inputs.", inputs, Contract(columns=columns, expected=expected), {reason: wrong}
    )


SKILLS = {
    "fasta-iupac-gc": ("ambiguity", "wrapped-fasta", "denominators"),
    "fasta-six-frame-translation": ("reading-frames", "reverse-complement", "genetic-code"),
    "fasta-orf-selection": ("orf-selection", "stop-codons", "coordinates"),
    "fasta-restriction-fragments": ("restriction-sites", "linear-boundaries"),
    "fasta-motif-hits": ("overlapping-motifs", "strand-coordinates"),
    "fasta-kmer-jaccard": ("canonical-kmers", "set-similarity", "ambiguous-bases"),
    "protein-molecular-mass": ("residue-masses", "terminal-water", "units"),
    "protein-tryptic-digest": ("enzyme-specificity", "proline-exception", "peptide-coordinates"),
    "protein-charge": ("ionization", "termini", "ph"),
    "protein-hydropathy-windows": ("sliding-windows", "hydropathy", "tie-breaking"),
}
RECIPES = tuple(
    Recipe(
        name,
        "1",
        Difficulty.EASY,
        skills,
        ("fasta", "csv-header") if name.startswith("protein") else ("fasta",),
        ("https://biopython.org/docs/latest/Tutorial/chapter_seq_objects.html",),
        partial(generate_sequence, operation=name),
    )
    for name, skills in SKILLS.items()
)
