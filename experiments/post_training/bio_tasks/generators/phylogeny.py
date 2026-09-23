# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rooted tree and multiple-sequence alignment operations."""

import random
from functools import partial
from itertools import combinations

from experiments.post_training.bio_tasks.contract import Column, Contract
from experiments.post_training.bio_tasks.recipe_types import Difficulty, Instance, Recipe, csv_text


def generate_phylogeny(seed: int, operation: str) -> Instance:
    rng = random.Random(seed)
    expected, inputs = {}, {}
    if operation.startswith("newick"):
        x, y, z, w, u, v, e = [rng.randint(1, 9) / 10 for _ in range(7)]
        inputs = {"tree.nwk": f"((a:{x},b:{y})AB:{u},(c:{z},d:{w})CD:{v},e:{e})ROOT;\n"}
        paths = {
            "a": {"ab": u, "a": x},
            "b": {"ab": u, "b": y},
            "c": {"cd": v, "c": z},
            "d": {"cd": v, "d": w},
            "e": {"e": e},
        }
        if operation == "newick-distances":
            for left, right in combinations(sorted(paths), 2):
                common = set(paths[left]) & set(paths[right])
                expected[left + ":" + right] = {
                    "distance": (
                        sum(paths[left].values())
                        + sum(paths[right].values())
                        - 2 * sum(paths[left][edge] for edge in common)
                    )
                }
            columns = {
                "distance": Column(
                    kind="number",
                    unit="substitutions per site",
                    description="patristic path length",
                    atol=1e-10,
                    rtol=1e-8,
                )
            }
            prompt = (
                "Compute patristic distances for all unordered tip pairs in tree.nwk, summing branch "
                "lengths along the connecting path. Internal labels are not tips. Use lexically ordered "
                "tip1:tip2 IDs. The supplied Newick has unquoted labels and a rooted topology; the root has"
                " no incoming edge."
            )
            wrong = [{"id": k, "distance": v["distance"] + u} for k, v in expected.items()]
            reason = "counted_shared_ancestral_edge"
        elif operation == "newick-monophyly":
            groups = {
                "ab": ["a", "b"],
                "ac": ["a", "c"],
                "cd": ["c", "d"],
                "abcd": ["a", "b", "c", "d"],
                "all": ["a", "b", "c", "d", "e"],
                "missing": ["a", "absent"],
            }
            inputs["groups.csv"] = csv_text(
                [{"group": name, "tip": tip} for name, tips in groups.items() for tip in tips]
            )
            for name, size, status in [
                ("ab", 2, "monophyletic"),
                ("ac", 5, "non_monophyletic"),
                ("cd", 2, "monophyletic"),
                ("abcd", 5, "non_monophyletic"),
                ("all", 5, "monophyletic"),
                ("missing", None, "unknown_taxa"),
            ]:
                expected[name] = {"status": status, "mrca_tips": size}
            columns = {
                "status": Column(
                    kind="text", unit="classification", description="monophyletic, non_monophyletic, or unknown_taxa"
                ),
                "mrca_tips": Column(
                    kind="integer",
                    unit="tips",
                    description="number of tips below MRCA; null for unknown taxa",
                    nullable=True,
                ),
            }
            prompt = (
                "Assess each groups.csv tip set on rooted tree.nwk. A set is monophyletic exactly when it "
                "equals all tips descended from its MRCA. Report status=monophyletic or non_monophyletic "
                "and MRCA descendant tip count. If any requested tip is absent, report unknown_taxa and "
                "null count; do not silently intersect. Internal node labels are not tips. Use group id."
            )
            wrong = [{"id": k, **row, "status": "monophyletic"} for k, row in expected.items()]
            reason = "assumed_every_named_group_is_a_clade"
        else:
            inputs["keep.txt"] = "a\nc\nd\n"
            expected = {
                "a": {"length": x + u},
                "c": {"length": z},
                "d": {"length": w},
                "c,d": {"length": v},
                "a,c,d": {"length": 0.0},
            }
            columns = {
                "length": Column(
                    kind="number",
                    unit="substitutions per site",
                    description="incoming branch length in pruned tree; root=0",
                    atol=1e-10,
                    rtol=1e-8,
                )
            }
            prompt = (
                "Prune rooted tree.nwk to tips listed in keep.txt. Suppress internal nodes with one "
                "remaining child by adding their incoming length to the child edge, preserving distances. "
                "Return one row per surviving tip/internal node, identified by the comma-joined lexically "
                "sorted descendant tips, with incoming branch length. The retained root has length 0; "
                "ignore internal labels. Do not output removed clades."
            )
            wrong = [{"id": k, "length": x if k == "a" else row["length"]} for k, row in expected.items()]
            reason = "lost_branch_length_when_collapsing_unary_node"
    else:
        column_values = ["AAC-", "CCCN", "GGTN", "AT--", "NN--", "ACGT", "TTTN", "GGGG"]
        rng.shuffle(column_values)
        if operation == "alignment-p-distance":
            order = rng.sample(range(4), 4)
            column_values = ["".join(column[i] for i in order) for column in column_values]
        sequences = {f"s{i}": "".join(column[i] for column in column_values) for i in range(4)}
        inputs = {"alignment.fa": "".join(f">{name}\n{sequence}\n" for name, sequence in sequences.items())}
        if operation == "alignment-consensus":
            known = {
                "AAC-": ("A", 3),
                "CCCN": ("C", 3),
                "GGTN": ("G", 3),
                "AT--": ("W", 2),
                "NN--": ("N", 0),
                "ACGT": ("N", 4),
                "TTTN": ("T", 3),
                "GGGG": ("G", 4),
            }
            expected = {
                str(i): {"consensus": known[column][0], "called": known[column][1]}
                for i, column in enumerate(column_values)
            }
            columns = {
                "consensus": Column(
                    kind="text",
                    unit="IUPAC base",
                    description="majority base or ambiguity code for tied maxima; N if none",
                ),
                "called": Column(kind="integer", unit="sequences", description="number of A/C/G/T calls"),
            }
            prompt = (
                "Build a columnwise consensus of alignment.fa. Ignore gaps and N when counting bases; "
                "choose the most frequent A/C/G/T, and use the IUPAC ambiguity code for tied "
                "maximum-frequency bases. No called bases gives N. Report called count and consensus for "
                "every 0-based column id."
            )
            wrong = [
                {"id": k, **v, "consensus": "A" if v["consensus"] == "W" else v["consensus"]}
                for k, v in expected.items()
            ]
            reason = "broke_consensus_tie_arbitrarily"
        elif operation == "alignment-p-distance":
            for a, b in combinations(range(4), 2):
                usable = [column for column in column_values if column[a] in "ACGT" and column[b] in "ACGT"]
                mismatch = sum(column[a] != column[b] for column in usable)
                expected[f"s{a}:s{b}"] = {
                    "sites": len(usable),
                    "differences": mismatch,
                    "p_distance": mismatch / len(usable) if usable else None,
                }
            columns = {
                "sites": Column(kind="integer", unit="columns", description="pairwise callable columns"),
                "differences": Column(kind="integer", unit="columns", description="different A/C/G/T bases"),
                "p_distance": Column(
                    kind="number",
                    unit="fraction",
                    description="differences / sites; null if no sites",
                    nullable=True,
                    atol=1e-10,
                    rtol=1e-8,
                ),
            }
            prompt = (
                "Compute uncorrected pairwise nucleotide p-distances for all unordered pairs in "
                "alignment.fa using pairwise deletion: a column is usable only when both sequences have "
                "A/C/G/T. Ignore N and - separately for each pair. Return usable site and mismatch counts "
                "and distance (null if no usable sites). Use lexically ordered sequence1:sequence2."
            )
            wrong = [{"id": k, **v, "p_distance": v["differences"] / len(column_values)} for k, v in expected.items()]
            reason = "used_full_alignment_denominator"
        else:
            acceptable = {"AAC-", "CCCN", "GGTN", "ACGT", "TTTN", "GGGG"}
            indices = [i for i, column in enumerate(column_values) if column in acceptable]
            expected = {
                name: {"sequence": "".join(sequence[i] for i in indices), "columns": ",".join(map(str, indices))}
                for name, sequence in sequences.items()
            }
            columns = {
                "sequence": Column(
                    kind="text", unit="aligned bases", description="sequence after shared column filtering"
                ),
                "columns": Column(
                    kind="text", unit="0-based indices", description="comma-separated retained original columns"
                ),
            }
            prompt = (
                "Filter alignment.fa columns, retaining a column only if gap (-) fraction <=0.25 and "
                "ambiguous (N) fraction <=0.25, each calculated over all sequences. Apply the same mask to "
                "every sequence without ungapping individual sequences. Report filtered sequence and "
                "ascending original 0-based retained-column indices for every sequence id."
            )
            wrong = [{"id": name, **row, "sequence": row["sequence"].replace("-", "")} for name, row in expected.items()]
            reason = "ungapped_individual_sequences"
    return Instance(
        prompt + " Inputs are in /app/inputs.", inputs, Contract(columns=columns, expected=expected), {reason: wrong}
    )


SKILLS = {
    "newick-distances": ("newick", "patristic-distance", "shared-ancestry"),
    "newick-monophyly": ("rooted-clades", "mrca", "missing-taxa"),
    "newick-pruning": ("tree-pruning", "branch-length-preservation"),
    "alignment-consensus": ("iupac", "majority-consensus", "ties"),
    "alignment-p-distance": ("pairwise-deletion", "sequence-distance"),
    "alignment-column-filter": ("alignment-mask", "gaps", "ambiguity"),
}
RECIPES = tuple(
    Recipe(
        name,
        "1",
        Difficulty.MEDIUM,
        skills,
        ("newick",) if name.startswith("newick") else ("aligned-fasta",),
        ("https://phylipweb.github.io/phylip/newicktree.html",),
        partial(generate_phylogeny, operation=name),
    )
    for name, skills in SKILLS.items()
)
