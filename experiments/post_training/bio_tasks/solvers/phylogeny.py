# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Small Newick parser and alignment reference calculations."""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import partial
from itertools import combinations
from pathlib import Path

from experiments.post_training.bio_tasks.solvers.formats import fasta, table


@dataclass
class Node:
    label: str
    length: float
    children: list["Node"] = field(default_factory=list)


def newick(text: str) -> Node:
    tokens = re.findall(r"[(),:;]|[^\s(),:;]+", text)
    index = 0

    def subtree() -> Node:
        nonlocal index
        children = []
        if tokens[index] == "(":
            index += 1
            children.append(subtree())
            while tokens[index] == ",":
                index += 1
                children.append(subtree())
            assert tokens[index] == ")"
            index += 1
        label = ""
        if tokens[index] not in {",", ")", ":", ";"}:
            label = tokens[index]
            index += 1
        length = 0.0
        if tokens[index] == ":":
            length = float(tokens[index + 1])
            index += 2
        return Node(label, length, children)

    root = subtree()
    assert tokens[index:] == [";"]
    return root


def leaves(node: Node) -> set[str]:
    return {node.label} if not node.children else set().union(*(leaves(child) for child in node.children))


def clades(node: Node) -> list[Node]:
    return [node] + [descendant for child in node.children for descendant in clades(child)]


def prune(node: Node, keep: set[str]) -> Node | None:
    if not node.children:
        return node if node.label in keep else None
    children = [result for child in node.children if (result := prune(child, keep)) is not None]
    if not children:
        return None
    if len(children) == 1:
        child = children[0]
        return Node(child.label, child.length + node.length, child.children)
    return Node(node.label, node.length, children)


def solve_phylogeny(inputs: Path, operation: str) -> list[dict]:
    answer = []
    if operation.startswith("newick"):
        root = newick((inputs / "tree.nwk").read_text())
        nodes = clades(root)
        if operation == "newick-distances":
            for a, b in combinations(sorted(leaves(root)), 2):
                distance = sum(node.length for node in nodes if (a in leaves(node)) != (b in leaves(node)))
                answer.append({"id": a + ":" + b, "distance": distance})
        elif operation == "newick-monophyly":
            groups = defaultdict(set)
            for row in table(inputs / "groups.csv"):
                groups[row["group"]].add(row["tip"])
            for name, tips in groups.items():
                if not tips <= leaves(root):
                    answer.append({"id": name, "status": "unknown_taxa", "mrca_tips": None})
                    continue
                descendants = min((leaves(node) for node in nodes if tips <= leaves(node)), key=len)
                answer.append(
                    {
                        "id": name,
                        "status": "monophyletic" if descendants == tips else "non_monophyletic",
                        "mrca_tips": len(descendants),
                    }
                )
        else:
            root = prune(root, set((inputs / "keep.txt").read_text().splitlines()))
            assert root is not None
            for node in clades(root):
                answer.append({"id": ",".join(sorted(leaves(node))), "length": 0.0 if node is root else node.length})
        return answer
    sequences = fasta(inputs / "alignment.fa")
    columns = list(zip(*sequences.values(), strict=True))
    if operation == "alignment-consensus":
        codes = {
            "A": "A",
            "C": "C",
            "G": "G",
            "T": "T",
            "AG": "R",
            "CT": "Y",
            "CG": "S",
            "AT": "W",
            "GT": "K",
            "AC": "M",
            "CGT": "B",
            "AGT": "D",
            "ACT": "H",
            "ACG": "V",
            "ACGT": "N",
            "": "N",
        }
        for i, column in enumerate(columns):
            counts = Counter(base for base in column if base in "ACGT")
            maximum = max(counts.values(), default=0)
            bases = "".join(sorted(base for base, count in counts.items() if count == maximum))
            answer.append({"id": str(i), "consensus": codes[bases], "called": sum(counts.values())})
    elif operation == "alignment-p-distance":
        for a, b in combinations(sorted(sequences), 2):
            paired = [(x, y) for x, y in zip(sequences[a], sequences[b], strict=True) if x in "ACGT" and y in "ACGT"]
            mismatches = sum(x != y for x, y in paired)
            answer.append(
                {
                    "id": a + ":" + b,
                    "sites": len(paired),
                    "differences": mismatches,
                    "p_distance": mismatches / len(paired) if paired else None,
                }
            )
    else:
        retained = [
            i
            for i, column in enumerate(columns)
            if column.count("-") / len(column) <= 0.25 and column.count("N") / len(column) <= 0.25
        ]
        for name, sequence in sequences.items():
            answer.append(
                {"id": name, "sequence": "".join(sequence[i] for i in retained), "columns": ",".join(map(str, retained))}
            )
    return answer


NAMES = (
    "newick-distances",
    "newick-monophyly",
    "newick-pruning",
    "alignment-consensus",
    "alignment-p-distance",
    "alignment-column-filter",
)
SOLVERS = {name: partial(solve_phylogeny, operation=name) for name in NAMES}
