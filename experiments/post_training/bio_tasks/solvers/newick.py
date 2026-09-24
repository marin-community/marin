# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Newick parsing and root-independent weighted tree edges."""

import math
import re
from dataclasses import dataclass, field


@dataclass
class Node:
    label: str
    length: float | None
    children: list["Node"] = field(default_factory=list)


def newick(text: str) -> Node:
    """Parse one tree, retaining absent lengths separately from zero lengths."""
    tokens = re.findall(r"\[[^\[\]]*\]|'(?:[^']|'')*'|[(),:;]|[^\s(),:;\[\]']+|\S", text)
    tokens = [token for token in tokens if not (token.startswith("[") and token.endswith("]"))]
    index = 0

    def take() -> str:
        nonlocal index
        if index >= len(tokens):
            raise ValueError("truncated_newick")
        token = tokens[index]
        index += 1
        return token

    def subtree() -> Node:
        nonlocal index
        children = []
        label = ""
        token = take()
        if token == "(":
            children.append(subtree())
            token = take()
            while token == ",":
                children.append(subtree())
                token = take()
            if token != ")":
                raise ValueError("malformed_newick_children")
            token = take()
        if token not in {",", ")", ":", ";"}:
            if token in {"(", "[", "]", "'"}:
                raise ValueError("malformed_newick_label")
            label = token[1:-1].replace("''", "'") if token.startswith("'") else token
            token = take()
        length = None
        if token == ":":
            length = float(take())
        else:
            index -= 1
        return Node(label, length, children)

    try:
        root = subtree()
    except RecursionError as error:
        raise ValueError("newick_too_deep") from error
    if tokens[index:] != [";"]:
        raise ValueError("expected_one_newick_tree")
    return root


def leaves(node: Node) -> set[str]:
    return {node.label} if not node.children else set().union(*(leaves(child) for child in node.children))


def clades(node: Node) -> list[Node]:
    return [node] + [descendant for child in node.children for descendant in clades(child)]


def weighted_splits(root: Node) -> tuple[list[str], dict[str, float]]:
    """Canonicalize unrooted edges; suppress a degree-two root by adding its edges."""
    nodes = clades(root)
    tips = [node.label for node in nodes if not node.children]
    if len(tips) != len(set(tips)) or not 3 <= len(tips) <= 256:
        raise ValueError("tree_duplicate_or_invalid_tip_count")
    if any(not re.fullmatch(r"[A-Za-z0-9_.|+-]+", tip) for tip in tips):
        raise ValueError("tree_invalid_tip_label")
    if any(node.children and len(node.children) < 2 for node in nodes):
        raise ValueError("tree_unary_node")
    if root.length not in {None, 0.0}:
        raise ValueError("unrooted_tree_has_stem_length")
    taxa = set(tips)
    edges = {}
    for node in nodes[1:]:
        if node.length is None or not math.isfinite(node.length) or node.length < 0:
            raise ValueError("tree_missing_or_invalid_branch_length")
        descendants = leaves(node)
        sides = (tuple(sorted(descendants)), tuple(sorted(taxa - descendants)))
        key = ",".join(min(sides, key=lambda side: (len(side), side)))
        edges[key] = edges.get(key, 0.0) + node.length
        if not math.isfinite(edges[key]):
            raise ValueError("tree_nonfinite_path")
    return sorted(tips), edges
