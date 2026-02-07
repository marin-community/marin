# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TreeDiff algorithm for computing edit paths between Python ASTs.

Given a corrupted program and a target program, computes a sequence of small
edits that transforms the corrupted program toward the target. This provides
cleaner training supervision than reversing the random forward walk, following
Algorithm 1 of Tree Diffusion (Kapur et al., 2024).

Adapted for Python ASTs: instead of operating on a small CFG, we compare
AST nodes structurally and produce edits that replace differing subtrees
with the corresponding target subtrees, subject to a maximum edit size.
"""

import ast
import logging
from dataclasses import dataclass

from experiments.kelp.tree.mutation import Mutation, _node_source_span
from experiments.kelp.tree.subtree_bank import count_statements

logger = logging.getLogger(__name__)


def _nodes_equal(a: ast.AST, b: ast.AST) -> bool:
    """Check structural equality of two AST nodes (recursively).

    Uses ast.dump for a deep comparison that ignores source positions
    and formatting differences.
    """
    return ast.dump(a) == ast.dump(b)


@dataclass(frozen=True)
class Edit:
    """A single edit in a TreeDiff path.

    Represents replacing a subtree in the source with the corresponding
    subtree from the target.
    """

    source_start: int
    """Character start offset in source."""

    source_end: int
    """Character end offset in source (exclusive)."""

    target_fragment: str
    """Replacement text from the target program."""

    node_type: str
    """AST node type of the edit."""

    stmt_count: int
    """Statement count of the target fragment."""

    def to_mutation(self, source: str) -> Mutation:
        """Convert to a Mutation that can be applied to the source."""
        return Mutation(
            start=self.source_start,
            end=self.source_end,
            replacement=self.target_fragment,
            node_type=self.node_type,
            original=source[self.source_start : self.source_end],
        )


def tree_diff(
    source: str,
    target: str,
    max_edit_stmts: int = 3,
) -> list[Edit]:
    """Compute a sequence of edits transforming source toward target.

    Recursively compares the ASTs of source and target, finding differing
    subtrees small enough to replace in a single edit.

    Args:
        source: Current (corrupted) Python source code.
        target: Target (clean) Python source code.
        max_edit_stmts: Maximum statement count per edit. Edits larger
            than this are broken into smaller sub-edits by descending
            into children.

    Returns:
        List of Edits in tree-traversal order. Each edit replaces a
        differing subtree in source with the corresponding target subtree.
        Returns an empty list if source and target are structurally equal.
    """
    try:
        source_tree = ast.parse(source)
        target_tree = ast.parse(target)
    except SyntaxError:
        return []

    edits: list[Edit] = []
    _diff_nodes(source, target, source_tree, target_tree, max_edit_stmts, edits)
    return edits


def _diff_nodes(
    source: str,
    target: str,
    source_node: ast.AST,
    target_node: ast.AST,
    max_edit_stmts: int,
    edits: list[Edit],
) -> None:
    """Recursively diff two AST nodes, collecting edits.

    Always tries to descend into children first to find the most specific
    (smallest) edits. Only emits a whole-node edit when no children can be
    matched, or the node is a leaf-level expression with no sub-structure
    to descend into.
    """
    if _nodes_equal(source_node, target_node):
        return

    source_stmts = count_statements(source_node)
    target_stmts = count_statements(target_node)

    source_span = _node_source_span(source, source_node)
    target_span = _node_source_span(target, target_node)

    # Try to descend into children to find more specific diffs.
    # We match children by field name (the AST's structural slots).
    child_edits: list[Edit] = []
    matched_any = False
    for field_name, source_value in ast.iter_fields(source_node):
        target_value = _get_field(target_node, field_name)
        if target_value is None:
            continue

        if isinstance(source_value, list) and isinstance(target_value, list):
            matched_any = True
            _diff_lists(source, target, source_value, target_value, max_edit_stmts, child_edits)
        elif isinstance(source_value, ast.AST) and isinstance(target_value, ast.AST):
            matched_any = True
            _diff_nodes(source, target, source_value, target_value, max_edit_stmts, child_edits)

    if child_edits:
        # Descending found specific sub-edits; use those.
        edits.extend(child_edits)
        return

    # Leaf case: no decomposable children, or node types are incompatible.
    # Emit a single edit replacing this whole node.
    if source_span is not None and target_span is not None:
        # Only emit if within the size limit, or as a last resort when
        # we couldn't decompose at all.
        if (source_stmts <= max_edit_stmts and target_stmts <= max_edit_stmts) or not matched_any:
            target_text = target[target_span[0] : target_span[1]]
            edits.append(
                Edit(
                    source_start=source_span[0],
                    source_end=source_span[1],
                    target_fragment=target_text,
                    node_type=type(source_node).__name__,
                    stmt_count=target_stmts,
                )
            )


def _diff_lists(
    source: str,
    target: str,
    source_list: list,
    target_list: list,
    max_edit_stmts: int,
    edits: list[Edit],
) -> None:
    """Diff two lists of AST nodes (e.g., function body statements).

    Uses a simple paired comparison: match by position index up to the
    shorter list's length, then handle insertions/deletions for the
    remaining elements.
    """
    min_len = min(len(source_list), len(target_list))

    for i in range(min_len):
        s_item = source_list[i]
        t_item = target_list[i]

        if isinstance(s_item, ast.AST) and isinstance(t_item, ast.AST):
            _diff_nodes(source, target, s_item, t_item, max_edit_stmts, edits)


def _get_field(node: ast.AST, field_name: str):
    """Get a field value from an AST node, or None if missing."""
    try:
        return getattr(node, field_name)
    except AttributeError:
        return None


def one_step_edit(
    source: str,
    target: str,
    max_edit_stmts: int = 3,
) -> Mutation | None:
    """Compute the single best next edit from source toward target.

    Returns the first (deepest, most specific) edit from tree_diff,
    which is the most targeted change. Returns None if source and target
    are structurally equal.

    Args:
        source: Current Python source code.
        target: Target Python source code.
        max_edit_stmts: Maximum statement count per edit.

    Returns:
        A Mutation for the next edit, or None if no edit is needed.
    """
    edits = tree_diff(source, target, max_edit_stmts=max_edit_stmts)
    if not edits:
        return None

    # Return the first edit (tree-traversal order = most specific first
    # for depth-first traversal).
    return edits[0].to_mutation(source)


def find_path(
    source: str,
    target: str,
    max_edit_stmts: int = 3,
    max_steps: int = 20,
) -> list[Mutation]:
    """Compute a sequence of mutations transforming source into target.

    Iteratively applies one_step_edit until source matches target or
    max_steps is reached.

    Args:
        source: Starting Python source code.
        target: Target Python source code.
        max_edit_stmts: Maximum statement count per edit.
        max_steps: Maximum number of edit steps.

    Returns:
        List of Mutations in application order.
    """
    mutations: list[Mutation] = []
    current = source

    for _ in range(max_steps):
        mutation = one_step_edit(current, target, max_edit_stmts=max_edit_stmts)
        if mutation is None:
            break

        mutations.append(mutation)
        current = mutation.apply(current)

        # Check if we've reached the target.
        try:
            if _nodes_equal(ast.parse(current), ast.parse(target)):
                break
        except SyntaxError:
            break

    return mutations
