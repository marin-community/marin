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

"""AST-based mutation for tree diffusion forward process.

Corrupts Python programs by replacing random AST subtrees with alternatives
drawn from a SubtreeBank. Every intermediate program produced by this process
is syntactically valid, following the key invariant of Tree Diffusion
(Kapur et al., 2024).
"""

import ast
import logging
import random
from dataclasses import dataclass

from experiments.kelp.tree.subtree_bank import (
    EXTRACTABLE_TYPES,
    STATEMENT_TYPES,
    SubtreeBank,
    count_statements,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Mutation:
    """A single tree edit: replace a span of source code.

    Character offsets are into the source string at the time the mutation
    was computed. After applying a mutation, all subsequent offsets shift.
    """

    start: int
    """Character start offset in source."""

    end: int
    """Character end offset in source (exclusive)."""

    replacement: str
    """Replacement source code string."""

    node_type: str
    """AST node type that was replaced (e.g., 'If', 'Call')."""

    original: str
    """Original source code that was replaced."""

    def apply(self, source: str) -> str:
        """Apply this mutation to a source string."""
        return source[: self.start] + self.replacement + source[self.end :]


def _linecol_to_offset(source: str, line: int, col: int) -> int:
    """Convert 1-based line and 0-based column to a character offset.

    Args:
        source: The source string.
        line: 1-based line number (as returned by ast nodes).
        col: 0-based column offset.

    Returns:
        0-based character offset into source.
    """
    current_line = 1
    offset = 0
    for i, ch in enumerate(source):
        if current_line == line:
            return i + col
        if ch == "\n":
            current_line += 1
    return offset + col


def _node_source_span(source: str, node: ast.AST) -> tuple[int, int] | None:
    """Get the (start, end) character offsets for an AST node.

    Returns None if the node lacks position info.
    """
    if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
        return None
    if node.end_lineno is None or node.end_col_offset is None:
        return None

    start = _linecol_to_offset(source, node.lineno, node.col_offset)
    end = _linecol_to_offset(source, node.end_lineno, node.end_col_offset)
    return (start, end)


@dataclass
class _Candidate:
    """A candidate AST node for mutation."""

    node: ast.AST
    node_type: str
    start: int
    end: int
    stmt_count: int


def _find_candidates(
    source: str,
    tree: ast.Module,
    max_edit_stmts: int,
    bank: SubtreeBank,
) -> list[_Candidate]:
    """Find all AST nodes eligible for mutation.

    A node is eligible if:
    1. Its type is in EXTRACTABLE_TYPES.
    2. Its statement count is <= max_edit_stmts.
    3. The bank has replacement candidates of the same type.
    4. It has valid source position info.
    5. Its source segment is non-trivial (>= 5 chars).
    """
    candidates = []

    for node in ast.walk(tree):
        type_name = type(node).__name__
        if type_name not in EXTRACTABLE_TYPES:
            continue
        if not bank.has_type(type_name):
            continue

        span = _node_source_span(source, node)
        if span is None:
            continue

        start, end = span
        if end - start < 5:
            continue

        stmt_count = count_statements(node)
        if stmt_count > max_edit_stmts:
            continue

        candidates.append(
            _Candidate(
                node=node,
                node_type=type_name,
                start=start,
                end=end,
                stmt_count=stmt_count,
            )
        )

    return candidates


def random_mutation(
    source: str,
    bank: SubtreeBank,
    max_edit_stmts: int = 3,
    rng: random.Random | None = None,
    max_retries: int = 10,
) -> Mutation | None:
    """Select a random AST subtree and replace it with a bank sample.

    Following the paper's approach: pick a random eligible node, sample a
    replacement of the same AST type from the subtree bank.

    Args:
        source: Python source code.
        bank: Subtree bank to draw replacements from.
        max_edit_stmts: Maximum statement count for editable subtrees.
        rng: Random number generator.
        max_retries: Maximum attempts to find a valid mutation.

    Returns:
        A Mutation, or None if no valid mutation could be found.
    """
    if rng is None:
        rng = random.Random()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    candidates = _find_candidates(source, tree, max_edit_stmts, bank)
    if not candidates:
        return None

    # The paper's improved variant: sample a node type first (uniform over
    # types), then sample a node of that type. This avoids bias toward types
    # that appear many times in a single program.
    type_to_candidates: dict[str, list[_Candidate]] = {}
    for c in candidates:
        if c.node_type not in type_to_candidates:
            type_to_candidates[c.node_type] = []
        type_to_candidates[c.node_type].append(c)

    for _ in range(max_retries):
        chosen_type = rng.choice(list(type_to_candidates.keys()))
        candidate = rng.choice(type_to_candidates[chosen_type])

        replacement_entry = bank.sample_with_size(
            candidate.node_type,
            max_stmts=max_edit_stmts,
            rng=rng,
        )
        if replacement_entry is None:
            continue

        original = source[candidate.start : candidate.end]

        # Skip if replacement is identical to the original.
        if replacement_entry.source == original:
            continue

        # For statements, we need to match the indentation of the original.
        replacement_source = _match_indentation(source, candidate.start, replacement_entry.source, candidate.node_type)

        # Verify the mutation produces valid Python.
        mutated = source[: candidate.start] + replacement_source + source[candidate.end :]
        try:
            ast.parse(mutated)
        except SyntaxError:
            continue

        return Mutation(
            start=candidate.start,
            end=candidate.end,
            replacement=replacement_source,
            node_type=candidate.node_type,
            original=original,
        )

    return None


def _match_indentation(source: str, insert_offset: int, replacement: str, node_type: str) -> str:
    """Adjust indentation of the replacement to match the insertion point.

    For statements, the replacement's indentation is normalized to match the
    indentation level at the insertion point in the source. Expressions don't
    need indentation adjustment.
    """
    if node_type not in STATEMENT_TYPES:
        return replacement

    # Find the indentation of the line at insert_offset.
    line_start = source.rfind("\n", 0, insert_offset) + 1
    target_indent = ""
    for ch in source[line_start:insert_offset]:
        if ch in (" ", "\t"):
            target_indent += ch
        else:
            break

    # Determine the replacement's current base indentation (first line).
    repl_lines = replacement.split("\n")
    if not repl_lines:
        return replacement

    repl_base_indent = ""
    for ch in repl_lines[0]:
        if ch in (" ", "\t"):
            repl_base_indent += ch
        else:
            break

    # Re-indent: replace the base indentation with the target indentation.
    result_lines = []
    for i, line in enumerate(repl_lines):
        if i == 0:
            result_lines.append(target_indent + line.lstrip())
        elif line.strip():
            # Remove original base indent, add target indent + relative indent.
            if line.startswith(repl_base_indent):
                relative = line[len(repl_base_indent) :]
                result_lines.append(
                    target_indent + "    " + relative.lstrip()
                    if relative.startswith(" ") or relative.startswith("\t")
                    else target_indent + relative
                )
            else:
                result_lines.append(target_indent + "    " + line.lstrip())
        else:
            result_lines.append("")

    return "\n".join(result_lines)


def corrupt_program(
    source: str,
    num_steps: int,
    bank: SubtreeBank,
    max_edit_stmts: int = 3,
    rng: random.Random | None = None,
) -> tuple[str, list[Mutation]]:
    """Apply multiple random mutations to corrupt a program.

    The forward (noise) process for tree diffusion. Each mutation replaces
    a random AST subtree with an alternative from the subtree bank, producing
    a valid but semantically different program.

    Args:
        source: Clean Python source code.
        num_steps: Number of mutations to apply.
        bank: Subtree bank to draw replacements from.
        max_edit_stmts: Maximum statement count per edit.
        rng: Random number generator.

    Returns:
        Tuple of (corrupted_source, list_of_mutations_applied).
        The mutations are in application order. If fewer than num_steps
        mutations could be found, fewer are returned.
    """
    if rng is None:
        rng = random.Random()

    current = source
    mutations: list[Mutation] = []

    for _ in range(num_steps):
        mutation = random_mutation(
            current,
            bank,
            max_edit_stmts=max_edit_stmts,
            rng=rng,
        )
        if mutation is None:
            break

        current = mutation.apply(current)
        mutations.append(mutation)

    return current, mutations
