# Copyright 2026 The Marin Authors
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

"""
Tree mutation operations for tree diffusion.

This module implements the forward diffusion process from Berkeley's tree
diffusion paper (arXiv:2405.20519). The key idea is:

1. Define a size function σ(z) that measures program complexity
2. Find candidate nodes where σ(subtree) ≤ σ_small (typically 2)
3. Replace a randomly selected mutation node with a grammatically valid alternative

These mutations serve as the "noise" in the diffusion process.
"""

import ast
import copy
import random
from dataclasses import dataclass


from experiments.kelp.ast_utils import TreeTensors, count_nodes, get_subtree_size


@dataclass
class MutationConfig:
    """Configuration for tree mutations."""

    sigma_small: int = 2  # Max subtree size for mutation candidates
    s_max: int = 5  # Max mutation steps in training


@dataclass
class MutationRecord:
    """Record of a single mutation for training."""

    original_node_idx: int  # Index of the node that was mutated
    original_node_type: int  # Original node type ID
    original_subtree_size: int  # Size of the original subtree
    new_node_type: int  # New node type ID after mutation
    mutation_type: str  # Type of mutation applied


def find_mutation_candidates(tensors: TreeTensors, sigma_small: int = 2) -> list[int]:
    """Find nodes that are candidates for mutation.

    A node is a mutation candidate if its subtree size is <= sigma_small.
    This follows the Berkeley tree diffusion approach where we only mutate
    "small" subtrees to ensure local, grammatically valid changes.

    Args:
        tensors: The tree tensor representation
        sigma_small: Maximum subtree size for candidates

    Returns:
        List of node indices that are mutation candidates
    """
    candidates = []
    for idx in range(tensors.num_valid_nodes):
        if tensors.node_mask[idx] == 0:
            continue
        subtree_size = get_subtree_size(tensors, idx)
        if subtree_size <= sigma_small:
            candidates.append(idx)
    return candidates


# Small replacement subtrees for different node types
# These are grammatically valid replacements for common AST patterns
REPLACEMENT_PATTERNS: dict[str, list[ast.AST]] = {
    # Expression replacements
    "Name": [
        ast.Name(id="x", ctx=ast.Load()),
        ast.Name(id="y", ctx=ast.Load()),
        ast.Name(id="n", ctx=ast.Load()),
        ast.Name(id="i", ctx=ast.Load()),
        ast.Constant(value=0),
        ast.Constant(value=1),
        ast.Constant(value=True),
        ast.Constant(value=False),
    ],
    "Constant": [
        ast.Constant(value=0),
        ast.Constant(value=1),
        ast.Constant(value=-1),
        ast.Constant(value=2),
        ast.Constant(value=True),
        ast.Constant(value=False),
        ast.Constant(value=None),
        ast.Name(id="x", ctx=ast.Load()),
        ast.Name(id="n", ctx=ast.Load()),
    ],
    "BinOp": [
        ast.BinOp(left=ast.Name(id="x", ctx=ast.Load()), op=ast.Add(), right=ast.Constant(value=1)),
        ast.BinOp(left=ast.Name(id="x", ctx=ast.Load()), op=ast.Sub(), right=ast.Constant(value=1)),
        ast.BinOp(left=ast.Name(id="x", ctx=ast.Load()), op=ast.Mult(), right=ast.Constant(value=2)),
        ast.Name(id="x", ctx=ast.Load()),
        ast.Constant(value=0),
    ],
    "UnaryOp": [
        ast.UnaryOp(op=ast.USub(), operand=ast.Name(id="x", ctx=ast.Load())),
        ast.UnaryOp(op=ast.Not(), operand=ast.Name(id="x", ctx=ast.Load())),
        ast.Name(id="x", ctx=ast.Load()),
        ast.Constant(value=0),
    ],
    "Compare": [
        ast.Compare(
            left=ast.Name(id="x", ctx=ast.Load()),
            ops=[ast.Eq()],
            comparators=[ast.Constant(value=0)],
        ),
        ast.Compare(
            left=ast.Name(id="x", ctx=ast.Load()),
            ops=[ast.Gt()],
            comparators=[ast.Constant(value=0)],
        ),
        ast.Compare(
            left=ast.Name(id="x", ctx=ast.Load()),
            ops=[ast.Lt()],
            comparators=[ast.Constant(value=0)],
        ),
        ast.Constant(value=True),
        ast.Constant(value=False),
    ],
    "Call": [
        ast.Call(func=ast.Name(id="len", ctx=ast.Load()), args=[ast.Name(id="x", ctx=ast.Load())], keywords=[]),
        ast.Call(func=ast.Name(id="sum", ctx=ast.Load()), args=[ast.Name(id="x", ctx=ast.Load())], keywords=[]),
        ast.Name(id="x", ctx=ast.Load()),
        ast.Constant(value=0),
    ],
    # Statement replacements
    "Return": [
        ast.Return(value=ast.Name(id="x", ctx=ast.Load())),
        ast.Return(value=ast.Constant(value=0)),
        ast.Return(value=ast.Constant(value=None)),
        ast.Return(value=ast.Constant(value=True)),
    ],
    "Pass": [
        ast.Pass(),
        ast.Return(value=ast.Constant(value=None)),
    ],
    "Expr": [
        ast.Expr(value=ast.Name(id="x", ctx=ast.Load())),
        ast.Expr(value=ast.Constant(value=0)),
        ast.Pass(),
    ],
    # Operator replacements
    "Add": [ast.Add(), ast.Sub(), ast.Mult()],
    "Sub": [ast.Sub(), ast.Add(), ast.Mult()],
    "Mult": [ast.Mult(), ast.Add(), ast.Div()],
    "Div": [ast.Div(), ast.Mult(), ast.FloorDiv()],
    "Mod": [ast.Mod(), ast.FloorDiv(), ast.Mult()],
    "Eq": [ast.Eq(), ast.NotEq(), ast.Lt(), ast.Gt()],
    "NotEq": [ast.NotEq(), ast.Eq(), ast.Lt(), ast.Gt()],
    "Lt": [ast.Lt(), ast.LtE(), ast.Gt(), ast.Eq()],
    "Gt": [ast.Gt(), ast.GtE(), ast.Lt(), ast.Eq()],
    "LtE": [ast.LtE(), ast.Lt(), ast.GtE(), ast.Eq()],
    "GtE": [ast.GtE(), ast.Gt(), ast.LtE(), ast.Eq()],
    "And": [ast.And(), ast.Or()],
    "Or": [ast.Or(), ast.And()],
}


def get_replacement_for_node(node: ast.AST) -> ast.AST | None:
    """Get a random grammatically valid replacement for a node.

    Args:
        node: The AST node to replace

    Returns:
        A replacement node, or None if no replacement available
    """
    node_type = type(node).__name__
    if node_type not in REPLACEMENT_PATTERNS:
        return None

    candidates = REPLACEMENT_PATTERNS[node_type]
    # Filter out the original node if it matches any candidate
    valid_candidates = [c for c in candidates if not _ast_equal(c, node)]

    if not valid_candidates:
        return None

    return copy.deepcopy(random.choice(valid_candidates))


def _ast_equal(a: ast.AST, b: ast.AST) -> bool:
    """Check if two AST nodes are structurally equal."""
    if type(a) is not type(b):
        return False
    if isinstance(a, ast.Constant) and isinstance(b, ast.Constant):
        return a.value == b.value
    if isinstance(a, ast.Name) and isinstance(b, ast.Name):
        return a.id == b.id
    # For other types, just check type equality (simplified)
    return type(a) is type(b)


def mutate_ast(tree: ast.AST, config: MutationConfig | None = None) -> tuple[ast.AST, list[int]]:
    """Apply a random mutation to an AST.

    Finds candidate nodes with small subtrees and replaces one with
    a grammatically valid alternative.

    Args:
        tree: The AST to mutate
        config: Mutation configuration

    Returns:
        Tuple of (mutated_tree, [mutated_node_indices])
    """
    if config is None:
        config = MutationConfig()

    # Make a deep copy to avoid modifying the original
    tree = copy.deepcopy(tree)

    # Find all nodes with small subtrees
    candidates: list[tuple[ast.AST, ast.AST | None, str | None]] = []  # (node, parent, field_name)

    def find_candidates(node: ast.AST, parent: ast.AST | None = None, field: str | None = None):
        size = count_nodes(node)
        if size <= config.sigma_small:
            # Check if we have a replacement for this node type
            if type(node).__name__ in REPLACEMENT_PATTERNS:
                candidates.append((node, parent, field))
        for name, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        find_candidates(item, node, name)
            elif isinstance(value, ast.AST):
                find_candidates(value, node, name)

    find_candidates(tree)

    if not candidates:
        return tree, []

    # Pick a random candidate to mutate
    node, parent, field = random.choice(candidates)
    replacement = get_replacement_for_node(node)

    if replacement is None or parent is None or field is None:
        return tree, []

    # Perform the replacement
    field_value = getattr(parent, field)
    if isinstance(field_value, list):
        # Find and replace in list
        for i, item in enumerate(field_value):
            if item is node:
                field_value[i] = replacement
                break
    else:
        setattr(parent, field, replacement)

    # Fix locations
    ast.fix_missing_locations(tree)

    return tree, [0]  # Return dummy index for now


def apply_n_mutations(
    tree: ast.AST, n: int, config: MutationConfig | None = None
) -> tuple[ast.AST, list[MutationRecord]]:
    """Apply n mutations to an AST.

    Args:
        tree: The AST to mutate
        n: Number of mutations to apply
        config: Mutation configuration

    Returns:
        Tuple of (mutated_tree, list_of_mutation_records)
    """
    if config is None:
        config = MutationConfig()

    records = []
    current_tree = copy.deepcopy(tree)

    for _ in range(n):
        mutated_tree, indices = mutate_ast(current_tree, config)
        if indices:
            records.append(
                MutationRecord(
                    original_node_idx=indices[0] if indices else -1,
                    original_node_type=0,  # Would need to track properly
                    original_subtree_size=0,
                    new_node_type=0,
                    mutation_type="replace",
                )
            )
        current_tree = mutated_tree

    return current_tree, records


def corrupt_tree(code: str, num_steps: int, config: MutationConfig | None = None) -> tuple[str, str, int]:
    """Apply corruption (forward diffusion) to Python code.

    Args:
        code: Original Python code
        num_steps: Number of mutation steps
        config: Mutation configuration

    Returns:
        Tuple of (original_code, corrupted_code, actual_steps_applied)
    """
    if config is None:
        config = MutationConfig()

    tree = ast.parse(code)
    corrupted_tree, records = apply_n_mutations(tree, num_steps, config)

    try:
        corrupted_code = ast.unparse(corrupted_tree)
    except Exception:
        # If unparsing fails, return original
        return code, code, 0

    return code, corrupted_code, len(records)
