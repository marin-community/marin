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

"""
Edit path computation for tree diffusion training.

This module implements the edit path algorithm from Berkeley's tree diffusion
paper. The key insight is that training with edit paths provides cleaner
signals than naive mutation reversal.

An edit path is a sequence of single-edit operations that transform one
tree into another. The model learns to predict the first step of this path.
"""

import ast
from dataclasses import dataclass
from typing import Any

import numpy as np

from experiments.kelp.ast_utils import TreeTensors, ast_to_tensors
from experiments.kelp.python_grammar import PythonNodeVocab, PythonValueVocab


@dataclass
class EditStep:
    """A single edit step in an edit path.

    Attributes:
        node_idx: Index of the node to edit in the source tree
        edit_type: Type of edit ('replace', 'insert', 'delete')
        old_value: The original value/subtree at this position
        new_value: The new value/subtree to place here
    """

    node_idx: int
    edit_type: str  # 'replace', 'insert', 'delete'
    old_value: Any  # Could be AST node or tensor representation
    new_value: Any


@dataclass
class EditPath:
    """A sequence of edit steps transforming source -> target."""

    source_tree: ast.AST
    target_tree: ast.AST
    steps: list[EditStep]

    @property
    def num_steps(self) -> int:
        return len(self.steps)


def compute_tree_diff(source: ast.AST, target: ast.AST) -> list[tuple[str, Any, Any]]:
    """Compute the differences between two ASTs.

    This is a simplified tree diff that identifies nodes that differ
    between source and target. A full implementation would use
    tree edit distance algorithms.

    Args:
        source: Source AST
        target: Target AST

    Returns:
        List of (path, source_value, target_value) tuples for differing nodes
    """
    diffs = []

    def compare_nodes(s: ast.AST | None, t: ast.AST | None, path: str = "root"):
        if s is None and t is None:
            return
        if s is None or t is None:
            diffs.append((path, s, t))
            return
        if type(s) is not type(t):
            diffs.append((path, s, t))
            return

        # Compare node-specific values
        if isinstance(s, ast.Name) and isinstance(t, ast.Name):
            if s.id != t.id:
                diffs.append((path, s, t))
                return
        elif isinstance(s, ast.Constant) and isinstance(t, ast.Constant):
            if s.value != t.value:
                diffs.append((path, s, t))
                return

        # Recursively compare children
        s_fields = dict(ast.iter_fields(s))
        t_fields = dict(ast.iter_fields(t))

        for field_name in set(s_fields.keys()) | set(t_fields.keys()):
            s_val = s_fields.get(field_name)
            t_val = t_fields.get(field_name)

            if isinstance(s_val, list) and isinstance(t_val, list):
                for i, (si, ti) in enumerate(zip(s_val, t_val)):
                    if isinstance(si, ast.AST) and isinstance(ti, ast.AST):
                        compare_nodes(si, ti, f"{path}.{field_name}[{i}]")
                # Handle length differences
                if len(s_val) != len(t_val):
                    diffs.append((f"{path}.{field_name}", s_val, t_val))
            elif isinstance(s_val, ast.AST) and isinstance(t_val, ast.AST):
                compare_nodes(s_val, t_val, f"{path}.{field_name}")
            elif s_val != t_val:
                diffs.append((f"{path}.{field_name}", s_val, t_val))

    compare_nodes(source, target)
    return diffs


def compute_edit_path(
    corrupted: ast.AST,
    original: ast.AST,
    node_vocab: PythonNodeVocab | None = None,
    value_vocab: PythonValueVocab | None = None,
) -> EditPath:
    """Compute an edit path from corrupted tree to original tree.

    This implements the reverse direction: given a corrupted tree,
    find the edits needed to recover the original.

    Args:
        corrupted: The corrupted/noisy tree (source)
        original: The original clean tree (target)
        node_vocab: Vocabulary for node types
        value_vocab: Vocabulary for node values

    Returns:
        EditPath with steps to transform corrupted -> original
    """
    if node_vocab is None:
        node_vocab = PythonNodeVocab()
    if value_vocab is None:
        value_vocab = PythonValueVocab()

    # Find all differences
    diffs = compute_tree_diff(corrupted, original)

    # Convert diffs to edit steps
    steps = []
    for i, (path, old_val, new_val) in enumerate(diffs):
        steps.append(
            EditStep(
                node_idx=i,  # Simplified - would need proper index mapping
                edit_type="replace",
                old_value=old_val,
                new_value=new_val,
            )
        )

    return EditPath(source_tree=corrupted, target_tree=original, steps=steps)


@dataclass
class TrainingExample:
    """A single training example for tree diffusion.

    Contains the corrupted tree, target tree, and the first edit step
    the model should predict.
    """

    corrupted_tensors: TreeTensors
    target_tensors: TreeTensors
    edit_location: int  # Index of node to edit
    edit_type: int  # Type of edit (encoded)
    replacement_type: int  # Node type of replacement
    replacement_value: np.ndarray  # Value tokens for replacement
    num_corruption_steps: int  # How many steps of corruption were applied


def create_training_example(
    original_code: str,
    num_corruption_steps: int,
    node_vocab: PythonNodeVocab,
    value_vocab: PythonValueVocab,
    max_nodes: int = 256,
    max_children: int = 16,
    max_value_len: int = 32,
) -> TrainingExample | None:
    """Create a training example from Python code.

    1. Parse the original code to AST
    2. Apply num_corruption_steps mutations
    3. Compute edit path from corrupted -> original
    4. Extract the first edit step as the training target

    Args:
        original_code: Clean Python source code
        num_corruption_steps: Number of mutation steps to apply
        node_vocab: Vocabulary for node types
        value_vocab: Vocabulary for node values
        max_nodes: Maximum nodes in tensor representation
        max_children: Maximum children per node
        max_value_len: Maximum value encoding length

    Returns:
        TrainingExample or None if creation fails
    """
    from experiments.kelp.mutations import apply_n_mutations

    try:
        original_tree = ast.parse(original_code)
    except SyntaxError:
        return None

    # Apply corruption
    corrupted_tree, _records = apply_n_mutations(original_tree, num_corruption_steps)

    # Convert to tensors
    original_tensors = ast_to_tensors(original_tree, node_vocab, value_vocab, max_nodes, max_children, max_value_len)
    corrupted_tensors = ast_to_tensors(corrupted_tree, node_vocab, value_vocab, max_nodes, max_children, max_value_len)

    # Compute edit path
    edit_path = compute_edit_path(corrupted_tree, original_tree, node_vocab, value_vocab)

    if not edit_path.steps:
        # No edits needed (corruption was identity)
        # Use self-prediction as fallback
        return TrainingExample(
            corrupted_tensors=corrupted_tensors,
            target_tensors=original_tensors,
            edit_location=0,
            edit_type=0,  # 'replace'
            replacement_type=int(original_tensors.node_types[0]),
            replacement_value=original_tensors.node_values[0],
            num_corruption_steps=num_corruption_steps,
        )

    # Take the first edit step
    first_step = edit_path.steps[0]

    # Encode the replacement node
    if isinstance(first_step.new_value, ast.AST):
        replacement_type = node_vocab.encode_node(first_step.new_value)
        from experiments.kelp.ast_utils import get_node_value

        value = get_node_value(first_step.new_value)
        replacement_value = np.array(value_vocab.encode_value(value if value else "", max_value_len), dtype=np.int32)
    else:
        replacement_type = node_vocab.unk_id
        replacement_value = np.zeros(max_value_len, dtype=np.int32)

    return TrainingExample(
        corrupted_tensors=corrupted_tensors,
        target_tensors=original_tensors,
        edit_location=first_step.node_idx,
        edit_type=0,  # 'replace' encoded as 0
        replacement_type=replacement_type,
        replacement_value=replacement_value,
        num_corruption_steps=num_corruption_steps,
    )


def batch_training_examples(
    examples: list[TrainingExample],
) -> dict[str, np.ndarray]:
    """Batch a list of training examples into arrays.

    Args:
        examples: List of training examples

    Returns:
        Dictionary of batched arrays
    """
    if not examples:
        raise ValueError("Cannot batch empty list of examples")

    # Stack corrupted tensors
    corrupted_node_types = np.stack([ex.corrupted_tensors.node_types for ex in examples])
    corrupted_node_values = np.stack([ex.corrupted_tensors.node_values for ex in examples])
    corrupted_parent_indices = np.stack([ex.corrupted_tensors.parent_indices for ex in examples])
    corrupted_child_indices = np.stack([ex.corrupted_tensors.child_indices for ex in examples])
    corrupted_num_children = np.stack([ex.corrupted_tensors.num_children for ex in examples])
    corrupted_node_mask = np.stack([ex.corrupted_tensors.node_mask for ex in examples])
    corrupted_depth = np.stack([ex.corrupted_tensors.depth for ex in examples])

    # Stack target tensors
    target_node_types = np.stack([ex.target_tensors.node_types for ex in examples])
    target_node_values = np.stack([ex.target_tensors.node_values for ex in examples])
    target_node_mask = np.stack([ex.target_tensors.node_mask for ex in examples])

    # Stack labels
    edit_locations = np.array([ex.edit_location for ex in examples], dtype=np.int32)
    edit_types = np.array([ex.edit_type for ex in examples], dtype=np.int32)
    replacement_types = np.array([ex.replacement_type for ex in examples], dtype=np.int32)
    replacement_values = np.stack([ex.replacement_value for ex in examples])
    num_corruption_steps = np.array([ex.num_corruption_steps for ex in examples], dtype=np.int32)

    return {
        # Corrupted tree
        "corrupted_node_types": corrupted_node_types,
        "corrupted_node_values": corrupted_node_values,
        "corrupted_parent_indices": corrupted_parent_indices,
        "corrupted_child_indices": corrupted_child_indices,
        "corrupted_num_children": corrupted_num_children,
        "corrupted_node_mask": corrupted_node_mask,
        "corrupted_depth": corrupted_depth,
        # Target tree (for reference during training)
        "target_node_types": target_node_types,
        "target_node_values": target_node_values,
        "target_node_mask": target_node_mask,
        # Labels
        "edit_location": edit_locations,
        "edit_type": edit_types,
        "replacement_type": replacement_types,
        "replacement_value": replacement_values,
        "num_corruption_steps": num_corruption_steps,
    }
