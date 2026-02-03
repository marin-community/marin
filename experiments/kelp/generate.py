#!/usr/bin/env python3
# Copyright 2026 The Marin Authors
from __future__ import annotations
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
Code generation using Kelp tree diffusion.

This module implements the reverse diffusion process for generating
Python code from conditioning (signature + docstring).

Generation process:
1. Start with a minimal/noisy AST
2. Iteratively denoise by predicting and applying edits
3. Stop when no more edits predicted or max steps reached
4. Convert final AST to code

Unlike AR generation which produces tokens left-to-right, tree diffusion
can edit any part of the tree at each step, enabling parallel refinement.
"""

import ast
import logging
from dataclasses import dataclass

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

import haliax as hax

from experiments.kelp.ast_utils import (
    TreeTensors,
    parse_python_to_tensors,
    tensors_to_code,
)
from experiments.kelp.conditioning import (
    create_condition_mask,
    tokenize_condition,
)
from experiments.kelp.python_grammar import PythonNodeVocab, PythonValueVocab
from experiments.kelp.tree_diffusion import TreeDiffusionConfig, TreeDiffusionModel

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for code generation."""

    # Generation steps
    max_steps: int = 20
    min_steps: int = 1

    # Sampling
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95

    # Early stopping
    stop_on_no_change: bool = True
    max_no_change_steps: int = 3

    # Output
    return_intermediate: bool = False


def create_initial_tree(
    signature: str,
    node_vocab: PythonNodeVocab,
    value_vocab: PythonValueVocab,
    max_nodes: int = 128,
    max_children: int = 16,
    max_value_len: int = 32,
) -> TreeTensors | None:
    """Create an initial AST for generation.

    Starts with the function signature and a minimal body (just 'pass').

    Args:
        signature: Function signature (e.g., "def add(a, b):")
        node_vocab: Node vocabulary
        value_vocab: Value vocabulary
        max_nodes: Maximum nodes in tree

    Returns:
        TreeTensors for the initial tree
    """
    # Create minimal function with signature
    # Remove trailing colon if present
    sig = signature.rstrip(":")

    initial_code = f"{sig}:\n    pass"

    try:
        return parse_python_to_tensors(
            initial_code,
            node_vocab,
            value_vocab,
            max_nodes=max_nodes,
            max_children=max_children,
            max_value_len=max_value_len,
        )
    except Exception as e:
        logger.warning(f"Failed to create initial tree: {e}")
        return None


def sample_from_logits(
    logits: jnp.ndarray,
    key: jrandom.PRNGKey,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
) -> int:
    """Sample from logits with temperature and top-k/top-p filtering.

    Args:
        logits: Logits array
        key: Random key
        temperature: Sampling temperature
        top_k: Top-k filtering (0 to disable)
        top_p: Top-p (nucleus) filtering (1.0 to disable)

    Returns:
        Sampled index
    """
    # Apply temperature
    if temperature != 1.0:
        logits = logits / temperature

    # Convert to numpy for easier manipulation
    logits = np.array(logits)

    # Top-k filtering
    if top_k > 0 and top_k < len(logits):
        indices_to_remove = logits < np.partition(logits, -top_k)[-top_k]
        logits[indices_to_remove] = -float("inf")

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_indices = np.argsort(logits)[::-1]
        sorted_logits = logits[sorted_indices]
        cumulative_probs = np.cumsum(np.exp(sorted_logits - np.max(sorted_logits)))
        cumulative_probs = cumulative_probs / cumulative_probs[-1]

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift to keep at least one token
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
        sorted_indices_to_remove[0] = False

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float("inf")

    # Sample
    probs = np.exp(logits - np.max(logits))
    probs = probs / probs.sum()

    key_np = int(jrandom.randint(key, (), 0, 2**31))
    rng = np.random.RandomState(key_np)
    return rng.choice(len(probs), p=probs)


def apply_edit_to_tree(
    tree: TreeTensors,
    edit_location: int,
    replacement_type: int,
    replacement_value: np.ndarray,
    node_vocab: PythonNodeVocab,
    value_vocab: PythonValueVocab,
) -> TreeTensors:
    """Apply an edit to the tree tensors.

    This modifies the tree at the specified location with the new node.

    Note: This is a simplified implementation that just replaces the node
    type and value. A full implementation would handle structural changes.

    Args:
        tree: Current tree tensors
        edit_location: Index of node to edit
        replacement_type: New node type ID
        replacement_value: New value token IDs
        node_vocab: Node vocabulary
        value_vocab: Value vocabulary

    Returns:
        Modified TreeTensors
    """
    # Create copies of arrays
    new_node_types = tree.node_types.copy()
    new_node_values = tree.node_values.copy()

    # Apply edit
    new_node_types[edit_location] = replacement_type
    new_node_values[edit_location] = replacement_value

    return TreeTensors(
        node_types=new_node_types,
        node_values=new_node_values,
        parent_indices=tree.parent_indices,
        child_indices=tree.child_indices,
        node_mask=tree.node_mask,
        depth=tree.depth,
    )


def generate_step(
    model: TreeDiffusionModel,
    tree: TreeTensors,
    condition_tokens: np.ndarray,
    key: jrandom.PRNGKey,
    config: GenerationConfig,
) -> tuple[TreeTensors, int, int, np.ndarray]:
    """Perform one generation step.

    Args:
        model: TreeDiffusionModel
        tree: Current tree state
        condition_tokens: Conditioning token IDs
        key: Random key
        config: Generation config

    Returns:
        (new_tree, edit_location, replacement_type, replacement_value)
    """
    model_config = model.config
    k1, k2, k3 = jrandom.split(key, 3)

    # Convert to NamedArrays
    node_types = hax.named(jnp.array(tree.node_types), (model_config.Nodes,))
    node_values = hax.named(jnp.array(tree.node_values), (model_config.Nodes, model_config.ValueLen))
    depth = hax.named(jnp.array(tree.depth), (model_config.Nodes,))
    mask = hax.named(jnp.array(tree.node_mask), (model_config.Nodes,))

    # Conditioning
    cond_tokens = hax.named(jnp.array(condition_tokens), (model_config.CondLen,))
    cond_mask_arr = create_condition_mask(condition_tokens)
    cond_mask = hax.named(cond_mask_arr, (model_config.CondLen,))

    # Forward pass
    location_logits, type_logits, value_logits = model(
        node_types,
        node_values,
        depth,
        mask=mask,
        condition_tokens=cond_tokens,
        condition_mask=cond_mask,
        key=None,
    )

    # Sample edit location
    loc_logits = np.array(location_logits.array)
    # Mask invalid locations
    loc_logits[tree.node_mask == 0] = -float("inf")
    edit_location = sample_from_logits(
        loc_logits, k1, config.temperature, config.top_k, config.top_p
    )

    # Sample replacement type
    type_logits_arr = np.array(type_logits.array)
    replacement_type = sample_from_logits(
        type_logits_arr, k2, config.temperature, config.top_k, config.top_p
    )

    # Sample replacement value (for each position)
    value_logits_arr = np.array(value_logits.array)  # (ValueLen, ValueVocab)
    replacement_value = np.zeros(model_config.max_value_len, dtype=np.int32)
    for i in range(model_config.max_value_len):
        k3, k_pos = jrandom.split(k3)
        replacement_value[i] = sample_from_logits(
            value_logits_arr[i], k_pos, config.temperature, config.top_k, config.top_p
        )

    return tree, edit_location, replacement_type, replacement_value


def generate_code(
    model: TreeDiffusionModel,
    condition_text: str,
    signature: str,
    node_vocab: PythonNodeVocab,
    value_vocab: PythonValueVocab,
    config: GenerationConfig | None = None,
    key: jrandom.PRNGKey | None = None,
) -> tuple[str | None, list[str]]:
    """Generate code using tree diffusion.

    Args:
        model: Trained TreeDiffusionModel
        condition_text: Conditioning text (signature + docstring)
        signature: Just the function signature
        node_vocab: Node vocabulary
        value_vocab: Value vocabulary
        config: Generation configuration
        key: Random key

    Returns:
        (generated_code, intermediate_codes) tuple
    """
    if config is None:
        config = GenerationConfig()

    if key is None:
        key = jrandom.PRNGKey(42)

    model_config = model.config

    # Tokenize conditioning
    condition_tokens = tokenize_condition(
        condition_text,
        max_len=model_config.max_condition_len,
    )

    # Create initial tree
    tree = create_initial_tree(
        signature,
        node_vocab,
        value_vocab,
        max_nodes=model_config.max_nodes,
        max_children=model_config.max_children,
        max_value_len=model_config.max_value_len,
    )

    if tree is None:
        return None, []

    intermediate_codes = []
    no_change_count = 0
    prev_code = None

    for step in range(config.max_steps):
        key, step_key = jrandom.split(key)

        # Generate step
        tree, edit_loc, repl_type, repl_value = generate_step(
            model, tree, condition_tokens, step_key, config
        )

        # Apply edit
        tree = apply_edit_to_tree(
            tree, edit_loc, repl_type, repl_value, node_vocab, value_vocab
        )

        # Convert to code
        try:
            current_code = tensors_to_code(tree, node_vocab, value_vocab)
        except Exception:
            current_code = None

        if config.return_intermediate and current_code:
            intermediate_codes.append(current_code)

        # Check for early stopping
        if config.stop_on_no_change:
            if current_code == prev_code:
                no_change_count += 1
                if no_change_count >= config.max_no_change_steps:
                    logger.debug(f"Early stopping at step {step} (no change)")
                    break
            else:
                no_change_count = 0

        prev_code = current_code

    # Final conversion
    try:
        final_code = tensors_to_code(tree, node_vocab, value_vocab)
        return final_code, intermediate_codes
    except Exception as e:
        logger.warning(f"Failed to convert final tree to code: {e}")
        return None, intermediate_codes


def generate_function_body(
    model: TreeDiffusionModel,
    prompt: str,
    node_vocab: PythonNodeVocab,
    value_vocab: PythonValueVocab,
    config: GenerationConfig | None = None,
    key: jrandom.PRNGKey | None = None,
) -> str | None:
    """Generate a function body given a HumanEval-style prompt.

    This is a convenience function for evaluation.

    Args:
        model: Trained TreeDiffusionModel
        prompt: HumanEval prompt (signature + docstring)
        node_vocab: Node vocabulary
        value_vocab: Value vocabulary
        config: Generation configuration
        key: Random key

    Returns:
        Generated function body (indented), or None if failed
    """
    # Extract signature and docstring
    lines = prompt.strip().split("\n")
    signature = lines[0].strip()

    # Build condition text
    condition_text = prompt.strip()

    # Generate
    code, _ = generate_code(
        model,
        condition_text,
        signature,
        node_vocab,
        value_vocab,
        config,
        key,
    )

    if code is None:
        return None

    # Extract just the body (remove signature)
    try:
        tree = ast.parse(code)
        if tree.body and isinstance(tree.body[0], ast.FunctionDef):
            func = tree.body[0]
            # Get body statements (skip docstring if present)
            body = func.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                if isinstance(body[0].value.value, str):
                    body = body[1:]  # Skip docstring

            if not body:
                return "    pass"

            # Unparse body
            body_lines = [ast.unparse(stmt) for stmt in body]
            return "\n".join(f"    {line}" for line in body_lines)
    except Exception:
        pass

    return None


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    print("Testing code generation...")

    node_vocab = PythonNodeVocab()
    value_vocab = PythonValueVocab()

    # Create a simple model for testing
    from experiments.kelp.conditioning import CONDITION_VOCAB_SIZE

    config = TreeDiffusionConfig(
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        mlp_dim=256,
        max_nodes=64,
        node_vocab_size=node_vocab.vocab_size,
        value_vocab_size=value_vocab.vocab_size,
        use_conditioning=True,
        condition_vocab_size=CONDITION_VOCAB_SIZE,
        max_condition_len=64,
    )

    key = jrandom.PRNGKey(42)
    model = TreeDiffusionModel.init(config, key=key)

    # Test generation
    prompt = '''def add(a, b):
    """Add two numbers and return the sum."""'''

    gen_config = GenerationConfig(max_steps=5)
    body = generate_function_body(
        model, prompt, node_vocab, value_vocab, gen_config, key
    )

    print(f"Generated body: {body}")
