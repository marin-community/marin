#!/usr/bin/env python3
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
Conditioning utilities for Kelp tree diffusion.

This module provides utilities to extract and encode conditioning information
from Python code for conditional tree diffusion generation.

Conditioning sources:
1. Docstrings - function documentation
2. Signatures - function name, arguments, type hints
3. Comments - preceding comments

The conditioning is encoded as text and embedded using a simple tokenizer
or optionally using Marin 8B embeddings.
"""

import ast
import logging
import re
from dataclasses import dataclass

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FunctionCondition:
    """Conditioning information extracted from a Python function."""

    # Full source code
    code: str

    # Extracted components
    name: str
    signature: str  # "def name(args) -> return_type:"
    docstring: str | None
    body_code: str  # Code without docstring

    # Combined conditioning text
    condition_text: str

    @property
    def has_docstring(self) -> bool:
        return self.docstring is not None and len(self.docstring.strip()) > 0


def extract_function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Extract the function signature as a string.

    Returns something like: "def add(a: int, b: int) -> int:"
    """
    # Build argument string
    args = []

    # Regular arguments
    for arg in node.args.args:
        arg_str = arg.arg
        if arg.annotation:
            try:
                arg_str += f": {ast.unparse(arg.annotation)}"
            except Exception:
                pass
        args.append(arg_str)

    # *args
    if node.args.vararg:
        vararg = f"*{node.args.vararg.arg}"
        if node.args.vararg.annotation:
            try:
                vararg += f": {ast.unparse(node.args.vararg.annotation)}"
            except Exception:
                pass
        args.append(vararg)

    # **kwargs
    if node.args.kwarg:
        kwarg = f"**{node.args.kwarg.arg}"
        if node.args.kwarg.annotation:
            try:
                kwarg += f": {ast.unparse(node.args.kwarg.annotation)}"
            except Exception:
                pass
        args.append(kwarg)

    args_str = ", ".join(args)

    # Build signature
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    sig = f"{prefix} {node.name}({args_str})"

    # Add return type if present
    if node.returns:
        try:
            sig += f" -> {ast.unparse(node.returns)}"
        except Exception:
            pass

    sig += ":"

    return sig


def extract_docstring(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    """Extract docstring from a function node."""
    if not node.body:
        return None

    first_stmt = node.body[0]
    if isinstance(first_stmt, ast.Expr) and isinstance(first_stmt.value, ast.Constant):
        value = first_stmt.value.value
        if isinstance(value, str):
            return value.strip()

    return None


def extract_body_without_docstring(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Extract function body code without the docstring."""
    if not node.body:
        return "pass"

    # Check if first statement is docstring
    body = node.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        if isinstance(body[0].value.value, str):
            body = body[1:]  # Skip docstring

    if not body:
        return "pass"

    # Unparse the body statements
    try:
        body_lines = [ast.unparse(stmt) for stmt in body]
        return "\n    ".join(body_lines)
    except Exception:
        return "pass"


def extract_function_condition(code: str) -> FunctionCondition | None:
    """Extract conditioning information from a Python function.

    Args:
        code: Python source code containing a function definition

    Returns:
        FunctionCondition with extracted information, or None if extraction fails
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    # Find the first function definition
    func_node = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_node = node
            break

    if func_node is None:
        return None

    # Extract components
    name = func_node.name
    signature = extract_function_signature(func_node)
    docstring = extract_docstring(func_node)
    body_code = extract_body_without_docstring(func_node)

    # Build conditioning text
    # Format: "signature\ndocstring" or just "signature" if no docstring
    if docstring:
        # Clean up docstring - take first paragraph or first N chars
        doc_lines = docstring.split("\n\n")[0]  # First paragraph
        doc_clean = " ".join(doc_lines.split())  # Normalize whitespace
        if len(doc_clean) > 200:
            doc_clean = doc_clean[:200] + "..."
        condition_text = f"{signature}\n\"\"\"{doc_clean}\"\"\""
    else:
        condition_text = signature

    return FunctionCondition(
        code=code,
        name=name,
        signature=signature,
        docstring=docstring,
        body_code=body_code,
        condition_text=condition_text,
    )


def extract_conditions_from_programs(programs: list[str]) -> list[FunctionCondition]:
    """Extract conditioning information from a list of programs.

    Filters to only include functions with docstrings for better conditioning.
    """
    conditions = []
    for code in programs:
        cond = extract_function_condition(code)
        if cond is not None:
            conditions.append(cond)
    return conditions


# ============================================================
# Simple character-level tokenizer for conditioning text
# ============================================================

# Special tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"

# Build vocabulary from printable ASCII + common unicode
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
PRINTABLE_CHARS = [chr(i) for i in range(32, 127)]  # Space to ~
EXTRA_CHARS = ["\n", "\t", "→", "…"]  # Common extras

CONDITION_VOCAB = SPECIAL_TOKENS + PRINTABLE_CHARS + EXTRA_CHARS
CONDITION_VOCAB_SIZE = len(CONDITION_VOCAB)

CHAR_TO_ID = {c: i for i, c in enumerate(CONDITION_VOCAB)}
ID_TO_CHAR = {i: c for i, c in enumerate(CONDITION_VOCAB)}


def tokenize_condition(
    text: str,
    max_len: int = 128,
    add_special_tokens: bool = True,
) -> np.ndarray:
    """Tokenize conditioning text to character IDs.

    Args:
        text: Conditioning text (e.g., signature + docstring)
        max_len: Maximum sequence length
        add_special_tokens: Whether to add BOS/EOS tokens

    Returns:
        Array of token IDs, padded to max_len
    """
    pad_id = CHAR_TO_ID[PAD_TOKEN]
    unk_id = CHAR_TO_ID[UNK_TOKEN]
    bos_id = CHAR_TO_ID[BOS_TOKEN]
    eos_id = CHAR_TO_ID[EOS_TOKEN]

    # Tokenize characters
    tokens = []
    if add_special_tokens:
        tokens.append(bos_id)

    for char in text:
        token_id = CHAR_TO_ID.get(char, unk_id)
        tokens.append(token_id)
        if len(tokens) >= max_len - 1:  # Leave room for EOS
            break

    if add_special_tokens:
        tokens.append(eos_id)

    # Pad to max_len
    while len(tokens) < max_len:
        tokens.append(pad_id)

    return np.array(tokens[:max_len], dtype=np.int32)


def detokenize_condition(token_ids: np.ndarray) -> str:
    """Convert token IDs back to text."""
    chars = []
    for tid in token_ids:
        tid = int(tid)
        if tid in ID_TO_CHAR:
            char = ID_TO_CHAR[tid]
            if char in SPECIAL_TOKENS:
                continue
            chars.append(char)
    return "".join(chars)


# ============================================================
# Conditioning encoder using learned embeddings
# ============================================================


class ConditioningConfig:
    """Configuration for conditioning encoder."""

    def __init__(
        self,
        vocab_size: int = CONDITION_VOCAB_SIZE,
        max_len: int = 128,
        embed_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
    ):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads


def create_condition_embeddings(
    condition_tokens: jnp.ndarray,
    embed_weights: jnp.ndarray,
    position_weights: jnp.ndarray,
) -> jnp.ndarray:
    """Create embeddings for conditioning tokens.

    Args:
        condition_tokens: Token IDs (batch, max_len) or (max_len,)
        embed_weights: Token embedding weights (vocab_size, embed_dim)
        position_weights: Position embedding weights (max_len, embed_dim)

    Returns:
        Embeddings (batch, max_len, embed_dim) or (max_len, embed_dim)
    """
    # Token embeddings
    token_embeds = embed_weights[condition_tokens]

    # Position embeddings
    if condition_tokens.ndim == 1:
        positions = jnp.arange(condition_tokens.shape[0])
        pos_embeds = position_weights[positions]
    else:
        positions = jnp.arange(condition_tokens.shape[1])
        pos_embeds = position_weights[positions]

    return token_embeds + pos_embeds


def create_condition_mask(condition_tokens: jnp.ndarray) -> jnp.ndarray:
    """Create attention mask for conditioning (1 for valid, 0 for padding)."""
    pad_id = CHAR_TO_ID[PAD_TOKEN]
    return (condition_tokens != pad_id).astype(jnp.float32)


# ============================================================
# Utility functions for training
# ============================================================


@dataclass
class ConditionalTrainingExample:
    """A training example with conditioning."""

    # Conditioning
    condition_text: str
    condition_tokens: np.ndarray  # (max_cond_len,)

    # Target (from edit_path.TrainingExample)
    corrupted_tensors: "TreeTensors"  # noqa: F821
    edit_location: int
    replacement_type: int
    replacement_value: np.ndarray

    # Original function info
    function_name: str
    has_docstring: bool


def create_conditional_training_example(
    code: str,
    num_corruption_steps: int,
    node_vocab: "PythonNodeVocab",  # noqa: F821
    value_vocab: "PythonValueVocab",  # noqa: F821
    max_nodes: int = 128,
    max_children: int = 16,
    max_value_len: int = 32,
    max_cond_len: int = 128,
) -> ConditionalTrainingExample | None:
    """Create a conditional training example.

    Extracts conditioning from the function's signature/docstring,
    then creates the tree diffusion training example.
    """
    from experiments.kelp.edit_path import create_training_example

    # Extract conditioning
    condition = extract_function_condition(code)
    if condition is None:
        return None

    # Create base training example
    base_example = create_training_example(
        code,
        num_corruption_steps,
        node_vocab,
        value_vocab,
        max_nodes=max_nodes,
        max_children=max_children,
        max_value_len=max_value_len,
    )
    if base_example is None:
        return None

    # Tokenize conditioning
    condition_tokens = tokenize_condition(condition.condition_text, max_len=max_cond_len)

    return ConditionalTrainingExample(
        condition_text=condition.condition_text,
        condition_tokens=condition_tokens,
        corrupted_tensors=base_example.corrupted_tensors,
        edit_location=base_example.edit_location,
        replacement_type=base_example.replacement_type,
        replacement_value=base_example.replacement_value,
        function_name=condition.name,
        has_docstring=condition.has_docstring,
    )


# ============================================================
# Statistics and analysis
# ============================================================


def analyze_conditioning_coverage(programs: list[str]) -> dict:
    """Analyze how many programs have good conditioning information."""
    total = len(programs)
    with_docstring = 0
    with_type_hints = 0
    avg_docstring_len = []

    for code in programs:
        cond = extract_function_condition(code)
        if cond is None:
            continue

        if cond.has_docstring:
            with_docstring += 1
            avg_docstring_len.append(len(cond.docstring))

        # Check for type hints
        if "->" in cond.signature or ": " in cond.signature:
            with_type_hints += 1

    return {
        "total_programs": total,
        "with_docstring": with_docstring,
        "docstring_rate": with_docstring / total if total > 0 else 0,
        "with_type_hints": with_type_hints,
        "type_hint_rate": with_type_hints / total if total > 0 else 0,
        "avg_docstring_len": np.mean(avg_docstring_len) if avg_docstring_len else 0,
    }


if __name__ == "__main__":
    # Quick test
    test_code = '''
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together.

    Args:
        a: First number
        b: Second number

    Returns:
        The sum of a and b
    """
    return a + b
'''

    print("Testing conditioning extraction...")
    cond = extract_function_condition(test_code)
    if cond:
        print(f"Name: {cond.name}")
        print(f"Signature: {cond.signature}")
        print(f"Docstring: {cond.docstring[:50]}..." if cond.docstring else "No docstring")
        print(f"Condition text:\n{cond.condition_text}")
        print()

        # Test tokenization
        tokens = tokenize_condition(cond.condition_text)
        print(f"Tokenized shape: {tokens.shape}")
        print(f"First 20 tokens: {tokens[:20]}")

        # Round-trip
        decoded = detokenize_condition(tokens)
        print(f"Decoded: {decoded[:50]}...")

    # Test on toy dataset
    from experiments.kelp.toy_dataset import get_toy_programs

    programs = get_toy_programs()
    stats = analyze_conditioning_coverage(programs)
    print(f"\nToy dataset conditioning coverage:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
