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

"""Quine dataset: Extract Python functions from the Marin codebase.

This dataset uses the Marin repository itself as training data, allowing
the model to learn from real-world Python code.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import Array

from experiments.kelp.tree.parser import PythonTreeParser, extract_functions

logger = logging.getLogger(__name__)


@dataclass
class QuineExample:
    """A code example extracted from the Marin codebase."""

    name: str
    docstring: str
    signature: str
    body: str
    source_file: str

    @property
    def full_code(self) -> str:
        """Get the complete function code."""
        return f'{self.signature}\n    """{self.docstring}"""\n{self.body}'

    @property
    def prompt(self) -> str:
        """Get the conditioning prompt."""
        return f'"""{self.docstring}"""\n{self.signature}'


def find_python_files(root_dir: str, exclude_dirs: list[str] | None = None) -> list[Path]:
    """Find all Python files in a directory tree.

    Args:
        root_dir: Root directory to search.
        exclude_dirs: Directory names to exclude.

    Returns:
        List of paths to Python files.
    """
    if exclude_dirs is None:
        exclude_dirs = [".git", "__pycache__", ".venv", "venv", "node_modules", ".eggs"]

    python_files = []
    root = Path(root_dir)

    for path in root.rglob("*.py"):
        if any(excluded in path.parts for excluded in exclude_dirs):
            continue
        python_files.append(path)

    return python_files


def extract_from_file(file_path: Path) -> list[QuineExample]:
    """Extract functions with docstrings from a Python file.

    Args:
        file_path: Path to Python file.

    Returns:
        List of QuineExample instances.
    """
    try:
        code = file_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.debug(f"Could not read {file_path}: {e}")
        return []

    functions = extract_functions(code)
    examples = []

    for func in functions:
        if not func["docstring"]:
            continue

        examples.append(
            QuineExample(
                name=func["name"],
                docstring=func["docstring"],
                signature=func["signature"],
                body=func["body"],
                source_file=str(file_path),
            )
        )

    return examples


def extract_from_directory(root_dir: str, min_docstring_len: int = 10) -> list[QuineExample]:
    """Extract all functions with docstrings from a directory.

    Args:
        root_dir: Root directory to search.
        min_docstring_len: Minimum docstring length to include.

    Returns:
        List of QuineExample instances.
    """
    python_files = find_python_files(root_dir)
    logger.info(f"Found {len(python_files)} Python files in {root_dir}")

    all_examples = []
    for file_path in python_files:
        examples = extract_from_file(file_path)
        for ex in examples:
            if len(ex.docstring) >= min_docstring_len:
                all_examples.append(ex)

    logger.info(f"Extracted {len(all_examples)} functions with docstrings")
    return all_examples


def get_marin_root() -> Path:
    """Get the Marin repository root directory."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "CLAUDE.md").exists() or (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not find Marin repository root")


def create_quine_dataset() -> list[dict]:
    """Create the quine dataset from the Marin codebase.

    Returns:
        List of dicts with 'prompt', 'code', 'docstring', 'signature', 'body', 'source_file' keys.
    """
    root = get_marin_root()
    examples = extract_from_directory(str(root))

    return [
        {
            "prompt": ex.prompt,
            "code": ex.full_code,
            "docstring": ex.docstring,
            "signature": ex.signature,
            "body": ex.body,
            "source_file": ex.source_file,
        }
        for ex in examples
    ]


class SimpleTokenizer:
    """Simple tokenizer for quine dataset (same as toy_dataset)."""

    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.mask_token_id = vocab_size - 1
        self.unk_token_id = 1

    def encode(self, text: str) -> list[int]:
        ids = []
        for c in text:
            code = ord(c)
            if code < self.vocab_size - 2:
                ids.append(code + 2)
            else:
                ids.append(self.unk_token_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        chars = []
        for i in ids:
            if i == self.pad_token_id or i == self.mask_token_id:
                continue
            if i == self.unk_token_id:
                chars.append("?")
            elif i >= 2:
                chars.append(chr(i - 2))
        return "".join(chars)


def create_quine_data_iter(
    batch_size: int,
    max_seq_len: int,
    key: jax.Array,
    vocab_size: int = 256,
) -> Iterator[dict[str, Array]]:
    """Create an iterator over quine dataset batches.

    Args:
        batch_size: Batch size.
        max_seq_len: Maximum sequence length.
        key: PRNG key.
        vocab_size: Vocabulary size for tokenizer.

    Yields:
        Batches with 'tokens' and 'prefix_len' keys.
    """
    tokenizer = SimpleTokenizer(vocab_size=vocab_size)
    dataset = create_quine_dataset()

    encoded_examples = []
    for item in dataset:
        prompt_ids = tokenizer.encode(item["prompt"])
        code_ids = tokenizer.encode(item["code"])

        if len(code_ids) <= max_seq_len:
            encoded_examples.append({
                "tokens": code_ids,
                "prefix_len": min(len(prompt_ids), max_seq_len),
            })

    logger.info(f"Created {len(encoded_examples)} examples from quine dataset")

    if not encoded_examples:
        raise ValueError("No valid examples found in quine dataset")

    num_examples = len(encoded_examples)

    while True:
        key, shuffle_key = random.split(key)
        indices = random.permutation(shuffle_key, jnp.arange(num_examples))

        for i in range(0, num_examples, batch_size):
            batch_indices = indices[i : i + batch_size]

            if len(batch_indices) < batch_size:
                key, extra_key = random.split(key)
                extra_indices = random.randint(extra_key, (batch_size - len(batch_indices),), 0, num_examples)
                batch_indices = jnp.concatenate([batch_indices, extra_indices])

            batch_tokens = []
            batch_prefix_lens = []

            for idx in batch_indices:
                example = encoded_examples[int(idx)]
                tokens = example["tokens"]

                padded = tokens + [tokenizer.pad_token_id] * (max_seq_len - len(tokens))
                padded = padded[:max_seq_len]

                batch_tokens.append(padded)
                batch_prefix_lens.append(example["prefix_len"])

            yield {
                "tokens": jnp.array(batch_tokens),
                "prefix_len": jnp.array(batch_prefix_lens),
            }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    dataset = create_quine_dataset()
    print(f"Extracted {len(dataset)} examples")

    for i, example in enumerate(dataset[:5]):
        print(f"\n--- Example {i + 1} ---")
        print(f"Source: {example['source_file']}")
        print(f"Docstring: {example['docstring'][:100]}...")
        print(f"Signature: {example['signature']}")
