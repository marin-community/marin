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
Dataset definitions for Kelp tree diffusion.

This module provides utilities to load Python code datasets for training
tree diffusion models. It integrates with Marin's data pipeline.
"""

import ast
import gzip
import json
import logging
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import fsspec

logger = logging.getLogger(__name__)

# Stack-Edu Python data location (from Marin's common_pile)
# This path matches the output of stackv2_edu_filtered_python ExecutorStep
STACKV2_EDU_PYTHON_GCS_PATH = "gs://marin-us-central2/documents/common_pile/stackv2_edu_filtered_python/"


@dataclass
class CodeExample:
    """A single code example from the dataset."""

    code: str
    source: str  # File path or identifier
    num_nodes: int = 0  # AST node count (computed)


def is_simple_function(code: str, max_nodes: int = 100, max_lines: int = 30) -> bool:
    """Check if code is a simple function suitable for tree diffusion.

    Criteria:
    - Must parse as valid Python
    - Should be a single function definition (optionally with docstring)
    - Not too complex (limited AST nodes)
    - Not too long (limited lines)
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    # Count nodes
    node_count = sum(1 for _ in ast.walk(tree))
    if node_count > max_nodes:
        return False

    # Check line count
    lines = code.strip().split("\n")
    if len(lines) > max_lines:
        return False

    # Check structure: should be a module with function definitions
    if not isinstance(tree, ast.Module):
        return False

    # Allow only function definitions and simple expressions
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Expr)):
            # Has imports, classes, or other complex structures
            return False

    # Should have at least one function
    has_function = any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in tree.body)
    return has_function


def extract_functions_from_code(code: str, max_nodes: int = 100) -> list[str]:
    """Extract individual function definitions from code.

    Returns a list of function source code strings.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    functions = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            try:
                # Get the function source
                func_code = ast.unparse(node)
                func_tree = ast.parse(func_code)
                node_count = sum(1 for _ in ast.walk(func_tree))

                if node_count <= max_nodes:
                    functions.append(func_code)
            except Exception:
                continue

    return functions


def load_jsonl_gz(path: str) -> Iterator[dict]:
    """Load a gzipped JSONL file."""
    with fsspec.open(path, "rb") as f:
        with gzip.open(f, "rt", encoding="utf-8") as gz:
            for line in gz:
                if line.strip():
                    yield json.loads(line)


def load_jsonl(path: str) -> Iterator[dict]:
    """Load a JSONL file (gzipped or not)."""
    if path.endswith(".gz"):
        yield from load_jsonl_gz(path)
    else:
        with fsspec.open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def stream_stackv2_edu_python(
    gcs_path: str = STACKV2_EDU_PYTHON_GCS_PATH,
    max_files: int | None = None,
    shuffle_files: bool = True,
) -> Iterator[CodeExample]:
    """Stream Python code examples from Stack-Edu dataset.

    Args:
        gcs_path: GCS path to the filtered Python dataset
        max_files: Maximum number of files to process (None for all)
        shuffle_files: Whether to shuffle file order

    Yields:
        CodeExample objects
    """
    fs = fsspec.filesystem("gcs")

    # List all JSONL files
    try:
        files = fs.glob(f"{gcs_path.rstrip('/')}/*.json.gz")
        if not files:
            files = fs.glob(f"{gcs_path.rstrip('/')}/**/*.json.gz")
    except Exception as e:
        logger.warning(f"Could not list files at {gcs_path}: {e}")
        return

    if shuffle_files:
        random.shuffle(files)

    if max_files is not None:
        files = files[:max_files]

    logger.info(f"Found {len(files)} files in {gcs_path}")

    for file_path in files:
        full_path = f"gs://{file_path}" if not file_path.startswith("gs://") else file_path
        logger.debug(f"Processing {full_path}")

        try:
            for doc in load_jsonl(full_path):
                code = doc.get("text", doc.get("content", ""))
                source = doc.get("id", doc.get("source", full_path))

                if code:
                    yield CodeExample(code=code, source=source)
        except Exception as e:
            logger.warning(f"Error processing {full_path}: {e}")
            continue


def stream_simple_functions(
    gcs_path: str = STACKV2_EDU_PYTHON_GCS_PATH,
    max_files: int | None = None,
    max_nodes: int = 80,
    max_lines: int = 25,
) -> Iterator[str]:
    """Stream simple Python functions from Stack-Edu.

    Filters for functions suitable for tree diffusion training.

    Args:
        gcs_path: GCS path to dataset
        max_files: Max files to process
        max_nodes: Maximum AST nodes per function
        max_lines: Maximum lines per function

    Yields:
        Function source code strings
    """
    for example in stream_stackv2_edu_python(gcs_path, max_files):
        # Try to extract individual functions
        functions = extract_functions_from_code(example.code, max_nodes)

        for func in functions:
            if is_simple_function(func, max_nodes, max_lines):
                yield func


def load_local_python_files(directory: str, max_nodes: int = 80) -> Iterator[str]:
    """Load Python functions from local directory.

    Useful for testing without GCS access.

    Args:
        directory: Path to directory with .py files
        max_nodes: Maximum AST nodes per function

    Yields:
        Function source code strings
    """
    path = Path(directory)
    for py_file in path.rglob("*.py"):
        try:
            code = py_file.read_text(encoding="utf-8")
            functions = extract_functions_from_code(code, max_nodes)
            for func in functions:
                if is_simple_function(func, max_nodes):
                    yield func
        except Exception as e:
            logger.debug(f"Error processing {py_file}: {e}")
            continue


class StackEduPythonDataset:
    """Dataset wrapper for Stack-Edu Python with caching.

    Loads functions into memory for efficient random access during training.
    """

    def __init__(
        self,
        gcs_path: str = STACKV2_EDU_PYTHON_GCS_PATH,
        max_files: int | None = 10,  # Start small
        max_functions: int = 10000,
        max_nodes: int = 80,
        max_lines: int = 25,
        cache_path: str | None = None,
    ):
        self.gcs_path = gcs_path
        self.max_files = max_files
        self.max_functions = max_functions
        self.max_nodes = max_nodes
        self.max_lines = max_lines
        self.cache_path = cache_path
        self.functions: list[str] = []
        self._loaded = False

    def load(self) -> None:
        """Load functions into memory."""
        if self._loaded:
            return

        # Try to load from cache
        if self.cache_path and Path(self.cache_path).exists():
            logger.info(f"Loading cached functions from {self.cache_path}")
            with open(self.cache_path, "r") as f:
                self.functions = json.load(f)
            self._loaded = True
            logger.info(f"Loaded {len(self.functions)} functions from cache")
            return

        # Load from GCS
        logger.info(f"Loading functions from {self.gcs_path}")
        count = 0
        for func in stream_simple_functions(
            self.gcs_path,
            max_files=self.max_files,
            max_nodes=self.max_nodes,
            max_lines=self.max_lines,
        ):
            self.functions.append(func)
            count += 1
            if count >= self.max_functions:
                break
            if count % 1000 == 0:
                logger.info(f"Loaded {count} functions...")

        self._loaded = True
        logger.info(f"Loaded {len(self.functions)} total functions")

        # Save to cache
        if self.cache_path:
            logger.info(f"Caching functions to {self.cache_path}")
            Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "w") as f:
                json.dump(self.functions, f)

    def __len__(self) -> int:
        if not self._loaded:
            self.load()
        return len(self.functions)

    def __getitem__(self, idx: int) -> str:
        if not self._loaded:
            self.load()
        return self.functions[idx]

    def sample(self, n: int = 1) -> list[str]:
        """Sample n random functions."""
        if not self._loaded:
            self.load()
        return random.choices(self.functions, k=n)

    def get_programs(self) -> list[str]:
        """Get all programs (for compatibility with toy_dataset)."""
        if not self._loaded:
            self.load()
        return self.functions


# Convenience function for quick loading
def load_stackv2_edu_python(
    max_files: int = 5,
    max_functions: int = 5000,
    cache_path: str | None = None,
) -> StackEduPythonDataset:
    """Load Stack-Edu Python dataset.

    Args:
        max_files: Maximum GCS files to process
        max_functions: Maximum functions to load
        cache_path: Optional path to cache loaded functions

    Returns:
        StackEduPythonDataset instance (call .load() or access items to load)
    """
    dataset = StackEduPythonDataset(
        max_files=max_files,
        max_functions=max_functions,
        cache_path=cache_path,
    )
    return dataset
