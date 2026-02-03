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

"""Tests for Kelp conditioning utilities."""

import numpy as np
import pytest

from experiments.kelp.conditioning import (
    CONDITION_VOCAB_SIZE,
    FunctionCondition,
    analyze_conditioning_coverage,
    create_condition_mask,
    detokenize_condition,
    extract_function_condition,
    extract_function_signature,
    tokenize_condition,
)


class TestFunctionConditionExtraction:
    def test_extract_simple_function(self):
        code = "def add(a, b): return a + b"
        cond = extract_function_condition(code)

        assert cond is not None
        assert cond.name == "add"
        assert "def add(a, b)" in cond.signature
        assert cond.docstring is None
        assert not cond.has_docstring

    def test_extract_function_with_docstring(self):
        code = '''def add(a, b):
    """Add two numbers together."""
    return a + b'''
        cond = extract_function_condition(code)

        assert cond is not None
        assert cond.name == "add"
        assert cond.has_docstring
        assert "Add two numbers" in cond.docstring
        assert '"""Add two numbers' in cond.condition_text

    def test_extract_function_with_type_hints(self):
        code = "def add(a: int, b: int) -> int: return a + b"
        cond = extract_function_condition(code)

        assert cond is not None
        assert ": int" in cond.signature
        assert "-> int" in cond.signature

    def test_extract_async_function(self):
        code = "async def fetch(url): return await get(url)"
        cond = extract_function_condition(code)

        assert cond is not None
        assert "async def" in cond.signature

    def test_invalid_code_returns_none(self):
        code = "this is not valid python"
        cond = extract_function_condition(code)
        assert cond is None

    def test_no_function_returns_none(self):
        code = "x = 1 + 2"
        cond = extract_function_condition(code)
        assert cond is None

    def test_multiline_docstring(self):
        code = '''def process(data):
    """Process the input data.

    Args:
        data: The data to process

    Returns:
        Processed result
    """
    return data'''
        cond = extract_function_condition(code)

        assert cond is not None
        assert cond.has_docstring
        # Should take first paragraph
        assert "Process the input data" in cond.condition_text


class TestTokenization:
    def test_tokenize_simple_text(self):
        text = "def add(a, b):"
        tokens = tokenize_condition(text, max_len=64)

        assert tokens.shape == (64,)
        assert tokens.dtype == np.int32

    def test_roundtrip_tokenization(self):
        text = "def multiply(x: int, y: int) -> int:"
        tokens = tokenize_condition(text, max_len=128)
        decoded = detokenize_condition(tokens)

        assert text in decoded or decoded.strip() == text.strip()

    def test_padding(self):
        text = "x"
        tokens = tokenize_condition(text, max_len=64)

        # Should have BOS, 'x', EOS, then padding
        assert tokens.shape == (64,)
        # Most tokens should be padding (id 0)
        assert np.sum(tokens == 0) > 60

    def test_truncation(self):
        text = "a" * 200  # Very long
        tokens = tokenize_condition(text, max_len=64)

        assert tokens.shape == (64,)

    def test_vocab_size(self):
        # All printable ASCII + special tokens
        assert CONDITION_VOCAB_SIZE > 95  # At least printable ASCII


class TestConditionMask:
    def test_mask_with_padding(self):
        tokens = np.array([1, 2, 3, 0, 0, 0], dtype=np.int32)  # 0 is PAD
        mask = create_condition_mask(tokens)

        assert mask.shape == tokens.shape
        assert mask[0] == 1.0
        assert mask[1] == 1.0
        assert mask[2] == 1.0
        assert mask[3] == 0.0
        assert mask[4] == 0.0
        assert mask[5] == 0.0


class TestConditioningCoverage:
    def test_analyze_toy_programs(self):
        from experiments.kelp.toy_dataset import get_toy_programs

        programs = get_toy_programs()
        stats = analyze_conditioning_coverage(programs)

        assert stats["total_programs"] == len(programs)
        assert "with_docstring" in stats
        assert "docstring_rate" in stats
        assert "with_type_hints" in stats

        # After our updates, should have some docstrings
        assert stats["with_docstring"] > 0

    def test_empty_programs(self):
        stats = analyze_conditioning_coverage([])

        assert stats["total_programs"] == 0
        assert stats["docstring_rate"] == 0


class TestConditionalTrainingExample:
    def test_create_conditional_example(self):
        from experiments.kelp.conditioning import create_conditional_training_example
        from experiments.kelp.python_grammar import PythonNodeVocab, PythonValueVocab

        code = '''def add(a, b):
    """Add two numbers."""
    return a + b'''

        node_vocab = PythonNodeVocab()
        value_vocab = PythonValueVocab()

        example = create_conditional_training_example(
            code,
            num_corruption_steps=1,
            node_vocab=node_vocab,
            value_vocab=value_vocab,
        )

        assert example is not None
        assert example.has_docstring
        assert "add" in example.function_name
        assert example.condition_tokens.shape[0] > 0
