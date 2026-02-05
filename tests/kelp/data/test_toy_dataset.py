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

"""Tests for Kelp toy dataset."""

import pytest

from experiments.kelp.data.toy_dataset import (
    TOY_PROGRAMS,
    ToyProgram,
    create_toy_dataset,
    get_toy_examples,
)


class TestToyProgram:
    def test_full_code(self):
        prog = ToyProgram(
            docstring="Add two numbers.",
            signature="def add(a, b):",
            body="    return a + b",
        )
        full = prog.full_code
        assert "def add" in full
        assert '"""Add two numbers."""' in full
        assert "return a + b" in full

    def test_prompt(self):
        prog = ToyProgram(
            docstring="Add two numbers.",
            signature="def add(a, b):",
            body="    return a + b",
        )
        prompt = prog.prompt
        assert "Add two numbers." in prompt
        assert "def add(a, b):" in prompt


class TestToyDataset:
    def test_toy_programs_not_empty(self):
        assert len(TOY_PROGRAMS) > 0

    def test_toy_programs_has_variety(self):
        assert len(TOY_PROGRAMS) >= 50

    def test_all_programs_valid_python(self):
        import ast

        for prog in TOY_PROGRAMS:
            try:
                ast.parse(prog.full_code)
            except SyntaxError as e:
                pytest.fail(f"Invalid Python in {prog.signature}: {e}")

    def test_create_toy_dataset(self):
        dataset = create_toy_dataset()
        assert len(dataset) == len(TOY_PROGRAMS)
        for item in dataset:
            assert "prompt" in item
            assert "code" in item
            assert "docstring" in item
            assert "signature" in item
            assert "body" in item

    def test_get_toy_examples(self):
        all_examples = get_toy_examples()
        assert len(all_examples) == len(TOY_PROGRAMS)

        subset = get_toy_examples(5)
        assert len(subset) == 5
