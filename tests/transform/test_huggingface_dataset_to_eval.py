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

"""End-to-end tests for HuggingFace dataset transformation to evaluation/decontamination formats."""

import gzip
import json
import tempfile
from pathlib import Path

try:
    import pytest
except ImportError:
    pytest = None

from marin.transform.huggingface.dataset_to_eval import (
    DatasetConversionConfig,
    DatasetWithMetaData,
    OutputFormatOptions,
    hf_dataset_to_jsonl,
)


class MockDataset:
    """Mock HuggingFace dataset for testing."""

    def __init__(self, examples):
        self.examples = examples

    def __iter__(self):
        return iter(self.examples)


def test_hf_dataset_to_jsonl_evaluation_format(tmp_path):
    """Test end-to-end transformation to evaluation format."""
    # Create mock dataset
    examples = [
        {"question": "What is 2+2?", "choices": ["3", "4", "5"], "answer": 1},
        {"question": "What is 3+3?", "choices": ["5", "6", "7"], "answer": 1},
    ]
    mock_dataset = MockDataset(examples)

    # Create config
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    cfg = DatasetConversionConfig(
        dataset_name="test/math",
        subsets=["arithmetic"],
        splits=["test"],
        input_path="test_path",
        hf_path="test/math",
        output_path=str(output_dir),
        output_format=OutputFormatOptions.evaluation,
        prompt_key="question",
        options_key="choices",
        answer_idx_key="answer",
        answer_labels=["A", "B", "C"],
    )

    # Mock load_datasets to return our mock dataset
    import marin.transform.huggingface.dataset_to_eval as module

    original_load_datasets = module.load_datasets

    def mock_load_datasets(config):
        return [DatasetWithMetaData(mock_dataset, "arithmetic", "test", "main")]

    module.load_datasets = mock_load_datasets

    try:
        # Run transformation
        hf_dataset_to_jsonl(cfg)

        # Check output file exists
        output_file = output_dir / "test/math-arithmetic-test-evaluation.jsonl.gz"
        assert output_file.exists(), f"Output file not created: {output_file}"

        # Read and verify contents
        with gzip.open(output_file, "rt") as f:
            lines = f.readlines()
            assert len(lines) == 2, f"Expected 2 lines, got {len(lines)}"

            # Check first example
            first = json.loads(lines[0])
            assert "What is 2+2?" in first["prompt"], "Question not in prompt"
            assert first["response"] == "B. 4", f"Expected 'B. 4', got '{first['response']}'"
            assert first["metadata"]["answer_idx"] == 1, "Answer index mismatch"
            assert first["source"] == "test/math", "Source mismatch"

            # Check second example
            second = json.loads(lines[1])
            assert "What is 3+3?" in second["prompt"], "Question not in prompt"
            assert second["response"] == "B. 6", f"Expected 'B. 6', got '{second['response']}'"
    finally:
        # Restore original function
        module.load_datasets = original_load_datasets


def test_hf_dataset_to_jsonl_decontamination_format(tmp_path):
    """Test end-to-end transformation to decontamination format."""
    # Create mock dataset
    examples = [
        {"question": "Question 1", "choices": ["A", "B"], "answer": 0},
        {"question": "Question 2", "choices": ["C", "D"], "answer": 1},
    ]
    mock_dataset = MockDataset(examples)

    # Create config
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    cfg = DatasetConversionConfig(
        dataset_name="test/dataset",
        subsets=["subset1"],
        splits=["train"],
        input_path="test_path",
        hf_path="test/dataset",
        output_path=str(output_dir),
        output_format=OutputFormatOptions.decontamination,
        prompt_key="question",
        options_key="choices",
        answer_idx_key="answer",
        answer_labels=["A", "B"],
    )

    # Mock load_datasets
    import marin.transform.huggingface.dataset_to_eval as module

    original_load_datasets = module.load_datasets

    def mock_load_datasets(config):
        return [DatasetWithMetaData(mock_dataset, "subset1", "train", "main")]

    module.load_datasets = mock_load_datasets

    try:
        # Run transformation
        hf_dataset_to_jsonl(cfg)

        # Check output file
        output_file = output_dir / "test/dataset-subset1-train-decontamination.jsonl.gz"
        assert output_file.exists(), f"Output file not created: {output_file}"

        # Read and verify contents
        with gzip.open(output_file, "rt") as f:
            lines = f.readlines()
            assert len(lines) == 2, f"Expected 2 lines, got {len(lines)}"

            first = json.loads(lines[0])
            assert first["text"] == "Question 1", f"Expected 'Question 1', got '{first['text']}'"
            assert "prompt" not in first, "Should not have 'prompt' field in decontamination format"
            assert "response" not in first, "Should not have 'response' field in decontamination format"

            second = json.loads(lines[1])
            assert second["text"] == "Question 2", f"Expected 'Question 2', got '{second['text']}'"
    finally:
        module.load_datasets = original_load_datasets


if __name__ == "__main__":
    if pytest is not None:
        pytest.main([__file__, "-v"])
    else:
        print("pytest not available, run with: uv run pytest tests/transform/test_huggingface_dataset_to_eval.py")
