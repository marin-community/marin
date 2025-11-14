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

"""Tests for HuggingFace dataset transformation to evaluation/decontamination formats."""

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
    format_prompt_response,
    get_nested_item,
    hf_dataset_to_jsonl,
    is_kv_list,
    standardize_options,
    transform_example_to_qa,
)


class MockDataset:
    """Mock HuggingFace dataset for testing."""

    def __init__(self, examples):
        self.examples = examples

    def __iter__(self):
        return iter(self.examples)


class TestHelperFunctions:
    """Test helper functions for data transformation."""

    def test_get_nested_item(self):
        """Test retrieving nested items from dictionaries."""
        data = {"a": {"b": {"c": "value"}}, "list": [1, 2, 3]}

        assert get_nested_item(data, "a.b.c") == "value"
        assert get_nested_item(data, "a.b") == {"c": "value"}
        assert get_nested_item(data, "nonexistent", "default") == "default"
        assert get_nested_item(data, "") == None

    def test_is_kv_list(self):
        """Test key-value list detection."""
        # Valid kv list
        assert is_kv_list([{"key": "A", "value": "Option 1"}, {"key": "B", "value": "Option 2"}])

        # Invalid - missing value
        assert not is_kv_list([{"key": "A"}, {"key": "B", "value": "Option 2"}])

        # Not a list
        assert not is_kv_list("not a list")
        assert not is_kv_list({"key": "A", "value": "B"})

    def test_standardize_options_dict(self):
        """Test standardizing options from dictionary format."""
        options_dict = {"B": "Canada", "A": "France", "D": "UK", "C": "USA"}
        result = standardize_options(options_dict)
        assert result == ["France", "Canada", "USA", "UK"]

    def test_standardize_options_list(self):
        """Test standardizing options from list format."""
        options_list = ["France", "Canada", "USA", "UK"]
        result = standardize_options(options_list)
        assert result == ["France", "Canada", "USA", "UK"]

    def test_standardize_options_kv_list(self):
        """Test standardizing options from key-value list format."""
        options_kv = [
            {"key": "B", "value": "Canada"},
            {"key": "A", "value": "France"},
            {"key": "D", "value": "UK"},
            {"key": "C", "value": "USA"},
        ]
        result = standardize_options(options_kv)
        assert result == ["France", "Canada", "USA", "UK"]

    def test_format_prompt_response(self):
        """Test formatting prompt and response."""
        question = "What is the capital of France?"
        options = ["London", "Paris", "Berlin", "Madrid"]
        labels = ["A", "B", "C", "D"]
        answer_idx = 1
        answer_text = "Paris"

        prompt, response = format_prompt_response(question, options, labels, answer_idx, answer_text)

        assert "What is the capital of France?" in prompt
        assert "A. London" in prompt
        assert "B. Paris" in prompt
        assert "C. Berlin" in prompt
        assert "D. Madrid" in prompt
        assert "Answer:" in prompt
        assert response == "B. Paris"


class TestTransformExampleToQA:
    """Test the transform_example_to_qa function."""

    def test_transform_multiple_choice_evaluation(self):
        """Test transforming a multiple choice question to evaluation format."""
        example = {
            "question": "What is 2+2?",
            "choices": ["3", "4", "5", "6"],
            "answer": 1,  # Index of correct answer
        }

        dataset_meta = DatasetWithMetaData(
            dataset=None, subset="test_subset", split="test", revision="main"
        )

        cfg = DatasetConversionConfig(
            dataset_name="test/dataset",
            subsets=["test_subset"],
            splits=["test"],
            input_path="test_path",
            hf_path="test/dataset",
            output_path="/tmp/test",
            output_format=OutputFormatOptions.evaluation,
            prompt_key="question",
            options_key="choices",
            answer_idx_key="answer",
            answer_labels=["A", "B", "C", "D"],
        )

        result = transform_example_to_qa(example, 0, dataset_meta, cfg)

        assert result is not None
        assert result["id"] == "test/dataset-test_subset-test-evaluation-0"
        assert result["source"] == "test/dataset"
        assert "What is 2+2?" in result["prompt"]
        assert "A. 3" in result["prompt"]
        assert "B. 4" in result["prompt"]
        assert result["response"] == "B. 4"
        assert result["metadata"]["answer_idx"] == 1
        assert result["metadata"]["answer_label"] == "B"
        assert result["metadata"]["answer"] == "4"

    def test_transform_decontamination_format(self):
        """Test transforming to decontamination format."""
        example = {
            "question": "What is the capital of France?",
            "choices": ["London", "Paris", "Berlin", "Madrid"],
            "answer": 1,
        }

        dataset_meta = DatasetWithMetaData(
            dataset=None, subset="test_subset", split="test", revision="main"
        )

        cfg = DatasetConversionConfig(
            dataset_name="test/dataset",
            subsets=["test_subset"],
            splits=["test"],
            input_path="test_path",
            hf_path="test/dataset",
            output_path="/tmp/test",
            output_format=OutputFormatOptions.decontamination,
            prompt_key="question",
            options_key="choices",
            answer_idx_key="answer",
            answer_labels=["A", "B", "C", "D"],
        )

        result = transform_example_to_qa(example, 0, dataset_meta, cfg)

        assert result is not None
        assert result["text"] == "What is the capital of France?"
        assert "prompt" not in result
        assert "response" not in result

    def test_transform_with_answer_label(self):
        """Test transforming when answer is provided as label instead of index."""
        example = {
            "question": "What is 2+2?",
            "choices": ["3", "4", "5", "6"],
            "answerKey": "B",  # Label instead of index
        }

        dataset_meta = DatasetWithMetaData(
            dataset=None, subset="test_subset", split="test", revision="main"
        )

        cfg = DatasetConversionConfig(
            dataset_name="test/dataset",
            subsets=["test_subset"],
            splits=["test"],
            input_path="test_path",
            hf_path="test/dataset",
            output_path="/tmp/test",
            output_format=OutputFormatOptions.evaluation,
            prompt_key="question",
            options_key="choices",
            answer_label_key="answerKey",
            answer_labels=["A", "B", "C", "D"],
        )

        result = transform_example_to_qa(example, 0, dataset_meta, cfg)

        assert result is not None
        assert result["metadata"]["answer_idx"] == 1
        assert result["metadata"]["answer_label"] == "B"
        assert result["metadata"]["answer"] == "4"

    def test_transform_with_nested_keys(self):
        """Test transforming with nested dictionary keys."""
        example = {
            "question": "Select the correct answer",
            "options": {"text": ["Option A", "Option B", "Option C"]},
            "correct": {"index": 1},
        }

        dataset_meta = DatasetWithMetaData(
            dataset=None, subset="test_subset", split="test", revision="main"
        )

        cfg = DatasetConversionConfig(
            dataset_name="test/dataset",
            subsets=["test_subset"],
            splits=["test"],
            input_path="test_path",
            hf_path="test/dataset",
            output_path="/tmp/test",
            output_format=OutputFormatOptions.evaluation,
            prompt_key="question",
            options_key="options.text",
            answer_idx_key="correct.index",
            answer_labels=["1", "2", "3"],
        )

        result = transform_example_to_qa(example, 0, dataset_meta, cfg)

        assert result is not None
        assert result["metadata"]["answer"] == "Option B"


class TestHfDatasetToJsonl:
    """Test the main hf_dataset_to_jsonl function."""

    def test_end_to_end_evaluation_format(self, tmp_path):
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
            assert output_file.exists()

            # Read and verify contents
            with gzip.open(output_file, "rt") as f:
                lines = f.readlines()
                assert len(lines) == 2

                # Check first example
                first = json.loads(lines[0])
                assert "What is 2+2?" in first["prompt"]
                assert first["response"] == "B. 4"
                assert first["metadata"]["answer_idx"] == 1

                # Check second example
                second = json.loads(lines[1])
                assert "What is 3+3?" in second["prompt"]
                assert second["response"] == "B. 6"
        finally:
            # Restore original function
            module.load_datasets = original_load_datasets

    def test_end_to_end_decontamination_format(self, tmp_path):
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
            assert output_file.exists()

            # Read and verify contents
            with gzip.open(output_file, "rt") as f:
                lines = f.readlines()
                assert len(lines) == 2

                first = json.loads(lines[0])
                assert first["text"] == "Question 1"
                assert "prompt" not in first
                assert "response" not in first

                second = json.loads(lines[1])
                assert second["text"] == "Question 2"
        finally:
            module.load_datasets = original_load_datasets


if __name__ == "__main__":
    if pytest is not None:
        pytest.main([__file__, "-v"])
    else:
        print("pytest not available, tests can be run via test_runner.py")
