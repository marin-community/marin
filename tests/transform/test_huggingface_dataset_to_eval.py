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
from unittest.mock import patch

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
    examples = [
        {"question": "What is 2+2?", "choices": ["3", "4", "5"], "answer": 1},
        {"question": "What is 3+3?", "choices": ["5", "6", "7"], "answer": 1},
    ]
    mock_dataset = MockDataset(examples)

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

    def mock_load_datasets(config):
        return [DatasetWithMetaData(mock_dataset, "arithmetic", "test", "main")]

    with patch("marin.transform.huggingface.dataset_to_eval.load_datasets", mock_load_datasets):
        hf_dataset_to_jsonl(cfg)

        # Find all shard files
        shard_files = sorted((output_dir / "test").glob("math-arithmetic-test-evaluation-*.jsonl.gz"))
        assert len(shard_files) > 0, f"No output files created in {output_dir / 'test'}"

        # Read and combine all shards
        all_lines = []
        for shard_file in shard_files:
            with gzip.open(shard_file, "rt") as f:
                all_lines.extend(f.readlines())

        assert len(all_lines) == 2, f"Expected 2 lines total, got {len(all_lines)}"

        # Check first example
        first = json.loads(all_lines[0])
        assert "What is 2+2?" in first["prompt"], "Question not in prompt"
        assert first["response"] == "B. 4", f"Expected 'B. 4', got '{first['response']}'"
        assert first["metadata"]["answer_idx"] == 1, "Answer index mismatch"
        assert first["source"] == "test/math", "Source mismatch"

        # Check second example
        second = json.loads(all_lines[1])
        assert "What is 3+3?" in second["prompt"], "Question not in prompt"
        assert second["response"] == "B. 6", f"Expected 'B. 6', got '{second['response']}'"


def test_hf_dataset_to_jsonl_decontamination_format(tmp_path):
    """Test end-to-end transformation to decontamination format."""
    examples = [
        {"question": "Question 1", "choices": ["A", "B"], "answer": 0},
        {"question": "Question 2", "choices": ["C", "D"], "answer": 1},
    ]
    mock_dataset = MockDataset(examples)

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

    def mock_load_datasets(config):
        return [DatasetWithMetaData(mock_dataset, "subset1", "train", "main")]

    with patch("marin.transform.huggingface.dataset_to_eval.load_datasets", mock_load_datasets):
        hf_dataset_to_jsonl(cfg)

        # Find all shard files
        shard_files = sorted((output_dir / "test").glob("dataset-subset1-train-decontamination-*.jsonl.gz"))
        assert len(shard_files) > 0, f"No output files created in {output_dir / 'test'}"

        # Read and combine all shards
        all_lines = []
        for shard_file in shard_files:
            with gzip.open(shard_file, "rt") as f:
                all_lines.extend(f.readlines())

        assert len(all_lines) == 2, f"Expected 2 lines total, got {len(all_lines)}"

        first = json.loads(all_lines[0])
        assert first["text"] == "Question 1", f"Expected 'Question 1', got '{first['text']}'"
        assert "prompt" not in first, "Should not have 'prompt' field in decontamination format"
        assert "response" not in first, "Should not have 'response' field in decontamination format"

        second = json.loads(all_lines[1])
        assert second["text"] == "Question 2", f"Expected 'Question 2', got '{second['text']}'"
