# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for HuggingFace dataset transformation to evaluation/decontamination formats."""

import gzip
import json

import pytest
from datasets import Dataset
from marin.transform.huggingface.dataset_to_eval import (
    DatasetConversionConfig,
    OutputFormatOptions,
    hf_dataset_to_jsonl,
)


def _write_local_hf_dataset(root, subset: str, examples_by_split: dict[str, list[dict]]) -> None:
    """Write a tiny on-disk HF dataset with a README declaring a named config.

    `datasets.load_dataset(root, subset, split=split)` resolves the config via the
    README's `configs:` YAML frontmatter.
    """
    subset_dir = root / subset
    subset_dir.mkdir(parents=True, exist_ok=True)
    data_files = ""
    for split, examples in examples_by_split.items():
        Dataset.from_list(examples).to_parquet(str(subset_dir / f"{split}.parquet"))
        data_files += f"      - split: {split}\n        path: {subset}/{split}.parquet\n"
    entry = f"  - config_name: {subset}\n    data_files:\n{data_files}"
    (root / "README.md").write_text("---\nconfigs:\n" + entry + "---\n")


def test_hf_dataset_to_jsonl_evaluation_format(tmp_path):
    """Test end-to-end transformation to evaluation format."""
    examples = [
        {"question": "What is 2+2?", "choices": ["3", "4", "5"], "answer": 1},
        {"question": "What is 3+3?", "choices": ["5", "6", "7"], "answer": 1},
    ]
    dataset_root = tmp_path / "input"
    _write_local_hf_dataset(dataset_root, "arithmetic", {"test": examples})

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    cfg = DatasetConversionConfig(
        dataset_name="test/math",
        subsets=["arithmetic"],
        splits=["test"],
        input_path=str(dataset_root),
        hf_path="test/math",
        output_path=str(output_dir),
        output_format=OutputFormatOptions.evaluation,
        prompt_key="question",
        options_key="choices",
        answer_idx_key="answer",
        answer_labels=["A", "B", "C"],
    )

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
    dataset_root = tmp_path / "input"
    _write_local_hf_dataset(dataset_root, "subset1", {"train": examples})

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    cfg = DatasetConversionConfig(
        dataset_name="test/dataset",
        subsets=["subset1"],
        splits=["train"],
        input_path=str(dataset_root),
        hf_path="test/dataset",
        output_path=str(output_dir),
        output_format=OutputFormatOptions.decontamination,
        prompt_key="question",
        options_key="choices",
        answer_idx_key="answer",
        answer_labels=["A", "B"],
    )

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


def test_hf_dataset_to_jsonl_converts_every_requested_split(tmp_path):
    """A config requesting several splits converts all of them instead of failing."""
    dataset_root = tmp_path / "input"
    _write_local_hf_dataset(
        dataset_root,
        "arithmetic",
        {
            "train": [{"question": "What is 2+2?", "choices": ["3", "4"], "answer": 1}],
            "test": [{"question": "What is 3+3?", "choices": ["6", "7"], "answer": 0}],
        },
    )

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    cfg = DatasetConversionConfig(
        dataset_name="test/math",
        subsets=["arithmetic"],
        splits=["train", "test"],
        input_path=str(dataset_root),
        hf_path="test/math",
        output_path=str(output_dir),
        output_format=OutputFormatOptions.evaluation,
        prompt_key="question",
        options_key="choices",
        answer_idx_key="answer",
        answer_labels=["A", "B"],
    )

    hf_dataset_to_jsonl(cfg)

    for split, question in (("train", "What is 2+2?"), ("test", "What is 3+3?")):
        shard_files = sorted((output_dir / "test").glob(f"math-arithmetic-{split}-evaluation-*.jsonl.gz"))
        assert shard_files, f"No output files created for split {split}"
        lines = []
        for shard_file in shard_files:
            with gzip.open(shard_file, "rt") as f:
                lines.extend(f.readlines())
        assert len(lines) == 1, f"Expected 1 line for split {split}, got {len(lines)}"
        record = json.loads(lines[0])
        assert question in record["prompt"]
        assert record["metadata"]["split"] == split


def test_hf_dataset_to_jsonl_reports_unloadable_split(tmp_path):
    """A split that cannot be loaded names the offending subset/split pair."""
    dataset_root = tmp_path / "input"
    _write_local_hf_dataset(
        dataset_root,
        "arithmetic",
        {"train": [{"question": "What is 2+2?", "choices": ["3", "4"], "answer": 1}]},
    )

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    cfg = DatasetConversionConfig(
        dataset_name="test/math",
        subsets=["arithmetic"],
        splits=["train", "absent"],
        input_path=str(dataset_root),
        hf_path="test/math",
        output_path=str(output_dir),
        output_format=OutputFormatOptions.evaluation,
        prompt_key="question",
        options_key="choices",
        answer_idx_key="answer",
        answer_labels=["A", "B"],
    )

    with pytest.raises(ValueError, match="arithmetic/absent"):
        hf_dataset_to_jsonl(cfg)
