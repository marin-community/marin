# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections import Counter

import pytest
from marin.transform.code_instruction_midtraining import (
    CodeInstructionMidtrainingConfig,
    iter_code_instruction_documents,
    render_code_instruction_row,
)


def test_opencode_instruct_filter_renders_high_score_dolma_document() -> None:
    cfg = CodeInstructionMidtrainingConfig(
        source="nvidia/OpenCodeInstruct",
        revision="8f3ba5b",
        output_path="/tmp/opencode",
        instruction_column="input",
        output_column="output",
        metadata_columns=[
            "domain",
            "generation_algorithm",
            "unit_tests",
            "tests_execution_status",
            "average_test_score",
        ],
        subsets=["train"],
        average_test_score_column="average_test_score",
        min_average_test_score=0.9,
    )
    rows = [
        {
            "id": "good",
            "input": "Write a Python function that returns x + 1.",
            "output": "def inc(x):\n    return x + 1",
            "domain": "algorithmic",
            "generation_algorithm": "self-instruct",
            "unit_tests": "assert inc(1) == 2",
            "tests_execution_status": "passed",
            "average_test_score": 1.0,
        },
        {
            "id": "low-score",
            "input": "Implement subtraction.",
            "output": "def sub(a, b):\n    return a - b",
            "average_test_score": 0.5,
        },
        {
            "id": "nan-score",
            "input": "Implement multiplication.",
            "output": "def mul(a, b):\n    return a * b",
            "average_test_score": "nan",
        },
    ]

    counters: Counter[str] = Counter()
    docs = list(iter_code_instruction_documents(rows, cfg, subset="train", split="train", counters=counters))

    assert len(docs) == 1
    assert docs[0]["source"] == "nvidia/OpenCodeInstruct"
    assert (
        docs[0]["text"]
        == "Problem:\nWrite a Python function that returns x + 1.\n\nAnswer:\ndef inc(x):\n    return x + 1"
    )
    assert "assert inc" not in docs[0]["text"]
    assert docs[0]["metadata"] == {
        "hf_dataset_id": "nvidia/OpenCodeInstruct",
        "hf_revision": "8f3ba5b",
        "hf_subset": "train",
        "hf_split": "train",
        "row_id": "good",
        "domain": "algorithmic",
        "generation_algorithm": "self-instruct",
        "unit_tests": "assert inc(1) == 2",
        "tests_execution_status": "passed",
        "average_test_score": 1.0,
    }
    assert counters["input_rows"] == 3
    assert counters["output_rows"] == 1
    assert counters["dropped_below_score"] == 1
    assert counters["dropped_missing_score"] == 1


def test_opencode_genetic_requires_solution_and_keeps_generation_metadata() -> None:
    cfg = CodeInstructionMidtrainingConfig(
        source="nvidia/OpenCodeGeneticInstruct",
        revision="9c95d36",
        output_path="/tmp/opencode-genetic",
        instruction_column="input",
        solution_column="solution",
        output_column="output",
        metadata_columns=["generation_model", "last_operation"],
        subsets=["qwen2.5-32b-instruct"],
    )
    rows = [
        {
            "id": "kept",
            "input": "Solve two-sum.",
            "solution": "Use a hash map from values to indices.",
            "output": "def two_sum(nums, target):\n    seen = {}\n    return []",
            "generation_model": "qwen2.5-32b-instruct",
            "last_operation": "mutation",
        },
        {
            "id": "missing-solution",
            "input": "Solve fizzbuzz.",
            "solution": "",
            "output": "for i in range(1, 101): print(i)",
            "generation_model": "qwen2.5-32b-instruct",
            "last_operation": "crossover",
        },
    ]

    counters: Counter[str] = Counter()
    docs = list(
        iter_code_instruction_documents(rows, cfg, subset="qwen2.5-32b-instruct", split="train", counters=counters)
    )

    assert len(docs) == 1
    assert docs[0]["text"] == (
        "Problem:\nSolve two-sum.\n\n"
        "Reference solution:\nUse a hash map from values to indices.\n\n"
        "Answer:\ndef two_sum(nums, target):\n    seen = {}\n    return []"
    )
    assert docs[0]["metadata"]["generation_model"] == "qwen2.5-32b-instruct"
    assert docs[0]["metadata"]["last_operation"] == "mutation"
    assert counters["dropped_missing_solution"] == 1


def test_sampling_uses_row_identity_before_document_dedup() -> None:
    cfg = CodeInstructionMidtrainingConfig(
        source="nvidia/OpenCodeGeneticInstruct",
        revision="9c95d36",
        output_path="/tmp/opencode-genetic",
        instruction_column="input",
        solution_column="solution",
        output_column="output",
        sample_fraction=0.1,
        sample_seed="test-seed",
    )
    duplicate_rows = [
        {"id": "keep", "input": "Problem", "solution": "Reference", "output": "Answer"},
        {"id": "b", "input": "Problem", "solution": "Reference", "output": "Answer"},
    ]

    counters: Counter[str] = Counter()
    docs = list(iter_code_instruction_documents(duplicate_rows, cfg, subset="default", split="train", counters=counters))

    assert len(docs) == 1
    assert docs[0]["metadata"]["row_id"] == "keep"
    assert counters["dropped_sample"] == 1
    assert counters["dropped_duplicate_text"] == 0

    no_sampling_cfg = CodeInstructionMidtrainingConfig(
        source="nvidia/OpenCodeGeneticInstruct",
        revision="9c95d36",
        output_path="/tmp/opencode-genetic",
        instruction_column="input",
        solution_column="solution",
        output_column="output",
        sample_fraction=1.0,
    )
    dedup_counters: Counter[str] = Counter()
    docs = list(
        iter_code_instruction_documents(
            duplicate_rows,
            no_sampling_cfg,
            subset="default",
            split="train",
            counters=dedup_counters,
        )
    )

    assert len(docs) == 1
    assert dedup_counters["dropped_duplicate_text"] == 1


@pytest.mark.parametrize("sample_fraction", [1.1, float("nan")])
def test_invalid_sample_fraction_fails_fast(sample_fraction: float) -> None:
    cfg = CodeInstructionMidtrainingConfig(
        source="nvidia/OpenCodeInstruct",
        revision="8f3ba5b",
        output_path="/tmp/opencode",
        instruction_column="input",
        output_column="output",
        sample_fraction=sample_fraction,
    )

    with pytest.raises(ValueError, match="sample_fraction"):
        render_code_instruction_row({"input": "p", "output": "a"}, cfg, subset="train", split="train")
