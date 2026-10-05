# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise pinned Ultra selections against local staged source files."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from taskcompendium.grader import grader_config
from taskcompendium.models import Source, TextMessage
from taskcompendium.pipeline.models import NormalizedTask, RawRow
from taskcompendium.pipeline.sources import staged_file_rows

from experiments.post_training.task_curation.nemotron import RECIPES


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _source() -> Source:
    return Source(dataset="fixture/ultra", revision="a" * 40, row="fixture:0", importer_revision="fixture-v1")


def test_safety_selection_preserves_request_for_normalization(tmp_path):
    name = "nemotron_ultra_mopd_ultra_sft_step3200_jailbreak"
    recipe = RECIPES[name]
    _write_jsonl(
        tmp_path / "mopd.jsonl",
        [
            {
                "dataset": "ultra_sft_step3200_jailbreak",
                "agent_ref": {"name": "safety_agent"},
                "responses_create_params": {"input": [{"role": "user", "content": "Explain safe handling."}]},
                "response_policy_mapped": "helpful",
            },
            {
                "dataset": "another_component",
                "agent_ref": {"name": "other_agent"},
                "responses_create_params": {"input": [{"role": "user", "content": "Other request"}]},
            },
        ],
    )
    records = list(staged_file_rows(str(tmp_path), "mopd.jsonl", recipe.inputs.files))
    assert len(records) == 1
    assert records[0]["locator"] == "mopd.jsonl:0"
    result = recipe.pipeline.normalize(RawRow("safety-1", _source(), records[0]["data"]))
    assert isinstance(result, NormalizedTask)
    assert result.task.context.events == (TextMessage(role="user", content="Explain safe handling."),)


def test_swe_components_split_by_pinned_membership(tmp_path):
    membership = tmp_path / "swe-gym-membership/data/train-00000-of-00001.parquet"
    membership.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([{"instance_id": "gym-1"}]), membership)
    selector = "swe_pivot_len40k"
    _write_jsonl(
        tmp_path / "mopd.jsonl",
        [
            {"dataset": selector, "metadata": {"instance_id": "gym-1"}},
            {"dataset": selector, "metadata": {"instance_id": "rebench-1"}},
        ],
    )
    gym = RECIPES["nemotron_ultra_mopd_swe_pivot_len40k_swe_gym_swe_gym"]
    rebench = RECIPES["nemotron_ultra_mopd_swe_pivot_len40k_nebius_swe_rebench_v2"]
    assert [row["locator"] for row in staged_file_rows(str(tmp_path), "mopd.jsonl", gym.inputs.files)] == [
        "mopd.jsonl:0"
    ]
    assert [row["locator"] for row in staged_file_rows(str(tmp_path), "mopd.jsonl", rebench.inputs.files)] == [
        "mopd.jsonl:1"
    ]


@pytest.mark.parametrize(
    "question,ground_truth,expected",
    [
        ("What is 2 + 3?", "5", "5"),
        ("What is 2 + 3?", '["5"]', "5"),
        ("Give the set of roots of x^2 - 3x + 2 = 0.", "{1, 2}", "{1, 2}"),
    ],
)
def test_math_placeholder_reconstructs_question_and_answer(tmp_path, question, ground_truth, expected):
    placeholder_file = tmp_path / "placeholder-dapo/data/dapo-math-17k.parquet"
    placeholder_file.parent.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist([{"prompt": [{"content": question}], "reward_model": {"ground_truth": ground_truth}}]),
        placeholder_file,
    )
    _write_jsonl(
        tmp_path / "mopd.jsonl",
        [
            {
                "dataset": "ultra_sft_step3200_math_cot",
                "agent_ref": {"name": "math_agent"},
                "_hf_question_placeholder": {
                    "dataset": "BytedTsinghua-SIA/DAPO-Math-17k",
                    "split": "train",
                    "row": 0,
                    "mode": "canonical",
                },
                "responses_create_params": {"input": [{"role": "user", "content": "placeholder"}]},
            }
        ],
    )
    recipe = RECIPES["nemotron_ultra_mopd_ultra_sft_step3200_math_cot"]
    record = next(staged_file_rows(str(tmp_path), "mopd.jsonl", recipe.inputs.files))
    assert record["data"]["placeholder_source"]["record"]["reward_model"]["ground_truth"] == ground_truth
    result = recipe.pipeline.normalize(RawRow("math-1", _source(), record["data"]))
    assert isinstance(result, NormalizedTask)
    assert result.task.context.events == (TextMessage(role="user", content=question),)
    assert grader_config(result.task)["contract"]["expected_answer"] == expected
    assert [change.field for change in result.changes] == ["question", "expected_answer"]
