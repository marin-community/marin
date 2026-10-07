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

from experiments.post_training.task_curation.datasets.nemotron_ultra.inputs import bind_reference_paths
from experiments.post_training.task_curation.pipeline import SourceRuntimeConfig
from experiments.post_training.task_curation.sources import rl_data_pipelines

SOURCES = {source.source_key: source for source in rl_data_pipelines().values()}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _source() -> Source:
    return Source(dataset="fixture/ultra", revision="a" * 40, row="fixture:0", importer_revision="fixture-v1")


def test_safety_selection_preserves_request_for_normalization(tmp_path):
    name = "MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_jailbreak"
    recipe = SOURCES[name].recipe(SourceRuntimeConfig(images={}, controller_url=None))
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
    result = recipe.policy.normalize(RawRow("safety-1", _source(), records[0]["data"]))
    assert isinstance(result, NormalizedTask)
    assert result.task.context.events == (TextMessage(role="user", content="Explain safe handling."),)


@pytest.mark.parametrize("separate_acquisition", [False, True])
def test_swe_components_split_by_pinned_membership(tmp_path, separate_acquisition):
    reference_root = tmp_path / "separate-membership" if separate_acquisition else tmp_path / "swe-gym-membership"
    membership = reference_root / "data/train-00000-of-00001.parquet"
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
    gym = SOURCES["MarinSkyRL:nemotron_ultra_mopd/swe_pivot_len40k/SWE-Gym/SWE-Gym"].recipe(
        SourceRuntimeConfig(images={}, controller_url=None)
    )
    rebench = SOURCES["MarinSkyRL:nemotron_ultra_mopd/swe_pivot_len40k/nebius/SWE-rebench-V2"].recipe(
        SourceRuntimeConfig(images={}, controller_url=None)
    )
    gym_files, rebench_files = gym.inputs.files, rebench.inputs.files
    if separate_acquisition:
        roots = {"swe-gym-membership": str(reference_root)}
        gym_files = bind_reference_paths(gym_files, roots)
        rebench_files = bind_reference_paths(rebench_files, roots)
    assert [row["locator"] for row in staged_file_rows(str(tmp_path), "mopd.jsonl", gym_files)] == ["mopd.jsonl:0"]
    assert [row["locator"] for row in staged_file_rows(str(tmp_path), "mopd.jsonl", rebench_files)] == ["mopd.jsonl:1"]


@pytest.mark.parametrize(
    "question,ground_truth,expected",
    [
        ("What is 2 + 3?", "5", "5"),
        ("What is 2 + 3?", '["5"]', "5"),
        ("Give the set of roots of x^2 - 3x + 2 = 0.", "{1, 2}", "{1, 2}"),
    ],
)
@pytest.mark.parametrize("separate_acquisition", [False, True])
def test_math_placeholder_reconstructs_question_and_answer(
    tmp_path, question, ground_truth, expected, separate_acquisition
):
    reference_root = tmp_path / "separate-dapo" if separate_acquisition else tmp_path / "placeholder-dapo"
    placeholder_file = reference_root / "data/dapo-math-17k.parquet"
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
    recipe = SOURCES["MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_math_cot"].recipe(
        SourceRuntimeConfig(images={}, controller_url=None)
    )
    files = recipe.inputs.files
    if separate_acquisition:
        files = bind_reference_paths(
            files, {"placeholder-dapo": str(reference_root), "placeholder-skywork": str(tmp_path / "skywork")}
        )
    record = next(staged_file_rows(str(tmp_path), "mopd.jsonl", files))
    assert record["data"]["placeholder_source"]["record"]["reward_model"]["ground_truth"] == ground_truth
    result = recipe.policy.normalize(RawRow("math-1", _source(), record["data"]))
    assert isinstance(result, NormalizedTask)
    assert result.task.context.events == (TextMessage(role="user", content=question),)
    assert grader_config(result.task)["contract"]["expected_answer"] == expected
    assert [change.field for change in result.changes] == ["question", "expected_answer"]
