# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gzip
import json
from pathlib import Path

from taskcompendium.grader import grader_config
from taskcompendium.models import TaskSpec

from experiments.post_training import iceball_micro


def test_gsm8k_record_matches_skyrl_reward_schema() -> None:
    record = iceball_micro._gsm8k_record(
        {"question": "What is 20 + 22?", "answer": "Add the values. #### 42"},
        "train",
        3,
    )

    task = TaskSpec.model_validate_json(record["task_spec"])
    assert (
        task.context.events[0].content
        == 'What is 20 + 22? Let\'s think step by step and output the final answer after "####".'
    )
    assert grader_config(task)["extras"]["reward_spec"]["ground_truth"] == "42"
    assert grader_config(task)["extras"]["extra_info"]["index"] == 3


def test_fineweb_slice_streams_only_the_declared_prefix(tmp_path: Path, monkeypatch) -> None:
    texts = (f"document {index}" for index in range(10))
    monkeypatch.setattr(iceball_micro, "_fineweb_texts", lambda _config: texts)

    iceball_micro.write_fineweb_slice(iceball_micro.FineWebSliceConfig(output_path=str(tmp_path), rows=3))

    with gzip.open(tmp_path / "train.jsonl.gz", "rt") as source:
        written = [json.loads(line) for line in source]
    assert written == [{"text": "document 0"}, {"text": "document 1"}, {"text": "document 2"}]


def test_workflow_is_one_dependency_chain_through_both_evaluations(monkeypatch) -> None:
    monkeypatch.setattr("marin.experiment.namespacing.username_segment", lambda: "alice")
    workflow = iceball_micro.build_workflow(version="2026.08.01")

    assert workflow.pretrain in workflow.sft.deps
    assert workflow.sft in workflow.rl.deps
    assert workflow.gsm8k in workflow.rl.deps
    assert workflow.evaluation.deps == (workflow.rl,)
