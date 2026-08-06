# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gzip
import json
from pathlib import Path

from click.testing import CliRunner
from marin.execution.lazy import materialized_config

from experiments.post_training import iceball_micro


def test_qwen_architecture_matches_0_6b_parameter_scale() -> None:
    assert iceball_micro.ICEBALL_QWEN3_CONFIG.total_trainable_params(151669) == 595_769_344


def test_gsm8k_record_matches_skyrl_reward_schema() -> None:
    record = iceball_micro._gsm8k_record(
        {"question": "What is 20 + 22?", "answer": "Add the values. #### 42"},
        "train",
        3,
    )

    assert record["prompt"] == [
        {
            "role": "user",
            "content": 'What is 20 + 22? Let\'s think step by step and output the final answer after "####".',
        }
    ]
    assert record["reward_spec"] == {"method": "rule", "ground_truth": "42"}
    assert record["extra_info"]["index"] == 3


def test_fineweb_slice_streams_only_the_declared_prefix(tmp_path: Path, monkeypatch) -> None:
    texts = (f"document {index}" for index in range(10))
    monkeypatch.setattr(iceball_micro, "_fineweb_texts", lambda _config: texts)

    iceball_micro.write_fineweb_slice(iceball_micro.FineWebSliceConfig(output_path=str(tmp_path), rows=3))

    with gzip.open(tmp_path / "train.jsonl.gz", "rt") as source:
        written = [json.loads(line) for line in source]
    assert written == [{"text": "document 0"}, {"text": "document 1"}, {"text": "document 2"}]


def test_workflow_is_one_dependency_chain_through_both_evaluators() -> None:
    workflow = iceball_micro.build_workflow(version="2026.08.01")

    assert workflow.pretrain in workflow.sft.deps
    assert workflow.sft in workflow.rl.deps
    assert workflow.gsm8k in workflow.rl.deps
    assert workflow.evaluation.deps == (workflow.rl,)
    assert workflow.evaluation.name.endswith("gsm8k-smoke,aime-smoke")


def test_rl_separates_policy_and_rollout_gpus(monkeypatch) -> None:
    monkeypatch.setattr(
        "marin.rl.skyrl.discover_hf_checkpoints",
        lambda _artifact_path: ["gs://test-prefix/checkpoints/iceball-micro-sft/hf/step-8"],
    )
    workflow = iceball_micro.build_workflow(version="2026.08.01")

    request = materialized_config(workflow.rl, "gs://test-prefix").request
    assert request.topology.num_nodes == 2
    assert request.topology.role_plan.colocate_all is False


def test_cli_can_run_rl_as_the_terminal_stage(monkeypatch) -> None:
    submitted = []
    monkeypatch.setattr("marin.experiment.cli.run", lambda *handles, max_concurrent: submitted.extend(handles))

    result = CliRunner().invoke(iceball_micro.main, ["--version", "2026.08.01", "--stage", "rl", "--run"])

    assert result.exit_code == 0
    assert len(submitted) == 1
    assert submitted[0].name == "checkpoints/iceball-micro-rl"
