# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gzip
import json
from pathlib import Path

from marin.execution.lazy import StepContext
from marin.rl.skyrl import SkyRLCheckpoint, SkyRLModel

from experiments.post_training import iceball_micro


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


def test_workflow_is_one_dependency_chain_through_both_evaluators(monkeypatch) -> None:
    monkeypatch.setattr("marin.experiment.namespacing.username_segment", lambda: "alice")
    workflow = iceball_micro.build_workflow(version="2026.08.01")

    assert workflow.pretrain.deps == (workflow.fineweb,)
    assert workflow.pretrain in workflow.sft.deps
    (native_training,) = workflow.rl.deps
    assert native_training.artifact_type is SkyRLCheckpoint
    assert native_training.deps == (workflow.sft, workflow.gsm8k)
    assert workflow.rl.artifact_type is SkyRLModel
    assert workflow.evaluation.deps == (workflow.rl,)
    assert workflow.evaluation.name.endswith("gsm8k-smoke,aime-smoke")
    assert workflow.rl.name == f"users/alice/checkpoints/{iceball_micro.ICEBALL_MODEL_NAME}-rl"
    training_config = native_training.build_config(
        StepContext.for_fingerprint(native_training.runtime_args, native_training.deps)
    )
    request = training_config.request
    assert request.completion_mode == "checkpoint"
    assert "++trainer.max_ckpts_to_keep=1" in request.overrides
    assert "++trainer.hf_save_interval=-1" in request.overrides
    assert request.model.identity == f"{workflow.sft.name}@{workflow.sft.version}:{workflow.sft.fingerprint()}"
    (train_data,) = request.train_data
    (validation_data,) = request.validation_data
    assert (
        train_data.identity
        == validation_data.identity
        == (f"{workflow.gsm8k.name}@{workflow.gsm8k.version}:{workflow.gsm8k.fingerprint()}")
    )
    assert (train_data.relative_path, validation_data.relative_path) == ("train.parquet", "validation.parquet")
    export_config = workflow.rl.build_config(StepContext.for_fingerprint(workflow.rl.runtime_args, workflow.rl.deps))
    assert export_config.request.training_manifest_uri == (
        f"{native_training.name}@{native_training.version}:{native_training.fingerprint()}/terminal.json"
    )
    evaluation_config = workflow.evaluation.build_config(
        StepContext.for_fingerprint(workflow.evaluation.runtime_args, workflow.evaluation.deps)
    )
    assert evaluation_config.evals == "gsm8k-smoke,aime-smoke"
    assert evaluation_config.model.location == (
        f"{workflow.rl.name}@{workflow.rl.version}:{workflow.rl.fingerprint()}/policy"
    )
