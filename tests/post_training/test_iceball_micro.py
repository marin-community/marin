# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gzip
import json
from pathlib import Path

from marin.execution.lazy import StepContext

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

    assert workflow.pretrain in workflow.sft.deps
    assert workflow.sft in workflow.rl.deps
    assert workflow.gsm8k in workflow.rl.deps
    assert workflow.evaluation.deps == (workflow.rl,)
    assert workflow.evaluation.name.endswith("gsm8k-smoke,aime-smoke")
    assert workflow.rl.name == f"users/alice/checkpoints/{iceball_micro.ICEBALL_MODEL_NAME}-rl"
    rl_config = workflow.rl.build_config(StepContext.for_fingerprint(workflow.rl.runtime_args, workflow.rl.deps))
    assert "++trainer.max_ckpts_to_keep=1" in rl_config.request.overrides
