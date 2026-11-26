# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import yaml
from marin.execution.artifact import write_artifact
from marin.execution.lazy import StepContext

from experiments.post_training.skyrl_port_smoke import build_smoke


def test_resume_uses_the_completed_training_checkpoint_in_the_launch_document(tmp_path: Path) -> None:
    smoke = build_smoke("2026.10.19.10306")
    prefix = str(tmp_path)
    previous_checkpoint = str(tmp_path / "saved-checkpoints")
    write_artifact(
        {
            "hf_model_uri": str(tmp_path / "policy"),
            "global_step": 2,
            "tokenizer_uri": "Qwen/Qwen2.5-0.5B-Instruct",
            "tokenizer_revision": "pinned",
            "checkpoint_root": previous_checkpoint,
            "draft_checkpoint_root": None,
            "terminal_manifest_uri": str(tmp_path / "terminal.json"),
            "iris_job_id": "/test/train",
        },
        smoke.train.path(prefix),
    )
    config = smoke.resume.build_config(
        StepContext.for_run(
            output_path=smoke.resume.path(prefix),
            prefix=prefix,
            runtime_args=smoke.resume.runtime_args,
            deps=smoke.resume.deps,
        )
    )
    launch = yaml.safe_load(config.launch_config_yaml)
    assert launch["artifacts"]["checkpoint_root"] == previous_checkpoint
    assert config.output.checkpoint_root == previous_checkpoint
    assert launch["skyrl"]["trainer"]["resume_mode"] == "latest"
    assert launch["skyrl"]["trainer"]["max_steps"] > 2
