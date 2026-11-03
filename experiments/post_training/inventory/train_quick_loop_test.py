# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise public arm selection and its constructed launch/report contract."""

import pytest
import yaml
from click.testing import CliRunner
from marin.execution.lazy import StepContext

from experiments.post_training.inventory.train_quick_loop import main


@pytest.mark.parametrize("arm", ["a", "b", "c", "d"])
def test_public_arm_cli_builds_matched_training_before_publication(arm, monkeypatch, tmp_path):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    submissions = []
    monkeypatch.setattr(
        "marin.rl.cli._submit_or_run", lambda module, handles, concurrency, env: submissions.append(handles)
    )
    result = CliRunner().invoke(main, ["--version", "2026.11.02", "--arm", arm, "--run"])
    assert result.exit_code == 0, result.output
    [publication] = submissions[0]
    [training] = publication.deps
    config = training.build_config(StepContext.for_fingerprint(training.runtime_args, training.deps))
    launch = yaml.safe_load(config.skyrl.launch_config_yaml)
    trainer = launch["skyrl"]["trainer"]
    generator = launch["skyrl"]["generator"]
    assert config.expected_steps == trainer["max_steps"] == 25
    assert trainer["train_batch_size"] * generator["n_samples_per_prompt"] == 512
    assert trainer["policy"]["optimizer_config"]["lr"] == (2e-6 if arm == "b" else 1e-6)
    assert trainer["algorithm"]["loss_reduction"] == ("sequence_mean" if arm == "c" else "token_mean")
    assert trainer["rollout_buffer"]["max_staleness_steps"] == (2 if arm == "d" else 0)
    assert trainer["rollout_buffer"]["max_in_flight"] == (192 if arm == "d" else 128)
    assert launch["iris"]["allocation"]["num_nodes"] * launch["iris"]["allocation"]["gpus_per_node"] == 40
    assert trainer["resume_mode"] == "none"
    assert config.skyrl.export_hf is False
