# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess

import pytest
from marin.execution.build_context import BuildContext, VersionCodex, build_context
from marin.execution.lazy import ArtifactStep, StepContext
from marin.rl.skyrl import _launcher_command as launcher_command_for_test
from marin.rl.skyrl import skyrl_step
from marin.training.training import LevanterCheckpoint

from experiments.post_training.phase1a.experts import (
    EXPERT_CLUSTER,
    EXPERT_SEQUENCES_PER_STEP,
    ExpertPlan,
    expert_recipe,
    expert_spec,
    expert_topology,
)
from experiments.post_training.phase1a.launch import execution
from experiments.post_training.phase1a.launch_test_support import COMPOSE
from experiments.post_training.phase1a.matrix import Task


def test_an_expert_trains_three_epochs_of_full_batches_and_saves_one_full_checkpoint_at_the_end():
    plan = ExpertPlan(Task.R, rows=3280, learning_rate=1e-5)
    assert plan.steps == (3280 // EXPERT_SEQUENCES_PER_STEP) * 3 == 153
    recipe = expert_recipe(plan)
    trainer = recipe["trainer"]
    assert (trainer["epochs"], trainer["max_steps"]) == (3, 153)
    assert trainer["checkpoint_steps"] == trainer["optimizer_checkpoint_steps"] == trainer["hf_export_steps"] == [153]
    assert trainer["ckpt_interval"] == 153
    assert trainer["policy"]["megatron_config"]["optimizer_checkpoint_sharding_type"] == "dp_reshardable"
    assert trainer["policy"]["optimizer_config"]["lr"] == 1e-5
    assert trainer["algorithm"]["policy_loss_type"] == "sft" and recipe["environment"]["env_class"] == "fixed_completion"
    assert recipe["data"]["shuffle"] is True and not trainer["use_sample_packing"]
    assert recipe["generator"]["max_generate_length_by_data_source"] == {"task_R": 16_384}
    topology = expert_topology(plan)
    assert topology.role_plan.train_batch_size == topology.role_plan.policy_mini_batch_size == EXPERT_SEQUENCES_PER_STEP
    assert topology.role_plan.num_inference_engines == 0


def test_too_few_rows_for_one_step_raise():
    with pytest.raises(ValueError, match="fewer than one"):
        _ = ExpertPlan(Task.X, rows=10, learning_rate=1e-5).steps


@pytest.mark.timeout(300)  # A cold cache installs the independently locked launcher.
def test_expert_configs_compose_through_skyrls_launcher(tmp_path):
    model = ArtifactStep.adopt("models/phase0c-test-b", "2026.10.24", "s3://bucket/b", kind=LevanterCheckpoint)
    rows = ArtifactStep.adopt("documents/phase0c-test-rows", "2026.10.24", "s3://bucket/rows")
    paths = []
    with build_context(BuildContext(versions=VersionCodex(default="2026.10.24"))):
        for plan in (ExpertPlan(Task.X, 5177, 1e-5), ExpertPlan(Task.R, 3280, 1e-5)):
            step = skyrl_step(expert_spec(plan, model=model, rows=rows), execution(EXPERT_CLUSTER), export_hf=True)
            config = step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps))
            path = tmp_path / f"{plan.name}.yaml"
            path.write_text(config.launch_config_yaml)
            paths.append(str(path))
    command = launcher_command_for_test(paths[0])
    bootstrap = command[: command.index("marinskyrl")]
    result = subprocess.run([*bootstrap, "python", "-c", COMPOSE, *paths], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr[-4000:]
    x, r = json.loads(result.stdout)
    assert x["steps"] == [240, 3, 1] and r["steps"] == [153, 3, 1]
    assert x["sequences_per_step"] == r["sequences_per_step"] == EXPERT_SEQUENCES_PER_STEP
    assert x["order"][0] is True and x["env_class"] == "fixed_completion"
    assert x["checkpoints"] == [[240], [240]] and r["response_caps"] == {"task_R": 16_384}
