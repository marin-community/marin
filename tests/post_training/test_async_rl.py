# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import click
import pytest
import yaml
from click.testing import CliRunner

from experiments.post_training import async_rl
from experiments.post_training.curriculum_rl.launch import QWEN_POLICY, SNOWBALL_POLICY


def rendered(preset=async_rl.DEFAULT, policy=SNOWBALL_POLICY, settings=()):
    return yaml.safe_load(async_rl.training_config(policy, preset, async_rl.RECIPES[policy.label], settings))


def test_default_preset_is_the_measured_async_recipe_on_megatron():
    config = rendered()
    assert config["entrypoint"] == "fully_async"
    trainer = config["trainer"]
    assert trainer["strategy"] == "megatron"
    assert "fsdp_config" not in trainer["policy"]
    geometry = {
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 2,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 8,
        "expert_tensor_parallel_size": 1,
    }
    assert trainer["policy"]["megatron_config"] == geometry
    assert trainer["ref"]["megatron_config"] == geometry
    assert trainer["policy"]["optimizer_config"]["lr"] == 1e-6
    assert trainer["fully_async"] == {
        "max_staleness_steps": 4,
        "num_parallel_generation_workers": 160,
        "max_buffered_groups": 32,
        "pause_mode": "abort",
        "clear_kv_cache_on_weight_sync": True,
        "first_token_admission": True,
    }
    assert (
        trainer["train_batch_size"],
        trainer["policy_mini_batch_size"],
        trainer["micro_train_batch_size_per_gpu"],
    ) == (32, 32, 1)
    assert config["generator"]["n_samples_per_prompt"] == 4
    assert (trainer["max_steps"], trainer["eval_interval"], trainer["eval_before_train"]) == (100, 20, True)
    assert trainer["algorithm"]["policy_loss_type"] == "regular" and trainer["algorithm"]["use_tis"] is False
    assert all(
        trainer[gate] for gate in ("training_metrics", "async_spans", "policy_train_spans", "optimizer_state_metrics")
    )
    assert trainer["algorithm"]["ratio_diagnostics"] == {"pooled": True}
    assert trainer["algorithm"]["grad_cosine"] == {"enabled": False}
    assert config["generator"]["sampling_params"]["logprobs"] == 0
    assert config["generator"]["engine_init_kwargs"]["moe_backend"] == "triton"


def test_smoke_and_on_policy_presets_change_only_what_they_name():
    smoke = rendered(async_rl.SMOKE_PRESET)["trainer"]
    assert (smoke["max_steps"], smoke["eval_interval"], smoke["eval_before_train"], smoke["ckpt_interval"]) == (
        2,
        -1,
        False,
        2,
    )
    assert smoke["fully_async"]["max_staleness_steps"] == 4
    on_policy = rendered(async_rl.ON_POLICY)["trainer"]["fully_async"]
    assert (on_policy["max_staleness_steps"], on_policy["num_parallel_generation_workers"]) == (0, 32)


def test_qwen_keeps_fsdp_and_its_curriculum_learning_rate():
    trainer = rendered(policy=QWEN_POLICY)["trainer"]
    assert async_rl.DEFAULT.scale(QWEN_POLICY).num_nodes == 2
    assert async_rl.DEFAULT.scale(SNOWBALL_POLICY).num_nodes == 5
    assert trainer["strategy"] == "fsdp2" and "megatron_config" not in trainer["policy"] and "ref" not in trainer
    assert trainer["policy"]["optimizer_config"]["lr"] == 2e-6
    assert trainer["optimizer_state_metrics"] is False and trainer["algorithm"]["ratio_diagnostics"] == {"pooled": False}


def test_settings_change_existing_keys_and_reject_unknown_ones():
    config = rendered(settings=("trainer.fully_async.max_staleness_steps=1", "generator.gpu_memory_utilization=0.8"))
    assert config["trainer"]["fully_async"]["max_staleness_steps"] == 1
    assert config["generator"]["gpu_memory_utilization"] == 0.8
    with pytest.raises(click.BadParameter, match="unknown setting"):
        rendered(settings=("trainer.fully_async.max_stalness_steps=1",))
    with pytest.raises(click.BadParameter, match="entrypoint cannot change"):
        rendered(settings=("entrypoint=standard",))
    with pytest.raises(click.BadParameter, match=r"dotted\.key=value"):
        rendered(settings=("trainer.max_steps",))
    with pytest.raises(click.BadParameter, match="holds a value, not a section"):
        rendered(settings=("+trainer.max_steps.more=1",))
    added = rendered(settings=("+trainer.fully_async.weight_sync_interval=1",))
    assert added["trainer"]["fully_async"]["weight_sync_interval"] == 1


def test_presets_score_the_right_suites_and_fit_the_prompt_budget():
    assert async_rl.DEFAULT.scale(SNOWBALL_POLICY).evals == "math500,gsm8k-0shot"
    assert async_rl.SMOKE_PRESET.scale(SNOWBALL_POLICY).evals == "gsm8k-smoke"


def test_run_wires_the_curriculum_pool_and_snowball_export_into_one_chain(monkeypatch):
    monkeypatch.setattr("marin.experiment.namespacing.username_segment", lambda: "alice")
    monkeypatch.setattr(async_rl, "username_segment", lambda: "alice")
    run = async_rl.build_run(policy=SNOWBALL_POLICY, preset=async_rl.SMOKE_PRESET, version="2026.09.18")
    assert run.rl.name == "users/alice/checkpoints/async-rl/snowball-smoke"
    assert any(dep.name == async_rl.POOL_ARTIFACT_NAME for dep in run.rl.deps)
    assert any(dep.name == SNOWBALL_POLICY.adopted_model.name for dep in run.rl.deps)
    assert run.evaluation.deps == (run.rl,)
    changed = async_rl.build_run(
        policy=SNOWBALL_POLICY,
        preset=async_rl.SMOKE_PRESET,
        version="2026.09.18",
        settings=("trainer.fully_async.max_staleness_steps=1",),
    )
    assert changed.rl.name.startswith("users/alice/checkpoints/async-rl/snowball-smoke-set-")
    assert changed.rl.name != run.rl.name


def test_command_plans_without_running(monkeypatch):
    monkeypatch.setattr("marin.experiment.namespacing.username_segment", lambda: "alice")
    monkeypatch.setattr(async_rl, "username_segment", lambda: "alice")
    result = CliRunner().invoke(async_rl.main, ["--version", "2026.09.18", "--preset", "smoke"])
    assert result.exit_code == 0, result.output
    assert "snowball-smoke" in result.output
