# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import yaml
from click.testing import CliRunner
from marin.execution.lazy import materialized_config

from experiments.post_training import async_rl as recipe


def test_four_minibatches_resolve_eight_updates_to_two_rollout_batches():
    step, _ = recipe.build_experiment(
        version="2026.09.08.90",
        cluster="cw-us-east-02a",
        runner=recipe.Runner.SYNC,
        scale=recipe.Scale.SCREENING,
        completion="metrics",
        minibatches=4,
        updates=8,
        eval_updates=8,
        epoch_seeded_shuffle=True,
    )
    request = materialized_config(step, "s3://marin-us-east-02a/marin").request
    cfg = yaml.safe_load(request.config_yaml)
    assert cfg["trainer"]["train_batch_size"] == request.topology.role_plan.train_batch_size == 256
    assert cfg["trainer"]["policy_mini_batch_size"] == request.topology.role_plan.policy_mini_batch_size == 64
    assert cfg["trainer"]["max_steps"] == cfg["trainer"]["eval_interval"] == cfg["trainer"]["ckpt_interval"] == 2
    assert cfg["trainer"]["update_epochs_per_batch"] == 1
    assert cfg["entrypoint"] == "standard"


@pytest.mark.parametrize(
    "options",
    [
        {"minibatches": 4},
        {"minibatches": 4, "updates": 9},
        {"minibatches": 4, "updates": 8, "eval_updates": 2},
        {"updates": 8, "eval_interval": 2},
        {"updates": 8, "screening_steps": 8},
        {"eval_updates": 8},
        {"minibatches": True},
    ],
)
def test_incompatible_update_and_batch_units_reject(options):
    with pytest.raises(ValueError):
        recipe.training_config(recipe.Runner.SYNC, recipe.Scale.SCREENING, spans=True, staleness=0, **options)


def test_async_more_than_two_minibatches_are_rejected():
    with pytest.raises(ValueError, match="one or two"):
        recipe.training_config(
            recipe.Runner.ASYNC, recipe.Scale.SCREENING, spans=True, staleness=0, minibatches=4, updates=8
        )


def test_async_n2_keeps_successful_update_schedule_and_complete_cohort():
    step, _ = recipe.build_experiment(
        version="2026.09.08.90",
        cluster="cw-us-east-02a",
        runner=recipe.Runner.ASYNC,
        scale=recipe.Scale.SCREENING,
        completion="metrics",
        minibatches=2,
        updates=96,
        eval_updates=32,
        staleness=0,
        weight_sync_interval=1,
        epoch_seeded_shuffle=True,
        kl_loss=False,
    )
    request = materialized_config(step, "s3://marin-us-east-02a/marin").request
    cfg = yaml.safe_load(request.config_yaml)
    assert cfg["trainer"]["train_batch_size"] == request.topology.role_plan.train_batch_size == 128
    assert cfg["trainer"]["policy_mini_batch_size"] == request.topology.role_plan.policy_mini_batch_size == 64
    assert cfg["trainer"]["max_steps"] == cfg["trainer"]["ckpt_interval"] == 96
    assert cfg["trainer"]["eval_interval"] == 32
    assert cfg["trainer"]["fully_async"].get("weight_sync_interval", 1) == 1
    assert cfg["trainer"]["fully_async"]["first_token_admission"] is True
    assert cfg["trainer"]["algorithm"].get("loss_reduction", "token_mean") == "token_mean"
    assert cfg["entrypoint"] == "fully_async"


@pytest.mark.parametrize("enabled,store", [(True, "gpu_fp32"), (False, "cpu_bf16"), (False, "off")])
def test_explicit_gradient_monitoring_reaches_native_config(enabled, store):
    step, _ = recipe.build_experiment(
        version="2026.09.08.90",
        cluster="cw-us-east-02a",
        runner=recipe.Runner.SYNC,
        scale=recipe.Scale.SCREENING,
        completion="metrics",
        grad_cosine=enabled,
        grad_cosine_store=store,
    )
    cfg = yaml.safe_load(materialized_config(step, "s3://marin-us-east-02a/marin").request.config_yaml)
    assert cfg["trainer"]["algorithm"]["grad_cosine"] == {"enabled": enabled, "store": store}


def test_gradient_override_absence_preserves_runtime_default():
    cfg = yaml.safe_load(recipe.training_config(recipe.Runner.SYNC, recipe.Scale.SCREENING, spans=True, staleness=0))
    assert "grad_cosine" not in cfg["trainer"]["algorithm"]


@pytest.mark.parametrize("runner,steps,eval_steps", [("sync", 48, 16), ("async", 96, 32)])
def test_actual_cli_preserves_optimizer_budget_with_runner_specific_native_clock(runner, steps, eval_steps):
    result = CliRunner().invoke(
        recipe.main,
        [
            "--version",
            "2026.09.10.1",
            "--runner",
            runner,
            "--stage",
            "rl",
            "--scale",
            "screening",
            "--completion",
            "metrics",
            "--minibatches",
            "2",
            "--updates",
            "96",
            "--eval-updates",
            "32",
            "--staleness",
            "0",
            "--weight-sync-interval",
            "1",
            "--no-kl-loss",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output + repr(result.exception)
    request = json.loads(result.output)["request"]
    cfg = yaml.safe_load(request["config_yaml"])
    assert cfg["trainer"]["max_steps"] == cfg["trainer"]["ckpt_interval"] == steps
    assert cfg["trainer"]["eval_interval"] == eval_steps
    assert cfg["trainer"]["train_batch_size"] == 128
    assert cfg["trainer"]["policy_mini_batch_size"] == 64
    assert request["topology"]["role_plan"]["train_batch_size"] == 128


@pytest.mark.parametrize("runner", ["sync", "async"])
def test_new_screen_default_uses_two_partitions_without_changing_update_count(runner):
    arguments = [
        "--version",
        "2026.09.10.1",
        "--runner",
        runner,
        "--scale",
        "screening",
        "--stage",
        "rl",
        "--completion",
        "metrics",
        "--updates",
        "8",
        "--no-kl-loss",
    ]
    default = CliRunner().invoke(recipe.main, arguments)
    explicit = CliRunner().invoke(recipe.main, [*arguments, "--minibatches", "2"])
    historical = CliRunner().invoke(recipe.main, [*arguments, "--minibatches", "1"])
    assert default.exit_code == explicit.exit_code == historical.exit_code == 0
    request, control, old = [json.loads(result.output)["request"] for result in (default, explicit, historical)]
    request.pop("attempt_id")
    control.pop("attempt_id")
    assert request == control
    assert request["run_id"] != old["run_id"]
    config = yaml.safe_load(request["config_yaml"])
    assert config["trainer"]["train_batch_size"] == 128
    assert config["trainer"]["max_steps"] == (4 if runner == "sync" else 8)


@pytest.mark.parametrize("arguments", [[], ["--updates", "8"], ["--no-kl-loss"]])
def test_legacy_screen_scopes_retain_single_partition_default(arguments):
    base = ["--version", "2026.09.10.1", "--stage", "rl", "--completion", "metrics", *arguments]
    default = CliRunner().invoke(recipe.main, base)
    explicit = CliRunner().invoke(recipe.main, [*base, "--minibatches", "1"])
    assert default.exit_code == explicit.exit_code == 0
    request, control = [json.loads(result.output)["request"] for result in (default, explicit)]
    request.pop("attempt_id")
    control.pop("attempt_id")
    assert request == control
