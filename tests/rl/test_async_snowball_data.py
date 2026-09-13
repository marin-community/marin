# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import yaml
from click.testing import CliRunner
from marin.execution.lazy import StepContext

from experiments.post_training import async_snowball


def test_explicit_snowball_rows_reach_data_writer_and_identity():
    args = dict(
        version="2026.09.08.34", scale=async_snowball.Scale.CADENCE_GATE, timeout_seconds=1350, completion="metrics"
    )
    baseline = async_snowball.build_experiment(**args)
    selected = async_snowball.build_experiment(**args, train_rows=128, validation_rows=128, dataloader_workers=0)
    data = selected.deps[1]
    config = data.build_config(StepContext.for_fingerprint(data.runtime_args, data.deps))
    assert isinstance(config, async_snowball.SnowballDataConfig)
    assert config.train_rows == 128 and config.validation_rows == 128
    assert selected.fingerprint() != baseline.fingerprint()
    request = selected.build_config(StepContext.for_fingerprint(selected.runtime_args, selected.deps)).request
    assert yaml.safe_load(request.config_yaml)["data"]["num_workers"] == 0


def test_snowball_row_and_worker_defaults_are_cli_equivalent():
    args = ["--version", "2026.09.08.34", "--scale", "cadence-gate", "--completion", "metrics"]
    default = CliRunner().invoke(async_snowball.main, args)
    explicit = CliRunner().invoke(
        async_snowball.main,
        [
            *args,
            "--train-rows",
            "1024",
            "--validation-rows",
            "128",
            "--dataloader-workers",
            "0",
            "--minibatches",
            "1",
        ],
    )
    assert default.exit_code == explicit.exit_code == 0
    assert json.loads(default.output) == json.loads(explicit.output)


@pytest.mark.parametrize("minibatches", [1, 2, 4])
def test_snowball_explicit_update_geometry_reaches_request(minibatches):
    step = async_snowball.build_experiment(
        version="2026.09.13.1",
        scale=async_snowball.Scale.QUALIFICATION,
        timeout_seconds=3600,
        completion="metrics",
        minibatches=minibatches,
        updates=28,
        eval_interval=5,
    )
    request = step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps)).request
    config = yaml.safe_load(request.config_yaml)
    assert config["entrypoint"] == "fully_async"
    assert config["trainer"]["max_steps"] == config["trainer"]["ckpt_interval"] == 28
    assert config["trainer"]["eval_interval"] == 5
    assert config["trainer"]["train_batch_size"] == request.topology.role_plan.train_batch_size == 32 * minibatches
    assert config["trainer"]["policy_mini_batch_size"] == request.topology.role_plan.policy_mini_batch_size == 32
    assert config["trainer"]["update_epochs_per_batch"] == 1


def test_snowball_n4_update_geometry_reaches_request_through_cli():
    result = CliRunner().invoke(
        async_snowball.main,
        [
            "--version",
            "2026.09.13.1",
            "--scale",
            "qualification",
            "--completion",
            "metrics",
            "--minibatches",
            "4",
            "--updates",
            "28",
            "--eval-interval",
            "5",
        ],
    )
    assert result.exit_code == 0, result.output
    request = json.loads(result.output)["request"]
    config = yaml.safe_load(request["config_yaml"])
    assert config["trainer"]["max_steps"] == 28
    assert config["trainer"]["eval_interval"] == 5
    assert request["topology"]["role_plan"]["train_batch_size"] == 128
    assert request["topology"]["role_plan"]["policy_mini_batch_size"] == 32


@pytest.mark.parametrize("minibatches", [2, 4])
def test_snowball_incomplete_prepared_cohort_rejects_before_request(minibatches):
    with pytest.raises(ValueError, match="complete prepared cohort"):
        async_snowball.build_experiment(
            version="2026.09.13.1",
            scale=async_snowball.Scale.QUALIFICATION,
            timeout_seconds=3600,
            completion="metrics",
            minibatches=minibatches,
            updates=25,
            eval_interval=5,
        )


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"minibatches": 3, "updates": 24}, "one of 1, 2 or 4"),
        ({"runner": async_snowball.Runner.SYNC, "minibatches": 2, "updates": 24}, "async runner"),
        ({"backend": async_snowball.Backend.FSDP2, "minibatches": 2, "updates": 24}, "Megatron backend"),
        ({"minibatches": 2}, "explicit total updates"),
        ({"updates": 24, "study_steps": 24}, "updates or study_steps"),
    ],
)
def test_snowball_rejects_unsupported_update_geometry(changes, message):
    with pytest.raises(ValueError, match=message):
        async_snowball.training_config(async_snowball.Scale.QUALIFICATION, **changes)


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_snowball_rejects_invalid_loader_workers(value):
    with pytest.raises(ValueError, match="nonnegative integer"):
        async_snowball.training_config(async_snowball.Scale.CADENCE_GATE, dataloader_workers=value)


@pytest.mark.parametrize("mode", ["blocking", "background"])
def test_snowball_eval_scheduling_is_opt_in_and_fingerprinted(mode):
    args = dict(
        version="2026.09.08.35", scale=async_snowball.Scale.CADENCE_GATE, completion="metrics", timeout_seconds=1350
    )

    def request(**extra):
        step = async_snowball.build_experiment(**args, **extra)
        return step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps)).request

    baseline = request()
    explicit = request(eval_on_installed_weights=False, eval_mode="blocking")
    selected = request(eval_on_installed_weights=True, eval_mode=mode)
    assert baseline.run_id == explicit.run_id and baseline.config_yaml == explicit.config_yaml
    assert selected.run_id != baseline.run_id
    config = yaml.safe_load(selected.config_yaml)
    assert config["trainer"]["fully_async"].pop("eval_on_installed_weights") is True
    assert config["trainer"]["fully_async"].pop("eval_mode") == mode
    assert config == yaml.safe_load(baseline.config_yaml)


@pytest.mark.parametrize(
    "changes",
    [
        {"runner": async_snowball.Runner.SYNC, "eval_on_installed_weights": True},
        {"eval_mode": "background"},
        {"eval_mode": "invalid"},
        {"eval_on_installed_weights": 1},
    ],
)
def test_snowball_eval_scheduling_rejects_unsupported_modes(changes):
    with pytest.raises(ValueError, match=r"Evaluation|evaluation"):
        async_snowball.training_config(async_snowball.Scale.CADENCE_GATE, **changes)


def test_snowball_background_eval_cli_reaches_native_recipe():
    result = CliRunner().invoke(
        async_snowball.main,
        [
            "--version",
            "2026.09.08.35",
            "--scale",
            "cadence-gate",
            "--completion",
            "metrics",
            "--eval-on-installed-weights",
            "--eval-mode",
            "background",
        ],
    )
    assert result.exit_code == 0, result.output
    config = yaml.safe_load(json.loads(result.output)["request"]["config_yaml"])
    assert config["trainer"]["fully_async"]["eval_on_installed_weights"] is True
    assert config["trainer"]["fully_async"]["eval_mode"] == "background"


@pytest.mark.parametrize("scale", list(async_snowball.Scale))
@pytest.mark.parametrize("runner", list(async_snowball.Runner))
def test_snowball_native_recipe_disables_unqualified_gradient_history(scale, runner):
    step = async_snowball.build_experiment(
        version="2026.09.08.35", scale=scale, runner=runner, completion="metrics", timeout_seconds=900
    )
    request = step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps)).request
    config = yaml.safe_load(request.config_yaml)
    assert config["trainer"]["algorithm"]["grad_cosine"] == {"enabled": False, "store": "cpu_bf16"}
