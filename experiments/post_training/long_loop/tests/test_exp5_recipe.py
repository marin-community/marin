# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exp5 recipe behavior: base-rate special-token updates with whole-run annealing."""

import json

import draccus
import pytest
from fray.types import ResourceConfig
from iris.rpc.proto_display import priority_band_value
from levanter.main.train_lm import TrainLmConfig
from marin.execution.lazy import StepContext

from experiments.post_training.long_loop.exp3_recipe import exp3_controls
from experiments.post_training.long_loop.exp5 import (
    SPECIAL_TOKEN_LR_MULTIPLIER,
    exp5_sft_controls,
    exp5_sft_spec,
)
from experiments.post_training.long_loop.tests.test_exp3_production import (
    preview_release as preview_release,
    production_recipe as production_recipe,
)
from experiments.post_training.long_loop.tests.test_initializer_stage import (
    prepared_sources as prepared_sources,
    staged_initializer as staged_initializer,
)
from experiments.sft.launcher import levanter_train_request, sft_step

SFT_UPDATES = 360

ANNEALED_SCHEDULE = {
    "horizon": SFT_UPDATES,
    "warmup": 0,
    "decay": SFT_UPDATES,
    "min_lr_ratio": 0.1,
    "lr_schedule": "linear",
}


def exp4_update_transforms(recipe: dict) -> dict:
    """Apply the Exp4 .76 SFT update transforms that Exp5 copies forward."""
    recipe["stop_after_steps"] = SFT_UPDATES
    recipe["trainer"]["num_train_steps"] = SFT_UPDATES
    recipe["optimizer_schedule"] = dict(ANNEALED_SCHEDULE)
    return recipe


def test_exp5_controls_hold_the_ruled_multiplier_by_construction():
    controls = exp5_sft_controls()
    baseline = exp3_controls()
    assert SPECIAL_TOKEN_LR_MULTIPLIER == 1.0
    assert controls.special_token_lr_multiplier == 1.0
    assert controls.special_token_ids == baseline.special_token_ids
    assert controls.anchor_token_ids == baseline.anchor_token_ids
    assert controls.freeze_router_bias == baseline.freeze_router_bias
    override = exp5_sft_controls(special_token_lr_multiplier=2.0)
    assert override.special_token_lr_multiplier == 2.0
    assert override.special_token_ids == baseline.special_token_ids


def test_exp5_spec_binds_the_base_rate_and_the_annealing_rule(production_recipe, tmp_path):
    recipe = exp4_update_transforms(production_recipe)
    spec = exp5_sft_spec("exp5-recipe-test", "2026.11.19.1", recipe)
    assert spec.snowball_sft.special_token_lr_multiplier == 1.0
    assert spec.num_train_steps == SFT_UPDATES
    assert spec.optimizer_num_train_steps == SFT_UPDATES
    resources = draccus.decode(ResourceConfig, recipe["resources"])
    step = sft_step(spec, resources)
    context = StepContext.for_run(str(tmp_path), str(tmp_path), runtime_args=step.runtime_args)
    pod = step.build_config(context)
    request = levanter_train_request(
        pod, task_timeout_seconds=recipe["task_timeout_seconds"], priority=priority_band_value("production")
    )
    [worker] = request.entrypoint.callable_entrypoint.args
    config = draccus.decode(TrainLmConfig, json.loads(json.dumps(draccus.encode(worker.train_config))))
    assert config.snowball_sft.special_token_lr_multiplier == 1.0
    assert config.optimizer_num_train_steps == SFT_UPDATES
    assert config.trainer.num_train_steps == SFT_UPDATES
    assert config.trainer.training_budget.duration.total_seconds() == recipe["task_timeout_seconds"]


def test_exp5_spec_honors_an_explicit_multiplier_override(production_recipe):
    recipe = exp4_update_transforms(production_recipe)
    recipe["special_token_lr_multiplier"] = 2.0
    spec = exp5_sft_spec("exp5-recipe-test", "2026.11.19.1", recipe)
    assert spec.snowball_sft.special_token_lr_multiplier == 2.0


def test_exp5_spec_rejects_a_horizon_that_differs_from_the_updates_run(production_recipe):
    recipe = exp4_update_transforms(production_recipe)
    recipe["optimizer_schedule"] = {**ANNEALED_SCHEDULE, "horizon": 4750}
    with pytest.raises(ValueError, match="horizon must equal num_train_steps"):
        exp5_sft_spec("exp5-recipe-test", "2026.11.19.1", recipe)
