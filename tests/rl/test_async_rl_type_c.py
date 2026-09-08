# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import yaml
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


def test_async_multiple_minibatches_are_rejected():
    with pytest.raises(ValueError, match="synchronous"):
        recipe.training_config(
            recipe.Runner.ASYNC, recipe.Scale.SCREENING, spans=True, staleness=0, minibatches=4, updates=8
        )
