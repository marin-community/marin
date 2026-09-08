# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import yaml
from marin.execution.lazy import StepContext

from experiments.post_training import async_snowball


@pytest.mark.parametrize("scale", list(async_snowball.Scale))
@pytest.mark.parametrize("runner", list(async_snowball.Runner))
def test_snowball_native_recipe_disables_unqualified_gradient_history(scale, runner):
    step = async_snowball.build_experiment(
        version="2026.09.08.35", scale=scale, runner=runner, completion="metrics", timeout_seconds=900
    )
    request = step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps)).request
    config = yaml.safe_load(request.config_yaml)
    assert config["trainer"]["algorithm"]["grad_cosine"] == {"enabled": False, "store": "cpu_bf16"}
