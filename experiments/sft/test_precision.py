# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import jax
import jax.numpy as jnp
import jmp
import numpy as np
import pytest
from fray.types import ResourceConfig
from levanter.data.text.datasets import LmDataConfig
from levanter.trainer_state import TrainerState
from marin.execution.lazy import StepContext

from experiments.sft.configs.snowball_h100 import SPEC


@pytest.mark.parametrize("params_dtype", [jnp.bfloat16, jnp.float32])
def test_sft_precision_controls_trainable_weights_and_update(tmp_path, params_dtype):
    spec = replace(
        SPEC,
        mp=jmp.Policy(param_dtype=params_dtype, compute_dtype=jnp.bfloat16, output_dtype=jnp.bfloat16),
    )
    pod = spec.model.build_train_config(
        StepContext.for_run(str(tmp_path / "run"), str(tmp_path)),
        spec,
        LmDataConfig(),
        ResourceConfig.with_cpu(),
        num_train_steps=15,
    )
    weights = jnp.array([1.0, -1.0], dtype=jnp.float32)
    state = TrainerState.init(spec.optimizer.build(15), weights, key=jax.random.key(0), mp=pod.train_config.trainer.mp)
    assert state.model.dtype == params_dtype
    updated, _ = state.take_step(jnp.array([0.25, -0.25], dtype=params_dtype), key=jax.random.key(1))
    assert updated.model.dtype == params_dtype
    assert np.all(np.isfinite(updated.model))
    if params_dtype == jnp.bfloat16:
        np.testing.assert_array_equal(updated.model, weights)
    else:
        assert np.all(np.abs(updated.model) < np.abs(weights))
