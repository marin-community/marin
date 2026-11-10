# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for non-finite Grug steps failing before checkpoint export."""

import jax.numpy as jnp
import pytest

from experiments.june_tpu_67b_a2b.moe.train import _check_step_numerics


def test_nonfinite_update_fails_before_next_step():
    metrics = {
        "train/loss": jnp.array(3.49),
        "train/loss_finite": jnp.array(True),
        "train/grads_finite": jnp.array(True),
        "train/updates_finite": jnp.array(False),
        "train/params_finite": jnp.array(False),
        "train/qb_betas_finite": jnp.array(True),
        "train/supervised_tokens": jnp.array(100),
    }
    with pytest.raises(FloatingPointError):
        _check_step_numerics(metrics, step=1, diagnose_numerics=True)
