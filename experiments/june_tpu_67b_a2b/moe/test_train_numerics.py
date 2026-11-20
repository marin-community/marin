# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for non-finite Grug steps failing before checkpoint export."""

import re

import jax.numpy as jnp
import pytest

from experiments.june_tpu_67b_a2b.moe.train import _check_step_numerics


def _metrics(*, loss: float = 3.49, params_finite: bool = True) -> dict:
    return {
        "train/loss": jnp.array(loss),
        "train/params_finite": jnp.array(params_finite),
        "train/qb_betas_finite": jnp.array(True),
    }


@pytest.mark.parametrize("loss", [jnp.nan, jnp.inf])
def test_nonfinite_loss_fails_training(loss):
    with pytest.raises(FloatingPointError, match="loss"):
        _check_step_numerics(_metrics(loss=loss), step=2, diagnose_numerics=False)


def test_nonfinite_params_fail_without_numeric_diagnostics():
    with pytest.raises(FloatingPointError, match="update"):
        _check_step_numerics(_metrics(params_finite=False), step=1, diagnose_numerics=False)


def test_numeric_diagnostics_fail_on_a_nonfinite_update_that_left_params_finite():
    metrics = {
        **_metrics(),
        "train/grads_finite": jnp.array(True),
        "train/updates_finite": jnp.array(False),
        "train/supervised_tokens": jnp.array(100),
    }
    with pytest.raises(FloatingPointError, match=re.escape("'train/updates_finite': 0.0")):
        _check_step_numerics(metrics, step=1, diagnose_numerics=True)
