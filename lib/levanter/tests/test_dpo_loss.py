# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp

from levanter.main.train_dpo import dpo_loss_from_logps


def test_dpo_loss_decreases_with_margin():
    delta_ref = jnp.array([0.0, 0.0])

    loss_small, metrics_small = dpo_loss_from_logps(jnp.array([0.1, 0.2]), delta_ref, beta=1.0)
    loss_large, metrics_large = dpo_loss_from_logps(jnp.array([1.0, 1.2]), delta_ref, beta=1.0)

    assert float(loss_large) < float(loss_small)
    assert float(metrics_large["dpo_accuracy"]) >= float(metrics_small["dpo_accuracy"])
