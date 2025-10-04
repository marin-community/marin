# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
from levanter.optim.model_averaging import EmaDecaySqrtModelAveraging


def _dummy(v):
    return {"p": jnp.array(v, jnp.float32)}


def test_ema_phase():
    ma = EmaDecaySqrtModelAveraging(model=_dummy(0.0), beta=0.5, switch_step=5, decay_steps=10)
    ma = ma.update(_dummy(1.0), step=0)
    assert jnp.allclose(ma.model["p"], 1.0)
    assert jnp.isclose(ma.tot_weight, 0.5)


def test_decay_phase():
    ma = EmaDecaySqrtModelAveraging(model=_dummy(0.0), beta=0.0, switch_step=0, decay_steps=4)
    ma = ma.update(_dummy(1.0), step=0)
    ma_end = ma.update(_dummy(10.0), step=4)
    assert jnp.allclose(ma_end.model["p"], 1.0)
    assert jnp.isclose(ma_end.tot_weight, 1.0)
