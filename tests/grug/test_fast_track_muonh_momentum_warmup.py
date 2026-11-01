# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track MuonH momentum warmup: the MuonH momentum starts at ``MUONH_MOMENTUM_WARMUP_START`` and rises
linearly to ``momentum``."""

import dataclasses

import jax
import numpy as np

from experiments.grug.fast_track.optimizer import MUONH_MOMENTUM_WARMUP_START, GrugMoeMuonHConfig

TOTAL_STEPS = 100


def _params():
    return {"proj": jax.random.normal(jax.random.key(0), (16, 8))}


def _grads(step: int):
    return {"proj": jax.random.normal(jax.random.key(step + 1), (16, 8))}


def _updates(config: GrugMoeMuonHConfig, num_steps: int):
    optimizer = config.build(TOTAL_STEPS)
    params = _params()
    state = optimizer.init(params)
    updates = []
    for step in range(num_steps):
        update, state = optimizer.update(_grads(step), state, params)
        updates.append(np.asarray(update["proj"], np.float32))
    return updates, state


def test_momentum_warmup_starts_low_and_reaches_momentum():
    base = GrugMoeMuonHConfig(learning_rate=0.01, momentum=0.98, warmup=0.0)
    warm = dataclasses.replace(base, muonh_momentum_warmup=0.1)

    warm_updates, warm_state = _updates(warm, 1)
    start_updates, _ = _updates(dataclasses.replace(base, momentum=MUONH_MOMENTUM_WARMUP_START), 1)
    final_updates, _ = _updates(base, 1)

    # Step 0 uses the warmup start, not the final momentum (Nesterov reads it on the first step).
    np.testing.assert_allclose(warm_updates[0], start_updates[0], rtol=1e-6, atol=1e-6)
    assert not np.allclose(warm_updates[0], final_updates[0], rtol=1e-3, atol=1e-5)

    schedule_end = int(0.1 * TOTAL_STEPS)
    _, state_at_end = _updates(warm, schedule_end + 1)
    assert float(state_at_end.hyperparams["muonh_momentum"]) == np.float32(0.98)
    assert float(warm_state.hyperparams["muonh_momentum"]) > MUONH_MOMENTUM_WARMUP_START
