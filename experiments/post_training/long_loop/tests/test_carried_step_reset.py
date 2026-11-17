"""Carried-state replay changes only the trainer/data index."""

import jax.numpy as jnp
import numpy as np
import optax
import pytest

from experiments.post_training.long_loop.sft_runtime.train import GrugTrainState, reset_carried_trainer_step


def state_at(step):
    params = {"w": jnp.array([1.0, 2.0])}
    tx = optax.adam(5e-6)
    initial = tx.init(params)
    _, nonzero = tx.update({"w": jnp.array([0.25, -0.5])}, initial, params)
    return GrugTrainState(
        step=jnp.array(step, dtype=jnp.int32),
        params=params,
        master_params=None,
        opt_state=nonzero,
        ema_params=None,
        pending_qb_betas=jnp.array([0.0]),
    )


def test_reset_preserves_nonzero_adam_moments_and_counter():
    before = state_at(161750)
    after = reset_carried_trainer_step(before, load_checkpoint=True, allow_partial=False)
    assert int(after.step) == 0
    assert after.params is before.params
    assert after.opt_state is before.opt_state
    assert int(after.opt_state[0].count) == 1
    np.testing.assert_array_equal(after.opt_state[0].mu["w"], before.opt_state[0].mu["w"])
    assert np.any(np.asarray(after.opt_state[0].mu["w"]) != 0)
    np.testing.assert_array_equal(after.opt_state[0].nu["w"], before.opt_state[0].nu["w"])
    # The next optimizer update carries the count rather than reinitializing it.
    _, updated = optax.adam(5e-6).update({"w": jnp.array([0.1, 0.2])}, after.opt_state, after.params)
    assert int(updated[0].count) == 2


@pytest.mark.parametrize(
    "step,load,partial", [(0, True, False), (161750, False, False), (161750, None, False), (161750, True, True)]
)
def test_reset_rejects_non_restored_or_partial_state(step, load, partial):
    with pytest.raises(ValueError, match="fully restored"):
        reset_carried_trainer_step(state_at(step), load_checkpoint=load, allow_partial=partial)
