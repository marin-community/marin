# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ID Balancing (``id_balance_gain``): the integral-derivative bias update, and the train step that carries it."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
from jax.sharding import AxisType, Mesh
from levanter.data.text.examples import GrugLmExample
from levanter.grug.attention import AttentionMask

from experiments.grug.fast_track.id_balancing import id_balance_update
from experiments.grug.fast_track.model import GrugModelConfig
from experiments.grug.fast_track.train import _make_train_step, initial_state

_SEQ = 32
_BATCH = 4
_GAIN = 0.5


def test_update_integrates_the_error_and_gates_the_derivative():
    gain = 0.1
    zeros = jnp.zeros((1, 4))
    # Step 0: no previous error, so the derivative gate is closed and the bias is the integral term alone.
    counts0 = jnp.array([[6.0, 2.0, 4.0, 4.0]])  # mean 4: errors -0.5, 0.5, 0, 0
    bias, error, gate = id_balance_update(zeros, zeros, counts0, gain)
    np.testing.assert_allclose(error, [[-0.5, 0.5, 0.0, 0.0]])
    np.testing.assert_array_equal(gate, zeros)
    np.testing.assert_allclose(bias, [[-0.05, 0.05, 0.0, 0.0]], atol=1e-7)

    # Step 1: expert 0's overload grows (gate opens), expert 1's underload shrinks (closed), and experts 2 and 3 had
    # no previous error (closed) though they are now overloaded and underloaded.
    counts1 = jnp.array([[8.0, 3.0, 4.5, 0.5]])  # mean 4: errors -1, 0.25, -0.125, 0.875
    bias1, error1, gate1 = id_balance_update(bias, error, counts1, gain)
    np.testing.assert_array_equal(gate1, [[1.0, 0.0, 0.0, 0.0]])
    raw = np.asarray(bias) + gain * np.asarray(error1)
    raw[0, 0] += gain * (float(error1[0, 0]) - float(error[0, 0]))
    np.testing.assert_allclose(bias1, raw - raw.mean(axis=-1, keepdims=True), atol=1e-7)
    np.testing.assert_allclose(jnp.mean(bias1, axis=-1), 0.0, atol=1e-7)


def _config(**overrides) -> GrugModelConfig:
    kwargs = dict(
        vocab_size=64,
        hidden_dim=32,
        intermediate_dim=16,
        shared_expert_intermediate_dim=16,
        num_shared_experts=1,
        num_experts=8,
        num_experts_per_token=2,
        latent_dim=16,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        local_kv_heads=1,
        global_kv_heads=1,
        head_dim=16,
        max_seq_len=_SEQ,
        sliding_window=_SEQ,
        global_every=2,
        capacity_factor=8.0,
        pooled_transport_capacity_factor=8.0,
        initializer_std=0.5,
    )
    kwargs.update(overrides)
    return GrugModelConfig(**kwargs)


def test_train_steps_carry_the_id_bias_and_balance_the_load():
    """With the weights frozen, the train step feeds each step's routing counts through the ID update, the next
    step routes with that bias, and the controller drives a skewed router's load toward balance."""
    cfg = _config(id_balance_gain=_GAIN)
    mesh = Mesh(
        np.array(jax.devices()[:1], dtype=object).reshape((1, 1, 1, 1)),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    tx = optax.set_to_zero()
    mp = jmp.get_policy("params=float32,compute=float32,output=float32")
    step = eqx.filter_jit(_make_train_step(tx, mp, z_loss_weight=1e-4, id_balance_gain=_GAIN))
    tokens = jax.random.randint(jax.random.key(1), (_BATCH, _SEQ), 0, cfg.vocab_size)
    with jax.set_mesh(mesh):
        batch = GrugLmExample(
            tokens=tokens,
            loss_weight=jnp.ones(tokens.shape, jnp.float32),
            attn_mask=AttentionMask(is_causal=True),
        )
        state = initial_state(cfg, optimizer=tx, mp=mp, key=jax.random.key(0))
        max_vio = []
        for i in range(30):
            bias_before = -np.asarray(state.pending_qb_betas)
            prev_error = state.id_balance_prev_error
            state, metrics, _ = step(state, batch)
            counts = metrics["train/router/routing_counts_per_layer"]
            expected, error, _ = id_balance_update(jnp.asarray(bias_before), prev_error, counts, _GAIN)
            np.testing.assert_allclose(-np.asarray(state.pending_qb_betas), expected, atol=1e-6)
            np.testing.assert_allclose(state.id_balance_prev_error, error, atol=1e-6)
            if i == 0:
                np.testing.assert_array_equal(bias_before, 0.0)
                assert float(metrics["train/router/id_balance/gate_open_frac"]) == 0.0
            max_vio.append(float(metrics["train/router/id_balance/max_vio_max"]))

    assert max_vio[0] > 0.3, max_vio
    assert max(max_vio[-5:]) < 0.5 * max_vio[0], max_vio
