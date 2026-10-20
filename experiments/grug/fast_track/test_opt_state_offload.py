# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Optimizer-state placement is a memory/scheduling choice: keeping all or part of the MuonH state on device
(``OptStateOffload.NONE`` / ``EXPERT_DOWN``) trains exactly as offloading all of it to pinned host memory does."""

import pytest
from levanter.testing.cpu_devices import run_on_cpu_devices

from experiments.grug.fast_track.train import OptStateOffload, _offloads_leaf

_SCRIPT = """
import sys, types
import jax, jax.numpy as jnp, jmp, numpy as np

# The expert Newton-Schulz calls QuACK's GPU-only symmetric GEMM; X @ X^T stands in for it on CPU.
quack = types.ModuleType("levanter.grug._moe.quack_symmetric_cute")
quack.quack_symmetric_gemm = lambda x: jnp.einsum("lik,ljk->lij", x, x)
sys.modules[quack.__name__] = quack

from jax.sharding import AxisType, Mesh
from levanter.data.text.examples import GrugLmExample
from levanter.grug.attention import AttentionMask
import experiments.grug.fast_track.test_kda_local as t
from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig
from experiments.grug.fast_track.train import OptStateOffload, _flat_train_step, _make_train_step, initial_state

STEPS = 3
mesh = Mesh(np.array(jax.devices(), dtype=object).reshape((1, 2, 2, 1)), ("replica_dcn", "data", "expert", "model"),
            axis_types=(AxisType.Explicit,) * 4)

def train(offload):
    cfg = t._config(scan_layers=True)
    optimizer = GrugMoeMuonHConfig(learning_rate=0.02, adam_lr=0.01, warmup=0.0).build(STEPS)
    mp = jmp.get_policy("params=float32,compute=float32,output=float32")
    step_fn = _make_train_step(optimizer, mp, z_loss_weight=0.0, opt_state_offload=offload)
    with jax.set_mesh(mesh):
        state = jax.jit(lambda key: initial_state(cfg, optimizer=optimizer, mp=mp, key=key, ema_beta=None,
                                                  opt_state_offload=offload))(jax.random.PRNGKey(0))
        leaves, treedef = jax.tree.flatten(state)
        train_step = _flat_train_step(step_fn, treedef)
        losses = []
        for i in range(STEPS):
            tokens = jax.random.randint(jax.random.PRNGKey(10 + i), (4, t._SEQ), 0, t._VOCAB)
            tokens = jax.sharding.reshard(tokens, jax.P(("replica_dcn", "data", "expert"), None))
            batch = GrugLmExample(tokens=tokens, loss_weight=jnp.ones(tokens.shape), attn_mask=AttentionMask.causal())
            leaves, metrics, _ = train_step(leaves, batch)
            losses.append(float(metrics["train/loss"]))
        final = jax.tree.unflatten(treedef, leaves)
        return [np.asarray(x) for x in jax.tree.leaves(final.params)], losses

offloaded, offloaded_losses = train(OptStateOffload.ALL)
assert offloaded_losses[-1] != offloaded_losses[0]
for mode in (OptStateOffload.NONE, OptStateOffload.EXPERT_DOWN):
    params, losses = train(mode)
    assert losses == offloaded_losses, (mode, losses, offloaded_losses)
    for a, b in zip(offloaded, params, strict=True):
        np.testing.assert_array_equal(a, b)
"""


@pytest.mark.timeout(600)
def test_opt_state_placement_trains_like_the_offloaded_state():
    run_on_cpu_devices(_SCRIPT, device_count=4)


def test_expert_down_selects_only_the_expert_w_down_state():
    muon = ".inner_state.inner_states['muonh'].inner_state[0].momentum_buffer.stacked_blocks.stacked.mlp.expert_mlp"
    assert _offloads_leaf(OptStateOffload.EXPERT_DOWN, f"{muon}.w_down")
    assert not _offloads_leaf(OptStateOffload.EXPERT_DOWN, f"{muon}.w_up")
    assert not _offloads_leaf(
        OptStateOffload.EXPERT_DOWN, ".inner_state.inner_states['adam'].inner_state[0].mu.token_embed"
    )
