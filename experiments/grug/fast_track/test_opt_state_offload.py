# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Optimizer-state placement is a memory/scheduling choice: keeping the MuonH state on device
(``offload_opt_state=False``) trains exactly as offloading it to pinned host memory between steps does."""

import pytest
from levanter.testing.cpu_devices import run_on_cpu_devices

_SCRIPT = """
import jax, jax.numpy as jnp, jmp, numpy as np
from jax.sharding import AxisType, Mesh
from levanter.data.text.examples import GrugLmExample
from levanter.grug.attention import AttentionMask
import experiments.grug.fast_track.test_kda_local as t
from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig
from experiments.grug.fast_track.train import _flat_train_step, _make_train_step, initial_state

STEPS = 3
mesh = Mesh(np.array(jax.devices(), dtype=object).reshape((1, 2, 1, 1)), ("replica_dcn", "data", "expert", "model"),
            axis_types=(AxisType.Explicit,) * 4)

def train(offload):
    # Dense MLPs: the 4D expert stacks orthogonalize with the GPU-only QuACK symmetric GEMM.
    cfg = t._config(scan_layers=True, dense_mlp=True)
    optimizer = GrugMoeMuonHConfig(learning_rate=0.02, adam_lr=0.01, warmup=0.0).build(STEPS)
    mp = jmp.get_policy("params=float32,compute=float32,output=float32")
    step_fn = _make_train_step(optimizer, mp, z_loss_weight=0.0, offload_opt_state=offload)
    with jax.set_mesh(mesh):
        state = jax.jit(lambda key: initial_state(cfg, optimizer=optimizer, mp=mp, key=key, ema_beta=None,
                                                  offload_opt_state=offload))(jax.random.PRNGKey(0))
        leaves, treedef = jax.tree.flatten(state)
        train_step = _flat_train_step(step_fn, treedef)
        losses = []
        for i in range(STEPS):
            tokens = jax.random.randint(jax.random.PRNGKey(10 + i), (2, t._SEQ), 0, t._VOCAB)
            tokens = jax.sharding.reshard(tokens, jax.P(("replica_dcn", "data", "expert"), None))
            batch = GrugLmExample(tokens=tokens, loss_weight=jnp.ones(tokens.shape), attn_mask=AttentionMask.causal())
            leaves, metrics, _ = train_step(leaves, batch)
            losses.append(float(metrics["train/loss"]))
        final = jax.tree.unflatten(treedef, leaves)
        return [np.asarray(x) for x in jax.tree.leaves(final.params)], losses

offloaded, offloaded_losses = train(True)
device, device_losses = train(False)
assert offloaded_losses == device_losses, (offloaded_losses, device_losses)
assert offloaded_losses[-1] != offloaded_losses[0]
for a, b in zip(offloaded, device, strict=True):
    np.testing.assert_array_equal(a, b)
"""


@pytest.mark.timeout(600)
def test_device_opt_state_trains_like_the_offloaded_state():
    run_on_cpu_devices(_SCRIPT, device_count=2)
