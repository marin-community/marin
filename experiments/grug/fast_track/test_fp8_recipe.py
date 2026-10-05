# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``fp8_recipe=mxfp8_sim``: on a 4-device data x expert mesh, the model trains through simulated MXFP8
projections and pooled-wave experts, landing within MXFP8 rounding of the bf16 model, with exact first-layer
routing (the router GEMM stays unquantized)."""

import os
import pathlib
import subprocess
import sys
import textwrap

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

_SCRIPT = """
import equinox as eqx, jax, jax.numpy as jnp, numpy as np
from jax.sharding import AxisType, Mesh, PartitionSpec as P, reshard
import experiments.grug.fast_track.test_ngram_stat as t
from experiments.grug.fast_track.model import ROUTING_SELECTED_KEY, Fp8Recipe, Transformer

mesh = Mesh(np.array(jax.devices()[:4]).reshape(1, 2, 2, 1), ("replica_dcn", "data", "expert", "model"),
            axis_types=(AxisType.Explicit,) * 4)


def run(recipe):
    cfg = t._config(ngram_stat_rows=0, hidden_dim=64, moe_ungated_relu2=True, moe_ungated_kernel=True,
                    shared_ungated_relu2=True, capacity_factor=4.0, pooled_transport_capacity_factor=4.0,
                    fp8_recipe=recipe)
    with jax.set_mesh(mesh):
        model = Transformer.init(cfg, key=jax.random.PRNGKey(0))
        tokens = reshard(jax.random.randint(jax.random.PRNGKey(1), (4, t._SEQ), 0, t._VOCAB),
                         P(("replica_dcn", "data", "expert"), None))
        weight = reshard(jnp.ones(tokens.shape), P(("replica_dcn", "data", "expert"), None))
        loss, grads = eqx.filter_jit(eqx.filter_value_and_grad(lambda m: m.next_token_loss(tokens, weight)))(model)
        _, metrics = eqx.filter_jit(lambda m: m(tokens, return_routing=True))(model)
    leaves = [np.asarray(g) for g in jax.tree.leaves(eqx.filter(grads, eqx.is_inexact_array))]
    return float(loss), leaves, np.asarray(metrics[ROUTING_SELECTED_KEY])


loss_ref, grads_ref, sel_ref = run(Fp8Recipe.NONE)
loss_mx, grads_mx, sel_mx = run(Fp8Recipe.MXFP8_SIM)
assert loss_mx != loss_ref, loss_mx
assert abs(loss_mx - loss_ref) / loss_ref < 0.02, (loss_mx, loss_ref)
np.testing.assert_array_equal(np.sort(sel_mx[0], -1), np.sort(sel_ref[0], -1))
for g_ref, g_mx in zip(grads_ref, grads_mx, strict=True):
    norm = np.linalg.norm(g_ref)
    if norm > 0:
        assert np.linalg.norm(g_mx - g_ref) / norm < 0.3, (g_ref.shape, np.linalg.norm(g_mx - g_ref) / norm)
print("FP8_OK")
"""


def test_mxfp8_sim_trains_close_to_bf16_with_exact_first_layer_routing():
    root = str(_REPO_ROOT)
    env = dict(os.environ)
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    env["JAX_PLATFORMS"] = "cpu"
    env["PYTHONPATH"] = os.pathsep.join(
        [root, f"{root}/lib/levanter/src", f"{root}/lib/haliax/src", env.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_SCRIPT)], env=env, cwd=root, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr[-4000:]
    assert "FP8_OK" in result.stdout
