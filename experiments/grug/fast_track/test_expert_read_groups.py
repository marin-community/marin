# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``expert_read_groups``: each routed expert reads its group's normed slice of the stream through a real
``[E, W, I]`` ``w_up``, every new leaf trains and routes to the intended optimizer, and the pooled-wave EP
dispatch of the per-slot slices matches the explicit per-(token, expert) reference."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import experiments.grug.fast_track.test_kda_local as kda_test
from experiments.grug.fast_track.model import MoEMLP, _run_grouped_read_bank
from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TOKENS = 16
_TOPK = 2
_GROUPS = 2
_READ_GROUPS = dict(latent_dim=None, latent_out_dim=16, expert_read_groups=_GROUPS)


def reference_bank_output(mlp: MoEMLP, x, selected, weights, *, gated: bool) -> np.ndarray:
    """Explicit per-(token, slot) slice -> group norm -> expert MLP, summed with the combine weights."""
    em, norm = mlp.expert_mlp, mlp.expert_read_norm
    x, selected, weights = np.asarray(x, np.float64), np.asarray(selected), np.asarray(weights, np.float64)
    gain = np.asarray(norm.weight, np.float64)
    groups, width = gain.shape
    w_up, w_down = np.asarray(em.w_up, np.float64), np.asarray(em.w_down, np.float64)
    out = np.zeros((x.shape[0], w_down.shape[-1]))
    for t in range(x.shape[0]):
        for k in range(selected.shape[1]):
            e = selected[t, k]
            g = e % groups
            piece = x[t, g * width : (g + 1) * width]
            piece = piece / np.sqrt(np.mean(piece**2) + norm.eps) * gain[g]
            up = piece @ w_up[e]
            if gated:
                gate = piece @ np.asarray(em.w_gate, np.float64)[e]
                hidden = gate / (1.0 + np.exp(-gate)) * up
            else:
                hidden = np.maximum(up, 0.0) ** 2
            out[t] += weights[t, k] * (hidden @ w_down[e])
    return out


def _routing(num_experts: int):
    key_x, key_sel, key_w = jax.random.split(jax.random.PRNGKey(7), 3)
    x = jax.random.normal(key_x, (_TOKENS, 32), jnp.float32)
    selected = jnp.argsort(jax.random.uniform(key_sel, (_TOKENS, num_experts)), axis=-1)[:, :_TOPK].astype(jnp.int32)
    weights = jax.random.uniform(key_w, (_TOKENS, _TOPK), jnp.float32, 0.2, 1.0)
    return x, selected, weights


@pytest.mark.parametrize("ungated", [False, True])
def test_grouped_read_matches_explicit_slice_reference(ungated):
    cfg = kda_test._config(**_READ_GROUPS, moe_ungated_relu2=ungated, capacity_factor=4.0)
    mesh = kda_test._mesh()
    with jax.set_mesh(mesh):
        mlp = MoEMLP.init(cfg, key=jax.random.PRNGKey(3))
        gain = jax.random.uniform(jax.random.PRNGKey(4), mlp.expert_read_norm.weight.shape, jnp.float32, 0.5, 2.0)
        mlp = eqx.tree_at(lambda m: m.expert_read_norm.weight, mlp, gain)
        x, selected, weights = _routing(cfg.num_experts)
        with jax.default_matmul_precision("highest"):
            out, overflow = _run_grouped_read_bank(mlp.expert_mlp, cfg, mlp.expert_read_norm, x, selected, weights, None)
    assert mlp.expert_mlp.w_up.shape == (cfg.num_experts, 16, cfg.intermediate_dim)
    assert int(overflow.dropped) == 0
    expected = reference_bank_output(mlp, x, selected, weights, gated=not ungated)
    np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-4, atol=1e-5)


def test_every_group_norm_and_w_up_row_trains_and_routes():
    tokens = jax.random.randint(jax.random.PRNGKey(2), (2, kda_test._SEQ), 0, kda_test._VOCAB)
    loss_weight = jnp.ones(tokens.shape, jnp.float32)
    mesh, model = kda_test._model(**_READ_GROUPS)
    with jax.set_mesh(mesh):
        loss, grads = eqx.filter_jit(eqx.filter_value_and_grad(lambda m: m.next_token_loss(tokens, loss_weight)))(model)
    assert np.isfinite(float(loss))
    mask = GrugMoeMuonHConfig().create_mask(eqx.filter(model, eqx.is_inexact_array))
    for stack in ("stacked_blocks", "kda_blocks"):
        mlp_grads = getattr(grads, stack).stacked.mlp
        mlp_mask = getattr(mask, stack).stacked.mlp
        # [layers, G, W]: every group of every layer gets a gradient.
        assert np.all(np.abs(np.asarray(mlp_grads.expert_read_norm.weight)).sum(axis=-1) > 0)
        # [layers, E, W, I]: no dead input rows.
        assert np.all(np.abs(np.asarray(mlp_grads.expert_mlp.w_up)).sum(axis=-1) > 0)
        assert mlp_mask.expert_read_norm.weight == "adam"
        assert mlp_mask.expert_mlp.w_up == "muonh"


_EP_SCRIPT = """
import jax, jax.numpy as jnp, numpy as np
from jax.sharding import AxisType, Mesh, PartitionSpec as P, reshard
import experiments.grug.fast_track.test_kda_local as kda_test
from experiments.grug.fast_track.model import MoEMLP, _run_grouped_read_bank
from experiments.grug.fast_track.test_expert_read_groups import _READ_GROUPS, _routing, reference_bank_output

cfg = kda_test._config(**_READ_GROUPS, moe_ungated_relu2=True, capacity_factor=4.0,
                       pooled_transport_capacity_factor=4.0, moe_implementation="fixed_pooled_wave_all_to_all")
mesh = Mesh(np.array(jax.devices()[:4]).reshape(1, 2, 2, 1), ("replica_dcn", "data", "expert", "model"),
            axis_types=(AxisType.Explicit,) * 4)
with jax.set_mesh(mesh), jax.default_matmul_precision("highest"):
    mlp = MoEMLP.init(cfg, key=jax.random.PRNGKey(3))
    x, selected, weights = _routing(cfg.num_experts)
    batch = P(("replica_dcn", "data", "expert"), None)
    x, selected, weights = (reshard(a, batch) for a in (x, selected, weights))
    out, overflow = jax.jit(lambda m, *a: _run_grouped_read_bank(
        m.expert_mlp, cfg, m.expert_read_norm, *a, None))(mlp, x, selected, weights)
assert int(overflow.dropped) == 0
np.testing.assert_allclose(np.asarray(out), reference_bank_output(mlp, x, selected, weights, gated=False),
                           rtol=1e-4, atol=1e-5)
print("EP_OK")
"""


def test_grouped_read_pooled_wave_ep_matches_reference():
    root = str(_REPO_ROOT)
    env = dict(os.environ)
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    env["JAX_PLATFORMS"] = "cpu"
    env["PYTHONPATH"] = os.pathsep.join(
        [root, f"{root}/lib/levanter/src", f"{root}/lib/haliax/src", env.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_EP_SCRIPT)], env=env, cwd=root, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr[-4000:]
    assert "EP_OK" in result.stdout
