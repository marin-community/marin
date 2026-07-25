# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from jax.sharding import AxisType, Mesh
from jax.sharding import PartitionSpec as P

from experiments.grug.moe.model import CausalSelfAttention, DenseMLP, GrugModelConfig, MoEMLP
from experiments.grug.moe.train import _next_qb_betas


def test_nonexpert_weights_are_sharded_over_data_and_expert_axes():
    mesh = Mesh(
        np.asarray([jax.devices()[0]]).reshape((1, 1, 1)),
        ("data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )

    with jax.set_mesh(mesh):
        dense = DenseMLP.init(16, 32, 0.02, key=jax.random.key(0))
        config = GrugModelConfig(
            vocab_size=128,
            hidden_dim=16,
            intermediate_dim=8,
            shared_expert_intermediate_dim=16,
            num_experts=4,
            num_experts_per_token=2,
            num_layers=1,
            num_heads=2,
            num_kv_heads=1,
        )
        attention = CausalSelfAttention.init(config, key=jax.random.key(1))

    column_sharding = P(("data", "expert"), "model")
    row_sharding = P("model", ("data", "expert"))
    assert dense.w_gate.sharding.spec == column_sharding
    assert dense.w_up.sharding.spec == column_sharding
    assert dense.w_down.sharding.spec == row_sharding
    assert attention.w_q.sharding.spec == column_sharding
    assert attention.w_k.sharding.spec == column_sharding
    assert attention.w_v.sharding.spec == column_sharding
    assert attention.w_o.sharding.spec == row_sharding


def test_next_qb_betas_stock_gain_and_integral_rules(monkeypatch):
    """_next_qb_betas: stock passthrough, SCALE_QB_GAIN blend, SCALE_QB_INTEGRAL sign update.

    The integral arm also checks the DeepSeek-V3 direction end-to-end through the
    bias = -center(beta) convention: an overloaded expert's applied bias must go DOWN.
    """
    monkeypatch.delenv("SCALE_QB_GAIN", raising=False)
    monkeypatch.delenv("SCALE_QB_INTEGRAL", raising=False)
    pending = jnp.zeros((2, 4), dtype=jnp.float32)
    measured = jnp.asarray([[0.3, -0.1, 0.05, -0.25], [1.0, 2.0, 3.0, 4.0]], dtype=jnp.float32)

    # Stock: replace outright, pending ignored.
    np.testing.assert_array_equal(np.asarray(_next_qb_betas(pending, measured)), np.asarray(measured))

    # Gain: g * measured + (1 - g) * pending.
    monkeypatch.setenv("SCALE_QB_GAIN", "0.5")
    np.testing.assert_allclose(
        np.asarray(_next_qb_betas(pending + 1.0, measured)), np.asarray(0.5 * measured + 0.5 * (pending + 1.0))
    )

    # Integral: pending + gamma * sign(load - mean_load), SCALE_QB_GAIN ignored.
    monkeypatch.setenv("SCALE_QB_GAIN", "2")  # must be inert in integral mode
    monkeypatch.setenv("SCALE_QB_INTEGRAL", "0.1")
    loads = jnp.asarray([[10.0, 2.0, 4.0, 4.0], [0.0, 0.0, 8.0, 0.0]], dtype=jnp.float32)
    step1 = _next_qb_betas(pending, loads)
    expected1 = pending + 0.1 * jnp.sign(loads - jnp.mean(loads, axis=-1, keepdims=True))
    np.testing.assert_array_equal(np.asarray(step1), np.asarray(expected1))
    # Accumulation across steps (integral, not replacement).
    step2 = _next_qb_betas(step1, loads)
    expected2 = step1 + 0.1 * jnp.sign(loads - jnp.mean(loads, axis=-1, keepdims=True))
    np.testing.assert_array_equal(np.asarray(step2), np.asarray(expected2))
    # Direction through bias = -center(beta): overloaded expert 0 (layer 0) gets the lowest bias.
    bias = -np.asarray(step1)
    bias = bias - bias.mean(axis=-1, keepdims=True)
    assert bias[0, 0] == bias[0].min() and bias[0, 0] < 0
    assert bias[1, 2] == bias[1].min() and bias[1, 2] < 0


def test_qb_integral_loads_match_global_bincount():
    """_compute_qb_loads psums per-expert assignment counts across the batch axes.

    EP8 CPU mesh (2x2x2 over replica_dcn/data/expert), skewed selection so counts are
    non-uniform. Runs in a fresh 8-CPU-device interpreter (XLA device count is process-global).
    """
    script = textwrap.dedent(
        """
        import os
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
        os.environ["JAX_PLATFORMS"] = "cpu"
        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P
        from experiments.grug.moe.model import GrugModelConfig, _compute_qb_loads

        NUM_EXPERTS, TOKENS, TOPK = 16, 64, 4
        mesh = Mesh(
            np.array(jax.devices()).reshape((2, 2, 2)),
            ("replica_dcn", "data", "expert"),
            axis_types=(AxisType.Explicit,) * 3,
        )
        cfg = GrugModelConfig(vocab_size=128, num_experts=NUM_EXPERTS)
        logits = jax.random.normal(jax.random.key(3), (TOKENS, NUM_EXPERTS)) + jnp.linspace(0, 3.0, NUM_EXPERTS)
        selected = jax.lax.top_k(logits, TOPK)[1].astype(jnp.int32)
        with jax.set_mesh(mesh):
            sel = jax.device_put(selected, NamedSharding(mesh, P(("replica_dcn", "data", "expert"), None)))
            loads = _compute_qb_loads(sel, cfg)
        expected = np.bincount(np.asarray(selected).ravel(), minlength=NUM_EXPERTS).astype(np.float32)
        np.testing.assert_array_equal(np.asarray(loads), expected)
        print("OK")
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "OK" in result.stdout


def test_qb_integral_forward_keeps_routed_output(monkeypatch):
    """SCALE_QB_INTEGRAL changes only the qb_beta channel (quantile -> load counts).

    The routed output (selection + combine weights) must be bitwise identical to the stock
    QB path, and the shipped counts must equal the biased top-k bincount.
    """
    mesh = Mesh(
        np.asarray([jax.devices()[0]]).reshape((1, 1, 1, 1)),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    cfg = GrugModelConfig(
        vocab_size=128,
        hidden_dim=16,
        intermediate_dim=8,
        shared_expert_intermediate_dim=16,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        qb_routing=True,
    )
    with jax.set_mesh(mesh):
        mlp = MoEMLP.init(cfg, key=jax.random.key(0))
        x = jax.random.normal(jax.random.key(1), (2, 8, cfg.hidden_dim), dtype=jnp.float32)
        monkeypatch.delenv("SCALE_QB_INTEGRAL", raising=False)
        out_stock, _, _ = mlp(x)
        monkeypatch.setenv("SCALE_QB_INTEGRAL", "0.001")
        out_integral, loads, _ = mlp(x)

        np.testing.assert_array_equal(np.asarray(out_integral), np.asarray(out_stock))
        b, s, k = x.shape[0], x.shape[1], cfg.num_experts_per_token
        assert float(jnp.sum(loads)) == float(b * s * k)
        x_flat = rearrange(x, "b s d -> (b s) d")
        biased = jnp.einsum("td,de->te", x_flat, mlp.router).astype(jnp.float32) + mlp.router_bias
        selected = jax.lax.top_k(biased, k + 1)[1][:, :-1]
        expected = np.bincount(np.asarray(selected).ravel(), minlength=cfg.num_experts).astype(np.float32)
        np.testing.assert_array_equal(np.asarray(loads), expected)
