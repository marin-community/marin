# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from levanter.models.snowball import GatedNorm, RMSNorm

from experiments.interp.snowball_identity import levanter_l0_ops as lev
from experiments.interp.snowball_identity import moe_drop_diagnostic as diag
from experiments.interp.snowball_identity.sequences import Sequence

D, R, V = 16, 8, 64


@pytest.fixture(scope="module")
def setup():
    keys = iter(jax.random.split(jax.random.PRNGKey(0), 16))

    def bf16(shape, scale=0.5):
        return (scale * jax.random.normal(next(keys), shape)).astype(jnp.bfloat16)

    modules = {
        key: (
            RMSNorm(weight=1 + bf16((D,), 0.1), eps=1e-5),
            GatedNorm(w_down=bf16((D, R)), w_up=bf16((R, D))),
        )
        for key in lev.NORMS
    }
    with jax.sharding.set_mesh(Mesh(np.asarray(jax.devices()[:1]), ("data",))):
        yield bf16((V, D), 2.0), modules


def test_per_op_steps_rebuild_the_modules(setup):
    """Each step is its own program; the gated-norm step outputs equal ``GatedNorm.__call__`` run as separate ops, and
    the composite's outputs have the per-op names and shapes."""
    table, modules = setup
    ids = jnp.asarray([[3, 7, 1, 9, 2]])
    ops = lev.per_op(table, modules, ids, None)
    assert set(ops) == {
        "embed_tokens",
        *(name for (_, _, rms, gated, out) in lev.NORMS.values() for name in (rms, out)),
        *(f"{gated}/{step}" for (_, _, _, gated, _) in lev.NORMS.values() for step in lev.GATED),
    }
    rms = modules["embed"][0]
    assert jnp.array_equal(ops["embed_rmsnorm"], jax.jit(lambda m, x: m(x))(rms, table[ids]))
    assert ops["embed_gated/down"].dtype == jnp.bfloat16
    out = lev.composite(table, modules, ids)
    assert out["embed"].shape == out["L0/attn_norm"].shape == ops["embed"].shape


def test_isolated_steps_take_the_given_inputs(setup):
    """With torch inputs, a step's output depends only on the given input for that step."""
    table, modules = setup
    ids = jnp.asarray([[3, 7, 1]])
    chained = lev.per_op(table, modules, ids, None)
    given = dict(chained)
    given["embed"] = jnp.zeros_like(chained["embed"])
    isolated = lev.per_op(table, modules, ids, given)
    assert jnp.array_equal(isolated["embed_rmsnorm"], chained["embed_rmsnorm"])
    assert jnp.array_equal(isolated["L0/attn_rmsnorm"], jax.jit(lambda m, x: m(x))(modules["attn"][0], given["embed"]))


def test_trace_reads_the_rmsnorm_weight_multiply_dtype(setup):
    table, modules = setup
    report = lev.trace(table, modules, jnp.asarray([[1, 2]]))
    assert report["embed_norm_weight_multiply_in_fp32"]
    assert any(eqn["primitive"] == "dot_general" for eqn in report["embed_gated_norm"])


def test_grids_place_each_check_sequence(monkeypatch):
    monkeypatch.setattr(diag, "PAD_ID", 0)
    sequences = [
        Sequence(f"{g}|r{i}", g, tuple(range(1, n + 1)))
        for i, (g, n) in enumerate((("check", 5), ("contrastive", 9), ("check", 3), ("long", 12)))
    ]
    check = [s for s in sequences if s.group == "check"]
    found = lev.grids(sequences, check)
    assert [rows for _, rows in found["batch1"]] == [{"check|r0": 0}, {"check|r2": 0}]
    for grid, rows in found["gate"] + found["dump"]:
        for sid, row in rows.items():
            tokens = next(s.tokens for s in sequences if s.id == sid)
            assert grid[row, : len(tokens)].tolist() == list(tokens)
    assert found["dump"][0][0].shape == (diag.DUMP_ROWS, diag.PAD_MULTIPLE)


def test_compare_counts_differing_elements():
    ours = {"check|a": {"embed": np.array([[1.0, 2.0]], dtype=np.float32)}}
    reference = {"check|a": {"embed": np.array([[1.0, 2.5]], dtype=np.float32)}}
    report = lev.compare(ours, reference)["embed"]
    assert (report["differing"], report["elements"], report["max_abs"]) == (1, 2, 0.5)


def test_two_round_rms_rounds_before_the_weight_multiply(setup):
    """vLLM's native order: round the normalized value to BF16, then multiply by the BF16 weight. It equals that
    computed by hand, and differs from the production one-round RMSNorm on some elements."""
    table, modules = setup
    rms = modules["embed"][0]
    x = table[jnp.asarray([[3, 7, 1, 9, 2, 11, 5]])]
    xf = np.asarray(x, dtype=np.float32)
    normalized = (xf / np.sqrt(np.mean(np.square(xf), axis=-1, keepdims=True) + rms.eps)).astype(jnp.bfloat16)
    expected = (normalized.astype(np.float32) * np.asarray(rms.weight, dtype=np.float32)).astype(jnp.bfloat16)
    two_round = np.asarray(lev.two_round_rms(rms, x))
    assert np.array_equal(two_round.astype(np.float32), expected.astype(np.float32))
    one_round = np.asarray(jax.jit(lambda m, x: m(x))(rms, x))
    assert (two_round != one_round).any()
    ops = lev.per_op(table, modules, jnp.asarray([[3, 7, 1]]), None, two_round=True)
    assert jnp.array_equal(ops["embed_rmsnorm"], lev.two_round_rms(rms, table[jnp.asarray([[3, 7, 1]])]))


def test_composite_two_round_uses_the_two_round_norms(setup):
    table, modules = setup
    ids = jnp.asarray([[3, 7, 1, 9]])
    out = lev.composite_two_round(table, modules, ids)
    rms, gated = modules["embed"]
    expected = jax.jit(lambda g, x: g(x))(gated, lev.two_round_rms(rms, table[ids]))
    # One program vs separate programs may round the gated norm differently; the inputs to it are equal.
    assert out["embed"].shape == expected.shape
    assert float(jnp.max(jnp.abs(out["embed"].astype(jnp.float32) - expected.astype(jnp.float32)))) < 0.05


def test_fp32_activations_round_once(setup):
    """FP32 silu/sigmoid cast once equal the activation computed in FP32 by hand and rounded to BF16."""
    table, modules = setup
    ids = jnp.asarray([[3, 7, 1, 9]])
    gated = modules["embed"][1]
    x = jax.jit(lambda m, x: m(x))(modules["embed"][0], table[ids])
    steps = lev.gated_steps(gated, x, fp32=True)
    down = np.asarray(steps["down"], dtype=np.float32)
    expected_silu = (down / (1 + np.exp(-down))).astype(jnp.bfloat16)
    assert np.abs(np.asarray(steps["silu"], np.float32) - expected_silu.astype(np.float32)).max() <= 2.0**-6
    out = lev.composite_fp32_activations(table, modules, ids)
    assert out["embed"].dtype == jnp.bfloat16


def test_by_sub_step_feeds_each_given_input(setup):
    table, modules = setup
    ids = jnp.asarray([[3, 7, 1]])
    chained = lev.per_op(table, modules, ids, None)
    given = dict(chained)
    given["embed_gated/up"] = jnp.zeros_like(chained["embed_gated/up"])
    ops = lev.per_op(table, modules, ids, given, by_sub_step=True)
    assert jnp.array_equal(ops["embed_gated/sigmoid"], jnp.full_like(ops["embed_gated/sigmoid"], 0.5))
    assert jnp.array_equal(ops["embed_gated/silu"], chained["embed_gated/silu"])
