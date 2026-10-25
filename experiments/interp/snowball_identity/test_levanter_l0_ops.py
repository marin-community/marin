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
