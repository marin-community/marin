# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Row-sharded table lookups (``_row_sharded_embedding_gather``, ``_row_sharded_memory_bag``) match the
replicated lookups they replace under ``embed2_fsdp`` on an 8-device CPU mesh, and the model with them never
all-gathers the table."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]

_SCRIPT = """
import re
import equinox as eqx, jax, jax.numpy as jnp, numpy as np
from jax.sharding import AxisType, Mesh, PartitionSpec as P, reshard
import experiments.grug.fast_track.model as M
import experiments.grug.fast_track.test_kda_local as kda_test

REPLICATED, SHARDED = P(None, None), P(M._FSDP_AXES, None)


def mesh(shape):
    return Mesh(np.array(jax.devices()[:8]).reshape(shape), ("replica_dcn", "data", "expert", "model"),
                axis_types=(AxisType.Explicit,) * 4)


def gather_and_grad(fn, spec, table, ids, cot):
    def loss(t):
        return jnp.sum(fn(reshard(t, spec), ids).astype(jnp.float32) * cot)
    out = jax.jit(lambda t: fn(reshard(t, spec), ids))(table)
    return np.asarray(out.astype(jnp.float32)), np.asarray(jax.jit(jax.grad(loss))(table).astype(jnp.float32))


def bag_and_grads(fn, spec, values, slots, weights, cot):
    def loss(v, w):
        return jnp.sum(fn(reshard(v, spec), slots, w).astype(jnp.float32) * cot)
    out = jax.jit(lambda v, w: fn(reshard(v, spec), slots, w))(values, weights)
    d_v, d_w = jax.jit(jax.grad(loss, argnums=(0, 1)))(values, weights)
    return np.asarray(out), np.asarray(d_v), np.asarray(d_w)


rows, dim, b, s, r = 64, 16, 8, 12, 3
default_chunk_elems = M._ROW_GATHER_CHUNK_ELEMS
for shape in [(1, 4, 2, 1), (2, 2, 2, 1)]:
    for chunk_elems in [default_chunk_elems, 64]:  # 64 walks each shard's 12 tokens in 4 chunks
        M._ROW_GATHER_CHUNK_ELEMS = chunk_elems
        with jax.set_mesh(mesh(shape)):
            batch = P(M._BATCH_AXES, None)
            ids = jax.random.randint(jax.random.PRNGKey(1), (b, s), 0, rows)
            # Every shard's rows, plus one row repeated across every batch shard.
            ids = reshard(ids.at[:, 0].set(jnp.arange(b) * (rows // b)).at[:, 1:4].set(5), batch)
            cot = jax.random.normal(jax.random.PRNGKey(2), (b, s, dim), jnp.float32)
            for dtype in [jnp.float32, jnp.bfloat16]:
                table = jax.random.normal(jax.random.PRNGKey(0), (rows, dim), jnp.float32).astype(dtype)
                ref_y, ref_g = gather_and_grad(M._embedding_gather, REPLICATED, table, ids, cot)
                new_y, new_g = gather_and_grad(M._row_sharded_embedding_gather, SHARDED, table, ids, cot)
                assert np.array_equal(ref_y, new_y), (shape, chunk_elems, dtype)
                hit = np.zeros(rows, bool)
                hit[np.asarray(ids).reshape(-1)] = True
                assert np.array_equal(np.abs(new_g).sum(-1) > 0, hit)
                # fp32: reduction order only. bf16: the replicated path rounds each shard's sum before its psum.
                tol = 1e-5 if dtype == jnp.float32 else 2 ** -7 * np.abs(ref_g).max()
                np.testing.assert_allclose(new_g, ref_g, atol=tol)
                print("GATHER", shape, chunk_elems, dtype.__name__, np.abs(ref_g - new_g).max())

            values = jax.random.normal(jax.random.PRNGKey(3), (rows, dim), jnp.float32)
            slots = reshard(jax.random.randint(jax.random.PRNGKey(4), (b, s, r), 0, rows).at[:, :2].set(7),
                            P(M._BATCH_AXES, None, None))
            weights = reshard(jax.nn.softmax(jax.random.normal(jax.random.PRNGKey(5), (b, s, r))),
                              P(M._BATCH_AXES, None, None))
            ref = bag_and_grads(M._memory_bag, REPLICATED, values, slots, weights, cot)
            new = bag_and_grads(M._row_sharded_memory_bag, SHARDED, values, slots, weights, cot)
            diffs = [float(np.abs(x - y).max()) for x, y in zip(ref, new)]
            for x, y in zip(ref, new):
                np.testing.assert_allclose(y, x, atol=1e-5)
            print("BAG", shape, chunk_elems, diffs)

cfg = dict(second_embed=True, second_embed_bigram=True, embed2_rows=64, embed2_hash_heads=2, embed3_rows=64)
tokens = jax.random.randint(jax.random.PRNGKey(2), (8, kda_test._SEQ), 0, kda_test._VOCAB)
results = {}
for fsdp in [False, True]:
    with jax.set_mesh(mesh((1, 4, 2, 1))):
        config = kda_test._config(embed2_fsdp=fsdp, embed2_row_sharded_gather=fsdp, **cfg)
        model = M.Transformer.init(config, key=jax.random.PRNGKey(0))
        tk = reshard(tokens, P(M._BATCH_AXES, None))
        w = reshard(jnp.ones(tokens.shape, jnp.float32), P(M._BATCH_AXES, None))
        params, static = eqx.partition(model, eqx.is_array)
        step = jax.jit(lambda p: eqx.filter_value_and_grad(lambda m: m.next_token_loss(tk, w))(eqx.combine(p, static)))
        loss, grads = step(params)
        hlo = step.lower(params).compile().as_text()
    results[fsdp] = (float(loss), grads)
    if fsdp:
        # The [128, 16] / [64, 32] tables never appear in a collective, only their [16, 16] / [8, 32] shards.
        table_collectives = re.findall(r"f32\\[(?:128,16|64,32)\\]\\S* (?:all-gather|all-reduce)", hlo)
        assert not table_collectives, table_collectives
assert abs(results[True][0] - results[False][0]) < 1e-6, results
for name in ("token_embed2", "token_embed3"):
    ref_g, new_g = (np.asarray(getattr(results[k][1], name)) for k in (False, True))
    assert np.abs(new_g).max() > 0
    np.testing.assert_allclose(new_g, ref_g, atol=1e-7)
    print("MODEL", name, np.abs(ref_g - new_g).max())
print("ROW_SHARDED_OK")
"""


def test_row_sharded_lookups_match_replicated():
    root = str(_REPO_ROOT)
    env = dict(os.environ)
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    env["JAX_PLATFORMS"] = "cpu"
    env["PYTHONPATH"] = os.pathsep.join(
        [root, f"{root}/lib/levanter/src", f"{root}/lib/haliax/src", env.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_SCRIPT)], env=env, cwd=root, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr[-4000:]
    assert "ROW_SHARDED_OK" in result.stdout
