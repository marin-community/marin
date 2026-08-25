# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import importlib
import os
import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AbstractMesh, AxisType, PartitionSpec as P

import levanter.grug.attention._te_cp as te_cp
from levanter.grug.attention import (
    AttentionMask,
    ContextParallelStrategy,
    TeContextParallelConfig,
    attention,
    reference_attention,
)
from levanter.grug.attention._te_cp import gpu_te_cp_attention
from levanter.grug.sharding import compact_grug_mesh


def _mesh_without_context_axis() -> AbstractMesh:
    return AbstractMesh(
        axis_sizes=(1, 4, 1, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )


def _mesh_with_context_axis(context_axis_size: int) -> AbstractMesh:
    return AbstractMesh(
        axis_sizes=(1, 2, context_axis_size, 1, 1),
        axis_names=("replica_dcn", "data", "context", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 5,
    )


@pytest.mark.parametrize(("cp_size", "stripe_size"), [(2, 1), (3, 1), (2, 3), (4, 5)])
def test_striping_round_trips_tensors_and_metadata(cp_size, stripe_size):
    seq_len = cp_size * stripe_size * 3
    tensor = jax.random.normal(jax.random.PRNGKey(0), (3, seq_len, 5, 7))
    metadata = jnp.arange(3 * seq_len, dtype=jnp.int32).reshape(3, seq_len)

    for value in (tensor, metadata):
        striped = te_cp.stripe_for_cp(value, cp_size=cp_size, stripe_size=stripe_size, seq_dim=1)
        assert striped.shape == value.shape
        restored = te_cp.unstripe_from_cp(striped, cp_size=cp_size, stripe_size=stripe_size, seq_dim=1)
        np.testing.assert_array_equal(restored, value)


def test_striping_deals_stripes_to_context_ranks_round_robin():
    tokens = jnp.arange(12, dtype=jnp.int32)[None, :]

    striped = te_cp.stripe_for_cp(tokens, cp_size=3, stripe_size=2, seq_dim=1)

    # Rank r owns stripes r, r + 3, ... and receives them contiguously in shard r.
    np.testing.assert_array_equal(striped, jnp.array([[0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11]], dtype=jnp.int32))


def test_striping_rejects_a_sequence_the_context_shards_cannot_split():
    tokens = jnp.arange(10, dtype=jnp.int32)[None, :]

    with pytest.raises(ValueError, match="cp_size \\* stripe_size = 4"):
        te_cp.stripe_for_cp(tokens, cp_size=4, stripe_size=1, seq_dim=1)


def test_striped_metadata_gives_every_rank_a_causal_spread_of_both_documents():
    # Two packed documents plus a trailing padding token, in Grug's -1 padding convention.
    segment_ids = jnp.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, -1]], dtype=jnp.int32)
    ids = te_cp.te_segment_ids(segment_ids)
    positions = te_cp.segment_positions_from_segment_ids(segment_ids)
    stripe = {"cp_size": 3, "stripe_size": 2, "seq_dim": 1}

    striped_ids = te_cp.stripe_for_cp(ids, **stripe)
    striped_positions = te_cp.stripe_for_cp(positions, **stripe)

    # Rank 0 holds global tokens 0,1,6,7; rank 1 holds 2,3,8,9; rank 2 holds 4,5,10,11. Every rank
    # sees both documents, and each token keeps the in-segment position it started with.
    np.testing.assert_array_equal(striped_ids, np.array([[1, 1, 2, 2, 1, 1, 2, 2, 1, 2, 2, 0]], dtype=np.int32))
    np.testing.assert_array_equal(striped_positions, np.array([[0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 5, 0]], dtype=np.int32))


def test_segment_positions_restart_at_each_packed_document():
    segment_ids = jnp.array([[0, 0, 0, 1, 1, -1, -1], [3, 3, 4, 4, 4, 4, 4]], dtype=jnp.int32)

    positions = te_cp.segment_positions_from_segment_ids(segment_ids)

    np.testing.assert_array_equal(
        positions,
        np.array([[0, 1, 2, 0, 1, 0, 1], [0, 1, 0, 1, 2, 3, 4]], dtype=np.int32),
    )


def test_te_segment_ids_reserve_zero_for_padding():
    segment_ids = jnp.array([[0, 0, 1, 1, -1, -1]], dtype=jnp.int32)

    np.testing.assert_array_equal(
        te_cp.te_segment_ids(segment_ids),
        np.array([[1, 1, 2, 2, 0, 0]], dtype=np.int32),
    )


def test_documents_per_sequence_ignores_padding_runs():
    te_ids = jnp.array([[1, 1, 2, 2, 3, 0], [1, 1, 1, 1, 1, 0]], dtype=jnp.int32)

    np.testing.assert_array_equal(te_cp.documents_per_sequence(te_ids), np.array([3, 1], dtype=np.int32))


def test_te_window_size_keeps_grug_sliding_window_token_count():
    # Grug's sliding_window=W keeps W tokens including the query, so TE's inclusive left bound is W-1.
    assert te_cp.te_window_size(AttentionMask.causal(sliding_window=512)) == (511, 0)
    assert te_cp.te_window_size(AttentionMask.causal()) == (-1, -1)


def test_ring_strategy_rejects_wide_stripes():
    with pytest.raises(ValueError, match="only stripe_size=1"):
        TeContextParallelConfig(strategy=ContextParallelStrategy.RING, stripe_size=8)

    TeContextParallelConfig(strategy=ContextParallelStrategy.ALL_GATHER, stripe_size=8)


def test_striped_sequence_length_check_requires_one_stripe_per_rank():
    # Striped balancing needs cp_size * stripe_size, not the doubled DualChunkSwap factor.
    te_cp.check_striped_sequence_length(2048, cp_size=4, stripe_size=512)

    with pytest.raises(ValueError, match="cp_size \\* stripe_size = 4"):
        te_cp.check_striped_sequence_length(4094, cp_size=4, stripe_size=1)


def test_ring_attention_rejects_an_inherited_scan_implementation(monkeypatch):
    monkeypatch.setenv("NVTE_FUSED_RING_ATTENTION_USE_SCAN", "1")
    with pytest.raises(ValueError, match="does not support sliding windows"):
        te_cp._check_ring_attention_scan_env()

    monkeypatch.setenv("NVTE_FUSED_RING_ATTENTION_USE_SCAN", "0")
    te_cp._check_ring_attention_scan_env()


def test_context_parallel_size_requires_a_sharded_context_axis():
    config = TeContextParallelConfig()

    with pytest.raises(ValueError, match="compact_grug_mesh"):
        te_cp._context_parallel_size(_mesh_without_context_axis(), config)
    with pytest.raises(ValueError, match="axis size >= 2"):
        te_cp._context_parallel_size(_mesh_with_context_axis(1), config)

    assert te_cp._context_parallel_size(_mesh_with_context_axis(4), config) == 4


def test_context_parallel_spec_shards_the_sequence_and_keeps_other_axes():
    spec = P("data", None, None, "model")

    assert te_cp.context_parallel_spec(spec, seq_dim=1, context_axis="context") == P("data", "context", None, "model")


def test_context_parallel_spec_rejects_a_sequence_that_is_already_sharded():
    with pytest.raises(ValueError, match="already shards it"):
        te_cp.context_parallel_spec(P("data", "model"), seq_dim=1, context_axis="context")


def test_batch_sharding_check_rejects_a_compound_batch_axis():
    # TE's MeshResource.dp_resource is a single axis name, so Grug's usual compound batch is out.
    with pytest.raises(ValueError, match="single mesh axis"):
        te_cp.check_batch_sharding(P(("replica_dcn", "data"), None, None, "model"), data_axis="data")
    with pytest.raises(ValueError, match="single mesh axis"):
        te_cp.check_batch_sharding(P("expert", None, None, "model"), data_axis="data")

    te_cp.check_batch_sharding(P("data", None, None, "model"), data_axis="data")
    te_cp.check_batch_sharding(P(None, None, None, "model"), data_axis="data")


def _qkv(*, batch=2, seq_len=8, q_heads=4, kv_heads=2, head_dim=8, dtype=jnp.float32):
    q_key, k_key, v_key = jax.random.split(jax.random.PRNGKey(0), 3)
    return (
        jax.random.normal(q_key, (batch, seq_len, q_heads, head_dim), dtype=dtype),
        jax.random.normal(k_key, (batch, seq_len, kv_heads, head_dim), dtype=dtype),
        jax.random.normal(v_key, (batch, seq_len, kv_heads, head_dim), dtype=dtype),
    )


@pytest.mark.parametrize(
    ("mask", "message"),
    [
        (None, "requires an AttentionMask"),
        (AttentionMask(), "only causal self-attention"),
    ],
)
def test_backend_rejects_masks_transformer_engine_cannot_express(mask, message):
    q, k, v = _qkv()

    with pytest.raises(NotImplementedError, match=message):
        gpu_te_cp_attention(q, k, v, mask, config=TeContextParallelConfig())


def test_backend_rejects_dense_masks():
    q, k, v = _qkv()
    dense = jnp.ones((2, 8, 8), dtype=jnp.bool_)

    with pytest.raises(NotImplementedError, match="dense masks"):
        gpu_te_cp_attention(q, k, v, dense, config=TeContextParallelConfig())


def test_backend_rejects_a_mask_whose_window_hides_in_fa4_metadata():
    # The hero layer scan varies the per-layer window through fa4_bounds while sliding_window stays
    # None, which TE would otherwise read as full causal.
    q, k, v = _qkv()
    segment_ids = jnp.zeros((2, 8), dtype=jnp.int32)
    lower_bounds = jnp.zeros((2, 8), dtype=jnp.int32)
    valid = jnp.ones((2, 8), dtype=jnp.bool_)
    mask = AttentionMask.causal().with_segment_ids(segment_ids).with_fa4_bounds(lower_bounds, valid)

    with pytest.raises(NotImplementedError, match="FA4 metadata"):
        gpu_te_cp_attention(q, k, v, mask, config=TeContextParallelConfig())


def test_attention_dispatch_requires_a_context_parallel_config():
    q, k, v = _qkv()

    with pytest.raises(ValueError, match="requires a TeContextParallelConfig"):
        attention(q, k, v, AttentionMask.causal(), implementation="gpu_te_cp")


def test_attention_dispatch_names_the_setup_script_when_transformer_engine_is_missing(monkeypatch):
    q, k, v = _qkv()
    real_import_module = importlib.import_module

    def without_transformer_engine(name: str, *args, **kwargs):
        if name.startswith("transformer_engine"):
            raise ImportError(f"No module named {name!r}")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", without_transformer_engine)

    with pytest.raises(ImportError, match="te_setup.py"):
        attention(
            q,
            k,
            v,
            AttentionMask.causal(),
            implementation="gpu_te_cp",
            te_cp=TeContextParallelConfig(),
        )


_SHARDING_ROUND_TRIP_SCRIPT = """
    import contextlib
    import types

    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P

    import levanter.grug.attention._te_cp as te_cp
    from levanter.grug.attention import AttentionMask, TeContextParallelConfig
    from levanter.grug.sharding import compact_grug_mesh

    captured = {}


    def fake_transformer_engine():
        enums = types.SimpleNamespace(
            THD_THD_THD=0,
            NO_BIAS=0,
            PADDING_CAUSAL_MASK=0,
            VANILLA_SOFTMAX=0,
            RING=0,
            ALL_GATHER=1,
        )

        def fused_attn(qkv, bias, descriptor, rng, **kwargs):
            captured["window_size"] = kwargs["window_size"]
            captured["descriptor"] = descriptor
            return qkv[0]

        return te_cp.TransformerEngineApi(
            te=types.SimpleNamespace(autocast=lambda **kwargs: contextlib.nullcontext()),
            AttnBiasType=enums,
            AttnMaskType=enums,
            AttnSoftmaxType=enums,
            CPStrategy=enums,
            QKVLayout=enums,
            SequenceDescriptor=types.SimpleNamespace(from_segment_ids_and_pos=lambda ids, pos: (ids, pos)),
            fused_attn=fused_attn,
            is_fused_attn_kernel_available=lambda *args, **kwargs: True,
            MeshResource=lambda **kwargs: None,
        )


    te_cp.load_transformer_engine = fake_transformer_engine

    mesh = compact_grug_mesh(context_axis_size=2)
    config = TeContextParallelConfig(max_segments_per_seq=2)
    segment_ids = jnp.concatenate([jnp.zeros((2, 5), dtype=jnp.int32), jnp.ones((2, 3), dtype=jnp.int32)], axis=1)
    mask = AttentionMask.causal(sliding_window=4).with_segment_ids(segment_ids)
    qkv_spec = P("data", None, None, "model")
    q, k, v = (
        jax.device_put(jax.random.normal(jax.random.PRNGKey(seed), (2, 8, 4, 8)), NamedSharding(mesh, qkv_spec))
        for seed in range(3)
    )

    with jax.set_mesh(mesh):
        out = jax.jit(lambda *args: te_cp.gpu_te_cp_attention(*args, mask, config=config))(q, k, v)

    # The fake kernel is the identity on its striped q, so a correct stripe/shard/gather/unstripe
    # round trip has to return q itself, in natural token order and under q's own spec.
    np.testing.assert_array_equal(np.asarray(out), np.asarray(q))
    assert out.sharding.spec == qkv_spec, out.sharding.spec
    assert captured["window_size"] == (3, 0), captured["window_size"]
    # The metadata handed to TE is captured mid-trace, so its sharding lives on the aval.
    for metadata in captured["descriptor"]:
        spec = jax.typeof(metadata).sharding.spec
        assert spec[1] == "context", spec
"""


def test_context_parallel_sharding_round_trips_tokens_and_keeps_the_input_spec():
    """The striped, context-sharded detour must hand back exactly the tokens it was given.

    Runs on two forced CPU devices: the reshapes around the fused kernel are only expressible
    while the sequence is replicated, which a single-device mesh cannot exercise.
    """
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_SHARDING_ROUND_TRIP_SCRIPT)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_gpu_te_cp_attention_matches_reference_under_context_parallelism():
    """Forward and backward parity against the dense reference, on real TE context-parallel GPUs.

    Unverified so far: TE 2.17.1 fails cuDNN backward workspace sizing on Marin's GB200 image, so
    this test has never run to completion. It is the gate for the next TE build.
    """
    if jax.default_backend() != "gpu":
        pytest.skip("Transformer Engine context-parallel attention requires a GPU backend.")
    pytest.importorskip("transformer_engine.jax")
    if jax.device_count() < 2:
        pytest.skip("Context-parallel attention needs at least two devices.")

    config = TeContextParallelConfig(strategy=ContextParallelStrategy.RING, max_segments_per_seq=2)
    q, k, v = _qkv(batch=2, seq_len=64, q_heads=4, kv_heads=2, head_dim=64, dtype=jnp.bfloat16)
    segment_ids = jnp.concatenate(
        [jnp.zeros((2, 40), dtype=jnp.int32), jnp.ones((2, 24), dtype=jnp.int32)],
        axis=1,
    )
    mask = AttentionMask.causal(sliding_window=16).with_segment_ids(segment_ids)
    cotangent = jax.random.normal(jax.random.PRNGKey(1), q.shape, dtype=jnp.bfloat16)

    def loss(fn):
        def inner(q_arg, k_arg, v_arg):
            out = fn(q_arg, k_arg, v_arg)
            return jnp.sum(out.astype(jnp.float32) * cotangent.astype(jnp.float32))

        return inner

    with jax.set_mesh(compact_grug_mesh(context_axis_size=2)):
        actual = jax.jit(lambda *args: gpu_te_cp_attention(*args, mask, config=config))(q, k, v)
        actual_grads = jax.jit(
            jax.grad(loss(lambda *args: gpu_te_cp_attention(*args, mask, config=config)), argnums=(0, 1, 2))
        )(q, k, v)

    expected = reference_attention(q, k, v, mask, logits_dtype=jnp.float32)
    expected_grads = jax.jit(
        jax.grad(loss(lambda *args: reference_attention(*args, mask, logits_dtype=jnp.float32)), argnums=(0, 1, 2))
    )(q, k, v)

    np.testing.assert_allclose(actual, expected, atol=7e-2, rtol=7e-2)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        np.testing.assert_allclose(actual_grad, expected_grad, atol=7e-2, rtol=7e-2)
