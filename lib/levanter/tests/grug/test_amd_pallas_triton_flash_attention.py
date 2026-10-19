# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AbstractMesh, AxisType, Mesh, NamedSharding, reshard, use_abstract_mesh
from jax.sharding import PartitionSpec as P

from levanter.grug.attention import AttentionMask, fa4_cute_segment_bounds, reference_attention
from levanter.grug.attention._amd_pallas_triton_flash import TritonFlashBlockSizes, pallas_triton_flash_attention
from levanter.testing.cpu_devices import run_on_cpu_devices

# On a GPU the kernels compile through Triton; elsewhere they run in the Pallas interpreter.
INTERPRET = jax.default_backend() != "gpu"
SEQ_LEN = 256
# Small tiles so the short test sequence still crosses masked, unmasked and skipped key blocks.
SMALL_BLOCKS = TritonFlashBlockSizes(
    block_q=32, block_k=16, block_q_dq=32, block_k_dq=16, block_q_dkv=16, block_k_dkv=32
)
# Rows 0-89 are one document, 90-199 another, 200-255 padding (segment -1, its own run).
SEGMENTS = jnp.concatenate([jnp.zeros(90), jnp.ones(110), -jnp.ones(56)]).astype(jnp.int32)

MASKS = {
    "causal": AttentionMask.causal(),
    "window": AttentionMask.causal(sliding_window=40),
    "window_segments": AttentionMask.causal(sliding_window=40).with_segment_ids(
        jnp.stack([SEGMENTS, jnp.roll(SEGMENTS, 37).at[:37].set(0)])
    ),
}


def _qkv(dtype):
    """Two sequences with 8 query heads sharing 2 KV heads, head_dim 64."""
    kq, kk, kv = jax.random.split(jax.random.key(0), 3)
    q = jax.random.normal(kq, (2, SEQ_LEN, 8, 64), jnp.float32).astype(dtype)
    k = jax.random.normal(kk, (2, SEQ_LEN, 2, 64), jnp.float32).astype(dtype)
    v = jax.random.normal(kv, (2, SEQ_LEN, 2, 64), jnp.float32).astype(dtype)
    return q, k, v


@pytest.fixture(autouse=True)
def _interpreter_matmul_precision():
    """Run interpreted kernels at full f32 precision; TPU's default runs f32 matmuls in bf16 passes."""
    if not INTERPRET:
        yield
        return
    with jax.default_matmul_precision("highest"):
        yield


def _reference(q, k, v, mask):
    """f32 reference on the same (possibly bf16-rounded) inputs."""
    with jax.default_matmul_precision("highest"):
        f32 = [x.astype(jnp.float32) for x in (q, k, v)]
        return reference_attention(*f32, mask, logits_dtype=jnp.float32)


def _flash(q, k, v, mask):
    return pallas_triton_flash_attention(q, k, v, mask, block_sizes=SMALL_BLOCKS, interpret=INTERPRET)


def _f32_dots_use_tf32() -> bool:
    """Whether Triton runs the kernels' default-precision f32 dots in TF32: on NVIDIA GPUs and gfx942 (MI300X, MI325X)."""
    device = jax.devices()[0]
    if device.platform != "gpu":
        return False
    # ROCm reports the gfx architecture name as the compute capability, CUDA a version number.
    return device.compute_capability == "gfx942" or not str(device.compute_capability).startswith("gfx")


# Worst cases measured on MI350X and in the CPU interpreter: 9e-7 in f32, and 3.5e-3 in bf16, where P and dS are
# rounded to bf16 before their matmuls as in every flash kernel.
TOLERANCE = {jnp.float32: 1e-4, jnp.bfloat16: 1e-2}


def _assert_gradients_match(actual, expected, *, tol):
    for name, want, got in zip("qkv", expected, actual, strict=True):
        want = np.asarray(want, np.float32)
        got = np.asarray(got, np.float32)
        # Gradients of K/V sum over a GQA group and many queries, so compare relative to their scale.
        scale = max(1.0, float(np.max(np.abs(want))))
        np.testing.assert_allclose(got / scale, want / scale, atol=tol, rtol=tol, err_msg=f"d{name}")


@pytest.mark.parametrize("mask_name", list(MASKS))
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_flash_attention_forward_matches_reference(mask_name, dtype):
    if dtype == jnp.float32 and _f32_dots_use_tf32():
        # About 2.7e-3 off on MI300X.
        pytest.skip("f32 Triton dots use TF32 on NVIDIA and gfx942")
    q, k, v = _qkv(dtype)
    mask = MASKS[mask_name]
    actual = _flash(q, k, v, mask)
    expected = _reference(q, k, v, mask)
    assert actual.dtype == dtype
    tol = TOLERANCE[dtype]
    np.testing.assert_allclose(np.asarray(actual, np.float32), np.asarray(expected), atol=tol, rtol=tol)


@pytest.mark.parametrize("mask_name", list(MASKS))
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_flash_attention_gradients_match_reference(mask_name, dtype):
    if dtype == jnp.float32 and _f32_dots_use_tf32():
        # About 2.7e-3 off on MI300X.
        pytest.skip("f32 Triton dots use TF32 on NVIDIA and gfx942")
    q, k, v = _qkv(dtype)
    mask = MASKS[mask_name]
    cotangent = jax.random.normal(jax.random.key(1), q.shape, jnp.float32)

    def loss(attend, q, k, v):
        return jnp.sum(attend(q, k, v, mask).astype(jnp.float32) * cotangent)

    expected = jax.grad(lambda *a: loss(_reference, *a), argnums=(0, 1, 2))(q, k, v)
    actual = jax.grad(lambda *a: loss(_flash, *a), argnums=(0, 1, 2))(q, k, v)
    _assert_gradients_match(actual, expected, tol=TOLERANCE[dtype])


def test_flash_attention_batch_sharded_mesh_keeps_output_and_gradient_sharding():
    mesh = AbstractMesh(axis_sizes=(2,), axis_names=("data",), axis_types=(AxisType.Explicit,))
    sharding = NamedSharding(mesh, P("data", None, None, None))
    q = jax.ShapeDtypeStruct((4, SEQ_LEN, 8, 64), jnp.bfloat16, sharding=sharding)
    kv = jax.ShapeDtypeStruct((4, SEQ_LEN, 2, 64), jnp.bfloat16, sharding=sharding)
    mask = AttentionMask.causal(sliding_window=40)

    def attend(q, k, v):
        return pallas_triton_flash_attention(q, k, v, mask, block_sizes=SMALL_BLOCKS, interpret=True)

    def loss(q, k, v):
        return jnp.sum(attend(q, k, v).astype(jnp.float32))

    with use_abstract_mesh(mesh):
        output = jax.eval_shape(attend, q, kv, kv)
        grads = jax.eval_shape(jax.grad(loss, argnums=(0, 1, 2)), q, kv, kv)
    assert output.sharding == sharding
    assert [g.sharding for g in grads] == [sharding] * 3


@pytest.mark.parametrize("precomputed_bounds", [False, True])
def test_flash_attention_on_single_device_grug_mesh_matches_reference(precomputed_bounds):
    # A one-device Grug mesh keeps its length-1 axes, so every spec names mesh axes that partition nothing. With
    # precomputed bounds, moe_hero_ep reshards them over the batch axes too, so they name those axes as well.
    names = ("replica_dcn", "data", "context", "expert", "model")
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape((1,) * 5), names, axis_types=(AxisType.Explicit,) * 5)
    batch_axes = ("replica_dcn", "data", "expert")
    q, k, v = _qkv(jnp.bfloat16)
    mask = MASKS["window_segments"]
    # Padding queries attend only to themselves under precomputed bounds, so they get no cotangent and their
    # outputs are left out of the comparison.
    valid_rows = (mask.segment_ids[0] >= 0)[:, :, None, None] if precomputed_bounds else jnp.ones((), bool)
    cotangent = jnp.where(valid_rows, jax.random.normal(jax.random.key(1), q.shape, jnp.float32), 0.0)
    expected = jnp.where(valid_rows, _reference(q, k, v, mask), 0.0)
    expected_grads = jax.grad(lambda *a: jnp.sum(_reference(*a, mask) * cotangent), argnums=(0, 1, 2))(q, k, v)

    with jax.set_mesh(mesh):
        sharding = NamedSharding(mesh, P(batch_axes, None, "model", None))
        q, k, v, cotangent = (jax.device_put(x, sharding) for x in (q, k, v, cotangent))
        ids = jax.device_put(mask.segment_ids[0], NamedSharding(mesh, P(batch_axes, None)))

        def attend(q, k, v, ids):
            # Separate q and kv arrays, as a batch from the data loader carries them.
            sharded_mask = AttentionMask.causal(sliding_window=mask.sliding_window).with_segment_ids(
                ids, jnp.copy(ids)
            )
            if precomputed_bounds:
                lower, valid = fa4_cute_segment_bounds(
                    sharded_mask, batch_size=2, seq_len=SEQ_LEN, sliding_window=mask.sliding_window
                )
                bounds_spec = P(batch_axes, None)
                sharded_mask = AttentionMask.causal().with_segment_ids(ids, jnp.copy(ids))
                sharded_mask = sharded_mask.with_fa4_bounds(reshard(lower, bounds_spec), reshard(valid, bounds_spec))
            return _flash(q, k, v, sharded_mask)

        def loss(q, k, v, ids):
            return jnp.sum(attend(q, k, v, ids).astype(jnp.float32) * cotangent)

        actual = jax.jit(attend)(q, k, v, ids)
        actual_grads = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))(q, k, v, ids)

    assert actual.sharding.spec == sharding.spec
    tol = TOLERANCE[jnp.bfloat16]
    actual = jnp.where(valid_rows, np.asarray(actual, np.float32), 0.0)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=tol, rtol=tol)
    _assert_gradients_match(actual_grads, expected_grads, tol=tol)


_SHARDED_SCRIPT = """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

    from levanter.grug.attention import AttentionMask, reference_attention
    from levanter.grug.attention._amd_pallas_triton_flash import TritonFlashBlockSizes, pallas_triton_flash_attention

    blocks = TritonFlashBlockSizes(block_q=32, block_k=16, block_q_dq=32, block_k_dq=16, block_q_dkv=16, block_k_dkv=32)
    segments = np.concatenate([np.zeros(50), np.ones(60), -np.ones(18)]).astype(np.int32)
    segments = jnp.asarray(np.stack([segments, np.roll(segments, 20)] * 2))
    kq, kk, kv, kc = jax.random.split(jax.random.key(0), 4)
    q = jax.random.normal(kq, (4, 128, 8, 32))
    k = jax.random.normal(kk, (4, 128, 2, 32))
    v = jax.random.normal(kv, (4, 128, 2, 32))
    cotangent = jax.random.normal(kc, q.shape)
    reference_mask = AttentionMask.causal(sliding_window=40).with_segment_ids(segments)

    def reference_loss(q, k, v):
        return jnp.sum(reference_attention(q, k, v, reference_mask, logits_dtype=jnp.float32) * cotangent)

    with jax.default_matmul_precision("highest"):
        expected = reference_attention(q, k, v, reference_mask, logits_dtype=jnp.float32)
        expected_grads = jax.grad(reference_loss, argnums=(0, 1, 2))(q, k, v)

    def flash(q, k, v, mask):
        return pallas_triton_flash_attention(q, k, v, mask, block_sizes=blocks, interpret=True)

    def loss(q, k, v, mask, cotangent):
        return jnp.sum(flash(q, k, v, mask) * cotangent)

    mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2), ("data", "model"), axis_types=(AxisType.Explicit,) * 2)
    with jax.set_mesh(mesh), jax.default_matmul_precision("highest"):
        sharding = NamedSharding(mesh, P("data", None, "model", None))
        q, k, v, cotangent = (jax.device_put(x, sharding) for x in (q, k, v, cotangent))
        ids = jax.device_put(segments, NamedSharding(mesh, P("data", None)))
        # Separate q and kv arrays, as a batch from the data loader carries them.
        mask = AttentionMask.causal(sliding_window=40).with_segment_ids(ids, jnp.copy(ids))
        actual = jax.jit(flash)(q, k, v, mask)
        grads = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))(q, k, v, mask, cotangent)

    assert actual.sharding.spec == sharding.spec
    assert all(g.sharding.spec == sharding.spec for g in grads)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-4, rtol=1e-4)
    for name, got, want in zip("qkv", grads, expected_grads, strict=True):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), atol=1e-4, rtol=1e-4, err_msg=f"d{name}")
"""


def test_flash_attention_batch_and_head_sharded_packed_matches_reference():
    run_on_cpu_devices(_SHARDED_SCRIPT, device_count=4)


@pytest.mark.parametrize("window", [None, 40])
def test_flash_attention_reads_precomputed_fa4_bounds(window):
    # moe_hero_ep and moe_hero_fsdp precompute these bounds outside their layer scan and pass a full causal mask
    # alongside them, so the bounds alone must carry the segments and the window. Both runs use the same kernel on
    # the same per-query bounds, so they agree to rounding; the segment-ID path is checked against the reference above.
    q, k, v = _qkv(jnp.float32)
    segment_ids = MASKS["window_segments"].segment_ids
    segment_mask = AttentionMask(is_causal=True, sliding_window=window, segment_ids=segment_ids)
    lower, valid = fa4_cute_segment_bounds(segment_mask, batch_size=2, seq_len=SEQ_LEN, sliding_window=window)
    bounds_mask = AttentionMask(is_causal=True, segment_ids=segment_ids).with_fa4_bounds(lower, valid)
    # Padding queries attend only to themselves under precomputed bounds, so give their rows no cotangent: then
    # neither their outputs nor their gradients reach the comparison.
    valid_rows = (segment_ids[0] >= 0)[:, :, None, None]
    cotangent = jnp.where(valid_rows, jax.random.normal(jax.random.key(1), q.shape, jnp.float32), 0.0)

    def loss(mask, q, k, v):
        return jnp.sum(_flash(q, k, v, mask) * cotangent)

    expected = jnp.where(valid_rows, _flash(q, k, v, segment_mask), 0.0)
    actual = jnp.where(valid_rows, _flash(q, k, v, bounds_mask), 0.0)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-6, rtol=1e-6)
    expected_grads = jax.grad(lambda *a: loss(segment_mask, *a), argnums=(0, 1, 2))(q, k, v)
    actual_grads = jax.grad(lambda *a: loss(bounds_mask, *a), argnums=(0, 1, 2))(q, k, v)
    _assert_gradients_match(actual_grads, expected_grads, tol=1e-6)


# The moe_hero_ep layer scan: bounds for the full causal and the windowed layers are computed once outside the scan
# and sharded over the batch, and each layer selects its bounds with jnp.where on a scanned flag. Every layer attends
# the same q/k/v, so each one compares directly against the reference under its own mask.
_SHARDED_SCAN_SCRIPT = """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax import lax
    from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P, reshard

    from levanter.grug.attention import AttentionMask, fa4_cute_segment_bounds, reference_attention
    from levanter.grug.attention._amd_pallas_triton_flash import TritonFlashBlockSizes, pallas_triton_flash_attention

    blocks = TritonFlashBlockSizes(block_q=32, block_k=16, block_q_dq=32, block_k_dq=16, block_q_dkv=16, block_k_dkv=32)
    window = 40
    # Each segment ID is one contiguous run, the only layout precomputed bounds describe.
    segments = np.concatenate([np.zeros(50), np.ones(60), -np.ones(18)]).astype(np.int32)
    rolled = np.roll(segments, 20)
    rolled[:20] = 0
    segments = jnp.asarray(np.stack([segments, rolled] * 2))
    use_long = jnp.asarray([False, True, False])
    kq, kk, kv, kc = jax.random.split(jax.random.key(0), 4)
    q = jax.random.normal(kq, (4, 128, 8, 32))
    k = jax.random.normal(kk, (4, 128, 2, 32))
    v = jax.random.normal(kv, (4, 128, 2, 32))
    # Padding queries attend only to themselves under precomputed bounds, so they get no cotangent.
    valid_rows = (segments >= 0)[:, :, None, None]
    cotangents = jnp.where(valid_rows, jax.random.normal(kc, (3, *q.shape)), 0.0)

    def reference_loss(q, k, v):
        total = 0.0
        for layer_long, cotangent in zip(use_long.tolist(), cotangents):
            mask = AttentionMask.causal(sliding_window=None if layer_long else window).with_segment_ids(segments)
            total += jnp.sum(reference_attention(q, k, v, mask, logits_dtype=jnp.float32) * cotangent)
        return total

    with jax.default_matmul_precision("highest"):
        expected_loss, expected_grads = jax.value_and_grad(reference_loss, argnums=(0, 1, 2))(q, k, v)

    def loss(q, k, v, segment_ids, cotangents):
        mask = AttentionMask.causal().with_segment_ids(segment_ids, jnp.copy(segment_ids))
        long_lower, valid = fa4_cute_segment_bounds(mask, batch_size=4, seq_len=128, sliding_window=None)
        short_lower, _ = fa4_cute_segment_bounds(mask, batch_size=4, seq_len=128, sliding_window=window)
        bounds_spec = P("data", None)
        long_lower, short_lower, valid = (reshard(x, bounds_spec) for x in (long_lower, short_lower, valid))

        def layer(total, scan_inputs):
            layer_long, cotangent = scan_inputs
            layer_mask = mask.with_fa4_bounds(jnp.where(layer_long, long_lower, short_lower), valid)
            out = pallas_triton_flash_attention(q, k, v, layer_mask, block_sizes=blocks, interpret=True)
            return total + jnp.sum(out * cotangent), None

        total, _ = lax.scan(layer, jnp.zeros((), jnp.float32), (use_long, cotangents))
        return total

    mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2), ("data", "model"), axis_types=(AxisType.Explicit,) * 2)
    with jax.set_mesh(mesh), jax.default_matmul_precision("highest"):
        sharding = NamedSharding(mesh, P("data", None, "model", None))
        q, k, v = (jax.device_put(x, sharding) for x in (q, k, v))
        cotangents = jax.device_put(cotangents, NamedSharding(mesh, P(None, "data", None, "model", None)))
        ids = jax.device_put(segments, NamedSharding(mesh, P("data", None)))
        actual_loss, grads = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2)))(q, k, v, ids, cotangents)

    assert all(g.sharding.spec == sharding.spec for g in grads)
    np.testing.assert_allclose(np.asarray(actual_loss), np.asarray(expected_loss), atol=1e-3, rtol=1e-5)
    for name, got, want in zip("qkv", grads, expected_grads, strict=True):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), atol=1e-4, rtol=1e-4, err_msg=f"d{name}")
"""


def test_flash_attention_sharded_layer_scan_over_precomputed_bounds_matches_reference():
    run_on_cpu_devices(_SHARDED_SCAN_SCRIPT, device_count=4)


@pytest.mark.skipif(INTERPRET, reason="production tiles and shapes are too slow for the Pallas interpreter")
def test_flash_attention_bf16_default_tiles_on_precomputed_bounds_matches_reference():
    # The moe_hero_ep attention at its production sequence length and window, through the precomputed-bounds path
    # with the default tile sizes that a training run compiles.
    seq_len, window = 4096, 2048
    kq, kk, kv, kc = jax.random.split(jax.random.key(0), 4)
    q = jax.random.normal(kq, (2, seq_len, 8, 128), jnp.float32).astype(jnp.bfloat16)
    k = jax.random.normal(kk, (2, seq_len, 2, 128), jnp.float32).astype(jnp.bfloat16)
    v = jax.random.normal(kv, (2, seq_len, 2, 128), jnp.float32).astype(jnp.bfloat16)
    segments = jnp.concatenate([jnp.zeros(1500), jnp.ones(2300), -jnp.ones(296)]).astype(jnp.int32)
    segment_ids = jnp.stack([segments, jnp.roll(segments, 700).at[:700].set(0)])
    reference_mask = AttentionMask.causal(sliding_window=window).with_segment_ids(segment_ids)
    lower, valid = fa4_cute_segment_bounds(reference_mask, batch_size=2, seq_len=seq_len, sliding_window=window)
    bounds_mask = AttentionMask.causal().with_segment_ids(segment_ids).with_fa4_bounds(lower, valid)
    valid_rows = (segment_ids >= 0)[:, :, None, None]
    cotangent = jnp.where(valid_rows, jax.random.normal(kc, q.shape, jnp.float32), 0.0)

    def flash(q, k, v):
        return pallas_triton_flash_attention(q, k, v, bounds_mask)

    def loss(attend, q, k, v):
        return jnp.sum(attend(q, k, v).astype(jnp.float32) * cotangent)

    def reference(q, k, v):
        return _reference(q, k, v, reference_mask)

    actual = jnp.where(valid_rows, jax.jit(flash)(q, k, v).astype(jnp.float32), 0.0)
    expected = jnp.where(valid_rows, reference(q, k, v), 0.0)
    tol = TOLERANCE[jnp.bfloat16]
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=tol, rtol=tol)
    actual_grads = jax.jit(jax.grad(lambda *a: loss(flash, *a), argnums=(0, 1, 2)))(q, k, v)
    expected_grads = jax.grad(lambda *a: loss(reference, *a), argnums=(0, 1, 2))(q, k, v)
    _assert_gradients_match(actual_grads, expected_grads, tol=tol)
