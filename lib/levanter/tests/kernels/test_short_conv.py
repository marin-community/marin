# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Correctness gate for the fused depthwise causal short convolution (SConv).

The Pallas kernel targets a GPU and cannot execute on CPU as compiled code, but Pallas's
reference interpreter executes the *kernel body* -- grid, block specs, the neighbouring
sequence-block views that supply the halo, the edge masking, the register accumulation --
with plain XLA ops. Everything here therefore exercises the real algorithm on CPU. What
it does not and cannot cover is whether the kernel *lowers* on a given GPU architecture;
`test_pallas_short_conv_matches_reference_on_gpu` covers that when a GPU is present.

The bar is bitwise: the forward and `dx` must be bit-identical to the pad-and-shift
reference, because a fused conv changes only *when* bytes cross HBM, never the
arithmetic. `dw` is a reduction over 65,536 tokens whose association order XLA does not
define, so it is checked against a float64 oracle instead -- the kernel must be at least
as accurate as the reference, not identical to it.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.kernels.pallas.short_conv import (
    ShortConvBlockSizes,
    short_conv,
    short_conv_reference,
)
from levanter.kernels.pallas.short_conv.pallas_gpu import interpret_mode

# (batch, seq_len, channels, kernel_size, s_block, c_block)
SHAPES = [
    (1, 32, 8, 4, 8, 8),
    (2, 32, 8, 4, 8, 4),
    (3, 64, 16, 4, 16, 8),
    (2, 64, 32, 4, 32, 16),
    (2, 48, 8, 3, 16, 8),
    (1, 64, 8, 2, 32, 8),
    (1, 64, 8, 1, 32, 8),
    (2, 128, 16, 4, 128, 16),  # a single sequence block: no neighbour view is in range
]


def _packed_segment_ids(rng, batch, seq_len, min_run=1, max_run=None):
    """Contiguous document runs, deliberately including runs shorter than the kernel."""
    max_run = max_run or max(min_run + 1, seq_len // 4)
    out = np.zeros((batch, seq_len), np.int32)
    for b in range(batch):
        pos, sid = 0, 0
        while pos < seq_len:
            run = int(rng.integers(min_run, max_run + 1))
            out[b, pos : pos + run] = sid
            pos += run
            sid += 1
    return jnp.asarray(out)


def _segment_start_mask(segment_ids, width):
    """[B, S] mask over the first `width - 1` positions of every segment."""
    seg = np.asarray(jax.device_get(segment_ids))
    starts = np.ones_like(seg, dtype=bool)
    starts[:, 1:] = seg[:, 1:] != seg[:, :-1]
    mask = np.zeros_like(starts)
    for offset in range(max(width - 1, 1)):
        shifted = starts if offset == 0 else np.pad(starts[:, :-offset], ((0, 0), (offset, 0)))
        mask |= shifted.astype(bool)
    return mask


def _bits(array):
    array = np.asarray(jax.device_get(array))
    return array.view({2: np.uint16, 4: np.uint32, 8: np.uint64}[array.dtype.itemsize])


def _run_both(weight, x, segment_ids, cotangent, blocks):
    def kernel_fn(w, xx):
        return short_conv(w, xx, segment_ids, implementation="pallas_gpu", block_sizes=blocks)

    def reference_fn(w, xx):
        return short_conv_reference(w, xx, segment_ids)

    with interpret_mode():
        got = jax.jit(kernel_fn)(weight, x)
        _, kernel_vjp = jax.vjp(kernel_fn, weight, x)
        got_dw, got_dx = jax.jit(kernel_vjp)(cotangent)

    want = jax.jit(reference_fn)(weight, x)
    _, reference_vjp = jax.vjp(reference_fn, weight, x)
    want_dw, want_dx = jax.jit(reference_vjp)(cotangent)
    return (got, got_dx, got_dw), (want, want_dx, want_dw)


def _inputs(batch, seq_len, channels, width, seed, dtype, packed):
    rng = np.random.default_rng(seed)
    x = jnp.asarray(rng.standard_normal((batch, seq_len, channels)), dtype)
    weight = jnp.asarray(rng.standard_normal((width, channels)) * 0.5, dtype)
    cotangent = jnp.asarray(rng.standard_normal((batch, seq_len, channels)), dtype)
    segment_ids = _packed_segment_ids(rng, batch, seq_len) if packed else None
    return weight, x, segment_ids, cotangent


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(str(v) for v in s))
@pytest.mark.parametrize("packed", [True, False], ids=["packed", "unpacked"])
def test_forward_and_dx_are_bitwise_identical_to_reference(shape, packed):
    batch, seq_len, channels, width, s_block, c_block = shape
    weight, x, segment_ids, cotangent = _inputs(
        batch, seq_len, channels, width, seed=hash(shape) % 2**16, dtype=jnp.bfloat16, packed=packed
    )
    blocks = ShortConvBlockSizes(s_block_size=s_block, c_block_size=c_block)
    (got, got_dx, _), (want, want_dx, _) = _run_both(weight, x, segment_ids, cotangent, blocks)

    np.testing.assert_array_equal(_bits(got), _bits(want), err_msg="forward is not bit-identical")
    np.testing.assert_array_equal(_bits(got_dx), _bits(want_dx), err_msg="dx is not bit-identical")


@pytest.mark.parametrize("shape", SHAPES[:5], ids=lambda s: "x".join(str(v) for v in s))
def test_segment_boundaries_and_segment_starts_match_exactly(shape):
    """The first `kernel_size - 1` positions of every document are where taps get dropped.

    Those positions are checked on their own so a regression there cannot hide inside a
    whole-tensor max.
    """
    batch, seq_len, channels, width, s_block, c_block = shape
    weight, x, segment_ids, cotangent = _inputs(
        batch, seq_len, channels, width, seed=99, dtype=jnp.bfloat16, packed=True
    )
    blocks = ShortConvBlockSizes(s_block_size=s_block, c_block_size=c_block)
    (got, got_dx, _), (want, want_dx, _) = _run_both(weight, x, segment_ids, cotangent, blocks)

    mask = _segment_start_mask(segment_ids, width)
    assert mask.any(), "test fixture produced no segment starts"
    np.testing.assert_array_equal(_bits(got)[mask], _bits(want)[mask])
    np.testing.assert_array_equal(_bits(got_dx)[mask], _bits(want_dx)[mask])

    # And the taps really are being dropped: with a non-degenerate weight, masking must
    # make the output differ from an unmasked convolution somewhere on the boundary.
    unmasked = short_conv_reference(weight, x, None)
    assert not np.array_equal(_bits(unmasked)[mask], _bits(want)[mask])


@pytest.mark.parametrize("shape", SHAPES[:4], ids=lambda s: "x".join(str(v) for v in s))
def test_dw_is_at_least_as_accurate_as_the_reference(shape):
    """`dw` reduces 65,536 tokens in fp32; association order is not part of the contract.

    Both implementations are compared against a float64 oracle. The kernel is required to
    be no worse than the reference, which is the meaningful guarantee.
    """
    batch, seq_len, channels, width, s_block, c_block = shape
    weight, x, segment_ids, cotangent = _inputs(
        batch, seq_len, channels, width, seed=5, dtype=jnp.bfloat16, packed=True
    )
    blocks = ShortConvBlockSizes(s_block_size=s_block, c_block_size=c_block)
    (_, _, got_dw), (_, _, want_dw) = _run_both(weight, x, segment_ids, cotangent, blocks)

    # float64 oracle for dw, computed directly from the definition.
    x64 = np.asarray(jax.device_get(x), np.float64)
    ct64 = np.asarray(jax.device_get(cotangent), np.float64)
    seg = np.asarray(jax.device_get(segment_ids))
    oracle = np.zeros((width, channels), np.float64)
    for lag in range(width):
        shifted = np.zeros_like(x64)
        if lag == 0:
            shifted = x64
            keep = np.ones(seg.shape, bool)
        else:
            shifted[:, lag:, :] = x64[:, :-lag, :]
            seg_shifted = np.full(seg.shape, -1, seg.dtype)
            seg_shifted[:, lag:] = seg[:, :-lag]
            keep = seg_shifted == seg
        oracle[lag] = np.sum(ct64 * shifted * keep[..., None], axis=(0, 1))

    got_err = np.max(np.abs(np.asarray(jax.device_get(got_dw), np.float64) - oracle))
    want_err = np.max(np.abs(np.asarray(jax.device_get(want_dw), np.float64) - oracle))
    scale = max(np.max(np.abs(oracle)), 1e-30)
    assert got_err <= max(want_err * 1.5, 0.02 * scale), (
        f"kernel dw error {got_err:.3e} materially worse than reference {want_err:.3e} " f"(oracle scale {scale:.3e})"
    )


def test_float32_gradients_match_to_float32_tolerance():
    """fp32 inputs: the pad/shift chain and the kernel differ only by fp32 reassociation."""
    weight, x, segment_ids, cotangent = _inputs(2, 64, 16, 4, seed=17, dtype=jnp.float32, packed=True)
    blocks = ShortConvBlockSizes(s_block_size=16, c_block_size=8)
    (got, got_dx, got_dw), (want, want_dx, want_dw) = _run_both(weight, x, segment_ids, cotangent, blocks)
    for name, a, b in (("y", got, want), ("dx", got_dx, want_dx), ("dw", got_dw, want_dw)):
        np.testing.assert_allclose(
            jax.device_get(a), jax.device_get(b), rtol=1e-5, atol=1e-5, err_msg=f"{name} mismatch"
        )


def test_explicit_implementation_fails_fast_when_unsupported():
    """An explicitly requested backend must raise, never silently fall back (api-patterns)."""
    weight, x, segment_ids, _ = _inputs(2, 32, 8, 4, seed=1, dtype=jnp.bfloat16, packed=True)
    bad_blocks = ShortConvBlockSizes(s_block_size=7, c_block_size=8)  # 32 % 7 != 0
    with interpret_mode():
        with pytest.raises(RuntimeError, match="not divisible"):
            short_conv(weight, x, segment_ids, implementation="pallas_gpu", block_sizes=bad_blocks)


def test_ordered_implementation_sequence_falls_back_with_a_warning():
    weight, x, segment_ids, _ = _inputs(2, 32, 8, 4, seed=1, dtype=jnp.bfloat16, packed=True)
    bad_blocks = ShortConvBlockSizes(s_block_size=7, c_block_size=8)
    with interpret_mode():
        with pytest.warns(UserWarning, match="falling back"):
            got = short_conv(
                weight, x, segment_ids, implementation=("pallas_gpu", "reference"), block_sizes=bad_blocks
            )
    np.testing.assert_array_equal(_bits(got), _bits(short_conv_reference(weight, x, segment_ids)))


def test_default_implementation_on_cpu_is_the_reference():
    if jax.default_backend() == "gpu":
        pytest.skip("this asserts the non-GPU default")
    weight, x, segment_ids, _ = _inputs(2, 32, 8, 4, seed=2, dtype=jnp.bfloat16, packed=True)
    got = short_conv(weight, x, segment_ids)
    np.testing.assert_array_equal(_bits(got), _bits(short_conv_reference(weight, x, segment_ids)))


def test_kernel_call_is_wrapped_in_a_shard_map_under_a_mesh():
    """House rule: every Pallas call sits inside an explicit shard_map on a real mesh.

    Checked on the lowered HLO rather than by inspection, so a refactor that drops the
    manual region fails here.
    """
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("needs >= 2 devices to form a non-trivial mesh")
    mesh = jax.sharding.AbstractMesh(
        axis_sizes=(1, 2, 1, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 4,
    )
    weight, x, segment_ids, _ = _inputs(2, 32, 8, 4, seed=3, dtype=jnp.bfloat16, packed=True)
    blocks = ShortConvBlockSizes(s_block_size=8, c_block_size=8)

    def fn(w, xx, seg):
        return short_conv(w, xx, seg, implementation="pallas_gpu", block_sizes=blocks)

    with interpret_mode(), jax.sharding.use_abstract_mesh(mesh):
        jaxpr = jax.make_jaxpr(fn)(weight, x, segment_ids)
    text = str(jaxpr)
    assert "shard_map" in text, "kernel is not inside an explicit shard_map"
    # ...and the manual region must not contain a collective: the op is shard-local.
    for banned in ("all_gather", "all_reduce", "psum", "all_to_all", "reduce_scatter"):
        assert banned not in text, f"short_conv lowered through an unexpected {banned}"


def test_pallas_short_conv_matches_reference_on_gpu():
    """The compiled kernel, not the interpreter. Only meaningful with a GPU present."""
    if jax.default_backend() != "gpu":
        pytest.skip("requires the JAX GPU backend")
    weight, x, segment_ids, cotangent = _inputs(2, 512, 256, 4, seed=21, dtype=jnp.bfloat16, packed=True)
    blocks = ShortConvBlockSizes(s_block_size=128, c_block_size=128)

    def kernel_fn(w, xx):
        return short_conv(w, xx, segment_ids, implementation="pallas_gpu", block_sizes=blocks)

    def reference_fn(w, xx):
        return short_conv_reference(w, xx, segment_ids)

    got = jax.jit(kernel_fn)(weight, x)
    _, kernel_vjp = jax.vjp(kernel_fn, weight, x)
    got_dw, got_dx = jax.jit(kernel_vjp)(cotangent)

    want = jax.jit(reference_fn)(weight, x)
    _, reference_vjp = jax.vjp(reference_fn, weight, x)
    want_dw, want_dx = jax.jit(reference_vjp)(cotangent)

    np.testing.assert_array_equal(_bits(got), _bits(want))
    np.testing.assert_array_equal(_bits(got_dx), _bits(want_dx))
    np.testing.assert_allclose(
        jax.device_get(got_dw).astype(np.float32),
        jax.device_get(want_dw).astype(np.float32),
        rtol=5e-2,
        atol=5e-2,
    )


@pytest.mark.parametrize(
    ("model_size", "should_reject"),
    [(1, False), (2, True)],
    ids=["size-1 model axis is a no-op", "size-2 model axis really shards"],
)
def test_channel_axis_gate_consults_the_mesh_not_just_the_spec(model_size, should_reject):
    """A spec entry naming a size-1 mesh axis shards nothing, and must not be rejected.

    This is not hypothetical. The EP64 hero mesh is (replica_dcn=1, data=1, expert=64, model=1)
    and the attention projections are `P(_FSDP_AXES, "model")`, so every k/v activation reaching
    SConv carries "model" on its channel axis while being, in fact, unsharded there. Gating on
    the name alone rejects the production shape -- and the reshard the gate guards is a no-op in
    exactly that case, so there is nothing to guard.

    The size-2 leg keeps the gate honest: a genuinely sharded channel axis must still raise,
    because silently resharding it would hide a real all-gather inside the kernel wrapper.
    """
    if len(jax.devices()) < 2:
        pytest.skip("needs >= 2 devices to form a non-trivial mesh")
    mesh = jax.sharding.AbstractMesh(
        axis_sizes=(1, 2 // model_size, 2, model_size),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 4,
    )
    weight, x, segment_ids, _ = _inputs(4, 32, 8, 4, seed=11, dtype=jnp.bfloat16, packed=True)
    blocks = ShortConvBlockSizes(s_block_size=8, c_block_size=8)

    def fn(w, xx, seg):
        # Reproduce the hero's k_flat sharding: batch over the FSDP pair, channel named "model".
        xx = jax.sharding.reshard(xx, jax.sharding.PartitionSpec(("data", "expert"), None, "model"))
        seg = jax.sharding.reshard(seg, jax.sharding.PartitionSpec(("data", "expert"), None))
        return short_conv(w, xx, seg, implementation="pallas_gpu", block_sizes=blocks)

    with interpret_mode(), jax.sharding.use_abstract_mesh(mesh):
        if should_reject:
            with pytest.raises(ValueError, match="unsharded channel axis"):
                jax.make_jaxpr(fn)(weight, x, segment_ids)
        else:
            jaxpr = jax.make_jaxpr(fn)(weight, x, segment_ids)
            assert "shard_map" in str(jaxpr)
