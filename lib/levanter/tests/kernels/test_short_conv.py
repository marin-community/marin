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

The whole module is scoped to the backends this Triton kernel targets. `dx` is bitwise
only because `_dx_body` accumulates in the order XLA's transpose of the pad-and-shift
forward emits, and that order is a property of the backend's transpose, not of the
algorithm: on TPU the multi-tap shapes disagree in the last bit or two while `W=1`, which
has no accumulation to associate, still matches exactly. Running the interpreter on TPU
therefore measures XLA:TPU rather than the kernel that ships.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from levanter.kernels.pallas.short_conv import (
    ShortConvBlockSizes,
    short_conv,
    short_conv_reference,
)
from levanter.kernels.pallas.short_conv.pallas_gpu import interpret_mode
from levanter.testing.cpu_devices import run_on_cpu_devices

pytestmark = pytest.mark.skipif(
    jax.default_backend() == "tpu",
    reason="Triton kernel: dx parity is defined against XLA's CPU/GPU transpose of the reference",
)

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

    Checked on the lowered jaxpr rather than by inspection, so a refactor that drops the
    manual region fails here. The mesh is abstract and nothing is executed, so a
    single-device CPU runner is enough.
    """
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


@pytest.mark.parametrize(("model_size", "should_reject"), [(1, False), (2, True)])
def test_context_parallel_path_keeps_the_channel_axis_gate(model_size, should_reject):
    """The halo path shards the sequence by design, but a genuinely sharded channel axis must
    still raise there, exactly as on the unsharded path, instead of being silently all-gathered."""
    mesh = jax.sharding.AbstractMesh(
        axis_sizes=(1, 1, 2, 1, model_size),
        axis_names=("replica_dcn", "data", "context", "expert", "model"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 5,
    )
    weight, x, segment_ids, _ = _inputs(2, 32, 8, 4, seed=11, dtype=jnp.float32, packed=True)

    def fn(w, xx, seg):
        xx = jax.sharding.reshard(xx, jax.sharding.PartitionSpec(("data", "expert"), "context", "model"))
        seg = jax.sharding.reshard(seg, jax.sharding.PartitionSpec(("data", "expert"), "context"))
        return short_conv(w, xx, seg, implementation="reference", batch_axes=("data", "expert"))

    with jax.sharding.use_abstract_mesh(mesh):
        if should_reject:
            with pytest.raises(ValueError, match="unsharded channel axis"):
                jax.make_jaxpr(fn)(weight, x, segment_ids)
        else:
            text = str(jax.make_jaxpr(fn)(weight, x, segment_ids))
            assert "ppermute" in text
            for banned in ("all_gather", "all_reduce", "psum", "all_to_all", "reduce_scatter"):
                assert banned not in text, f"the halo exchange lowered through an unexpected {banned}"


def test_context_parallel_path_rejects_a_batch_axis_it_would_gather():
    """A batch sharded over an axis missing from ``batch_axes`` must raise, not be all-gathered.

    The shard-local path can skip the shard_map when no batch axis is active, but the halo
    path cannot: resharding to ``P(None, "context", None)`` would replicate the batch.
    """
    mesh = jax.sharding.AbstractMesh(
        axis_sizes=(2, 2, 2),
        axis_names=("fsdp", "context", "spare"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 3,
    )
    weight, x, segment_ids, _ = _inputs(2, 32, 8, 4, seed=11, dtype=jnp.float32, packed=True)

    def fn(w, xx, seg):
        xx = jax.sharding.reshard(xx, P("fsdp", "context", None))
        seg = jax.sharding.reshard(seg, P("fsdp", "context"))
        return short_conv(w, xx, seg, implementation="reference", batch_axes=("data",))

    with jax.sharding.use_abstract_mesh(mesh), pytest.raises(ValueError, match="not in batch_axes"):
        jax.make_jaxpr(fn)(weight, x, segment_ids)


def test_context_axis_named_in_batch_axes_shards_only_the_sequence():
    """Grug's token axes list ``context`` next to the batch axes; it must not be spelled twice."""
    mesh = jax.sharding.AbstractMesh(
        axis_sizes=(2, 2, 2),
        axis_names=("data", "context", "expert"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 3,
    )
    weight, x, segment_ids, _ = _inputs(4, 32, 8, 4, seed=11, dtype=jnp.float32, packed=True)

    def fn(w, xx, seg):
        xx = jax.sharding.reshard(xx, P(("data", "expert"), "context", None))
        seg = jax.sharding.reshard(seg, P(("data", "expert"), "context"))
        return short_conv(w, xx, seg, implementation="reference", batch_axes=("data", "expert", "context"))

    with jax.sharding.use_abstract_mesh(mesh):
        jaxpr = jax.make_jaxpr(fn)(weight, x, segment_ids)
    assert "ppermute" in str(jaxpr)
    assert "all_gather" not in str(jaxpr)


def test_short_conv_rejects_mixed_dtypes():
    """Mixed weight/activation dtypes are rejected at the boundary. The reference promotes
    (fp32) while the Pallas kernel outputs ``x.dtype``, so accepting mixed inputs would make
    the same call backend-dependent; the dtype gate runs before dispatch on every backend."""
    weight = jnp.ones((4, 8), dtype=jnp.float32)
    x = jnp.ones((1, 16, 8), dtype=jnp.bfloat16)
    with pytest.raises(ValueError, match="share a dtype"):
        short_conv(weight, x)


_HALO_SCRIPT = """
import itertools

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from levanter.kernels.pallas.short_conv import ShortConvBlockSizes, short_conv, short_conv_reference
from levanter.kernels.pallas.short_conv.pallas_gpu import interpret_mode

IMPLEMENTATION = "__IMPLEMENTATION__"
BATCH, SEQ, CHANNELS = 2, 32, 8
DEVICES = np.asarray(jax.devices())
assert DEVICES.size == 8


def bits(array):
    array = np.asarray(array)
    return array.view({2: np.uint16, 4: np.uint32}[array.dtype.itemsize])


def inputs(width, packed, dtype):
    rng = np.random.default_rng(width)
    weight = jnp.asarray(rng.standard_normal((width, CHANNELS)) * 0.5, dtype)
    x = jnp.asarray(rng.standard_normal((BATCH, SEQ, CHANNELS)), dtype)
    cotangent = jnp.asarray(rng.standard_normal((BATCH, SEQ, CHANNELS)), dtype)
    if not packed:
        return weight, x, cotangent, None
    # Short runs, and a document boundary inside the context=4 left halo.
    seg = np.concatenate([np.zeros((BATCH, 5)), np.full((BATCH, 2), 7), np.full((BATCH, SEQ - 7), 9)], axis=1)
    return weight, x, cotangent, jnp.asarray(seg, jnp.int32)


def check(context, packed, width):
    # `data` splits the batch; what is left goes on an axis nothing names, so every device is in
    # the mesh. Blocks are small so the padded local block spans several sequence tiles with a
    # ragged tail, and just wide enough for the kernel's `s_block_size >= kernel_size - 1` rule.
    mesh = Mesh(
        DEVICES.reshape(BATCH, context, 8 // (BATCH * context)),
        ("data", "context", "spare"),
        axis_types=(AxisType.Explicit,) * 3,
    )
    blocks = ShortConvBlockSizes(s_block_size=max(8, width - 1), c_block_size=8)

    def conv(w, xx, seg):
        return short_conv(w, xx, seg, implementation=IMPLEMENTATION, block_sizes=blocks, batch_axes=("data",))

    def loss(w, xx, seg):
        return jnp.sum(conv(w, xx, seg) * cotangent)

    def reference_loss(w, xx, seg):
        return jnp.sum(short_conv_reference(w, xx, seg) * cotangent)

    for dtype in (jnp.bfloat16, jnp.float32):
        weight, x, cotangent, segment_ids = inputs(width, packed, dtype)
        with jax.set_mesh(mesh), interpret_mode():
            x_sharded = jax.device_put(x, NamedSharding(mesh, P("data", "context", None)))
            seg_sharded = None
            if segment_ids is not None:
                seg_sharded = jax.device_put(segment_ids, NamedSharding(mesh, P("data", "context")))

            if width - 1 > SEQ // context:
                try:
                    conv(weight, x_sharded, seg_sharded)
                except ValueError as error:
                    assert "halo" in str(error), error
                else:
                    raise AssertionError("a halo longer than the local sequence must be rejected")
                return

            # The bf16 forward is bitwise. XLA:CPU contracts the fp32 reference into FMAs
            # differently per fusion, and dx reassociates at shard boundaries in any dtype
            # (see the `short_conv` docstring), so those are held to fp32 rounding instead.
            got = jax.jit(conv)(weight, x_sharded, seg_sharded)
            want = short_conv_reference(weight, x, segment_ids)
            if dtype == jnp.bfloat16:
                np.testing.assert_array_equal(bits(got), bits(want))
            else:
                np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-6, atol=1e-6)
                dw_want, dx_want = jax.jit(jax.grad(reference_loss, argnums=(0, 1)))(weight, x, segment_ids)
                dw_got, dx_got = jax.jit(jax.grad(loss, argnums=(0, 1)))(weight, x_sharded, seg_sharded)
                np.testing.assert_allclose(np.asarray(dx_got), np.asarray(dx_want), rtol=1e-6, atol=1e-6)
                np.testing.assert_allclose(np.asarray(dw_got), np.asarray(dw_want), rtol=1e-6, atol=1e-6)


for case in itertools.product((2, 4), (True, False), (1, 4, 17)):
    try:
        check(*case)
    except Exception as error:
        raise AssertionError(f"context, packed, width = {case}") from error
"""

_AUTO_MESH_GATE_SCRIPT = """
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from levanter.kernels.pallas.short_conv import ShortConvBlockSizes, short_conv
from levanter.kernels.pallas.short_conv.pallas_gpu import interpret_mode

# A concrete array on an Auto-axis mesh shows its placement only on `array.sharding`; the
# channel gate must still see it rather than reshard the axis away.
mesh = Mesh(np.asarray(jax.devices()).reshape(2, 4), ("data", "model"), axis_types=(AxisType.Auto,) * 2)
weight = jnp.ones((4, 8), jnp.bfloat16)
x = jax.device_put(jnp.ones((2, 32, 8), jnp.bfloat16), NamedSharding(mesh, P("data", None, "model")))
with jax.set_mesh(mesh), interpret_mode():
    try:
        short_conv(weight, x, implementation="pallas_gpu", block_sizes=ShortConvBlockSizes(8, 8))
    except ValueError as error:
        assert "unsharded channel axis" in str(error), error
    else:
        raise AssertionError("a concrete channel-sharded array slipped past the gate")
"""


@pytest.mark.parametrize("implementation", ["reference", "pallas_gpu"])
def test_context_parallel_halo_matches_the_unsharded_reference(implementation):
    """Packed and unpacked, halo 0, 3 and 16, on a real 8-device CPU mesh.

    One interpreter per backend runs the whole grid: the JAX import and backend start
    dominate a fresh process. The Pallas leg runs the kernel body under the interpreter with
    its halo, ragged tail pad and multi-block local sequence, which the reference backend
    never exercises.
    """
    run_on_cpu_devices(_HALO_SCRIPT.replace("__IMPLEMENTATION__", implementation), device_count=8)


def test_channel_axis_gate_reads_concrete_shardings_on_an_auto_mesh():
    run_on_cpu_devices(_AUTO_MESH_GATE_SCRIPT, device_count=8)


@pytest.mark.parametrize("width", [1, 4])
def test_context_parallel_pallas_matches_reference(width):
    if jax.default_backend() != "gpu" or jax.device_count() < 4:
        pytest.skip("requires four GPUs for context-parallel Pallas convolution")
    mesh = Mesh(np.asarray(jax.devices()[:4]), ("context",), axis_types=(AxisType.Explicit,))
    weight, x, segment_ids, cotangent = _inputs(1, 256, 128, width, seed=17, dtype=jnp.float32, packed=True)
    # local_seq 64 + halo rounds to 128: two sequence blocks, so the general programs run too.
    blocks = ShortConvBlockSizes(s_block_size=64, c_block_size=128)

    def reference_loss(w, xx):
        return jnp.sum(short_conv_reference(w, xx, segment_ids) * cotangent)

    want = short_conv_reference(weight, x, segment_ids)
    want_dw, want_dx = jax.grad(reference_loss, argnums=(0, 1))(weight, x)
    with jax.set_mesh(mesh):
        sharded_x = jax.device_put(x, NamedSharding(mesh, P(None, "context", None)))
        sharded_seg = jax.device_put(segment_ids, NamedSharding(mesh, P(None, "context")))

        def forward(w, xx):
            return short_conv(w, xx, sharded_seg, implementation="pallas_gpu", block_sizes=blocks)

        def loss(w, xx):
            return jnp.sum(forward(w, xx) * cotangent)

        got = forward(weight, sharded_x)
        got_dw, got_dx = jax.grad(loss, argnums=(0, 1))(weight, sharded_x)
    for actual, expected in ((got, want), (got_dw, want_dw), (got_dx, want_dx)):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)
