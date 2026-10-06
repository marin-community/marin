# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Correctness gate for the depthwise causal short convolution (SConv).

The Triton kernel runs only on a GPU, so its tests skip elsewhere. They hold it to the pad-and-shift
reference: the bf16 forward and ``dx`` bit for bit, and ``dw`` within the error of an fp32 sum rounded
once to bf16, because its association order is not part of the contract. The wrapper around the
kernel -- the halo exchange, padding, and the gates against hidden all-gathers -- runs on CPU, through
the reference or a traced kernel call.

``dx`` is bitwise only against XLA's CPU/GPU transpose of the reference, whose accumulation order the
kernel follows; on TPU the multi-tap shapes disagree in the last bit or two.
"""

import re
import zlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.extend.mlir import ir
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from levanter.kernels.triton.short_conv import api, short_conv, short_conv_reference, triton_gpu
from levanter.kernels.triton.short_conv.reference import OOB_SEGMENT
from levanter.testing.cpu_devices import run_on_cpu_devices

pytestmark = pytest.mark.skipif(
    jax.default_backend() == "tpu",
    reason="dx parity is defined against XLA's CPU/GPU transpose of the reference",
)


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


def _bits(array):
    array = np.asarray(jax.device_get(array))
    return array.view({2: np.uint16, 4: np.uint32, 8: np.uint64}[array.dtype.itemsize])


def _dw_oracle(x, segment_ids, cotangent, width):
    """float64 ``dw`` computed directly from the definition."""
    x64 = np.asarray(jax.device_get(x), np.float64)
    ct64 = np.asarray(jax.device_get(cotangent), np.float64)
    seg = np.zeros(x64.shape[:2], np.int32) if segment_ids is None else np.asarray(jax.device_get(segment_ids))
    oracle = np.zeros((width, x64.shape[2]), np.float64)
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
    return oracle


def _inputs(batch, seq_len, channels, width, seed, dtype, packed):
    rng = np.random.default_rng(seed)
    x = jnp.asarray(rng.standard_normal((batch, seq_len, channels)), dtype)
    weight = jnp.asarray(rng.standard_normal((width, channels)) * 0.5, dtype)
    cotangent = jnp.asarray(rng.standard_normal((batch, seq_len, channels)), dtype)
    segment_ids = _packed_segment_ids(rng, batch, seq_len) if packed else None
    return weight, x, segment_ids, cotangent


def _triton_segment_ids(kind, batch, seq_len, rng):
    if kind == "unpacked":
        return None
    if kind == "packed":
        return _packed_segment_ids(rng, batch, seq_len)
    if kind == "short_runs":
        # Documents of one to three tokens: every tap can cross a boundary, including the
        # boundaries between the rows a program walks and the halo rows it reloads.
        return _packed_segment_ids(rng, batch, seq_len, min_run=1, max_run=3)
    if kind == "padded":
        # Two documents, then padding carrying the out-of-range segment id.
        seg = np.full((batch, seq_len), OOB_SEGMENT, np.int32)
        for b in range(batch):
            valid = int(rng.integers(seq_len // 2, seq_len))
            seg[b, : valid // 3] = 0
            seg[b, valid // 3 : valid] = 1
        return jnp.asarray(seg)
    raise ValueError(kind)


def _triton_programs(fn, *args) -> list[bytes]:
    """The kernel calls, each with its compiled PTX, that lowering ``fn`` embeds."""
    programs = []

    def visit(op):
        for region in op.regions:
            for block in region.blocks:
                for child in block.operations:
                    child = child.operation
                    if child.name == "stablehlo.custom_call":
                        if ir.StringAttr(child.attributes["call_target_name"]).value == "triton_kernel_call":
                            programs.append(
                                zlib.decompress(ir.StringAttr(child.attributes["backend_config"]).value_bytes)
                            )
                    visit(child)

    visit(jax.jit(fn).lower(*args).compiler_ir("stablehlo").operation)
    return programs


@pytest.mark.parametrize("packed_bf16", [True, False], ids=["packed-bf16", "fp32-then-bf16"])
@pytest.mark.parametrize("segments", ["unpacked", "packed", "short_runs", "padded"])
@pytest.mark.parametrize(
    "shape", [(2, 512, 256), (1, 256, 1536), (2, 200, 256)], ids=lambda s: "x".join(str(v) for v in s)
)
def test_triton_short_conv_matches_reference_on_gpu(shape, segments, packed_bf16, monkeypatch):
    """Streaming Triton kernels: forward and ``dx`` bitwise, ``dw`` within fp32-accumulation error.

    512 rows span several of the kernel's sequence chunks, and 1536 channels span several
    channel blocks, so chunk halos and block edges are both exercised. 200 rows are padded to the
    kernel's sequence multiple. Unpacked inputs take
    the single-document steps only, short runs mostly the masked ones. Without packed bf16,
    the path GPUs older than SM90 take, the kernels round through fp32 and must give the same
    bits.
    """
    if jax.default_backend() != "gpu":
        pytest.skip("requires the JAX GPU backend")
    if packed_bf16 and not triton_gpu._packed_bf16_arithmetic():
        pytest.skip("packed bf16 arithmetic needs SM90 or newer")
    if not packed_bf16:
        monkeypatch.setattr(triton_gpu, "_packed_bf16_arithmetic", lambda: False)
    batch, seq_len, channels = shape
    weight, x, _, cotangent = _inputs(batch, seq_len, channels, 4, seed=31, dtype=jnp.bfloat16, packed=False)
    segment_ids = _triton_segment_ids(segments, batch, seq_len, np.random.default_rng(32))

    def kernel_fn(w, xx):
        return short_conv(w, xx, segment_ids, implementation="triton_gpu")

    def reference_fn(w, xx):
        return short_conv_reference(w, xx, segment_ids)

    got = jax.jit(kernel_fn)(weight, x)
    _, kernel_vjp = jax.vjp(kernel_fn, weight, x)
    got_dw, got_dx = jax.jit(kernel_vjp)(cotangent)
    programs = _triton_programs(kernel_fn, weight, x) + _triton_programs(kernel_vjp, cotangent)
    assert len(programs) == 2, "expected one forward and one backward kernel call"
    # The packed path's multiplies are inline PTX. The fallback's PTX can hold mul.rn.bf16x2 too, where
    # LLVM pairs two bf16 multiplies, but never inside an inline-asm block.
    for program in programs:
        packed = re.search(rb"// begin inline asm\s+mul\.rn\.bf16x2", program) is not None
        assert packed == packed_bf16, "the kernel did not take the requested rounding path"
    want = jax.jit(reference_fn)(weight, x)
    _, reference_vjp = jax.vjp(reference_fn, weight, x)
    _, want_dx = jax.jit(reference_vjp)(cotangent)

    np.testing.assert_array_equal(_bits(got), _bits(want), err_msg="forward is not bit-identical")
    np.testing.assert_array_equal(_bits(got_dx), _bits(want_dx), err_msg="dx is not bit-identical")
    # Each dw term is an exact fp32 product of two bf16 values. Their fp32 sum, in any order, differs
    # from the exact dw by at most gamma_n times the sum of the terms' magnitudes, and rounding it once
    # to bf16 adds at most 2^-8 of its magnitude. The bound holds elementwise, so it rejects a dw that
    # is uniformly scaled by more than about 2^-8.
    oracle = _dw_oracle(x, segment_ids, cotangent, 4)
    terms = batch * seq_len
    gamma = terms * 2.0**-24 / (1 - terms * 2.0**-24)
    absolute_sum = _dw_oracle(jnp.abs(x), segment_ids, jnp.abs(cotangent), 4)
    bound = 2.0**-8 * np.abs(oracle) + (1 + 2.0**-8) * gamma * absolute_sum
    error = np.abs(np.asarray(jax.device_get(got_dw), np.float64) - oracle)
    assert np.all(error <= bound), f"dw error is up to {np.max(error / bound):.2f}x its bound"


def test_triton_implementation_fails_fast_when_unsupported():
    """Off GPU the backend is missing; on GPU a kernel width other than 4 is unsupported."""
    width = 3 if jax.default_backend() == "gpu" else 4
    weight, x, segment_ids, _ = _inputs(2, 64, 8, width, seed=3, dtype=jnp.bfloat16, packed=True)
    with pytest.raises(RuntimeError, match="triton_gpu"):
        short_conv(weight, x, segment_ids, implementation="triton_gpu")
    with pytest.warns(UserWarning, match="triton_gpu"):
        got = short_conv(weight, x, segment_ids, implementation=("triton_gpu", "reference"))
    np.testing.assert_array_equal(_bits(got), _bits(short_conv_reference(weight, x, segment_ids)))


def _requires_gpu():
    if jax.default_backend() != "gpu":
        pytest.skip("requires the JAX GPU backend")


def _trace_kernel_off_gpu(monkeypatch):
    """Let the dispatch pick the kernel on CPU, so tests can trace the wrapper around its call."""
    monkeypatch.setattr(api, "triton_short_conv_available", lambda: True)


def test_float32_matches_the_reference_on_gpu():
    """Float32: forward and ``dx`` bitwise, ``dw`` within the error of an fp32 sum in any order."""
    _requires_gpu()
    batch, seq_len = 2, 384
    weight, x, segment_ids, cotangent = _inputs(batch, seq_len, 256, 4, seed=17, dtype=jnp.float32, packed=True)

    def kernel_fn(w, xx):
        return short_conv(w, xx, segment_ids, implementation="triton_gpu")

    def reference_fn(w, xx):
        return short_conv_reference(w, xx, segment_ids)

    got, kernel_vjp = jax.vjp(jax.jit(kernel_fn), weight, x)
    got_dw, got_dx = kernel_vjp(cotangent)
    want, reference_vjp = jax.vjp(jax.jit(reference_fn), weight, x)
    _, want_dx = reference_vjp(cotangent)

    np.testing.assert_array_equal(_bits(got), _bits(want), err_msg="forward is not bit-identical")
    np.testing.assert_array_equal(_bits(got_dx), _bits(want_dx), err_msg="dx is not bit-identical")
    oracle = _dw_oracle(x, segment_ids, cotangent, 4)
    terms = batch * seq_len
    # Each fp32 product rounds once (2^-24), then the sum adds at most gamma_n of the absolute sum.
    gamma = (terms + 1) * 2.0**-24 / (1 - (terms + 1) * 2.0**-24)
    bound = gamma * _dw_oracle(jnp.abs(x), segment_ids, jnp.abs(cotangent), 4)
    error = np.abs(np.asarray(jax.device_get(got_dw), np.float64) - oracle)
    assert np.all(error <= bound), f"dw error is up to {np.max(error / bound):.2f}x its bound"


def test_default_implementation_on_gpu_is_the_kernel():
    _requires_gpu()
    weight, x, segment_ids, _ = _inputs(2, 256, 256, 4, seed=2, dtype=jnp.bfloat16, packed=True)
    programs = _triton_programs(lambda w, xx: short_conv(w, xx, segment_ids), weight, x)
    assert len(programs) == 1


def test_default_implementation_on_cpu_is_the_reference():
    if jax.default_backend() == "gpu":
        pytest.skip("this asserts the non-GPU default")
    weight, x, segment_ids, _ = _inputs(2, 32, 8, 4, seed=2, dtype=jnp.bfloat16, packed=True)
    got = short_conv(weight, x, segment_ids)
    np.testing.assert_array_equal(_bits(got), _bits(short_conv_reference(weight, x, segment_ids)))


def test_kernel_call_is_wrapped_in_a_shard_map_under_a_mesh(monkeypatch):
    """Every kernel call sits inside an explicit shard_map on a real mesh.

    Checked on the traced jaxpr rather than by inspection, so a refactor that drops the manual
    region fails here. The mesh is abstract and nothing is executed.
    """
    pytest.importorskip("jax_triton")
    _trace_kernel_off_gpu(monkeypatch)
    mesh = jax.sharding.AbstractMesh(
        axis_sizes=(1, 2, 1, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 4,
    )
    weight, x, segment_ids, _ = _inputs(2, 128, 64, 4, seed=3, dtype=jnp.bfloat16, packed=True)

    def fn(w, xx, seg):
        return short_conv(w, xx, seg, implementation="triton_gpu")

    with jax.sharding.use_abstract_mesh(mesh):
        text = str(jax.make_jaxpr(fn)(weight, x, segment_ids))
    assert "shard_map" in text, "kernel is not inside an explicit shard_map"
    # ...and the manual region must not contain a collective: the op is shard-local.
    for banned in ("all_gather", "all_reduce", "psum", "all_to_all", "reduce_scatter"):
        assert banned not in text, f"short_conv lowered through an unexpected {banned}"


@pytest.mark.parametrize(
    ("model_size", "should_reject"),
    [(1, False), (2, True)],
    ids=["size-1 model axis is a no-op", "size-2 model axis really shards"],
)
def test_channel_axis_gate_consults_the_mesh_not_just_the_spec(model_size, should_reject, monkeypatch):
    """A spec entry naming a size-1 mesh axis shards nothing, and must not be rejected.

    The EP64 hero mesh is (replica_dcn=1, data=1, expert=64, model=1) and the attention projections
    are `P(_FSDP_AXES, "model")`, so every k/v activation reaching SConv carries "model" on its
    channel axis while being unsharded there. A genuinely sharded channel axis must still raise,
    because silently resharding it would hide an all-gather inside the kernel wrapper.
    """
    if not should_reject:
        pytest.importorskip("jax_triton")
    _trace_kernel_off_gpu(monkeypatch)
    mesh = jax.sharding.AbstractMesh(
        axis_sizes=(1, 2 // model_size, 2, model_size),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 4,
    )
    weight, x, segment_ids, _ = _inputs(4, 128, 128, 4, seed=11, dtype=jnp.bfloat16, packed=True)

    def fn(w, xx, seg):
        # Reproduce the hero's k_flat sharding: batch over the FSDP pair, channel named "model".
        xx = jax.sharding.reshard(xx, jax.sharding.PartitionSpec(("data", "expert"), None, "model"))
        seg = jax.sharding.reshard(seg, jax.sharding.PartitionSpec(("data", "expert"), None))
        return short_conv(w, xx, seg, implementation="triton_gpu")

    with jax.sharding.use_abstract_mesh(mesh):
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
    (fp32) while the kernel outputs ``x.dtype``, so accepting mixed inputs would make
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

from levanter.kernels.triton.short_conv import short_conv, short_conv_reference

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
    # the mesh.
    mesh = Mesh(
        DEVICES.reshape(BATCH, context, 8 // (BATCH * context)),
        ("data", "context", "spare"),
        axis_types=(AxisType.Explicit,) * 3,
    )
    def conv(w, xx, seg):
        return short_conv(w, xx, seg, implementation="reference", batch_axes=("data",))

    def loss(w, xx, seg):
        return jnp.sum(conv(w, xx, seg) * cotangent)

    def reference_loss(w, xx, seg):
        return jnp.sum(short_conv_reference(w, xx, seg) * cotangent)

    for dtype in (jnp.bfloat16, jnp.float32):
        weight, x, cotangent, segment_ids = inputs(width, packed, dtype)
        with jax.set_mesh(mesh):
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

from levanter.kernels.triton.short_conv import api, short_conv

# The gate runs before the kernel call, so it can run on CPU with the dispatch forced to the kernel.
api.triton_short_conv_available = lambda: True
# A concrete array on an Auto-axis mesh shows its placement only on `array.sharding`; the
# channel gate must still see it rather than reshard the axis away.
mesh = Mesh(np.asarray(jax.devices()).reshape(2, 4), ("data", "model"), axis_types=(AxisType.Auto,) * 2)
weight = jnp.ones((4, 128), jnp.bfloat16)
x = jax.device_put(jnp.ones((2, 128, 128), jnp.bfloat16), NamedSharding(mesh, P("data", None, "model")))
with jax.set_mesh(mesh):
    try:
        short_conv(weight, x, implementation="triton_gpu")
    except ValueError as error:
        assert "unsharded channel axis" in str(error), error
    else:
        raise AssertionError("a concrete channel-sharded array slipped past the gate")
"""


def test_context_parallel_halo_matches_the_unsharded_reference():
    """Check packed/unpacked values and gradients across context shards and halo sizes."""
    run_on_cpu_devices(_HALO_SCRIPT, device_count=8)


def test_channel_axis_gate_reads_concrete_shardings_on_an_auto_mesh():
    run_on_cpu_devices(_AUTO_MESH_GATE_SCRIPT, device_count=8)


def test_context_parallel_kernel_matches_reference_on_gpu():
    if jax.default_backend() != "gpu" or jax.device_count() < 4:
        pytest.skip("requires four GPUs for the context-parallel kernel")
    mesh = Mesh(np.asarray(jax.devices()[:4]), ("context",), axis_types=(AxisType.Explicit,))
    weight, x, segment_ids, cotangent = _inputs(1, 512, 128, 4, seed=17, dtype=jnp.float32, packed=True)

    def reference_loss(w, xx):
        return jnp.sum(short_conv_reference(w, xx, segment_ids) * cotangent)

    want = short_conv_reference(weight, x, segment_ids)
    want_dw, want_dx = jax.grad(reference_loss, argnums=(0, 1))(weight, x)
    with jax.set_mesh(mesh):
        sharded_x = jax.device_put(x, NamedSharding(mesh, P(None, "context", None)))
        sharded_seg = jax.device_put(segment_ids, NamedSharding(mesh, P(None, "context")))

        def forward(w, xx):
            # Each shard's 128 rows plus the 3-row halo are padded to 256.
            return short_conv(w, xx, sharded_seg, implementation="triton_gpu")

        def loss(w, xx):
            return jnp.sum(forward(w, xx) * cotangent)

        got = forward(weight, sharded_x)
        got_dw, got_dx = jax.grad(loss, argnums=(0, 1))(weight, sharded_x)
    for actual, expected in ((got, want), (got_dw, want_dw), (got_dx, want_dx)):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)
