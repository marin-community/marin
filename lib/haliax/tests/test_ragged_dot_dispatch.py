# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import importlib
from typing import Literal

import jax
import jax.numpy as jnp
import pytest

from haliax.nn import ragged_dot

ragged_dot_module = importlib.import_module("haliax.nn.ragged_dot")


def _inputs():
    contraction_dim = 32
    output_dim = 17
    lhs = jnp.arange(3 * contraction_dim, dtype=jnp.float32).reshape(3, contraction_dim) / 100
    rhs = jnp.arange(2 * contraction_dim * output_dim, dtype=jnp.float32).reshape(2, contraction_dim, output_dim) / 100
    group_sizes = jnp.array([2, 1], dtype=jnp.int32)
    return lhs, rhs, group_sizes


def _accelerator_implementation() -> Literal["megablox", "triton"]:
    backend = jax.default_backend()
    if backend == "gpu" and ragged_dot_module._has_pallas_triton:
        return "triton"
    if backend == "tpu" and ragged_dot_module._gmm_megablox is not None:
        return "megablox"
    pytest.skip("requires a supported accelerator ragged-dot implementation")


def test_accelerator_implementation_value_and_gradients_match_xla():
    lhs, rhs, group_sizes = _inputs()
    implementation = _accelerator_implementation()

    def loss(lhs, rhs, implementation):
        return jnp.sum(ragged_dot(lhs, rhs, group_sizes, implementation=implementation) ** 2)

    # At default precision, ROCm may run f32 dots in reduced-precision xf32 on MI300X, and its choice can
    # differ between the two paths and between processes, giving ~7e-4 relative error on either side.
    with jax.default_matmul_precision("highest"):
        actual_value, actual_gradients = jax.value_and_grad(loss, argnums=(0, 1))(lhs, rhs, implementation)
        expected_value, expected_gradients = jax.value_and_grad(loss, argnums=(0, 1))(lhs, rhs, "xla")

    assert jnp.allclose(actual_value, expected_value, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(actual_gradients[0], expected_gradients[0], rtol=1e-5, atol=1e-5)
    assert jnp.allclose(actual_gradients[1], expected_gradients[1], rtol=1e-5, atol=1e-5)


def test_triton_kernel_traces_with_jax_0_9_pallas_memory_api_on_cpu_interpreter():
    if not ragged_dot_module._has_pallas_triton:
        pytest.skip("Pallas Triton backend is not available")

    lhs, grouped_rhs, _ = _inputs()
    lhs = lhs[:2]
    rhs = grouped_rhs[0]
    lo = jnp.array(0, dtype=jnp.int32)
    hi = jnp.array(lhs.shape[0], dtype=jnp.int32)
    pallas_call = ragged_dot_module.pl.pallas_call(
        lambda a, b, lo, hi, out: ragged_dot_module._triton_ragged_dot_kernel(
            a, b, lo, hi, out, block_m=lhs.shape[0], block_k=lhs.shape[1], n=rhs.shape[1]
        ),
        out_shape=jax.ShapeDtypeStruct((lhs.shape[0], rhs.shape[1]), lhs.dtype),
        in_specs=[ragged_dot_module.pl.no_block_spec] * 4,
        out_specs=ragged_dot_module.pl.no_block_spec,
        grid=(1, 1),
        interpret=True,
    )

    assert jnp.allclose(pallas_call(lhs, rhs, lo, hi), lhs @ rhs, rtol=1e-5, atol=1e-5)


@dataclasses.dataclass(frozen=True)
class _FakeGpu:
    device_kind: str
    compute_capability: str


@pytest.mark.parametrize(
    "device, expected",
    [
        (_FakeGpu("AMD Instinct MI300X", "gfx942"), "xla"),
        (_FakeGpu("AMD Instinct MI325X", "gfx942"), "xla"),
        (_FakeGpu("AMD Instinct MI350X", "gfx950"), "triton"),
        (_FakeGpu("NVIDIA H100 80GB HBM3", "9.0"), "triton"),
        (_FakeGpu("NVIDIA B200", "10.0"), "triton"),
    ],
    ids=["mi300x", "mi325x", "mi350x", "h100", "b200"],
)
def test_gpu_auto_prefers_xla_only_on_gfx942(monkeypatch, device, expected):
    lhs, rhs, group_sizes = _inputs()
    monkeypatch.delenv("RAGGED_DOT_IMPL", raising=False)
    monkeypatch.setattr(ragged_dot_module.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(ragged_dot_module.jax, "devices", lambda backend=None: [device])
    monkeypatch.setattr(ragged_dot_module, "_has_pallas_triton", True)

    def constant_result(value):
        return lambda lhs, rhs, group_sizes: jnp.full((lhs.shape[0], rhs.shape[2]), value, dtype=lhs.dtype)

    monkeypatch.setattr(ragged_dot_module, "_ragged_dot_triton_impl", constant_result(1.0))
    monkeypatch.setattr(ragged_dot_module, "_ragged_dot_xla_impl", constant_result(2.0))

    auto_out = ragged_dot(lhs, rhs, group_sizes, implementation="auto")

    assert float(auto_out[0, 0]) == {"triton": 1.0, "xla": 2.0}[expected]


def test_triton_default_block_sizes_use_blackwell_n_tile(monkeypatch):
    monkeypatch.setattr(ragged_dot_module, "_is_blackwell_gpu_backend", lambda: True)

    assert ragged_dot_module._triton_default_block_sizes(32768, 5120, 5120) == (128, 256, 32)


def test_triton_default_block_sizes_keep_non_blackwell_n_tile(monkeypatch):
    monkeypatch.setattr(ragged_dot_module, "_is_blackwell_gpu_backend", lambda: False)

    assert ragged_dot_module._triton_default_block_sizes(32768, 5120, 5120) == (128, 128, 32)


def test_triton_custom_vjp_routes_backward_through_triton_layouts(monkeypatch):
    lhs, rhs, group_sizes = _inputs()
    calls = []

    def fake_triton_pallas_call(
        lhs,
        rhs,
        group_sizes,
        ragged_dot_dimension_numbers=ragged_dot_module._DEFAULT_DIM_NUMS,
    ):
        calls.append(ragged_dot_dimension_numbers)
        return jax.lax.ragged_dot_general(
            lhs=lhs,
            rhs=rhs,
            group_sizes=group_sizes,
            ragged_dot_dimension_numbers=ragged_dot_dimension_numbers,
        )

    monkeypatch.setattr(ragged_dot_module, "_has_pallas_triton", True)
    monkeypatch.setattr(ragged_dot_module, "_triton_pallas_call", fake_triton_pallas_call)

    def triton_loss(lhs, rhs):
        return jnp.sum(ragged_dot_module._ragged_dot_triton_impl(lhs, rhs, group_sizes))

    def xla_loss(lhs, rhs):
        return jnp.sum(ragged_dot(lhs, rhs, group_sizes, implementation="xla"))

    # At default precision, ROCm may run f32 dots in reduced-precision xf32 on MI300X, and its choice can
    # differ between the two paths and between processes, giving ~7e-4 relative error on either side.
    with jax.default_matmul_precision("highest"):
        triton_value, triton_grads = jax.value_and_grad(triton_loss, argnums=(0, 1))(lhs, rhs)
        xla_value, xla_grads = jax.value_and_grad(xla_loss, argnums=(0, 1))(lhs, rhs)

    assert jnp.allclose(triton_value, xla_value, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(triton_grads[0], xla_grads[0], rtol=1e-5, atol=1e-5)
    assert jnp.allclose(triton_grads[1], xla_grads[1], rtol=1e-5, atol=1e-5)
    assert calls == [
        ragged_dot_module._DEFAULT_DIM_NUMS,
        ragged_dot_module._DLHS_DIM_NUMS,
        ragged_dot_module._DRHS_DIM_NUMS,
    ]


@pytest.mark.parametrize(
    "device, expected",
    [
        (_FakeGpu("AMD Instinct MI350X", "gfx950"), "tile_map"),
        (_FakeGpu("AMD Instinct MI300X", "gfx942"), "tile_map"),
        (_FakeGpu("NVIDIA H100 80GB HBM3", "9.0"), "group_grid"),
        (_FakeGpu("NVIDIA B200", "10.0"), "group_grid"),
    ],
    ids=["mi350x", "mi300x", "h100", "b200"],
)
def test_triton_kernel_family_uses_tile_map_only_on_amd_instinct(monkeypatch, device, expected):
    monkeypatch.setattr(ragged_dot_module.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(ragged_dot_module.jax, "devices", lambda backend=None: [device])

    assert ragged_dot_module._triton_kernel_family() == ragged_dot_module.TritonKernelFamily(expected)


# Group sizes over 300 rows and 4 groups, chosen so that row tiles straddle group boundaries.
_GROUP_SIZE_CASES = {
    "unaligned": [37, 101, 0, 162],
    "empty_first_and_last": [0, 150, 150, 0],
    "single_group_all_rows": [0, 0, 300, 0],
    "rows_past_last_group": [64, 0, 100, 20],
    "all_empty": [0, 0, 0, 0],
}
# Odd sizes: rows, contraction and columns are not multiples of the tile-map blocks below.
_ROWS, _K, _N, _GROUPS = 300, 80, 200, 4
_LAYOUTS = [layout.value for layout in ragged_dot_module.RaggedLayout]


def _require_triton_gpu():
    if jax.default_backend() != "gpu" or not ragged_dot_module._has_pallas_triton:
        pytest.skip("requires the Pallas Triton GPU backend")


def _require_kernel_family(family: str):
    """Both families run on a GPU; the tile-map kernels also run on CPU through the Pallas interpreter."""
    if family == "tile_map" and jax.default_backend() == "cpu" and ragged_dot_module._has_pallas_triton:
        return
    _require_triton_gpu()


def _layout_operands(layout: str, dtype, group_sizes: list[int]):
    key_lhs, key_rhs = jax.random.split(jax.random.key(0))
    if layout == "fwd":
        lhs_shape, rhs_shape = (_ROWS, _K), (_GROUPS, _K, _N)
    elif layout == "dlhs":
        lhs_shape, rhs_shape = (_ROWS, _N), (_GROUPS, _K, _N)
    else:
        lhs_shape, rhs_shape = (_ROWS, _K), (_ROWS, _N)
    lhs = jax.random.normal(key_lhs, lhs_shape, jnp.float32).astype(dtype)
    rhs = jax.random.normal(key_rhs, rhs_shape, jnp.float32).astype(dtype)
    return lhs, rhs, jnp.array(group_sizes, dtype=jnp.int32)


def _to_cpu_f32(*arrays):
    cpu = jax.devices("cpu")[0]
    return tuple(jax.device_put(x.astype(jnp.float32) if x.dtype != jnp.int32 else x, cpu) for x in arrays)


def _cpu_f32_reference(lhs, rhs, group_sizes, dim_nums):
    """``jax.lax.ragged_dot_general`` in float32 on the CPU backend, as an independent oracle."""
    lhs, rhs, group_sizes = _to_cpu_f32(lhs, rhs, group_sizes)
    return jax.lax.ragged_dot_general(
        lhs, rhs, group_sizes, ragged_dot_dimension_numbers=dim_nums, precision=jax.lax.Precision.HIGHEST
    )


def _assert_close_to_reference(actual, expected, rtol):
    """Max abs error, relative to the reference's largest magnitude, stays within ``rtol``."""
    (actual,) = _to_cpu_f32(actual)
    error = jnp.abs(actual - expected)
    scale = max(float(jnp.max(jnp.abs(expected), initial=0.0)), 1.0)
    assert bool(jnp.all(jnp.isfinite(actual)))
    max_error = float(jnp.max(error, initial=0.0))
    assert max_error <= rtol * scale, f"max abs error {max_error:.3e}, scale {scale:.3e}"


def _run_kernel_family(family: str, lhs, rhs, sizes, layout: str):
    layout = ragged_dot_module.RaggedLayout(layout)
    if family == "group_grid":
        return jax.jit(lambda a, b, g: ragged_dot_module._group_grid_pallas_call(a, b, g, layout))(lhs, rhs, sizes)
    # Small blocks put several row and column tiles in each group and leave partial edge tiles. With
    # num_xcds=8 the launch grid is padded and renumbered, so padding programs must not write, and
    # group_m=3 leaves a partial band of tile rows.
    config = ragged_dot_module.TritonBlockConfig(
        block_m=64, block_n=64, block_k=32, num_warps=4, num_stages=2, num_xcds=8, group_m=3
    )
    return jax.jit(lambda a, b, g: ragged_dot_module._tile_map_pallas_call(a, b, g, layout, config))(lhs, rhs, sizes)


@pytest.mark.parametrize("dtype, rtol", [(jnp.float32, 1e-4), (jnp.bfloat16, 1e-2)], ids=["f32", "bf16"])
@pytest.mark.parametrize("group_sizes", list(_GROUP_SIZE_CASES.values()), ids=list(_GROUP_SIZE_CASES))
@pytest.mark.parametrize("layout", _LAYOUTS)
@pytest.mark.parametrize("family", ["group_grid", "tile_map"])
def test_triton_kernel_families_match_ragged_dot_general(family, layout, group_sizes, dtype, rtol):
    _require_kernel_family(family)
    reduced_f32 = (
        ragged_dot_module._GpuFamily.NVIDIA,
        ragged_dot_module._GpuFamily.NVIDIA_BLACKWELL,
        ragged_dot_module._GpuFamily.AMD_INSTINCT_GFX942,
    )
    if dtype == jnp.float32 and ragged_dot_module._gpu_family() in reduced_f32:
        # Triton runs f32 dots in TF32 on NVIDIA and xf32 on gfx942, 3e-4 to 1e-3 off here; XLA's own ragged
        # dot is off by as much. On gfx942 the group-grid f32 tiles also exceed the 64 KiB LDS.
        pytest.skip("f32 Triton dots use TF32 on NVIDIA and xf32 on MI300X")
    lhs, rhs, sizes = _layout_operands(layout, dtype, group_sizes)

    actual = _run_kernel_family(family, lhs, rhs, sizes, layout)
    expected = _cpu_f32_reference(
        lhs, rhs, sizes, ragged_dot_module._LAYOUT_DIM_NUMS[ragged_dot_module.RaggedLayout(layout)]
    )

    if family == "group_grid" and layout != "drhs":
        # The group-grid kernels leave rows past sum(group_sizes) unwritten; the tile-map kernels zero them.
        actual, expected = actual[: sum(group_sizes)], expected[: sum(group_sizes)]
    _assert_close_to_reference(actual, expected, rtol)


@pytest.mark.parametrize("group_sizes", [[512, 512, 512, 512], [5, 1000, 0, 1043]], ids=["balanced", "skewed"])
def test_triton_ragged_dot_value_and_gradients_match_reference_with_device_blocks(group_sizes):
    _require_triton_gpu()
    rows, k, n = sum(group_sizes), 256, 384
    key_lhs, key_rhs, key_dout = jax.random.split(jax.random.key(1), 3)
    lhs = jax.random.normal(key_lhs, (rows, k), jnp.float32).astype(jnp.bfloat16)
    rhs = jax.random.normal(key_rhs, (len(group_sizes), k, n), jnp.float32).astype(jnp.bfloat16)
    dout = jax.random.normal(key_dout, (rows, n), jnp.float32).astype(jnp.bfloat16)
    sizes = jnp.array(group_sizes, dtype=jnp.int32)

    out, vjp = jax.vjp(lambda a, b: ragged_dot(a, b, sizes, implementation="triton"), lhs, rhs)
    dlhs, drhs = vjp(dout)

    def reference(a, b):
        return _cpu_f32_reference(a, b, sizes, ragged_dot_module._DEFAULT_DIM_NUMS)

    expected_out, reference_vjp = jax.vjp(reference, *_to_cpu_f32(lhs, rhs))
    expected_dlhs, expected_drhs = reference_vjp(*_to_cpu_f32(dout))
    _assert_close_to_reference(out, expected_out, 1e-2)
    _assert_close_to_reference(dlhs, expected_dlhs, 1e-2)
    _assert_close_to_reference(drhs, expected_drhs, 1e-2)
