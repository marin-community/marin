# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import importlib
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
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


def test_gpu_auto_selects_triton_instead_of_xla(monkeypatch):
    lhs, rhs, group_sizes = _inputs()
    expected = jnp.full((lhs.shape[0], rhs.shape[2]), 17.0, dtype=lhs.dtype)
    monkeypatch.setattr(ragged_dot_module.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(ragged_dot_module, "_has_pallas_triton", True)

    def triton_result(lhs, rhs, group_sizes):
        return jnp.full((lhs.shape[0], rhs.shape[2]), expected[0, 0], dtype=lhs.dtype)

    def unexpected_xla(*args):
        raise AssertionError("GPU auto dispatch selected XLA instead of Triton")

    monkeypatch.setattr(ragged_dot_module, "_ragged_dot_triton_impl", triton_result)
    monkeypatch.setattr(ragged_dot_module, "_ragged_dot_xla_impl", unexpected_xla)

    auto_out = ragged_dot(lhs, rhs, group_sizes, implementation="auto")

    assert jnp.array_equal(auto_out, expected)


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


@dataclasses.dataclass(frozen=True)
class _FakeGpu:
    device_kind: str
    compute_capability: str | None = None


@pytest.mark.parametrize(
    "device, has_jax_triton, expected",
    [
        (_FakeGpu("AMD Instinct MI350X"), True, "tile_map"),
        (_FakeGpu("AMD Instinct MI300X"), True, "tile_map"),
        (_FakeGpu("AMD Instinct MI350X"), False, "group_grid"),
        (_FakeGpu("NVIDIA H100 80GB HBM3", "9.0"), True, "group_grid"),
        (_FakeGpu("NVIDIA B200", "10.0"), True, "group_grid"),
    ],
    ids=["mi350x", "mi300x", "mi350x_without_jax_triton", "h100", "b200"],
)
def test_triton_kernel_family_uses_tile_map_only_on_amd_instinct_with_jax_triton(
    monkeypatch, device, has_jax_triton, expected
):
    monkeypatch.setattr(ragged_dot_module.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(ragged_dot_module.jax, "devices", lambda backend=None: [device])
    # Stands in for the jax_triton module, which family selection only needs to have imported.
    monkeypatch.setattr(ragged_dot_module, "jt", object() if has_jax_triton else None)

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
_ROWS, _K, _N, _GROUPS = 300, 96, 200, 4
_LAYOUTS = [layout.value for layout in ragged_dot_module.RaggedLayout]


def _require_triton_gpu():
    if jax.default_backend() != "gpu" or not ragged_dot_module._has_pallas_triton:
        pytest.skip("requires the Pallas Triton GPU backend")


def _require_kernel_family(family: str):
    """Both families run on GPU only: the group-grid kernels need Pallas Triton, the tile-map kernels jax-triton."""
    if jax.default_backend() != "gpu":
        pytest.skip("the Triton kernels run on GPU only")
    if family == "group_grid" and not ragged_dot_module._has_pallas_triton:
        pytest.skip("the group-grid kernels need Pallas Triton")
    if family == "tile_map" and ragged_dot_module.jt is None:
        pytest.skip("the tile-map kernels need jax-triton")


@pytest.mark.parametrize("block_m", [16, 64, 512])
@pytest.mark.parametrize("group_sizes", list(_GROUP_SIZE_CASES.values()), ids=list(_GROUP_SIZE_CASES))
def test_row_tile_metadata_assigns_each_row_to_one_tile_of_its_group(group_sizes, block_m):
    tiles = ragged_dot_module._row_tile_metadata(jnp.array(group_sizes, dtype=jnp.int32), _ROWS, block_m)

    # Rows past sum(group_sizes) belong to the pseudo-group len(group_sizes), which the kernel zero-fills.
    row_group = np.repeat(np.arange(len(group_sizes) + 1), [*group_sizes, _ROWS - sum(group_sizes)])
    tiles_per_row = np.zeros(_ROWS, dtype=np.int32)
    for group, row_tile, lo, hi in np.asarray(tiles).T:
        if lo == hi:
            continue
        assert row_tile * block_m <= lo < hi <= (row_tile + 1) * block_m
        np.testing.assert_array_equal(row_group[lo:hi], group)
        tiles_per_row[lo:hi] += 1
    np.testing.assert_array_equal(tiles_per_row, 1)


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


def _triton_f32_dots_use_tf32() -> bool:
    """Triton's dot defaults f32 inputs to TF32 on NVIDIA GPUs and on gfx942 (MI300X, MI325X), and to IEEE on gfx950."""
    nvidia = (ragged_dot_module._GpuFamily.NVIDIA, ragged_dot_module._GpuFamily.NVIDIA_BLACKWELL)
    if ragged_dot_module._gpu_family() in nvidia:
        return True
    return any(name in jax.devices()[0].device_kind for name in ("MI300", "MI325"))


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
    return jax.jit(lambda a, b, g: ragged_dot_module._tile_map_triton_call(a, b, g, layout, config))(lhs, rhs, sizes)


@pytest.mark.parametrize("dtype, rtol", [(jnp.float32, 1e-4), (jnp.bfloat16, 1e-2)], ids=["f32", "bf16"])
@pytest.mark.parametrize("group_sizes", list(_GROUP_SIZE_CASES.values()), ids=list(_GROUP_SIZE_CASES))
@pytest.mark.parametrize("layout", _LAYOUTS)
@pytest.mark.parametrize("family", ["group_grid", "tile_map"])
def test_triton_kernel_families_match_ragged_dot_general(family, layout, group_sizes, dtype, rtol):
    _require_kernel_family(family)
    if dtype == jnp.float32 and _triton_f32_dots_use_tf32():
        # About 3e-4 off here on NVIDIA and 7e-4 on MI300X; XLA's own ragged dot on NVIDIA is off by as much.
        pytest.skip("f32 Triton dots use TF32 on NVIDIA and gfx942")
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
