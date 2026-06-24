# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import levanter.grug.attention._fa4_cute as fa4_cute
import levanter.grug.attention._fa4_cute_backend as fa4_cute_backend
import levanter.grug.attention._fa4_cute_config as fa4_cute_config
import levanter.grug.attention._fa4_cute_kernels as fa4_cute_kernels
from levanter.grug.attention import (
    AttentionMask,
    gpu_fa4_cute_attention,
    reference_attention,
)


def _make_qkv(*, batch: int = 2, q_len: int = 6, k_len: int = 6, q_heads: int = 4, kv_heads: int = 2):
    key = jax.random.PRNGKey(0)
    q_key, k_key, v_key = jax.random.split(key, 3)
    q = jax.random.normal(q_key, (batch, q_len, q_heads, 8), dtype=jnp.float32)
    k = jax.random.normal(k_key, (batch, k_len, kv_heads, 8), dtype=jnp.float32)
    v = jax.random.normal(v_key, (batch, k_len, kv_heads, 8), dtype=jnp.float32)
    return q, k, v


def test_gpu_fa4_cute_supports_hopper_kernel_config(monkeypatch):
    monkeypatch.setattr(fa4_cute, "_gpu_compute_arch", lambda: 90)

    config = fa4_cute._segmented_kernel_config(128)

    assert config.forward_tile == (128, 64)
    assert config.backward_tile == (64, 64)
    assert config.num_threads == 128
    assert config.backward_arch == 90
    assert config.sm90_backward is not None
    assert config.sm90_backward.tile == (64, 128)
    assert config.sm90_backward.num_threads == 384
    assert config.sm90_backward.num_stages_q == 2
    assert config.sm90_backward.num_stages_do == 2
    assert config.sm90_backward.num_stages_pds == 2
    assert config.sm90_backward.sdp_swap_ab is True
    assert config.sm90_backward.atom_layout_m_sdp == 1
    assert config.sm90_backward.atom_layout_n_dkv == 2
    assert config.sm90_backward.atom_layout_m_dq == 1


def test_sm90_native_backward_config_matches_upstream_sparse_q_rule():
    dense_noncausal = fa4_cute_config.sm90_flash4_cute_backward_config(
        128,
        schedule=fa4_cute_config.Sm90BackwardSchedule.DENSE,
    )
    sparse_noncausal = fa4_cute_config.sm90_flash4_cute_backward_config(
        128,
        schedule=fa4_cute_config.Sm90BackwardSchedule.DENSE,
        sparse_block_size_q=128,
    )

    assert dense_noncausal.tile == (80, 128)
    assert dense_noncausal.dq_swap_ab is True
    assert sparse_noncausal.tile == (64, 128)
    assert sparse_noncausal.dq_swap_ab is False


def test_packed_segment_backward_block_sparse_indices_are_q_direction():
    segment_ids = jnp.array([[0, 0, 0, 0, 1, 1, 1, -1]], dtype=jnp.int32)
    lower_bounds, valid = fa4_cute._packed_segment_causal_lower_bounds(
        segment_ids,
        batch_size=1,
        seq_len=8,
        sliding_window=None,
    )

    mask_block_cnt, mask_block_idx = fa4_cute_backend._packed_segment_backward_block_sparse_indices(
        lower_bounds,
        valid,
        tile_m=2,
        tile_n=4,
    )

    np.testing.assert_array_equal(mask_block_cnt, jnp.array([[[2, 2]]], dtype=jnp.int32))
    np.testing.assert_array_equal(
        mask_block_idx,
        jnp.array([[[[0, 1, 0, 0], [2, 3, 0, 0]]]], dtype=jnp.int32),
    )


def test_packed_segment_backward_block_sparse_indices_split_full_blocks():
    segment_ids = jnp.zeros((1, 8), dtype=jnp.int32)
    lower_bounds, valid = fa4_cute._packed_segment_causal_lower_bounds(
        segment_ids,
        batch_size=1,
        seq_len=8,
        sliding_window=None,
    )

    sparse_metadata = fa4_cute_backend._packed_segment_backward_block_sparse_indices_with_full(
        lower_bounds,
        valid,
        tile_m=2,
        tile_n=2,
    )

    np.testing.assert_array_equal(sparse_metadata.partial_block_cnt, jnp.array([[[1, 1, 1, 1]]], dtype=jnp.int32))
    np.testing.assert_array_equal(
        sparse_metadata.partial_block_idx,
        jnp.array([[[[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0]]]], dtype=jnp.int32),
    )
    np.testing.assert_array_equal(sparse_metadata.full_block_cnt, jnp.array([[[3, 2, 1, 0]]], dtype=jnp.int32))
    np.testing.assert_array_equal(
        sparse_metadata.full_block_idx,
        jnp.array([[[[1, 2, 3, 0], [2, 3, 0, 0], [3, 0, 0, 0], [0, 0, 0, 0]]]], dtype=jnp.int32),
    )


def test_sm90_native_backward_boundary_passes_sparse_metadata(monkeypatch):
    captured: dict[str, object] = {"calls": []}

    class FakeCjax:
        @staticmethod
        def TensorSpec(*args, **kwargs):
            del args, kwargs
            return object()

        @staticmethod
        def cutlass_call(*args, **kwargs):
            del args
            call_capture = {
                "input_spec_len": len(kwargs["input_spec"]),
                "output_shape_dtype": kwargs["output_shape_dtype"],
                "compile_options": kwargs.get("compile_options"),
                "input_output_aliases": kwargs.get("input_output_aliases"),
            }
            captured["calls"].append(call_capture)

            def call(*call_args):
                call_capture["call_arg_shapes"] = [arg.shape for arg in call_args]
                output_shapes = kwargs["output_shape_dtype"]
                return tuple(jnp.zeros(shape.shape, shape.dtype) for shape in output_shapes)

            return call

    modules = SimpleNamespace(cjax=FakeCjax)

    def fake_launcher(*args, **kwargs):
        del args
        captured["launcher_config"] = kwargs["config"]
        return object()

    def fake_preprocess_launcher(*args, **kwargs):
        del args
        captured["preprocess_tile_m"] = kwargs["tile_m"]
        return object()

    def fake_postprocess_launcher(*args, **kwargs):
        del args
        captured.setdefault("postprocess_configs", []).append(
            (
                kwargs["arch"],
                kwargs["tile_m"],
                kwargs["atom_layout_m"],
                kwargs.get("cluster_size"),
                kwargs.get("use_2cta_instrs"),
            )
        )
        return object()

    monkeypatch.setattr(fa4_cute_backend, "_import_cutlass_cute", lambda: modules)
    monkeypatch.setattr(fa4_cute_backend, "segmented_flash_attention_backward_sm90_launcher", fake_launcher)
    monkeypatch.setattr(
        fa4_cute_backend,
        "segmented_flash_attention_backward_sm90_preprocess_launcher",
        fake_preprocess_launcher,
    )
    monkeypatch.setattr(fa4_cute_backend, "flash_attention_backward_postprocess_launcher", fake_postprocess_launcher)

    q = jnp.ones((1, 8, 4, 128), dtype=jnp.bfloat16)
    k = jnp.ones((1, 8, 1, 128), dtype=jnp.bfloat16)
    v = jnp.ones((1, 8, 1, 128), dtype=jnp.bfloat16)
    out = jnp.ones_like(q)
    dout = jnp.ones_like(q)
    lse = jnp.ones((1, 4, 8), dtype=jnp.float32)
    lower_bounds = jnp.zeros((1, 8), dtype=jnp.int32)
    valid = jnp.ones((1, 8), dtype=jnp.bool_)
    config = fa4_cute_config.flash4_cute_kernel_config(128, arch=90)
    assert config.sm90_backward is not None
    sparse_metadata = fa4_cute_backend._packed_segment_backward_block_sparse_indices_with_full(
        lower_bounds,
        valid,
        tile_m=config.sm90_backward.tile[0],
        tile_n=config.sm90_backward.tile[1],
    )

    dq, dk, dv = fa4_cute_backend.segmented_flash_attention_backward_sm90_native(
        q,
        k,
        v,
        out,
        dout,
        lse,
        lower_bounds,
        valid,
        sparse_metadata.partial_block_cnt,
        sparse_metadata.partial_block_idx,
        sparse_metadata.full_block_cnt,
        sparse_metadata.full_block_idx,
        softmax_scale=1.0,
        kernel_config=config,
    )

    calls = captured["calls"]
    assert len(calls) == 5
    preprocess_call, backward_call, dq_postprocess_call, dk_postprocess_call, dv_postprocess_call = calls
    assert captured["launcher_config"] == config.sm90_backward
    assert captured["preprocess_tile_m"] == config.sm90_backward.tile[0]
    assert captured["postprocess_configs"] == [
        (90, 64, 1, 1, False),
        (90, 64, 1, 1, None),
        (90, 64, 1, 1, None),
    ]
    assert preprocess_call["call_arg_shapes"] == [out.shape, dout.shape, lse.shape]
    assert backward_call["call_arg_shapes"][-4:] == [(1, 4, 1), (1, 4, 1, 1), (1, 4, 1), (1, 4, 1, 1)]
    output_shape_dtype = backward_call["output_shape_dtype"]
    assert isinstance(output_shape_dtype, tuple)
    assert len(output_shape_dtype) == 3
    assert output_shape_dtype[1].shape == (1, 1, 128 * 128)
    assert dq_postprocess_call["call_arg_shapes"] == [(1, 4, 64 * 128)]
    assert dk_postprocess_call["call_arg_shapes"] == [(1, 1, 128 * 128)]
    assert dv_postprocess_call["call_arg_shapes"] == [(1, 1, 128 * 128)]
    assert dq.shape == q.shape
    assert dk.shape == k.shape
    assert dv.shape == v.shape


def test_segmented_backward_routes_hopper_d128_gqa_to_native(monkeypatch):
    captured: dict[str, object] = {}

    def fake_native(*args, **kwargs):
        q_arg, k_arg, v_arg = args[:3]
        mask_block_cnt, mask_block_idx = args[8:10]
        full_block_cnt, full_block_idx = args[10:12]
        captured["mask_block_cnt_shape"] = mask_block_cnt.shape
        captured["mask_block_idx_shape"] = mask_block_idx.shape
        captured["full_block_cnt_shape"] = full_block_cnt.shape
        captured["full_block_idx_shape"] = full_block_idx.shape
        captured["window_size_left"] = kwargs["window_size_left"]
        return jnp.zeros_like(q_arg), jnp.zeros_like(k_arg), jnp.zeros_like(v_arg)

    monkeypatch.setattr(fa4_cute_backend, "_import_cutlass_cute", lambda: object())
    monkeypatch.setattr(fa4_cute_backend, "segmented_flash_attention_backward_sm90_native", fake_native)

    q = jnp.ones((1, 8, 4, 128), dtype=jnp.bfloat16)
    k = jnp.ones((1, 8, 1, 128), dtype=jnp.bfloat16)
    v = jnp.ones((1, 8, 1, 128), dtype=jnp.bfloat16)
    out = jnp.ones_like(q)
    dout = jnp.ones_like(q)
    lse = jnp.ones((1, 4, 8), dtype=jnp.float32)
    lower_bounds = jnp.zeros((1, 8), dtype=jnp.int32)
    valid = jnp.ones((1, 8), dtype=jnp.bool_)
    config = fa4_cute_config.flash4_cute_kernel_config(128, arch=90)

    dq, dk, dv = fa4_cute_backend.segmented_flash_attention_backward(
        q,
        k,
        v,
        out,
        dout,
        lse,
        lower_bounds,
        valid,
        softmax_scale=1.0,
        kernel_config=config,
    )

    assert captured == {
        "mask_block_cnt_shape": (1, 1, 1),
        "mask_block_idx_shape": (1, 1, 1, 1),
        "full_block_cnt_shape": (1, 1, 1),
        "full_block_idx_shape": (1, 1, 1, 1),
        "window_size_left": None,
    }
    assert dq.shape == q.shape
    assert dk.shape == k.shape
    assert dv.shape == v.shape


def test_sm90_native_backward_launcher_uses_upstream_sm90_with_grug_mask(monkeypatch):
    captured: dict[str, object] = {}

    class FakeCutlass:
        BFloat16 = object()
        Float16 = object()
        Float32 = float
        Int32 = int
        Boolean = bool

        @staticmethod
        def const_expr(value):
            return value

    class FakeCute:
        Tensor = object()

        @staticmethod
        def jit(fn):
            return fn

        @staticmethod
        def kernel(fn):
            return fn

    class FakeCuda:
        CUstream = object()

    class FakePreprocess:
        def __init__(self, dtype, head_dim, head_dim_v, tile_m, *args, **kwargs):
            captured["preprocess"] = (dtype, head_dim, head_dim_v, tile_m, kwargs)

    class FakeBackwardSm90:
        def __init__(self, dtype, head_dim, head_dim_v, **kwargs):
            captured["backward"] = (dtype, head_dim, head_dim_v, kwargs)

    def fake_dependencies(modules):
        del modules
        return SimpleNamespace(cutlass=FakeCutlass, cute=FakeCute, cuda=FakeCuda)

    def fake_import_module(name):
        if name == "flash_attn.cute.flash_bwd_preprocess":
            return SimpleNamespace(FlashAttentionBackwardPreprocess=FakePreprocess)
        if name == "flash_attn.cute.flash_bwd_sm90":
            return SimpleNamespace(FlashAttentionBackwardSm90=FakeBackwardSm90)
        if name == "flash_attn.cute.block_sparsity":
            return SimpleNamespace(BlockSparseTensors=tuple)
        if name == "flash_attn.cute.utils":
            return SimpleNamespace(ssa_to_scalar=lambda value: value, scalar_to_ssa=lambda value, dtype: value)
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(fa4_cute_kernels, "_import_cute_dependencies", fake_dependencies)
    monkeypatch.setattr(fa4_cute_kernels.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(fa4_cute_kernels, "_patch_jax_array_list_tvm_ffi_converter", lambda: None)

    config = fa4_cute_config.sm90_flash4_cute_backward_config(
        128,
        schedule=fa4_cute_config.Sm90BackwardSchedule.CAUSAL_OR_LOCAL,
    )

    fa4_cute_kernels.segmented_flash_attention_backward_sm90_launcher(
        object(),
        dtype=jnp.dtype(jnp.bfloat16),
        head_dim=128,
        head_dim_v=128,
        qhead_per_kvhead=4,
        config=config,
    )

    _backward_dtype, _backward_head_dim, _backward_head_dim_v, backward_kwargs = captured["backward"]
    assert backward_kwargs["is_causal"] is False
    assert backward_kwargs["is_local"] is False
    assert backward_kwargs["mask_mod"] is not None
    assert backward_kwargs["has_aux_tensors"] is True


def test_sm90_native_backward_boundary_requires_sm90_config():
    q = jnp.ones((1, 8, 4, 128), dtype=jnp.bfloat16)
    k = jnp.ones((1, 8, 1, 128), dtype=jnp.bfloat16)
    v = jnp.ones((1, 8, 1, 128), dtype=jnp.bfloat16)
    lower_bounds = jnp.zeros((1, 8), dtype=jnp.int32)
    valid = jnp.ones((1, 8), dtype=jnp.bool_)
    mask_block_cnt = jnp.ones((1, 1, 1), dtype=jnp.int32)
    mask_block_idx = jnp.zeros((1, 1, 1, 1), dtype=jnp.int32)

    with pytest.raises(NotImplementedError, match="sm90_backward"):
        fa4_cute_backend.segmented_flash_attention_backward_sm90_native(
            q,
            k,
            v,
            q,
            q,
            jnp.ones((1, 4, 8), dtype=jnp.float32),
            lower_bounds,
            valid,
            mask_block_cnt,
            mask_block_idx,
            softmax_scale=1.0,
            kernel_config=fa4_cute_config.Flash4CuteKernelConfig(
                forward_tile=(128, 64),
                backward_tile=(128, 64),
                num_threads=128,
                backward_arch=80,
            ),
        )


def test_segmented_fa4_cute_hopper_backward_uses_sm120_compatible_postprocess(monkeypatch):
    captured: dict[str, object] = {"postprocess_arches": []}

    class FakeCutlass:
        BFloat16 = object()
        Float16 = object()
        Float32 = object()

    class FakeCute:
        Tensor = object()

        @staticmethod
        def jit(fn):
            return fn

        @staticmethod
        def kernel(fn):
            return fn

    class FakeCuda:
        CUstream = object()

    class FakePreprocess:
        def __init__(self, *args, **kwargs):
            pass

    class FakePostprocess:
        def __init__(self, dtype, head_dim, arch, tile_m, *args, **kwargs):
            captured["postprocess_arches"].append(arch)

    class FakeBackwardSm80:
        def __init__(self, *args, **kwargs):
            captured["backward_class"] = "sm80"

    class FakeBackwardSm120:
        def __init__(self, *args, **kwargs):
            captured["backward_class"] = "sm120"

    def fake_dependencies(_modules):
        return SimpleNamespace(cutlass=FakeCutlass, cute=FakeCute, cuda=FakeCuda)

    def fake_import_module(name):
        if name == "flash_attn.cute.flash_bwd_preprocess":
            return SimpleNamespace(FlashAttentionBackwardPreprocess=FakePreprocess)
        if name == "flash_attn.cute.flash_bwd_postprocess":
            return SimpleNamespace(FlashAttentionBackwardPostprocess=FakePostprocess, cpasync=SimpleNamespace())
        if name == "levanter.grug.attention._fa4_cute_segmented_bwd":
            return SimpleNamespace(
                SegmentedFlashAttentionBackwardSm80=FakeBackwardSm80,
                SegmentedFlashAttentionBackwardSm120=FakeBackwardSm120,
            )
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(fa4_cute_kernels, "_import_cute_dependencies", fake_dependencies)
    monkeypatch.setattr(fa4_cute_kernels.importlib, "import_module", fake_import_module)

    fa4_cute_kernels.segmented_flash_attention_backward_launcher(
        object(),
        dtype=jnp.dtype(jnp.bfloat16),
        head_dim=128,
        head_dim_v=128,
        qhead_per_kvhead=2,
        tile_m=64,
        tile_n=64,
        num_threads=128,
        compute_arch=90,
    )

    assert captured["backward_class"] == "sm120"
    assert captured["postprocess_arches"] == [120, 120, 120]


def test_segmented_fa4_cute_backward_uses_dense_dkv_accum_for_mha():
    q = jax.ShapeDtypeStruct((1, 64, 4, 128), jnp.bfloat16)
    k = jax.ShapeDtypeStruct((1, 64, 4, 128), jnp.bfloat16)
    v = jax.ShapeDtypeStruct((1, 64, 4, 128), jnp.bfloat16)

    output_shapes = fa4_cute_backend._cutlass_attention_backward_output_shapes(q, k, v, (64, 64))

    assert output_shapes[-2] == k
    assert output_shapes[-1] == v


def test_segmented_fa4_cute_backward_uses_float32_dkv_accum_for_gqa():
    q = jax.ShapeDtypeStruct((1, 64, 4, 128), jnp.bfloat16)
    k = jax.ShapeDtypeStruct((1, 64, 1, 128), jnp.bfloat16)
    v = jax.ShapeDtypeStruct((1, 64, 1, 128), jnp.bfloat16)

    output_shapes = fa4_cute_backend._cutlass_attention_backward_output_shapes(q, k, v, (64, 64))

    assert output_shapes[-2] == jax.ShapeDtypeStruct((1, 1, 64 * 128), jnp.float32)
    assert output_shapes[-1] == jax.ShapeDtypeStruct((1, 1, 64 * 128), jnp.float32)


def test_fa4_frontend_rejects_mismatched_q_kv_segment_ids():
    if jax.default_backend() != "gpu":
        pytest.skip("FA4/CuTe validation requires a GPU backend.")
    q, k, v = _make_qkv(batch=1, q_len=4, k_len=4, q_heads=2, kv_heads=1)
    q = q.astype(jnp.bfloat16)
    k = k.astype(jnp.bfloat16)
    v = v.astype(jnp.bfloat16)
    q_segment_ids = jnp.array([[1, 1, 2, 2]], dtype=jnp.int32)
    kv_segment_ids = jnp.array([[1, 1, 3, 3]], dtype=jnp.int32)
    mask = AttentionMask.causal().with_segment_ids(q_segment_ids, kv_segment_ids)

    with pytest.raises(Exception, match="requires matching q/kv segment_ids"):
        jax.block_until_ready(gpu_fa4_cute_attention(q, k, v, mask))


@pytest.mark.parametrize(("q_heads", "kv_heads"), [(4, 1), (2, 2)])
def test_real_gpu_fa4_cute_attention_matches_reference_for_valid_dynamic_packed_segments(q_heads, kv_heads):
    if jax.default_backend() != "gpu":
        pytest.skip("FA4/CuTe correctness requires a GPU backend.")
    pytest.importorskip("cutlass")
    pytest.importorskip("cutlass.cute")
    pytest.importorskip("flash_attn.cute.flash_bwd_preprocess")
    key = jax.random.PRNGKey(4)
    q_key, k_key, v_key, cotangent_key = jax.random.split(key, 4)
    q = jax.random.normal(q_key, (1, 64, q_heads, 64), dtype=jnp.bfloat16)
    k = jax.random.normal(k_key, (1, 64, kv_heads, 64), dtype=jnp.bfloat16)
    v = jax.random.normal(v_key, (1, 64, kv_heads, 64), dtype=jnp.bfloat16)
    segment_ids = jnp.array(
        [[37] * 17 + [42] * 23 + [43] * 21 + [-1] * 3],
        dtype=jnp.int32,
    )
    mask = AttentionMask.causal(sliding_window=5).with_segment_ids(segment_ids)

    actual = jax.jit(gpu_fa4_cute_attention)(q, k, v, mask)
    expected = reference_attention(q, k, v, mask, logits_dtype=jnp.float32)
    valid = segment_ids >= 0

    np.testing.assert_allclose(
        jnp.where(valid[..., None, None], actual, expected),
        expected,
        atol=7e-2,
        rtol=7e-2,
    )

    cotangent = jax.random.normal(cotangent_key, q.shape, dtype=jnp.bfloat16)
    cotangent = cotangent * valid[..., None, None].astype(jnp.bfloat16)

    def ref_loss(q_arg, k_arg, v_arg):
        out = reference_attention(q_arg, k_arg, v_arg, mask, logits_dtype=jnp.float32)
        return jnp.sum(out.astype(jnp.float32) * cotangent.astype(jnp.float32))

    def fa4_loss(q_arg, k_arg, v_arg):
        out = gpu_fa4_cute_attention(q_arg, k_arg, v_arg, mask)
        return jnp.sum(out.astype(jnp.float32) * cotangent.astype(jnp.float32))

    actual_grads = jax.jit(jax.grad(fa4_loss, argnums=(0, 1, 2)))(q, k, v)
    expected_grads = jax.jit(jax.grad(ref_loss, argnums=(0, 1, 2)))(q, k, v)

    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        np.testing.assert_allclose(actual_grad, expected_grad, atol=7e-2, rtol=7e-2)
