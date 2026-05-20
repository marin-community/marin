# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.grug import fa4_cute_backend as backend


def test_require_cutlass_cute_reports_optional_dependency_absence(monkeypatch):
    def fake_import_module(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(backend.importlib, "import_module", fake_import_module)

    assert not backend.cutlass_cute_available()
    with pytest.raises(RuntimeError, match="nvidia-cutlass-dsl"):
        backend.require_cutlass_cute()


def test_cute_vector_add_wires_cutlass_call(monkeypatch):
    calls = {}

    class FakeCute:
        Tensor = object

        @staticmethod
        def kernel(fn):
            calls["kernel_name"] = fn.__name__
            return fn

        @staticmethod
        def jit(fn):
            calls["jit_name"] = fn.__name__
            return fn

    class FakeCjax:
        @staticmethod
        def cutlass_call(launcher, *, output_shape_dtype, use_static_tensors):
            calls["launcher"] = launcher
            calls["output_shape_dtype"] = output_shape_dtype
            calls["use_static_tensors"] = use_static_tensors

            def call(a, b):
                return a + b

            return call

    fake_modules = backend._CutlassCuteModules(
        cute=FakeCute(),
        cjax=FakeCjax(),
        cuda=SimpleNamespace(CUstream=object),
    )
    monkeypatch.setattr(backend, "_import_cutlass_cute", lambda: fake_modules)

    a = jnp.arange(5, dtype=jnp.float16)
    b = jnp.full((5,), 2, dtype=jnp.float16)

    actual = backend.cute_vector_add(a, b, block_size=4)

    np.testing.assert_array_equal(actual, a + b)
    assert calls["kernel_name"] == "_vector_add_kernel"
    assert calls["jit_name"] == "_launch_vector_add"
    assert calls["output_shape_dtype"].shape == (1, 4, 2)
    assert calls["output_shape_dtype"].dtype == jnp.float16
    assert calls["use_static_tensors"] is True


def test_fa4_cute_attention_forward_validates_metadata_before_optional_import(monkeypatch):
    def fail_import():
        raise AssertionError("optional import should not run after metadata validation fails")

    monkeypatch.setattr(backend, "_import_cutlass_cute", fail_import)

    q = jnp.ones((2, 4, 4, 8), dtype=jnp.bfloat16)
    k = jnp.ones((2, 4, 2, 8), dtype=jnp.bfloat16)
    v = jnp.ones((2, 4, 2, 8), dtype=jnp.bfloat16)
    lower_bounds = jnp.zeros((2, 3), dtype=jnp.int32)
    valid = jnp.ones((2, 4), dtype=jnp.bool_)

    with pytest.raises(ValueError, match="lower_bounds"):
        backend.fa4_cute_attention_forward(q, k, v, lower_bounds, valid, sm_scale=1.0, kernel_config=object())


def test_fa4_cute_attention_forward_rejects_float32_before_optional_import(monkeypatch):
    def fail_import():
        raise AssertionError("optional import should not run after dtype validation fails")

    monkeypatch.setattr(backend, "_import_cutlass_cute", fail_import)

    q = jnp.ones((2, 4, 4, 8), dtype=jnp.float32)
    k = jnp.ones((2, 4, 2, 8), dtype=jnp.float32)
    v = jnp.ones((2, 4, 2, 8), dtype=jnp.float32)
    lower_bounds = jnp.zeros((2, 4), dtype=jnp.int32)
    valid = jnp.ones((2, 4), dtype=jnp.bool_)

    with pytest.raises(TypeError, match="bf16/fp16"):
        backend.fa4_cute_attention_forward(q, k, v, lower_bounds, valid, sm_scale=1.0, kernel_config=object())


def test_fa4_cute_attention_forward_wires_cutlass_call(monkeypatch):
    calls = {}

    class FakeCjax:
        class TensorSpec:
            def __init__(self, *, mode=None, divisibility=None, static=False):
                self.mode = mode
                self.divisibility = divisibility
                self.static = static

            def __eq__(self, other):
                return (
                    isinstance(other, FakeCjax.TensorSpec)
                    and self.mode == other.mode
                    and self.divisibility == other.divisibility
                    and self.static == other.static
                )

        @staticmethod
        def cutlass_call(launcher, *, output_shape_dtype, input_spec, output_spec, use_static_tensors, softmax_scale):
            calls["cutlass_call"] = {
                "launcher": launcher,
                "output_shape_dtype": output_shape_dtype,
                "input_spec": input_spec,
                "output_spec": output_spec,
                "use_static_tensors": use_static_tensors,
                "softmax_scale": softmax_scale,
            }

            def call(q, k, v, lower_bounds, valid):
                calls["call"] = {
                    "q_shape": q.shape,
                    "k_shape": k.shape,
                    "v_shape": v.shape,
                    "lower_bounds_shape": lower_bounds.shape,
                    "valid_shape": valid.shape,
                }
                out_shape_dtype, lse_shape_dtype = output_shape_dtype
                return (
                    jnp.zeros(out_shape_dtype.shape, out_shape_dtype.dtype),
                    jnp.zeros(lse_shape_dtype.shape, lse_shape_dtype.dtype),
                )

            return call

    fake_modules = backend._CutlassCuteModules(
        cute=SimpleNamespace(kernel=lambda fn: fn, jit=lambda fn: fn, Tensor=object),
        cjax=FakeCjax(),
        cuda=SimpleNamespace(CUstream=object),
    )
    monkeypatch.setattr(backend, "_import_cutlass_cute", lambda: fake_modules)

    def fake_launcher(modules, *, head_dim, head_dim_v, qhead_per_kvhead, tile_m, tile_n, num_threads):
        calls["launcher"] = {
            "modules": modules,
            "head_dim": head_dim,
            "head_dim_v": head_dim_v,
            "qhead_per_kvhead": qhead_per_kvhead,
            "tile_m": tile_m,
            "tile_n": tile_n,
            "num_threads": num_threads,
        }
        return object()

    monkeypatch.setattr(backend, "segmented_flash_attention_forward_launcher", fake_launcher)

    q = jnp.ones((2, 4, 4, 8), dtype=jnp.bfloat16)
    k = jnp.ones((2, 4, 2, 8), dtype=jnp.bfloat16)
    v = jnp.ones((2, 4, 2, 8), dtype=jnp.bfloat16)
    lower_bounds = jnp.zeros((2, 4), dtype=jnp.int32)
    valid = jnp.ones((2, 4), dtype=jnp.bool_)
    kernel_config = SimpleNamespace(forward_tile=(64, 32), num_threads=96)

    out = backend.fa4_cute_attention_forward(q, k, v, lower_bounds, valid, sm_scale=0.5, kernel_config=kernel_config)

    assert out.shape == q.shape
    assert out.dtype == q.dtype
    assert calls["launcher"] == {
        "modules": fake_modules,
        "head_dim": 8,
        "head_dim_v": 8,
        "qhead_per_kvhead": 2,
        "tile_m": 64,
        "tile_n": 32,
        "num_threads": 96,
    }
    out_shape_dtype, lse_shape_dtype = calls["cutlass_call"]["output_shape_dtype"]
    assert out_shape_dtype.shape == q.shape
    assert out_shape_dtype.dtype == q.dtype
    assert lse_shape_dtype.shape == (2, 4, 4)
    assert lse_shape_dtype.dtype == jnp.float32
    qkv_spec = FakeCjax.TensorSpec(mode=(1, 3, 2, 0), divisibility=(1, 1, 1, 8), static=True)
    lse_spec = FakeCjax.TensorSpec(divisibility=(1, 1, 1), static=True)
    metadata_spec = FakeCjax.TensorSpec(static=True)
    assert calls["cutlass_call"]["input_spec"] == (qkv_spec, qkv_spec, qkv_spec, metadata_spec, metadata_spec)
    assert calls["cutlass_call"]["output_spec"] == (qkv_spec, lse_spec)
    assert calls["cutlass_call"]["use_static_tensors"] is True
    assert calls["cutlass_call"]["softmax_scale"] == 0.5
    assert calls["call"] == {
        "q_shape": q.shape,
        "k_shape": k.shape,
        "v_shape": v.shape,
        "lower_bounds_shape": lower_bounds.shape,
        "valid_shape": valid.shape,
    }


def test_fa4_cute_attention_backward_wires_cutlass_call(monkeypatch):
    calls: dict[str, Any] = {"cutlass_call": []}

    class FakeCjax:
        class TensorSpec:
            def __init__(self, *, mode=None, divisibility=None, static=False):
                self.mode = mode
                self.divisibility = divisibility
                self.static = static

            def __eq__(self, other):
                return (
                    isinstance(other, FakeCjax.TensorSpec)
                    and self.mode == other.mode
                    and self.divisibility == other.divisibility
                    and self.static == other.static
                )

        @staticmethod
        def cutlass_call(launcher, *, output_shape_dtype, input_spec, output_spec, use_static_tensors, softmax_scale):
            calls["cutlass_call"].append(
                {
                    "launcher": launcher,
                    "output_shape_dtype": output_shape_dtype,
                    "input_spec": input_spec,
                    "output_spec": output_spec,
                    "use_static_tensors": use_static_tensors,
                    "softmax_scale": softmax_scale,
                }
            )

            def call(*args):
                calls.setdefault("call_arg_shapes", []).append(tuple(arg.shape for arg in args))
                if len(output_shape_dtype) == 2:
                    out_shape_dtype, lse_shape_dtype = output_shape_dtype
                    return (
                        jnp.ones(out_shape_dtype.shape, out_shape_dtype.dtype),
                        jnp.full(lse_shape_dtype.shape, 2.0, lse_shape_dtype.dtype),
                    )
                return tuple(jnp.full(spec.shape, i + 1, spec.dtype) for i, spec in enumerate(output_shape_dtype))

            return call

    fake_modules = backend._CutlassCuteModules(
        cute=SimpleNamespace(kernel=lambda fn: fn, jit=lambda fn: fn, Tensor=object),
        cjax=FakeCjax(),
        cuda=SimpleNamespace(CUstream=object),
    )
    monkeypatch.setattr(backend, "_import_cutlass_cute", lambda: fake_modules)
    monkeypatch.setattr(backend, "segmented_flash_attention_forward_launcher", lambda *args, **kwargs: "fwd")

    def fake_backward_launcher(modules, *, dtype, head_dim, head_dim_v, qhead_per_kvhead, tile_m, tile_n, num_threads):
        calls["backward_launcher"] = {
            "modules": modules,
            "dtype": dtype,
            "head_dim": head_dim,
            "head_dim_v": head_dim_v,
            "qhead_per_kvhead": qhead_per_kvhead,
            "tile_m": tile_m,
            "tile_n": tile_n,
            "num_threads": num_threads,
        }
        return "bwd"

    monkeypatch.setattr(backend, "segmented_flash_attention_backward_launcher", fake_backward_launcher)

    q = jnp.ones((1, 2, 2, 8), dtype=jnp.bfloat16)
    k = jnp.ones((1, 2, 1, 8), dtype=jnp.bfloat16)
    v = jnp.ones((1, 2, 1, 8), dtype=jnp.bfloat16)
    lower_bounds = jnp.zeros((1, 2), dtype=jnp.int32)
    valid = jnp.ones((1, 2), dtype=jnp.bool_)
    kernel_config = SimpleNamespace(forward_tile=(64, 32), backward_tile=(32, 32), num_threads=128)

    def loss(q_arg, k_arg, v_arg):
        out = backend.fa4_cute_attention_forward(
            q_arg,
            k_arg,
            v_arg,
            lower_bounds,
            valid,
            sm_scale=0.5,
            kernel_config=kernel_config,
        )
        return jnp.sum(out.astype(jnp.float32))

    dq, dk, dv = jax.grad(loss, argnums=(0, 1, 2))(q, k, v)

    assert dq.shape == q.shape
    assert dk.shape == k.shape
    assert dv.shape == v.shape
    assert calls["backward_launcher"] == {
        "modules": fake_modules,
        "dtype": q.dtype,
        "head_dim": 8,
        "head_dim_v": 8,
        "qhead_per_kvhead": 2,
        "tile_m": 32,
        "tile_n": 32,
        "num_threads": 128,
    }
    assert calls["call_arg_shapes"] == [
        (q.shape, k.shape, v.shape, lower_bounds.shape, valid.shape),
        (q.shape, k.shape, v.shape, q.shape, q.shape, (1, 2, 2), lower_bounds.shape, valid.shape),
    ]
    qkv_spec = FakeCjax.TensorSpec(mode=(0, 1, 2, 3), divisibility=(1, 1, 1, 8), static=True)
    lse_spec = FakeCjax.TensorSpec(mode=(0, 1, 2), divisibility=(1, 1, 1), static=True)
    metadata_spec = FakeCjax.TensorSpec(mode=(0, 1), static=True)
    scratch_spec = FakeCjax.TensorSpec(mode=(0, 1, 2), static=True)
    assert calls["cutlass_call"][1]["input_spec"] == (
        qkv_spec,
        qkv_spec,
        qkv_spec,
        qkv_spec,
        qkv_spec,
        lse_spec,
        metadata_spec,
        metadata_spec,
    )
    assert calls["cutlass_call"][1]["output_spec"] == (
        qkv_spec,
        qkv_spec,
        qkv_spec,
        scratch_spec,
        scratch_spec,
        scratch_spec,
        scratch_spec,
        scratch_spec,
    )
    assert tuple(spec.shape for spec in calls["cutlass_call"][1]["output_shape_dtype"]) == (
        q.shape,
        k.shape,
        v.shape,
        (1, 2, 32),
        (1, 2, 32),
        (1, 2, 1024),
        (1, 1, 1024),
        (1, 1, 1024),
    )
    assert calls["cutlass_call"][1]["softmax_scale"] == 0.5


def test_fa4_cute_attention_forward_rejects_mismatched_kv_heads_before_optional_import(monkeypatch):
    def fail_import():
        raise AssertionError("optional import should not run after shape validation fails")

    monkeypatch.setattr(backend, "_import_cutlass_cute", fail_import)

    q = jnp.ones((2, 4, 4, 8), dtype=jnp.bfloat16)
    k = jnp.ones((2, 4, 2, 8), dtype=jnp.bfloat16)
    v = jnp.ones((2, 4, 1, 8), dtype=jnp.bfloat16)
    lower_bounds = jnp.zeros((2, 4), dtype=jnp.int32)
    valid = jnp.ones((2, 4), dtype=jnp.bool_)

    with pytest.raises(ValueError, match="k/v head counts"):
        backend.fa4_cute_attention_forward(q, k, v, lower_bounds, valid, sm_scale=1.0, kernel_config=object())


def test_fa4_cute_attention_forward_rejects_different_value_head_dim_before_optional_import(monkeypatch):
    def fail_import():
        raise AssertionError("optional import should not run after shape validation fails")

    monkeypatch.setattr(backend, "_import_cutlass_cute", fail_import)

    q = jnp.ones((2, 4, 4, 8), dtype=jnp.bfloat16)
    k = jnp.ones((2, 4, 2, 8), dtype=jnp.bfloat16)
    v = jnp.ones((2, 4, 2, 16), dtype=jnp.bfloat16)
    lower_bounds = jnp.zeros((2, 4), dtype=jnp.int32)
    valid = jnp.ones((2, 4), dtype=jnp.bool_)

    with pytest.raises(NotImplementedError, match="Dv == D"):
        backend.fa4_cute_attention_forward(q, k, v, lower_bounds, valid, sm_scale=1.0, kernel_config=object())
