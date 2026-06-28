# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the mixed E4M3/E5M2 wgmma overlay (``haliax._src.mixed_fp8_wgmma_shim``).

These run on CPU: they verify the overlay installs and that its dtype-guard relaxation actually
changes behavior, without needing an H100. The end-to-end mixed-wgmma numerics (the mixed dgrad/
wgrad producing correct gradients) are validated on H100 by
``lib/levanter/scripts/bench/bench_ragged_mosaic_hybrid_e2e.py --grad-dtype e5m2 --mosaic-wgrad fp8``.

``activate`` monkeypatches process-global jax state (a strict superset of stock — same PTX for
same-dtype operands, verifier-disable scoped to Mosaic lowering), so these coexist with the rest
of the suite without disturbing other backends.
"""

import importlib
import importlib.util
import inspect

import jax
import jax.numpy as jnp
import pytest

from haliax._src import mixed_fp8_wgmma_shim as shim


def _operand(dtype_name: str):
    """A minimal stand-in carrying just a ``.dtype`` (what the relaxed guards inspect)."""
    return type("_Operand", (), {"dtype": jnp.dtype(dtype_name)})


def test_is_mixed_f8_predicate():
    assert shim._is_mixed_f8(_operand("float8_e4m3fn"), _operand("float8_e5m2"))
    assert shim._is_mixed_f8(_operand("float8_e5m2"), _operand("float8_e4m3fn"))
    assert shim._is_mixed_f8(_operand("float8_e4m3fn"), _operand("float8_e4m3fn"))
    assert not shim._is_mixed_f8(_operand("float8_e4m3fn"), _operand("bfloat16"))
    assert not shim._is_mixed_f8(_operand("bfloat16"), _operand("float8_e5m2"))


def _probe_module(tmp_path, name: str, src: str):
    path = tmp_path / f"{name}.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_relax_inline_dtype_guard_changes_behavior(tmp_path):
    """The relaxation must let a mixed FP8 pair through while still rejecting a real mismatch."""
    # Deeper indentation than the real jax guards (4 spaces) — the regex matcher must preserve it.
    module = _probe_module(
        tmp_path,
        "probe_guard",
        "def f(a, b):\n"
        "    if True:\n"
        "        if a.dtype != b.dtype:\n"
        "            raise ValueError('mismatch')\n"
        "    return 'ok'\n",
    )
    e5m2, e4m3, bf16 = _operand("float8_e5m2"), _operand("float8_e4m3fn"), _operand("bfloat16")

    with pytest.raises(ValueError):
        module.f(e5m2, e4m3)  # stock guard rejects the mixed pair

    shim._relax_inline_dtype_guard(module, "f", "a", "b")

    assert module.f(e5m2, e4m3) == "ok"  # the e4m3/e5m2 pair now lowers
    assert module.f(e4m3, e4m3) == "ok"  # same-dtype unaffected
    with pytest.raises(ValueError):
        module.f(e5m2, bf16)  # a genuine mismatch is still rejected


def test_relax_inline_dtype_guard_rejects_ambiguous_match(tmp_path):
    """A guard string that does not appear exactly once must fail loudly (drift guard)."""
    module = _probe_module(
        tmp_path,
        "probe_ambiguous",
        "def g(a, b):\n"
        "    if a.dtype != b.dtype:\n"
        "        return 1\n"
        "    if a.dtype != b.dtype:\n"
        "        return 2\n",
    )
    with pytest.raises(RuntimeError, match="exactly one"):
        shim._relax_inline_dtype_guard(module, "g", "a", "b")


def test_verification_disabled_restores():
    from jaxlib.mlir import ir, passmanager

    orig_verify, orig_parse = ir.Operation.verify, passmanager.PassManager.parse
    with shim._verification_disabled():
        assert ir.Operation.verify is not orig_verify
        assert passmanager.PassManager.parse is not orig_parse
    assert ir.Operation.verify is orig_verify
    assert passmanager.PassManager.parse is orig_parse


@pytest.mark.skipif(
    jax.__version__ != shim._SUPPORTED_JAX_VERSION,
    reason=f"overlay targets jax {shim._SUPPORTED_JAX_VERSION}",
)
def test_activate_installs_overlay():
    shim.activate()
    assert shim.is_active()

    wgmma_mod = importlib.import_module("jax.experimental.mosaic.gpu.wgmma")
    primitives = importlib.import_module("jax._src.pallas.mosaic_gpu.primitives")
    ragged = importlib.import_module("jax.experimental.pallas.ops.gpu.ragged_dot_mgpu")
    plgpu = importlib.import_module("jax.experimental.pallas.mosaic_gpu")
    core = importlib.import_module("jax.experimental.mosaic.gpu.core")

    # (2) emitter: independent a/b element types, resolving names against the stock module.
    params = set(inspect.signature(wgmma_mod.wgmma_m64).parameters)
    assert {"a_element_type", "b_element_type"} <= params
    assert wgmma_mod.wgmma.__globals__ is wgmma_mod.__dict__

    # (3) + (4) dtype gates relaxed (predicate baked into the compiled guard) + re-export repointed.
    assert "_haliax_is_mixed_f8" in primitives.wgmma.__code__.co_names
    assert plgpu.wgmma is primitives.wgmma
    assert "_haliax_is_mixed_f8" in ragged.ragged_dot.__code__.co_names

    # (1) verifier-disable wraps every Mosaic-GPU lowering entrypoint that verifies, including the
    # core._lower_as_gpu_kernel the pallas path hits directly and the pallas lowering entry itself.
    lowering = importlib.import_module("jax._src.pallas.mosaic_gpu.lowering")
    assert getattr(core._lower_as_gpu_kernel, "_haliax_mixed_f8_wrapped", False)
    assert getattr(core._kernel_to_module, "_haliax_mixed_f8_wrapped", False)
    assert getattr(core._run_serde_pass, "_haliax_mixed_f8_wrapped", False)
    assert getattr(lowering.lower_pipelined_jaxpr_to_module, "_haliax_mixed_f8_wrapped", False)
    wrapped = core._lower_as_gpu_kernel
    shim.activate()  # idempotent: no double-wrap
    assert core._lower_as_gpu_kernel is wrapped
