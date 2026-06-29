# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util
import numpy as np
import pytest
from chex import assert_trees_all_close

import haliax as hax
from haliax._src.fp8 import compute_scale, fp8_scaled_dot_general
from haliax.nn import Linear
from haliax.quantization import (
    Fp8DotGeneralOp,
    QuantizationConfig,
    apply_updates,
    partition_for_grad_overwrite,
    quantize_linear_layers,
)


def test_fp8_is_reasonable():
    In = hax.Axis("In", 8)
    Out = hax.Axis("Out", 8)
    linear = Linear.init(In, Out, key=jrandom.PRNGKey(0), init_scale=0.1)

    fp8_linear = Linear.init(In, Out, key=jrandom.PRNGKey(0), dot_general=Fp8DotGeneralOp.init(), init_scale=0.1)

    input = hax.random.normal(jrandom.PRNGKey(3), In)
    output = linear(input)
    fp8_output = fp8_linear(input)

    assert output.shape == fp8_output.shape
    assert output.dtype == fp8_output.dtype

    assert_trees_all_close(output.array, fp8_output.array, atol=2e-2, rtol=5e-2)


def _all_dot_general_eqns(jaxpr):
    """Recursively collect dot_general equations, descending into nested jaxprs
    (the forward dot lives inside the custom_vjp call's call_jaxpr)."""

    def _as_jaxprs(param):
        for c in param if isinstance(param, (list, tuple)) else [param]:
            inner = getattr(c, "jaxpr", c)  # unwrap ClosedJaxpr -> Jaxpr
            if hasattr(inner, "eqns"):
                yield inner

    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "dot_general":
            yield eqn
        for param in eqn.params.values():
            for sub in _as_jaxprs(param):
                yield from _all_dot_general_eqns(sub)


def test_fp8_feeds_fp8_operands_to_dot():
    # Assert the lowered jaxpr has a dot_general with float8 operands.
    In = hax.Axis("In", 16)
    Out = hax.Axis("Out", 8)
    fp8_linear = Linear.init(In, Out, key=jrandom.PRNGKey(0), dot_general=Fp8DotGeneralOp.init())
    input = hax.random.normal(jrandom.PRNGKey(3), In)

    jaxpr = jax.make_jaxpr(lambda m, x: m(x).array)(fp8_linear, input)
    dot_eqns = list(_all_dot_general_eqns(jaxpr.jaxpr))
    assert dot_eqns, "expected a dot_general in the lowered op"
    assert any(
        all(v.aval.dtype == jnp.float8_e4m3fn for v in eqn.invars) for eqn in dot_eqns
    ), "forward dot_general should contract two float8_e4m3fn operands"


def test_fp8_quant_dtype_overrides_reach_dot_operands():
    # The forward-operand and output-gradient quantization dtypes are
    # configurable (defaulting to E4M3 / E5M2). Swap the two so each override is
    # observable, then confirm the dtype fed to each dot_general follows: the
    # forward contracts two E5M2 operands and a backward dot contracts an E4M3
    # output gradient -- the reverse of the defaults asserted elsewhere.
    In = hax.Axis("In", 16)
    Out = hax.Axis("Out", 8)
    dot_general = Fp8DotGeneralOp.init(fwd_dtype=jnp.float8_e5m2, rev_dtype=jnp.float8_e4m3fn)
    fp8_linear = Linear.init(In, Out, key=jrandom.PRNGKey(0), dot_general=dot_general)
    x = hax.random.normal(jrandom.PRNGKey(3), In)

    def loss(model, x):
        return hax.sum(model(x) ** 2).scalar()

    jaxpr = jax.make_jaxpr(eqx.filter_grad(loss))(fp8_linear, x)
    dot_eqns = list(_all_dot_general_eqns(jaxpr.jaxpr))
    assert any(
        all(v.aval.dtype == jnp.float8_e5m2 for v in eqn.invars) for eqn in dot_eqns
    ), "forward dot_general should contract operands quantized to the configured fwd_dtype (here E5M2)"
    assert any(
        any(v.aval.dtype == jnp.float8_e4m3fn for v in eqn.invars) for eqn in dot_eqns
    ), "a backward dot_general should contract the output gradient quantized to the configured rev_dtype (here E4M3)"


def _gpu_is_fp8_capable() -> bool:
    # FP8 tensor-core GEMMs need CUDA compute capability >= 8.9 (Ada/Hopper/Blackwell).
    if jax.default_backend() != "gpu":
        return False
    # compute_capability is a string like "9.0" on a CUDA device.
    major, _, minor = jax.devices()[0].compute_capability.partition(".")
    return (int(major), int(minor or 0)) >= (8, 9)


@pytest.mark.skipif(
    not _gpu_is_fp8_capable(),
    reason="needs an FP8-capable GPU (CUDA sm89+: Ada/Hopper/Blackwell)",
)
def test_fp8_emits_fp8_kernels_on_gpu():
    # Both passes of the default recipe must run as FP8 GEMMs on tensor cores.
    # Compiling the gradient exercises the forward dot (e4m3 x e4m3) and the two
    # grad dots (e5m2 output grad x e4m3 operands) in one HLO, so none may fall
    # back to a bf16 cuBLASLt kernel that dequantizes first. Depending on shape,
    # XLA's autotuner lowers each to a cuBLASLt FP8 kernel (__cublas$lt$matmul$f8)
    # or a Triton FP8 gemm fusion (__triton_nested_gemm_fusion).
    Batch = hax.Axis("Batch", 512)
    In = hax.Axis("In", 512)
    Out = hax.Axis("Out", 512)
    fp8_linear = Linear.init(In, Out, key=jrandom.PRNGKey(0), use_bias=False, dot_general=Fp8DotGeneralOp.init())
    # Run in bf16 (the training compute dtype); FP8 state stays float32.
    fp8_linear = dataclasses.replace(fp8_linear, weight=fp8_linear.weight.astype(jnp.bfloat16))
    x = hax.random.normal(jrandom.PRNGKey(1), (Batch, In)).astype(jnp.bfloat16)

    # Partition so jax.jit sees only arrays (the static module part is closed
    # over); this also yields a real Compiled whose as_text() is the optimized HLO.
    params, static = eqx.partition(fp8_linear, eqx.is_array)

    def loss(params, x):
        return hax.sum(eqx.combine(params, static)(x) ** 2).scalar()

    # Differentiate w.r.t. both the weight and the input so both grad GEMMs
    # (wgrad needs d/d weight, dgrad needs d/d input) stay live; the forward dot
    # is present too, since its residuals feed the backward.
    hlo = jax.jit(jax.grad(loss, argnums=(0, 1))).lower(params, x).compile().as_text()

    # Forward operands reach the matmul as E4M3, the output gradient as E5M2
    # (a bf16 fallback would dequantize them first).
    assert "f8e4m3" in hlo, "forward should quantize operands to FP8 (e4m3)"
    assert "f8e5m2" in hlo, "backward should quantize the output gradient to FP8 (e5m2)"
    # ... fed to real FP8 GEMM kernels: cuBLASLt FP8 or a Triton FP8 gemm fusion.
    n_cublas = hlo.count("__cublas$lt$matmul")
    n_cublas_f8 = hlo.count("__cublas$lt$matmul$f8")
    n_triton_f8 = hlo.count("__triton_nested_gemm_fusion")
    # Every cuBLASLt matmul must be the $f8 variant (no bf16 cuBLASLt fallback).
    assert n_cublas == n_cublas_f8, "a matmul fell back to a bf16 __cublas$lt$matmul"
    # All three matmuls (forward + dgrad + wgrad) must lower to FP8 GEMMs. Counting
    # both backends keeps this non-vacuous when XLA picks Triton fusions, where the
    # cuBLASLt check above is trivially 0 == 0.
    assert (
        n_cublas_f8 + n_triton_f8 >= 3
    ), f"expected >= 3 FP8 GEMMs (fwd + 2 grad dots); got {n_cublas_f8} cuBLASLt + {n_triton_f8} Triton"


def test_fp8_grads_match_reference():
    Batch = hax.Axis("Batch", 64)
    In = hax.Axis("In", 32)
    Out = hax.Axis("Out", 16)
    ref = Linear.init(In, Out, key=jrandom.PRNGKey(0), init_scale=1.0)
    fp8_linear = Linear.init(In, Out, key=jrandom.PRNGKey(0), dot_general=Fp8DotGeneralOp.init(), init_scale=1.0)
    x = hax.random.normal(jrandom.PRNGKey(3), (Batch, In))

    def loss(model, xx):
        return hax.sum(model(xx) ** 2).scalar()

    def grads(model):
        g_w = eqx.filter_grad(lambda m: loss(m, x))(model).weight.array
        g_x = eqx.filter_grad(lambda xx: loss(model, xx))(x).array
        return np.asarray(g_w), np.asarray(g_x)

    gw_fp8, gx_fp8 = grads(fp8_linear)
    gw_ref, gx_ref = grads(ref)

    def assert_grad_close(got, want, name):
        rms = np.sqrt(np.mean(want**2))
        np.testing.assert_allclose(got, want, rtol=0.2, atol=0.5 * rms, err_msg=f"{name}: elementwise")
        norm_diff = np.linalg.norm(got - want) / np.linalg.norm(want)
        assert norm_diff < 0.15, f"{name}: relative difference norm {norm_diff:.3f} too large"
        mag = np.linalg.norm(got) / np.linalg.norm(want)
        assert 0.97 < mag < 1.03, f"{name}: magnitude ratio {mag:.3f} (systematic scale bias)"

    assert_grad_close(gw_fp8, gw_ref, "wgrad")
    assert_grad_close(gx_fp8, gx_ref, "dgrad")


def test_fp8_exact_for_representable_inputs():
    M, K, N = 8, 12, 4
    dimension_numbers = (((1,), (0,)), ((), ()))
    rep_op = jnp.array([-3.0, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 3.0])  # exact in e4m3
    rep_grad = jnp.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])  # exact in e5m2

    def pick(key, shape, table):
        return table[jrandom.randint(key, shape, 0, table.shape[0])]

    k_lhs, k_rhs, k_cot = jrandom.split(jrandom.PRNGKey(7), 3)
    lhs = pick(k_lhs, (M, K), rep_op)
    rhs = pick(k_rhs, (K, N), rep_op)
    cot = pick(k_cot, (M, N), rep_grad)

    zero_hist = jnp.zeros(16, jnp.float32)

    def fp8_matmul(a, b):
        return fp8_scaled_dot_general(
            a,
            b,
            dimension_numbers,
            preferred_element_type=jnp.float32,
            lhs_scale=jnp.float32(2.0),
            rhs_scale=jnp.float32(0.5),
            grad_scale=jnp.float32(4.0),
            lhs_amax_history=zero_hist,
            rhs_amax_history=zero_hist,
            grad_amax_history=zero_hist,
            quantize_compute_type=jnp.float32,
        )

    y, vjp_fn = jax.vjp(fp8_matmul, lhs, rhs)
    grad_lhs, grad_rhs = vjp_fn(cot)

    np.testing.assert_array_equal(np.asarray(y), np.asarray(lhs @ rhs))
    np.testing.assert_array_equal(np.asarray(grad_lhs), np.asarray(cot @ rhs.T))
    np.testing.assert_array_equal(np.asarray(grad_rhs), np.asarray(lhs.T @ cot))


def test_fp8_backward_uses_distinct_operand_scales():
    # quantized_dot_bwd dequantizes grad_lhs with rhs_scale and grad_rhs with
    # lhs_scale. With the single-step defaults every scale is exactly 1.0, so a
    # swapped assignment would be invisible. Give the operands very different
    # magnitudes and pre-seed their amax histories so input and kernel get
    # distinct, non-unit scales (here ~32x apart), then check both gradients
    # against an fp32 reference (the gradient of the exact matmul -- an oracle
    # independent of the FP8 implementation). A swap would be off by ~32x.
    M, K, N = 8, 16, 4
    lhs = jrandom.normal(jrandom.PRNGKey(0), (M, K))
    rhs = jrandom.normal(jrandom.PRNGKey(1), (K, N)) * 32.0
    cot = jrandom.normal(jrandom.PRNGKey(2), (M, N))
    dimension_numbers = (((1,), (0,)), ((), ()))

    n = 16
    one = jnp.ones(1, jnp.float32)
    lhs_hist = jnp.zeros(n, jnp.float32).at[0].set(jnp.max(jnp.abs(lhs)))
    rhs_hist = jnp.zeros(n, jnp.float32).at[0].set(jnp.max(jnp.abs(rhs)))
    grad_hist = jnp.zeros(n, jnp.float32)

    def fp8_matmul(lhs, rhs):
        return fp8_scaled_dot_general(
            lhs,
            rhs,
            dimension_numbers,
            preferred_element_type=jnp.float32,
            lhs_scale=one,
            rhs_scale=one,
            grad_scale=one,
            lhs_amax_history=lhs_hist,
            rhs_amax_history=rhs_hist,
            grad_amax_history=grad_hist,
        )

    _, vjp_fn = jax.vjp(fp8_matmul, lhs, rhs)
    grad_lhs, grad_rhs = vjp_fn(cot)

    def rel(a, b):
        return np.linalg.norm(a - b) / np.linalg.norm(b)

    assert rel(grad_lhs, cot @ rhs.T) < 0.2, f"grad_lhs rel err {rel(grad_lhs, cot @ rhs.T)}"
    assert rel(grad_rhs, lhs.T @ cot) < 0.2, f"grad_rhs rel err {rel(grad_rhs, lhs.T @ cot)}"


def test_fp8_forward_matches_reference_under_nonunit_scales():
    # test_fp8_is_reasonable only covers the unit-scale fresh state. Here the
    # delayed-scaling state carries non-unit scales: pre-seed the amax histories
    # from operands with large, distinct magnitudes so input and kernel get
    # non-unit scales ~32x apart. The out_dq dequantize must undo lhs_scale *
    # rhs_scale so the forward output tracks the exact fp32 matmul lhs @ rhs (an
    # oracle independent of the FP8 implementation); a missing/mismatched scale
    # would be off by orders of magnitude, not just quantization noise.
    M, K, N = 8, 16, 4
    lhs = jrandom.normal(jrandom.PRNGKey(0), (M, K)) * 4.0
    rhs = jrandom.normal(jrandom.PRNGKey(1), (K, N)) * 128.0
    dimension_numbers = (((1,), (0,)), ((), ()))

    n = 16
    one = jnp.ones(1, jnp.float32)
    lhs_hist = jnp.zeros(n, jnp.float32).at[0].set(jnp.max(jnp.abs(lhs)))
    rhs_hist = jnp.zeros(n, jnp.float32).at[0].set(jnp.max(jnp.abs(rhs)))
    grad_hist = jnp.zeros(n, jnp.float32)

    y = fp8_scaled_dot_general(
        lhs,
        rhs,
        dimension_numbers,
        preferred_element_type=jnp.float32,
        lhs_scale=one,
        rhs_scale=one,
        grad_scale=one,
        lhs_amax_history=lhs_hist,
        rhs_amax_history=rhs_hist,
        grad_amax_history=grad_hist,
    )

    ref = lhs @ rhs
    rel = np.linalg.norm(np.asarray(y) - ref) / np.linalg.norm(ref)
    assert rel < 0.1, f"forward rel err {rel} too large"


def test_fp8_backward_quantizes_grad_to_e5m2():
    # Mirror of the forward FP8-operand check for the backward pass: the output
    # gradient must reach the gradient GEMMs as float8_e5m2.
    In = hax.Axis("In", 16)
    Out = hax.Axis("Out", 8)
    fp8_linear = Linear.init(In, Out, key=jrandom.PRNGKey(0), dot_general=Fp8DotGeneralOp.init())
    x = hax.random.normal(jrandom.PRNGKey(3), In)

    def loss(model, x):
        return hax.sum(model(x) ** 2).scalar()

    jaxpr = jax.make_jaxpr(eqx.filter_grad(loss))(fp8_linear, x)
    dot_eqns = list(_all_dot_general_eqns(jaxpr.jaxpr))
    assert any(
        any(v.aval.dtype == jnp.float8_e5m2 for v in eqn.invars) for eqn in dot_eqns
    ), "a backward dot_general should contract a float8_e5m2 output-gradient operand"


# https://github.com/google/flax/blob/6f2b08e024c2fd2f8cec42a6c82408cb35412319/tests/linen/linen_test.py#L1222
# Validates the delayed-scaling state update against a manual reference: each
# step rolls the amax history and recomputes the scale with the TE formula on
# the input, kernel and output-gradient amax.
def test_fp_loop():
    key, init_key, random_key = jrandom.split(jrandom.PRNGKey(seed=123), 3)
    Batch = hax.Axis("Batch", 16)
    In = hax.Axis("In", 16)
    Out = hax.Axis("Out", 32)
    linear = Linear.init(In, Out, key=init_key, dot_general=Fp8DotGeneralOp.init())

    def _roll_and_update(amax_h, update):
        return jnp.roll(amax_h, shift=-1, axis=0).at[0].set(update)

    lr = 1e-3

    def apply_gradients(model, grads):
        overwrites, grads = partition_for_grad_overwrite(grads)
        updates = jax.tree_util.tree_map(lambda g: -lr * g, grads)
        model = apply_updates(model, updates, overwrites)
        return model

    def _train_step(model, x, dy):
        def loss_fn(lin):
            y = lin(x)
            loss = y * dy.astype(y.dtype)
            return hax.sum(loss).scalar()

        grad_fn = eqx.filter_grad(loss_fn)
        grads = grad_fn(model)
        return apply_gradients(model, grads)

    train_fn = eqx.filter_jit(_train_step)

    scale_x, amax_history_x = jnp.ones(()), jnp.zeros((1024,))
    scale_k, amax_history_k = jnp.ones(()), jnp.zeros((1024,))
    scale_g, amax_history_g = jnp.ones(()), jnp.zeros((1024,))
    e4m3_max = jnp.finfo(jnp.float8_e4m3fn).max.astype(jnp.float32)
    e5m2_max = jnp.finfo(jnp.float8_e5m2).max.astype(jnp.float32)

    for _ in range(5):
        key, random_key = jrandom.split(key, 2)
        # x = jrandom.normal(random_key, (16, 16), dtype=jnp.float32)
        # g = jrandom.normal(random_key, (16, 32), dtype=jnp.float32)
        x = hax.random.normal(
            random_key,
            (
                Batch,
                In,
            ),
        )
        g = hax.random.normal(
            random_key,
            (
                Batch,
                Out,
            ),
        )

        # Manually compute the expected amax history and scaling factors.
        amax_from_history_x = jnp.max(amax_history_x, axis=0)
        amax_from_history_k = jnp.max(amax_history_k, axis=0)
        amax_from_history_g = jnp.max(amax_history_g, axis=0)
        scale_x = compute_scale(amax_from_history_x, scale_x, e4m3_max)
        scale_k = compute_scale(amax_from_history_k, scale_k, e4m3_max)
        scale_g = compute_scale(amax_from_history_g, scale_g, e5m2_max)
        amax_history_x = _roll_and_update(amax_history_x, jnp.max(jnp.abs(x.array)))
        amax_history_k = _roll_and_update(amax_history_k, jnp.max(jnp.abs(linear.weight.array)))
        amax_history_g = _roll_and_update(amax_history_g, jnp.max(jnp.abs(g.array)))

        linear = train_fn(linear, x, g)

        rtol, atol = 0.001, 0.001
        np.testing.assert_allclose(
            linear.dot_general.input_amax_history,  # type: ignore
            amax_history_x,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            linear.dot_general.kernel_amax_history,  # type: ignore
            amax_history_k,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            linear.dot_general.output_grad_amax_history,  # type: ignore
            amax_history_g,
            rtol=rtol,
            atol=atol,
        )

        np.testing.assert_allclose(linear.dot_general.input_scale, scale_x, rtol=rtol, atol=atol)  # type: ignore
        np.testing.assert_allclose(linear.dot_general.kernel_scale, scale_k, rtol=rtol, atol=atol)  # type: ignore
        np.testing.assert_allclose(linear.dot_general.output_grad_scale, scale_g, rtol=rtol, atol=atol)  # type: ignore


def test_layer_splicing():
    key, init_key, random_key = jrandom.split(jrandom.PRNGKey(seed=123), 3)
    Input = hax.Axis("Input", 16)
    Hidden = hax.Axis("Hidden", 64)
    Output = hax.Axis("Output", 32)
    mlp = hax.nn.MLP.init(Input, Output, Hidden, 3, key=init_key, init_scale=0.1)

    mlp_q = quantize_linear_layers(mlp, QuantizationConfig(fp8=True))
    for layer in mlp_q.layers:
        assert isinstance(layer.dot_general, Fp8DotGeneralOp)

    input = hax.random.normal(jrandom.PRNGKey(0), Input) * 10  # 10 so we don't underflow
    output = mlp(input)
    output_q = mlp_q(input)
    chex.assert_trees_all_close(output.array, output_q.array, atol=1e-3, rtol=1e-3)
    assert not jnp.allclose(output_q.array, 0)  # don't want them to all underflow

    mlp_q = quantize_linear_layers(mlp, QuantizationConfig(targets="layers.0", fp8=True))
    for i, layer in enumerate(mlp_q.layers):
        if i == 0:
            assert isinstance(layer.dot_general, Fp8DotGeneralOp)
        else:
            assert not isinstance(layer.dot_general, Fp8DotGeneralOp)

    mlp_q = quantize_linear_layers(mlp, QuantizationConfig(targets=["0", "1"], fp8=True))
    for i, layer in enumerate(mlp_q.layers):
        if i < 2:
            assert isinstance(layer.dot_general, Fp8DotGeneralOp)
        else:
            assert not isinstance(layer.dot_general, Fp8DotGeneralOp)


def test_fp8ize_stacking():
    class Block(eqx.Module):
        up_proj: hax.nn.Linear
        down_proj: hax.nn.Linear

        @staticmethod
        def init(In, Out, key):
            up_proj = hax.nn.Linear.init(In, Out, key=key)
            down_proj = hax.nn.Linear.init(Out, In, key=key)
            return Block(up_proj, down_proj)

        def __call__(self, x):
            return self.down_proj(self.up_proj(x))

    Layer = hax.Axis("Layer", 3)

    class Tformer(eqx.Module):
        blocks: hax.nn.Stacked[Block]

        @staticmethod
        def init(In, Out, key):
            blocks = hax.nn.Stacked.init(Layer, Block)(In, Out, key=jax.random.split(key, Layer.size))
            return Tformer(blocks)

        def __call__(self, x):
            return self.blocks.fold(x)

    In = hax.Axis("In", 16)
    Out = hax.Axis("Out", 32)
    tformer = Tformer.init(In, Out, key=jrandom.PRNGKey(0))
    tformer_q = quantize_linear_layers(tformer, QuantizationConfig(fp8=True))

    # want to be sure this vmaps the dot_general to the right places
    dg = tformer_q.blocks.stacked.up_proj.dot_general
    assert isinstance(dg, Fp8DotGeneralOp)
    assert dg.input_scale.shape == (Layer.size, 1)
    assert dg.input_amax_history.shape == (Layer.size, 1024)
    dg = tformer_q.blocks.stacked.down_proj.dot_general
    assert isinstance(dg, Fp8DotGeneralOp)

    # just stack the up_proj
    tformer_q = quantize_linear_layers(tformer, QuantizationConfig(targets=["up_proj"], fp8=True))
    dg = tformer_q.blocks.stacked.up_proj.dot_general
    assert isinstance(dg, Fp8DotGeneralOp)
    dg = tformer_q.blocks.stacked.down_proj.dot_general
    assert not isinstance(dg, Fp8DotGeneralOp)
