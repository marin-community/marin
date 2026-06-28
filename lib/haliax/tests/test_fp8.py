# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

import chex
import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util
import numpy as np
import pytest
from chex import assert_trees_all_close

import haliax as hax
from haliax._src.fp8 import compute_scale
from haliax.nn import Linear
from haliax.quantization import (
    Fp8DirectDotGeneralOp,
    Fp8DotGeneralOp,
    QuantizationConfig,
    apply_updates,
    partition_for_grad_overwrite,
    quantize_linear_layers,
)

_NN_DIMS = (((1,), (0,)), ((), ()))

# Both delayed-scaling FP8 ops: the operand-QDQ op and Flax's direct-quant op. They
# share the per-tensor scale/amax-history math (E4M3 fwd, E5M2 output grad), so the
# numerics and the delayed-scaling state evolution are validated against both.
_FP8_OPS = [Fp8DotGeneralOp, Fp8DirectDotGeneralOp]


def _subjaxprs(value):
    if hasattr(value, "eqns"):
        return [value]
    inner = getattr(value, "jaxpr", None)
    if inner is not None and hasattr(inner, "eqns"):
        return [inner]
    if isinstance(value, (tuple, list)):
        return [sub for item in value for sub in _subjaxprs(item)]
    return []


def _dot_general_precisions(closed_jaxpr):
    """Every dot_general `precision` param in a (possibly nested) jaxpr, as strings, in trace order."""
    found: list[str] = []

    def walk(jaxpr):
        for eqn in jaxpr.eqns:
            if eqn.primitive.name == "dot_general":
                found.append(str(eqn.params.get("precision")))
            for param in eqn.params.values():
                for sub in _subjaxprs(param):
                    walk(sub)

    walk(closed_jaxpr.jaxpr)
    return found


def _fp8_dot_loss(forward_precision):
    op = Fp8DotGeneralOp.init(forward_precision=forward_precision)
    x = jnp.ones((8, 16), jnp.bfloat16)
    w = jnp.ones((16, 32), jnp.bfloat16)

    def loss(x, w):
        return jnp.sum(op(x, w, _NN_DIMS).astype(jnp.float32))

    return loss, x, w


@pytest.mark.parametrize("op_cls", _FP8_OPS)
def test_fp8_is_reasonable(op_cls):
    In = hax.Axis("In", 8)
    Out = hax.Axis("Out", 8)
    linear = Linear.init(In, Out, key=jrandom.PRNGKey(0), init_scale=0.1)

    fp8_linear = Linear.init(In, Out, key=jrandom.PRNGKey(0), dot_general=op_cls.init(), init_scale=0.1)

    input = hax.random.normal(jrandom.PRNGKey(3), In)
    output = linear(input)
    fp8_output = fp8_linear(input)

    assert output.shape == fp8_output.shape
    assert output.dtype == fp8_output.dtype

    assert_trees_all_close(output.array, fp8_output.array, atol=2e-2, rtol=5e-2)


# https://github.com/google/flax/blob/6f2b08e024c2fd2f8cec42a6c82408cb35412319/tests/linen/linen_test.py#L1222
@pytest.mark.parametrize("op_cls", _FP8_OPS)
def test_fp_loop(op_cls):
    key, init_key, random_key = jrandom.split(jrandom.PRNGKey(seed=123), 3)
    Batch = hax.Axis("Batch", 16)
    In = hax.Axis("In", 16)
    Out = hax.Axis("Out", 32)
    linear = Linear.init(In, Out, key=init_key, dot_general=op_cls.init())

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


def test_fp8_forward_precision_forward_only():
    # The standalone primal (fp8.py:133) governs the eval/inference forward.
    loss_d, x, w = _fp8_dot_loss(None)
    loss_h, _, _ = _fp8_dot_loss("highest")
    prec_d = _dot_general_precisions(jax.make_jaxpr(loss_d)(x, w))
    prec_h = _dot_general_precisions(jax.make_jaxpr(loss_h)(x, w))
    assert len(prec_d) == 1 and "DEFAULT" in prec_d[0]
    assert len(prec_h) == 1 and "HIGHEST" in prec_h[0]


def test_fp8_forward_precision_reaches_training_forward():
    # Regression for GFP8-012: dot_general_with_precision is a custom_jvp, so under
    # value_and_grad the forward VALUE is the jvp primal-recompute (fp8.py:146), not the
    # standalone primal (:133). forward_precision must reach it — else flipping only :133
    # would leave the training forward at DEFAULT (bf16) while passing a forward-only check.
    loss_d, x, w = _fp8_dot_loss(None)
    loss_h, _, _ = _fp8_dot_loss("highest")

    prec_d = _dot_general_precisions(jax.make_jaxpr(jax.value_and_grad(loss_d))(x, w))
    # forward value DEFAULT + two HIGHEST grad-dot tangents
    assert sum("DEFAULT" in p for p in prec_d) == 1
    assert sum("HIGHEST" in p for p in prec_d) == 2

    prec_h = _dot_general_precisions(jax.make_jaxpr(jax.value_and_grad(loss_h))(x, w))
    # forward value flipped to HIGHEST too -> no DEFAULT dot remains
    assert len(prec_h) == 3
    assert all("HIGHEST" in p for p in prec_h)


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="cuBLASLt $f8 only lowers on GPU")
def test_fp8_forward_fuses_cublas_f8_on_gpu():
    # Canary: forward_precision=HIGHEST must make the operand-QDQ forward re-fuse to a
    # $f8 cuBLASLt matmul (GFP8-012). Alarms if a future jaxlib silently regresses to bf16.
    op = Fp8DotGeneralOp.init(forward_precision="highest")
    x = jnp.ones((512, 512), jnp.bfloat16)
    w = jnp.ones((512, 512), jnp.bfloat16)
    hlo = jax.jit(lambda x, w: op(x, w, _NN_DIMS)).lower(x, w).compile().as_text()
    assert "__cublas$lt$matmul$f8" in hlo


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="cuBLASLt $f8 only lowers on GPU")
def test_fp8_direct_forward_fuses_cublas_f8_on_gpu():
    # The direct-quant op feeds genuine f8 operands into the dot, so the forward must fire
    # $f8 at DEFAULT precision — no forward_precision flip (GFP8-014). Backward fires $f8 too,
    # but the forward is the regression we care about; assert it on the forward HLO alone.
    op = Fp8DirectDotGeneralOp.init()
    x = jnp.ones((512, 512), jnp.bfloat16)
    w = jnp.ones((512, 512), jnp.bfloat16)
    hlo = jax.jit(lambda x, w: op(x, w, _NN_DIMS)).lower(x, w).compile().as_text()
    assert "__cublas$lt$matmul$f8" in hlo


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
