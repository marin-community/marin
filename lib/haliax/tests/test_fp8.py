# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


import chex
import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util
import numpy as np
from chex import assert_trees_all_close

import haliax as hax
from haliax._src.fp8 import compute_scale
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

    fp8_linear = Linear.init(
        In, Out, key=jrandom.PRNGKey(0), dot_general=hax.quantization.Fp8DotGeneralOp.init(), init_scale=0.1
    )

    input = hax.random.normal(jrandom.PRNGKey(3), In)
    output = linear(input)
    fp8_output = fp8_linear(input)

    assert output.shape == fp8_output.shape
    assert output.dtype == fp8_output.dtype

    assert_trees_all_close(output.array, fp8_output.array, atol=2e-2, rtol=5e-2)


# https://github.com/google/flax/blob/6f2b08e024c2fd2f8cec42a6c82408cb35412319/tests/linen/linen_test.py#L1222
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
