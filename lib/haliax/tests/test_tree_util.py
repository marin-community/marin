# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp

import haliax as hax
import haliax.tree_util as htu
from haliax import Axis


def test_resize_axis():
    A, B, C = hax.make_axes(A=10, B=20, C=30)

    class Module(eqx.Module):
        name1: hax.NamedArray
        name2: hax.NamedArray
        name3: hax.NamedArray

    module = Module(
        name1=hax.random.normal(jax.random.PRNGKey(0), (B, A, C)),
        name2=hax.zeros((B, C)),
        name3=hax.zeros((Axis("A", 20),)),
    )

    NewA = A.resize(15)

    module2 = htu.resize_axis(module, "A", 15, key=jax.random.PRNGKey(1))

    assert module2.name1.axes == (B, NewA, C)
    assert module2.name2.axes == (B, C)
    assert module2.name3.axes == (NewA,)

    # we don't mess with the mean or std of the original array too much
    assert jnp.allclose(module2.name1.mean().array, module.name1.mean().array, rtol=1e-1, atol=1e-2)


def test_scan_aware_tree_map():
    Embed = hax.Axis("embed", 10)
    Up = hax.Axis("up", 20)
    Block = hax.Axis("block", 4)

    class Module(eqx.Module):
        up: hax.nn.Linear
        down: hax.nn.Linear

        def __call__(self, x, *, key):
            return self.down(self.up(x), key=key)

        @staticmethod
        def init(layer_idx, *, key):
            k1, k2 = jax.random.split(key)
            up = hax.nn.Linear.init(Embed, Up, key=k1)
            down = hax.nn.Linear.init(Up, Embed, key=k2)

            up = dataclasses.replace(up, weight=up.weight + layer_idx)  # type: ignore
            down = dataclasses.replace(down, weight=down.weight + layer_idx)  # type: ignore

            return Module(up=up, down=down)

    class Model(eqx.Module):
        layers: hax.nn.Stacked[eqx.Module]

        def __call__(self, x, *, key):
            return self.layers.fold(x, key=jax.random.split(key, self.layers.Block.size))

        @staticmethod
        def init(Layers, *, key):
            stack = hax.nn.Stacked.init(Layers, Module)(
                layer_idx=hax.arange(Layers), key=jax.random.split(key, Layers.size)
            )
            return Model(layers=stack)

    model = Model.init(Block, key=jax.random.PRNGKey(0))

    def transform_linear(x):
        if not isinstance(x, hax.nn.Linear):
            return x

        # do something that distinguishes doing weights jointly from independently
        new_weight = x.weight - hax.mean(x.weight)
        return dataclasses.replace(x, weight=new_weight)  # type: ignore

    model2 = htu.scan_aware_tree_map(transform_linear, model, is_leaf=lambda x: isinstance(x, hax.nn.Linear))
    model3 = htu.tree_map(transform_linear, model, is_leaf=lambda x: isinstance(x, hax.nn.Linear))

    assert hax.all(model2.layers.stacked.up.weight != model3.layers.stacked.up.weight)
