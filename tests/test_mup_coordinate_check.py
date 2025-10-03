# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Coordinate check for µP modules built on Haliax primitives."""

from __future__ import annotations

import dataclasses
from typing import Any, Iterable

import equinox as eqx
import jax
import jax.random as jrandom

import haliax as hax
from haliax import Axis, NamedArray
from haliax.nn import Linear, activations
from haliax.nn.mup import InputLinearMup, HiddenLinearMup, OutputLinearMup


class TinyMLP(eqx.Module):
    """Minimal 2-hidden-layer MLP composed of Haliax Linear variants."""

    first: Linear
    second: Linear
    third: Linear

    @staticmethod
    def init(width: int, *, key: jax.Array, use_mup: bool) -> TinyMLP:
        in_axis = Axis("in", 2)
        hidden = Axis("hidden", width)
        hidden2 = hidden.alias("hidden2")
        out_axis = Axis("out", 1)

        k1, k2, k3 = jrandom.split(key, 3)

        if use_mup:
            first = Linear.init((in_axis,), hidden, key=k1, reparam_cls=InputLinearMup)
            second = Linear.init(hidden, hidden2, key=k2, reparam_cls=HiddenLinearMup)
            third = Linear.init(hidden2, (out_axis,), key=k3, reparam_cls=OutputLinearMup)
        else:
            first = Linear.init((in_axis,), hidden, key=k1)
            second = Linear.init(hidden, hidden2, key=k2)
            third = Linear.init(hidden2, (out_axis,), key=k3)

        return TinyMLP(first=first, second=second, third=third)

    def __call__(self, x: NamedArray) -> NamedArray:
        h = activations.relu(self.first(x))
        h = activations.relu(self.second(h))
        return self.third(h)


def _loss_fn(params: TinyMLP, x: NamedArray, y: NamedArray) -> jax.Array:
    preds = params(x)
    diff = preds - y
    return hax.mean(diff * diff).scalar()


_loss_and_grad = eqx.filter_jit(eqx.filter_value_and_grad(_loss_fn))
_loss_value = jax.jit(_loss_fn)


def _apply_sgd(module: TinyMLP, grads: TinyMLP, *, base_lr: float, use_mup: bool) -> TinyMLP:
    def update_linear(layer: Linear, grad_layer: Linear) -> Linear:
        lr_scale = layer.reparam.lr_scale
        new_weight = layer.weight - (base_lr * lr_scale) * grad_layer.weight
        if layer.bias is None or grad_layer.bias is None:
            new_bias = layer.bias
        else:
            new_bias = layer.bias - base_lr * grad_layer.bias
        return dataclasses.replace(layer, weight=new_weight, bias=new_bias)

    return TinyMLP(
        first=update_linear(module.first, grads.first),
        second=update_linear(module.second, grads.second),
        third=update_linear(module.third, grads.third),
    )


def _make_dataset(key: jax.Array, *, n_points: int = 2048) -> tuple[NamedArray, NamedArray]:
    data_axis = Axis("data", n_points)
    feature_axis = Axis("in", 2)
    out_axis = Axis("out", 1)

    xy = jrandom.uniform(key, (n_points, 2), minval=-1.0, maxval=1.0)
    inputs = hax.named(xy, (data_axis, feature_axis))
    targets = hax.named(xy[:, :1], (data_axis, out_axis))
    return inputs, targets


def _run_once(
    key: jax.Array,
    *,
    width: int,
    use_mup: bool,
    steps: int = 120,
    batch_size: int = 256,
    base_lr: float = 3e-3,
) -> float:
    data_key, model_key = jrandom.split(key)
    inputs, targets = _make_dataset(data_key)
    params = TinyMLP.init(width, key=model_key, use_mup=use_mup)

    def train_step(state: TinyMLP, xb: NamedArray, yb: NamedArray):
        loss, grads = _loss_and_grad(state, xb, yb)
        new_state = _apply_sgd(state, grads, base_lr=base_lr, use_mup=use_mup)
        return new_state, loss

    data_axis = inputs.axes[0]
    n = data_axis.size

    state = params
    for t in range(steps):
        start = (t * batch_size) % n
        end = start + batch_size
        batch_idx = (data_axis, slice(start, end))
        xb = inputs[batch_idx]
        yb = targets[batch_idx]
        state, _ = train_step(state, xb, yb)

    final_loss = _loss_value(state, inputs, targets)
    return float(final_loss)


def _span(values: Iterable[float]) -> float:
    seq = list(values)
    return max(seq) - min(seq)


def coord_check(
    widths: tuple[int, ...] = (32, 128, 512),
    *,
    steps: int = 120,
    base_lr: float = 3e-3,
) -> dict[str, Any]:
    seed = 0
    keys = jrandom.split(jrandom.PRNGKey(seed), len(widths))
    mup_losses = [
        _run_once(key, width=width, use_mup=True, steps=steps, base_lr=base_lr) for key, width in zip(keys, widths)
    ]
    ctrl_losses = [
        _run_once(key, width=width, use_mup=False, steps=steps, base_lr=base_lr) for key, width in zip(keys, widths)
    ]

    return {
        "widths": list(widths),
        "mup_losses": mup_losses,
        "ctrl_losses": ctrl_losses,
        "mup_span": _span(mup_losses),
        "ctrl_span": _span(ctrl_losses),
    }


def test_mup_coordinate_check_is_width_invariant():
    result = coord_check(widths=(32, 128, 512), steps=120, base_lr=3e-3)
    mup_span = result["mup_span"]
    ctrl_span = result["ctrl_span"]

    if ctrl_span < 1e-5:
        assert mup_span <= ctrl_span + 1e-6, f"μP not at least as invariant: {result}"
    else:
        assert mup_span <= 0.6 * ctrl_span, f"μP did not improve width invariance enough.\n{result}"
