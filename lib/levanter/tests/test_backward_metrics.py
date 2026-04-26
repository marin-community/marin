# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for scan-safe backward metric sinks."""

from dataclasses import dataclass
from typing import Optional, Type

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

import haliax as hax
from haliax import Axis, NamedArray

from levanter.backward_metrics import (
    empty_sink,
    grad_rms_from_sink,
    make_backward_observer,
    observe_grad_sumsq,
)
from levanter.layers.attention import AttentionMask
from levanter.models.lm_model import LmConfig, LmExample, LmHeadModel


def test_observe_basic():
    """custom_vjp backward injects gradient stats into sink cotangent."""
    sink = empty_sink("grad_sumsq", "grad_count")

    def f(x, sink):
        y, sink = observe_grad_sumsq(x, sink)
        return jnp.sum(y**2)

    grad_fn = jax.value_and_grad(f, argnums=(0, 1))
    x = jnp.array([1.0, 2.0, 3.0])
    loss, (g_x, g_sink) = grad_fn(x, sink)

    # loss = 1 + 4 + 9 = 14
    assert jnp.allclose(loss, 14.0)

    # g_x = d/dx sum(x^2) = 2x
    assert jnp.allclose(g_x, 2 * x)

    # backward metrics: sumsq of gradient = sum((2x)^2) = 4+16+36 = 56
    assert jnp.allclose(g_sink["grad_sumsq"], jnp.sum((2 * x) ** 2))
    assert jnp.allclose(g_sink["grad_count"], 3.0)


def test_observe_jit():
    """Backward metric sinks work inside jit."""
    sink = empty_sink("grad_sumsq", "grad_count")

    @jax.jit
    def f(x, sink):
        grad_fn = jax.value_and_grad(
            lambda x_, s_: jnp.sum(observe_grad_sumsq(x_, s_)[0] ** 2),
            argnums=(0, 1),
        )
        return grad_fn(x, sink)

    x = jnp.array([3.0, 4.0])
    loss, (g_x, g_sink) = f(x, sink)

    assert jnp.allclose(loss, 25.0)
    assert g_sink["grad_sumsq"] > 0
    assert jnp.allclose(g_sink["grad_count"], 2.0)


def test_observe_scan():
    """Backward metric sinks accumulate correctly under jax.lax.scan."""
    sink = empty_sink("grad_sumsq", "grad_count")

    def step(carry, x_i):
        param, sink = carry
        y = param * x_i
        y, sink = observe_grad_sumsq(y, sink)
        loss = jnp.sum(y**2)
        return (param, sink), loss

    def total_loss(param, sink, xs):
        (_, _), losses = jax.lax.scan(step, (param, sink), xs)
        return jnp.sum(losses)

    param = jnp.array(2.0)
    xs = jnp.array([1.0, 2.0, 3.0])

    grad_fn = jax.value_and_grad(total_loss, argnums=(0, 1))
    loss, (g_param, g_sink) = grad_fn(param, sink, xs)

    # loss = sum_i (param * x_i)^2 = 4*(1+4+9) = 56
    assert jnp.allclose(loss, 56.0)

    # Each step observes gradient of y = param * x_i w.r.t. loss_i = y^2
    # d(loss_i)/dy = 2y = 2*param*x_i, so grad sumsq per step = (2*param*x_i)^2
    # Total grad_sumsq across 3 steps = sum of per-step sumsq
    expected_sumsq = sum((2.0 * 2.0 * xi) ** 2 for xi in [1.0, 2.0, 3.0])
    assert jnp.allclose(g_sink["grad_sumsq"], expected_sumsq)
    assert jnp.allclose(g_sink["grad_count"], 3.0)  # one scalar per step


def test_observe_multiple_points():
    """Multiple observation points accumulate into the same sink."""
    sink = empty_sink("grad_sumsq", "grad_count")

    def f(x, sink):
        y, sink = observe_grad_sumsq(x, sink)
        z, sink = observe_grad_sumsq(y * 2, sink)
        return jnp.sum(z**2)

    grad_fn = jax.value_and_grad(f, argnums=(0, 1))
    x = jnp.array([1.0])
    _, (_, g_sink) = grad_fn(x, sink)

    # Two observation points: both contribute to the sink
    assert g_sink["grad_sumsq"] > 0
    # count = 1 (first observe) + 1 (second observe) = 2
    assert jnp.allclose(g_sink["grad_count"], 2.0)


def test_grad_rms_from_sink():
    """grad_rms_from_sink computes RMS correctly."""
    sink_grad = {"grad_sumsq": jnp.array(100.0), "grad_count": jnp.array(4.0)}
    rms = grad_rms_from_sink(sink_grad)
    assert jnp.allclose(rms, 5.0)  # sqrt(100/4) = 5


def test_make_backward_observer_custom_stats():
    """make_backward_observer works with custom stat functions."""

    def max_abs_stats(g):
        leaves = jax.tree.leaves(g)
        max_val = jnp.float32(0.0)
        for leaf in leaves:
            max_val = jnp.maximum(max_val, jnp.max(jnp.abs(leaf)))
        return {"grad_max_abs": max_val}

    observe_max = make_backward_observer(max_abs_stats)
    sink = {"grad_max_abs": jnp.zeros(())}

    def f(x, sink):
        y, sink = observe_max(x, sink)
        return jnp.sum(y**2)

    grad_fn = jax.value_and_grad(f, argnums=(0, 1))
    x = jnp.array([1.0, -3.0, 2.0])
    _, (_, g_sink) = grad_fn(x, sink)

    # gradient = 2*x, max abs = max(|2|, |-6|, |4|) = 6
    assert jnp.allclose(g_sink["grad_max_abs"], 6.0)


# ---------------------------------------------------------------------------
# Real pretraining callsite: LmHeadModel.compute_next_token_loss
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TinyLmConfig(LmConfig["_TinyLmHeadModel"]):
    max_seq_len: int = 8
    embed_dim: int = 16

    @property
    def model_type(self) -> Type["_TinyLmHeadModel"]:
        return _TinyLmHeadModel

    @property
    def Embed(self) -> Axis:
        return Axis("embed", self.embed_dim)


class _TinyLmHeadModel(LmHeadModel[_TinyLmConfig]):
    _config: _TinyLmConfig = eqx.field(static=True)
    _Vocab: Axis = eqx.field(static=True)
    embed_weight: NamedArray
    lm_head: NamedArray

    @property
    def config(self) -> _TinyLmConfig:
        return self._config

    @property
    def Vocab(self) -> Axis:
        return self._Vocab

    @classmethod
    def init(cls, Vocab: Axis, config: _TinyLmConfig, *, key: PRNGKeyArray) -> "_TinyLmHeadModel":
        k1, k2 = jax.random.split(key)
        embed = hax.random.normal(k1, (Vocab, config.Embed), dtype=jnp.float32)
        head = hax.random.normal(k2, (config.Embed, Vocab), dtype=jnp.float32)
        return cls(config, Vocab, embed, head)

    def activations(self, input_ids, attn_mask=None, *, key=None, pos_ids=None):
        del attn_mask, key, pos_ids
        return self.embed_weight.take(self.Vocab, input_ids)

    def get_lm_head(self):
        return self.lm_head

    def resize_vocab(self, new_size: int, key: Optional[PRNGKeyArray] = None) -> "_TinyLmHeadModel":
        raise NotImplementedError


def test_backward_metrics_lm_loss():
    """Backward metric sinks work through compute_next_token_loss."""
    Vocab = Axis("vocab", 32)
    cfg = _TinyLmConfig(max_seq_len=8, embed_dim=16)
    model = _TinyLmHeadModel.init(Vocab, cfg, key=jax.random.PRNGKey(0))

    Batch = Axis("batch", 2)
    Pos = cfg.max_Pos.resize(8)
    tokens = hax.random.randint(jax.random.PRNGKey(1), (Batch, Pos), 0, Vocab.size)
    loss_weight = hax.ones((Batch, Pos), dtype=jnp.float32)
    example = LmExample(tokens=tokens, loss_weight=loss_weight, attn_mask=AttentionMask.causal())

    sink = empty_sink("grad_sumsq", "grad_count")

    def loss_fn(model, sink):
        return model.compute_next_token_loss(example, backward_sink=sink).scalar()

    grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1))
    loss, (model_grad, backward_metrics) = grad_fn(model, sink)

    # Loss should be a finite scalar
    assert jnp.isfinite(loss)

    # Backward metrics should have non-zero sumsq and correct count
    assert backward_metrics["grad_sumsq"] > 0
    # count = Batch * Pos * Embed = 2 * 8 * 16 = 256
    expected_count = Batch.size * Pos.size * cfg.embed_dim
    assert jnp.allclose(backward_metrics["grad_count"], float(expected_count))

    # RMS should be finite and positive
    rms = grad_rms_from_sink(backward_metrics)
    assert jnp.isfinite(rms)
    assert rms > 0


def test_backward_metrics_lm_loss_scan():
    """Backward metric sinks accumulate across scan iterations with LM loss."""
    Vocab = Axis("vocab", 32)
    cfg = _TinyLmConfig(max_seq_len=8, embed_dim=16)
    model = _TinyLmHeadModel.init(Vocab, cfg, key=jax.random.PRNGKey(0))

    Batch = Axis("batch", 2)
    Pos = cfg.max_Pos.resize(8)
    num_steps = 2

    # Build a list of token arrays and stack along a raw (non-Named) axis for scan
    keys = jax.random.split(jax.random.PRNGKey(1), num_steps)
    all_tokens = [hax.random.randint(k, (Batch, Pos), 0, Vocab.size) for k in keys]
    # Stack the raw arrays: shape (num_steps, batch, pos)
    stacked = jnp.stack([t.array for t in all_tokens], axis=0)

    sink = empty_sink("grad_sumsq", "grad_count")

    def step(carry, raw_tokens_i):
        model, sink = carry
        tokens_i = hax.named(raw_tokens_i, (Batch, Pos))
        loss_weight = hax.ones((Batch, Pos), dtype=jnp.float32)
        example = LmExample(tokens=tokens_i, loss_weight=loss_weight, attn_mask=AttentionMask.causal())
        loss = model.compute_next_token_loss(example, backward_sink=sink).scalar()
        return (model, sink), loss

    def total_loss(model, sink, all_raw_tokens):
        (_, _), losses = jax.lax.scan(step, (model, sink), all_raw_tokens)
        return jnp.sum(losses)

    grad_fn = jax.value_and_grad(total_loss, argnums=(0, 1))
    loss, (_, backward_metrics) = grad_fn(model, sink, stacked)

    assert jnp.isfinite(loss)
    assert backward_metrics["grad_sumsq"] > 0
    # 2 scan steps * Batch * Pos * Embed = 2 * 2 * 8 * 16 = 512
    expected_count = num_steps * Batch.size * Pos.size * cfg.embed_dim
    assert jnp.allclose(backward_metrics["grad_count"], float(expected_count))
