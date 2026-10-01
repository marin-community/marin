# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``bigram_gate_act`` / ``bigram_gate_logit_scale``: content-gate variants that keep the gate's start."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import experiments.grug.fast_track.test_kda_local as t
from experiments.grug.fast_track.model import ContentGateActivation, _content_gate

_D, _R = 16, 4


def _gate(*args, **kwargs):
    with jax.set_mesh(t._mesh()):
        return _content_gate(*args, **kwargs)


def _inputs():
    k = jax.random.split(jax.random.PRNGKey(0), 4)
    hidden, source = (jax.random.normal(key, (2, 3, _D)) for key in k[:2])
    a = jax.random.normal(k[2], (_D, _R))
    b = jax.random.normal(k[3], (_R, _D))
    return hidden, source, a, b


@pytest.mark.parametrize("act", list(ContentGateActivation))
def test_zero_up_projection_starts_every_variant_at_sigmoid_bias(act):
    hidden, source, a, _ = _inputs()
    _, gate = _gate(hidden, source, None, jnp.array(2.0), a, jnp.zeros((_R, _D)), act=act, logit_scale=0.1)
    np.testing.assert_allclose(np.asarray(gate), float(jax.nn.sigmoid(2.0)), rtol=1e-6)


def test_activation_and_scale_change_the_logits_as_written():
    hidden, source, a, b = _inputs()
    low = (hidden * source) @ a
    for act, fn in ((ContentGateActivation.SILU, jax.nn.silu), (ContentGateActivation.GELU, jax.nn.gelu)):
        _, gate = _gate(hidden, source, None, jnp.array(0.5), a, b, act=act, logit_scale=0.1)
        np.testing.assert_allclose(np.asarray(gate), np.asarray(jax.nn.sigmoid(0.1 * (fn(low) @ b) + 0.5)), rtol=1e-5)


def test_activation_needs_the_low_rank_gate():
    hidden, source, _, _ = _inputs()
    with pytest.raises(ValueError):
        _gate(hidden, source, jnp.zeros(_D), jnp.array(2.0), None, None, act=ContentGateActivation.SILU)
