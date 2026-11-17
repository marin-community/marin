# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track ``mla_qk_norm_cap``: the per-vector L2 cap, and a tiny model that trains with it in place of the MLA
q/k RMS norms."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from experiments.belay.model_probe import tiny_config, tiny_mesh
from experiments.grug.fast_track.model import Transformer, l2_norm_cap


def test_cap_rescales_only_vectors_above_it():
    x = jnp.array([[3.0, 4.0], [0.3, 0.4], [6.0, 8.0]])  # norms 5, 0.5, 10

    got = np.asarray(l2_norm_cap(x, 5.0))

    np.testing.assert_allclose(got, [[3.0, 4.0], [0.3, 0.4], [3.0, 4.0]], rtol=1e-6)


def test_tiny_model_with_cap_differs_from_rms_norms_and_trains():
    base = dataclasses.replace(tiny_config(), mla_q_norm=True, mla_k_norm=True)
    capped = dataclasses.replace(base, mla_qk_norm_cap=2.0)
    with jax.set_mesh(tiny_mesh()):
        model = Transformer.init(capped, key=jax.random.key(0))
        plain = Transformer.init(base, key=jax.random.key(0))
        tokens = jax.random.randint(jax.random.key(1), (2, capped.max_seq_len), 0, capped.vocab_size)
        forward = eqx.filter_jit(lambda m, t: m(t)[0])
        assert not np.array_equal(forward(model, tokens), forward(plain, tokens))

        loss, grads = eqx.filter_jit(eqx.filter_value_and_grad(lambda m: jnp.sum(m(tokens)[0] ** 2)))(model)

    assert np.isfinite(float(loss))
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in leaves)
