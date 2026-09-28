# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V3 MTP (``mtp_mode``): t+2 targets stay inside their packed document, and evals ignore the MTP head."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from levanter.grug.attention import AttentionMask

from experiments.grug.fast_track.model import MtpMode, _mtp_targets
from experiments.grug.fast_track.test_kda_local import _SEQ, _VOCAB, _model


def test_mtp_targets_are_same_document_t_plus_2():
    # Documents [10..14] and [20..25], then two padding positions (segment -1).
    tokens = jnp.asarray([[10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 25, 0, 0]], jnp.int32)
    segments = jnp.asarray([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, -1, -1]], jnp.int32)
    loss_weight = jnp.ones(tokens.shape).at[:, -1].set(0) * (jnp.roll(segments, -1) >= 0)
    labels, weight = _mtp_targets(tokens, loss_weight, segments)
    assert np.asarray(weight[0]).tolist() == [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
    assert np.asarray(labels[0, :3]).tolist() == [12, 13, 14]
    assert np.asarray(labels[0, 5:9]).tolist() == [22, 23, 24, 25]


def test_mtp_is_training_only():
    tokens = jax.random.randint(jax.random.PRNGKey(2), (2, _SEQ), 0, _VOCAB)
    weight = jnp.ones(tokens.shape, jnp.float32).at[:, -1].set(0)
    segments = jnp.asarray(np.repeat([0, 1], [11, 13])[None].repeat(2, 0), jnp.int32)
    mask = AttentionMask(is_causal=True, segment_ids=(segments, segments))
    route_key = jax.random.PRNGKey(7)
    eval_losses = []
    for mode in (MtpMode.OFF, MtpMode.DEEPSEEK):
        mesh, model = _model(mtp_mode=mode, mtp_position_frac=0.5)
        with jax.set_mesh(mesh):
            if model.mtp is not None:
                # Evals must not depend on the MTP head.
                model = eqx.tree_at(lambda m: m.mtp.w_proj, model, model.mtp.w_proj * 5.0)
                loss, metrics = eqx.filter_jit(
                    lambda m: m.next_token_loss(
                        tokens, weight, mask=mask, train_terms=True, route_key=route_key, return_router_metrics=True
                    )
                )(model)
                expected = metrics["train/cross_entropy_loss"] + model.config.mtp_weight * metrics["train/aux/mtp_loss"]
                np.testing.assert_allclose(float(loss), float(expected), rtol=1e-5)
            eval_losses.append(float(eqx.filter_jit(lambda m: m.next_token_loss(tokens, weight, mask=mask))(model)))
    assert eval_losses[0] == eval_losses[1]
