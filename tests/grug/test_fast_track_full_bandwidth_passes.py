# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track ``full_bandwidth_passes``: pass 1 is the plain model, the training loss adds the later pass and evals
score it alone, the feedback reads only the previous token of the same document, the QB threshold pools every
pass's margins, and W_fuse leaves zero on its plain-Muon group."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard
from levanter.grug.attention import AttentionMask

from experiments.belay.model_probe import SEQ, tiny_config, tiny_mesh
from experiments.grug.fast_track.model import Transformer
from experiments.grug.fast_track.optimizer import FULL_BANDWIDTH_MUON_GROUP, GrugMoeMuonHConfig
from experiments.grug.fast_track.qb_balancing import qb_beta_hist, qb_beta_hist_pooled

_BATCH_AXES = ("replica_dcn", "data", "expert")


def _models():
    """The flag-on model with a random W_fuse, and the flag-off model with the same shared weights."""
    config = dataclasses.replace(tiny_config(), full_bandwidth_passes=2)
    model = Transformer.init(config, key=jax.random.key(0))
    fuse = 0.5 * jax.random.normal(jax.random.key(5), model.full_bandwidth_fuse.shape) / np.sqrt(config.hidden_dim)
    model = eqx.tree_at(lambda m: m.full_bandwidth_fuse, model, fuse)
    plain = Transformer.init(dataclasses.replace(config, full_bandwidth_passes=0), key=jax.random.key(0))
    return model, plain


def test_pass_one_is_the_plain_model_and_training_adds_the_later_pass():
    with jax.set_mesh(tiny_mesh()):
        model, plain = _models()
        tokens = jax.random.randint(jax.random.key(1), (2, SEQ), 0, model.config.vocab_size)
        weight = jnp.ones(tokens.shape, jnp.float32)

        def train_loss(m):
            return m.next_token_loss(tokens, weight, return_router_metrics=True, train_terms=True)

        loss, metrics = eqx.filter_jit(train_loss)(model)
        eval_loss = eqx.filter_jit(lambda m: m.next_token_loss(tokens, weight))(model)
        plain_loss = eqx.filter_jit(lambda m: m.next_token_loss(tokens, weight))(plain)

    first = float(metrics["train/full_bandwidth/pass1_cross_entropy"])
    second = float(metrics["train/full_bandwidth/pass2_cross_entropy"])
    np.testing.assert_allclose(first, float(plain_loss), rtol=1e-6)
    np.testing.assert_allclose(float(loss), first + second, rtol=1e-6)
    np.testing.assert_allclose(float(eval_loss), second, rtol=1e-6)
    assert abs(second - first) > 1e-4


def test_feedback_reads_only_earlier_tokens_of_the_same_document():
    half = SEQ // 2
    segments = jnp.asarray(np.repeat([0, 1], [half, half])[None].repeat(2, 0), jnp.int32)
    with jax.set_mesh(tiny_mesh()):
        model, _ = _models()
        vocab = model.config.vocab_size
        tokens = jax.random.randint(jax.random.key(1), (2, SEQ), 0, vocab)
        forward = eqx.filter_jit(lambda m, t: m(t, AttentionMask.causal().with_segment_ids(segments))[0])
        base = np.asarray(forward(model, tokens), np.float32)
        # New tokens in the second half of document 2, then in all of document 1.
        later = tokens.at[:, half + half // 2 :].set(jax.random.randint(jax.random.key(2), (2, half // 2), 0, vocab))
        earlier_doc = tokens.at[:, :half].set(jax.random.randint(jax.random.key(3), (2, half), 0, vocab))
        moved_later = np.asarray(forward(model, later), np.float32)
        moved_doc = np.asarray(forward(model, earlier_doc), np.float32)

    np.testing.assert_allclose(moved_later[:, : half + half // 2], base[:, : half + half // 2], atol=1e-5)
    np.testing.assert_allclose(moved_doc[:, half:], base[:, half:], atol=1e-5)
    assert np.abs(moved_doc[:, :half] - base[:, :half]).max() > 1e-3


def test_qb_threshold_pools_the_margins_of_every_pass():
    margins = jax.random.normal(jax.random.key(0), (2, 512, 8))
    margins = margins.at[1].multiply(3.0)  # the second pass spans a wider range
    with jax.set_mesh(tiny_mesh()):
        mesh = jax.sharding.get_abstract_mesh()
        settings = dict(target_share=2 / 8, num_experts=8, n_bins=1000, batch_axes=_BATCH_AXES)
        pooled = qb_beta_hist_pooled(margins, mesh, **settings)
        concatenated, _, _ = qb_beta_hist(reshard(margins.reshape(-1, 8), P(_BATCH_AXES, None)), mesh, **settings)
        first_pass, _, _ = qb_beta_hist(reshard(margins[0], P(_BATCH_AXES, None)), mesh, **settings)

    np.testing.assert_allclose(pooled, concatenated, rtol=1e-6)
    assert np.abs(np.asarray(pooled) - np.asarray(first_pass)).max() > 0.1


def test_fuse_matrix_leaves_zero_on_plain_muon_and_flag_off_keeps_its_groups():
    with jax.set_mesh(tiny_mesh()):
        model, _ = _models()
    optimizer_config = GrugMoeMuonHConfig(learning_rate=0.02, warmup=0.0, lr_schedule="constant")
    assert optimizer_config.create_mask(eqx.filter(model, eqx.is_array)).full_bandwidth_fuse == FULL_BANDWIDTH_MUON_GROUP

    plain = {"stacked_blocks": {"stacked": {"attn": {"w_o": jax.random.normal(jax.random.key(0), (2, 16, 16))}}}}
    params = {**plain, "full_bandwidth_fuse": jnp.zeros((16, 16))}
    grads = jax.tree.map(lambda p: jax.random.normal(jax.random.key(1), p.shape), params)
    optimizer = optimizer_config.build(num_train_steps=100)
    state = optimizer.init(params)
    updates, _ = optimizer.update(grads, state, params)
    plain_state = optimizer.init(plain)

    assert float(jnp.linalg.norm(updates["full_bandwidth_fuse"])) > 0
    on_groups = set(state.inner_state.inner_states)
    assert on_groups - set(plain_state.inner_state.inner_states) == {FULL_BANDWIDTH_MUON_GROUP}
