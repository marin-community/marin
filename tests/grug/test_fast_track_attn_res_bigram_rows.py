# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""fast_track ``attn_res_bigram_rows``: a hashed (previous, current) token-pair table read as an extra AttnRes
source, trained in the ``embed2`` optimizer group."""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh
from levanter.grug.attention import AttentionMask

from experiments.grug.fast_track.model import GrugModelConfig, LocalMixer, Transformer, _bigram_hash_ids
from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig

_HIDDEN = 64
_SEQ = 128
_ROWS = 4096


def _config() -> GrugModelConfig:
    return GrugModelConfig(
        vocab_size=128,
        hidden_dim=_HIDDEN,
        intermediate_dim=_HIDDEN // 2,
        shared_expert_intermediate_dim=_HIDDEN // 2,
        num_shared_experts=1,
        num_experts=4,
        num_experts_per_token=2,
        latent_dim=_HIDDEN // 2,
        num_layers=4,
        num_heads=2,
        num_kv_heads=2,
        local_kv_heads=None,
        global_kv_heads=None,
        head_dim=16,
        max_seq_len=_SEQ,
        sliding_window=_SEQ,
        global_every=2,
        capacity_factor=8.0,
        pooled_transport_capacity_factor=8.0,
        initializer_std=0.5 / math.sqrt(_HIDDEN),
        qk_norm=False,
        sconv_sites=("attn", "mlp"),
        mla=True,
        mla_kv_latent_dim=32,
        inkling_relpos=True,
        rel_dim=4,
        rel_extent=8,
        attn_res=True,
        attn_res_num_blocks=2,
        local_mixer=LocalMixer.KDA,
        attn_res_bigram_rows=_ROWS,
    )


def _mesh() -> Mesh:
    return Mesh(
        np.array(jax.devices()[:1], dtype=object).reshape((1, 1, 1, 1)),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )


def test_bigram_ids_read_the_previous_token_within_a_document():
    tokens = jnp.array([[5, 7, 7, 9, 5, 7]])
    segments = jnp.array([[0, 0, 0, 0, 1, 1]])
    ids = np.asarray(_bigram_hash_ids(tokens, segments, _ROWS))

    assert ((ids >= 0) & (ids < _ROWS)).all()
    # (5, 7) at positions 1 and 5; the document starts (positions 0 and 4) both read the sentinel before 5.
    assert ids[0, 1] == ids[0, 5]
    assert ids[0, 0] == ids[0, 4]
    # The same token after a different previous token reads a different row.
    assert ids[0, 1] != ids[0, 2]


def test_bigram_source_feeds_the_model_and_trains_in_embed2():
    cfg = _config()
    tokens = jax.random.randint(jax.random.key(1), (2, _SEQ), 0, cfg.vocab_size)
    segment_ids = jnp.asarray(np.repeat([0, 1], [_SEQ // 2, _SEQ // 2])[None].repeat(2, 0), jnp.int32)
    mask = AttentionMask(is_causal=True, segment_ids=(segment_ids, segment_ids))
    cotangent = jax.random.normal(jax.random.key(3), (2, _SEQ, _HIDDEN), jnp.float32)
    with jax.set_mesh(_mesh()):
        model = Transformer.init(cfg, key=jax.random.key(0))

        def loss_fn(m):
            return jnp.sum(m(tokens, mask)[0].astype(jnp.float32) * cotangent)

        loss, grads = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn))(model)
        ids = np.asarray(_bigram_hash_ids(tokens, segment_ids, _ROWS))

    assert np.isfinite(float(loss))
    table_grad = np.asarray(grads.bigram_embed)
    # Exactly the rows the batch's bigrams hash to receive gradient.
    touched = np.zeros(_ROWS, bool)
    touched[ids.ravel()] = True
    row_norms = np.linalg.norm(table_grad, axis=-1)
    assert (row_norms[touched] > 0).all()
    assert (row_norms[~touched] == 0).all()
    assert float(jnp.linalg.norm(grads.bigram_norm.weight)) > 0

    labels = GrugMoeMuonHConfig().create_mask(eqx.filter(model, eqx.is_array))
    assert labels.bigram_embed == "embed2"
    assert labels.bigram_norm.weight == "adam"
