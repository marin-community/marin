# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import glob

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from levanter.grug.attention import AttentionMask
from levanter.grug.attention._inkling_relpos import REL_BIAS_BLOCK, dense_rel_bias

import experiments.grug.fast_track.test_kda_local as t
from experiments.grug.fast_track.fact_probe import (
    RAW,
    FactProbeInput,
    FactProbeWriter,
    count_text_patterns,
    summarize_attention,
)
from experiments.grug.fast_track.model import (
    ATTN_PROBE_KEYS,
    ATTN_PROBE_STAT,
    PROBE_STAT,
    ForwardProbe,
    _probe_attention,
    forward_probe,
)

_EOS = 1


def _probe(tokens: np.ndarray, spans: list[tuple[int, int, int]]) -> FactProbeInput:
    row, start, end = (np.asarray(column) for column in zip(*spans, strict=True))
    return FactProbeInput(
        tokens=tokens.astype(np.int32),
        span_row=row,
        span_start=start,
        span_end=end,
        attn_queries=np.zeros((0, 2)),
        attn_full_queries=0,
        attn_key_sets=np.zeros((0, 0)),
        spot=np.zeros(0),
        spot_token_ids=np.zeros(0),
    )


def test_span_positions_are_the_predictions_of_its_tokens():
    probe = _probe(np.zeros((2, 8)), [(1, 3, 6), (0, 1, 2)])
    np.testing.assert_array_equal(probe.positions(), [[1, 2], [1, 3], [1, 4], [0, 0]])
    with pytest.raises(ValueError):
        _probe(np.zeros((2, 8)), [(0, 0, 3)])


def test_probe_rows_are_masked_by_document_like_the_loader():
    tokens = np.array([[0, 5, 6, _EOS, 0, 7, _EOS, 0]])
    example = _probe(tokens, [(0, 5, 6)]).example(_EOS)
    segments = np.asarray(example.attn_mask.segment_ids[0])
    np.testing.assert_array_equal(segments[0], [0, 0, 0, 0, 1, 1, 1, 2])
    np.testing.assert_array_equal(np.asarray(example.loss_weight)[0], [1, 1, 1, 1, 1, 1, 1, 0])


def test_writer_writes_full_chunks_then_the_rest(tmp_path):
    writer = FactProbeWriter(str(tmp_path), chunk_size=2)
    for step in (1, 2, 3):
        writer.add(
            RAW, step, np.full(3, step), np.zeros((3, 5)), np.zeros((3, 5)), np.zeros(4), {"a": np.ones((1, 2, 3))}
        )
    assert len(glob.glob(str(tmp_path / "*.npz"))) == 1
    writer.flush_all()
    chunks = sorted(glob.glob(str(tmp_path / "fact_probe_raw_*.npz")))
    np.testing.assert_array_equal(np.load(chunks[0])["steps"], [1, 2])
    np.testing.assert_array_equal(np.load(chunks[1])["loss"], [[3, 3, 3]])
    assert np.load(chunks[0])["probe/a"].shape == (2, 1, 2, 3)


def test_position_predictions_match_the_training_loss():
    mesh, model = t._model(logit_soft_cap=2.0)
    tokens = jax.random.randint(jax.random.PRNGKey(4), (2, t._SEQ), 0, t._VOCAB)
    positions = jnp.asarray([[0, 0], [0, 5], [1, 11], [1, t._SEQ - 2]], jnp.int32)
    with jax.set_mesh(mesh):
        per_token = jax.jit(lambda m, x: m.next_token_loss(x, jnp.ones(x.shape), reduction="none"))(model, tokens)
        loss, top_ids, top_probs, attention = jax.jit(lambda m, x, p: m.position_predictions(x, p, k=3))(
            model, tokens, positions
        )
    expected = np.asarray(per_token)[np.asarray(positions[:, 0]), np.asarray(positions[:, 1])]
    np.testing.assert_allclose(np.asarray(loss), expected, rtol=1e-4, atol=1e-4)
    probs = np.asarray(top_probs)
    assert np.all(np.diff(probs, axis=1) <= 0)
    assert np.all(probs[:, 0] >= np.exp(-np.asarray(loss)) - 1e-6)
    assert top_ids.shape == (4, 3)
    assert attention == {}


def _probe_forward(tokens, mask, spot=None, token_ids=()):
    mesh, model = t._model(mla=True)
    probe = ForwardProbe(attn_queries=((0, 9), (1, t._SEQ - 1)), spot=spot, spot_token_ids=token_ids)
    positions = jnp.asarray([[0, 0]], jnp.int32)
    with jax.set_mesh(mesh), forward_probe(probe):
        _, _, _, recorded = jax.jit(lambda m, x: m.position_predictions(x, positions, mask=mask, k=1))(model, tokens)
    return {name: np.asarray(value) for name, value in recorded.items()}


def _two_documents():
    tokens = jax.random.randint(jax.random.PRNGKey(5), (2, t._SEQ), 0, t._VOCAB)
    segments = jnp.asarray(np.repeat([0, 1], [6, t._SEQ - 6])[None].repeat(2, 0), jnp.int32)
    return tokens, AttentionMask(is_causal=True, segment_ids=(segments, segments))


def test_attention_probe_reports_causal_document_masked_probabilities():
    tokens, mask = _two_documents()
    attention = _probe_forward(tokens, mask)
    assert sorted(attention) == [f"{ATTN_PROBE_STAT}_L3", f"{ATTN_PROBE_STAT}_L5"]
    for probs in attention.values():
        assert probs.shape[0] == 2 and probs.shape[2] == ATTN_PROBE_KEYS
        np.testing.assert_allclose(probs.sum(-1), 1.0, rtol=1e-5)
        # Query (0, 9) sits in document 1 (positions 6..): keys 6..9 only, the last 4 slots of its window.
        np.testing.assert_array_equal(probs[0, :, :-4], 0.0)
        assert np.all(probs[0, :, -4:] > 0)
    # Changing a token after the query (position 9) leaves its attention unchanged.
    perturbed = _probe_forward(tokens.at[0, 12].set((tokens[0, 12] + 1) % t._VOCAB), mask)
    for name in attention:
        np.testing.assert_allclose(perturbed[name][0], attention[name][0], rtol=1e-5, atol=1e-6)


def test_spot_probe_records_every_layer_and_the_final_mix():
    tokens, mask = _two_documents()
    recorded = _probe_forward(tokens, mask, spot=(1, 10), token_ids=(3, 7))
    for layer in range(6):
        for part in ("attn_in", "attn_out", "mlp_in", "mlp_out"):
            assert recorded[f"{PROBE_STAT}{part}_L{layer}"].shape == (32,)
    sources = recorded[f"{PROBE_STAT}final_sources"]
    weights = recorded[f"{PROBE_STAT}final_gate_weights"]
    np.testing.assert_allclose(weights.sum(), 1.0, rtol=1e-5)
    # The final AttnRes output is the gate-weighted sum of the recorded sources.
    np.testing.assert_allclose(recorded[f"{PROBE_STAT}final_mixed"], weights @ sources, rtol=1e-4, atol=1e-5)
    assert recorded[f"{PROBE_STAT}lm_head_cols"].shape == (32, 2)
    assert recorded[f"{PROBE_STAT}head_in"].shape == (32,)


def test_text_counts_match_whole_words_per_row():
    words = {0: "India", 1: " Indiana", 2: " India", 3: "'s"}
    tokens = np.array([[0, 3, 1, 2], [1, 1, 2, 2]])
    counts = count_text_patterns(tokens, lambda ids: "".join(words[i] for i in ids), (r"\bIndia\b", r"Indiana"))
    np.testing.assert_array_equal(counts, [4, 3])


def test_probe_attention_matches_dense_attention_with_the_inkling_bias():
    batch, seq, heads, dim = 2, 24, 3, 8
    keys = jax.random.split(jax.random.PRNGKey(6), 3)
    q = jax.random.normal(keys[0], (batch, seq, heads, dim))
    k = jax.random.normal(keys[1], (batch, seq, heads, dim))
    band = jax.random.normal(keys[2], (batch, heads, seq, REL_BIAS_BLOCK + 8))
    segments = jnp.asarray(np.repeat([0, 1], [10, seq - 10])[None].repeat(batch, 0), jnp.int32)
    mask = AttentionMask(is_causal=True, segment_ids=(segments, segments))
    queries = ((0, 5), (1, 17), (1, seq - 1))
    with jax.set_mesh(t._mesh()):
        probs = np.asarray(_probe_attention(q, k, mask, band, queries))
    logits = np.einsum("bqhd,bkhd->bhqk", q, k) / np.sqrt(dim) + np.asarray(dense_rel_bias(band, seq, seq))
    seg = np.asarray(segments)
    allowed = (np.arange(seq)[None, :] <= np.arange(seq)[:, None])[None] & (seg[:, :, None] == seg[:, None, :])
    logits = np.where(allowed[:, None], logits, -np.inf)
    dense = np.exp(logits - logits.max(-1, keepdims=True))
    dense /= dense.sum(-1, keepdims=True)
    for index, (row, position) in enumerate(queries):
        window = probs[index, :, -(position + 1) :]  # keys 0..position sit in the window's last slots
        np.testing.assert_allclose(window, dense[row, :, position, : position + 1], rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(probs[index, :, : -(position + 1)], 0.0)


def test_spot_probe_runs_with_the_inkling_relative_position_bias():
    seq = REL_BIAS_BLOCK  # the banded bias needs whole 128-token blocks
    tokens = jax.random.randint(jax.random.PRNGKey(7), (2, seq), 0, t._VOCAB)
    segments = jnp.asarray(np.repeat([0, 1], [40, seq - 40])[None].repeat(2, 0), jnp.int32)
    mask = AttentionMask(is_causal=True, segment_ids=(segments, segments))
    mesh, model = t._model(mla=True, inkling_relpos=True, max_seq_len=seq, sliding_window=seq)
    probe = ForwardProbe(attn_queries=((1, 60),), spot=(1, 60), spot_token_ids=(3,))
    positions = jnp.zeros((1, 2), jnp.int32)
    with jax.set_mesh(mesh), forward_probe(probe):
        _, _, _, recorded = jax.jit(lambda m, x: m.position_predictions(x, positions, mask=mask, k=1))(model, tokens)
    for name in (f"{ATTN_PROBE_STAT}_L3", f"{ATTN_PROBE_STAT}_L5"):
        probs = np.asarray(recorded[name])
        np.testing.assert_allclose(probs.sum(-1), 1.0, rtol=1e-5)
        # Position 60 is in document 1 (positions 40..): exactly its 21 keys get weight.
        assert np.all(probs[0, :, -21:] > 0) and np.all(probs[0, :, :-21] == 0)


def test_attention_summary_is_mass_on_the_key_set_and_entropy():
    probs = np.zeros((2, 1, 4), np.float32)
    probs[0, 0] = [0.1, 0.2, 0.3, 0.4]  # query at position 10: keys 7..10
    probs[1, 0] = [0.0, 0.0, 0.5, 0.5]  # query at position 5: keys 2..5
    queries = np.array([[0, 10], [1, 5]])
    key_sets = np.array([[8, 10], [4, -1]])
    mass, entropy = summarize_attention(probs, queries, key_sets, num_keys=4)
    np.testing.assert_allclose(mass[:, 0], [0.2 + 0.4, 0.5])
    np.testing.assert_allclose(entropy[1, 0], np.log(2), rtol=1e-6)
    with pytest.raises(ValueError):
        summarize_attention(probs, queries, np.array([[1, -1], [4, -1]]), num_keys=4)
