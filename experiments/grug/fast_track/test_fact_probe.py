# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import glob

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import experiments.grug.fast_track.test_kda_local as t
from experiments.grug.fast_track.fact_probe import RAW, FactProbeInput, FactProbeWriter, count_text_patterns

_EOS = 1


def _probe(tokens: np.ndarray, spans: list[tuple[int, int, int]]) -> FactProbeInput:
    row, start, end = (np.asarray(column) for column in zip(*spans, strict=True))
    return FactProbeInput(tokens=tokens.astype(np.int32), span_row=row, span_start=start, span_end=end)


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
        writer.add(RAW, step, np.full(3, step), np.zeros((3, 5)), np.zeros((3, 5)), np.zeros(4))
    assert len(glob.glob(str(tmp_path / "*.npz"))) == 1
    writer.flush_all()
    chunks = sorted(glob.glob(str(tmp_path / "fact_probe_raw_*.npz")))
    np.testing.assert_array_equal(np.load(chunks[0])["steps"], [1, 2])
    np.testing.assert_array_equal(np.load(chunks[1])["loss"], [[3, 3, 3]])


def test_position_predictions_match_the_training_loss():
    mesh, model = t._model(logit_soft_cap=2.0)
    tokens = jax.random.randint(jax.random.PRNGKey(4), (2, t._SEQ), 0, t._VOCAB)
    positions = jnp.asarray([[0, 0], [0, 5], [1, 11], [1, t._SEQ - 2]], jnp.int32)
    with jax.set_mesh(mesh):
        per_token = jax.jit(lambda m, x: m.next_token_loss(x, jnp.ones(x.shape), reduction="none"))(model, tokens)
        loss, top_ids, top_probs = jax.jit(lambda m, x, p: m.position_predictions(x, p, k=3))(model, tokens, positions)
    expected = np.asarray(per_token)[np.asarray(positions[:, 0]), np.asarray(positions[:, 1])]
    np.testing.assert_allclose(np.asarray(loss), expected, rtol=1e-4, atol=1e-4)
    probs = np.asarray(top_probs)
    assert np.all(np.diff(probs, axis=1) <= 0)
    assert np.all(probs[:, 0] >= np.exp(-np.asarray(loss)) - 1e-6)
    assert top_ids.shape == (4, 3)


def test_text_counts_match_whole_words_per_row():
    words = {0: "India", 1: " Indiana", 2: " India", 3: "'s"}
    tokens = np.array([[0, 3, 1, 2], [1, 1, 2, 2]])
    counts = count_text_patterns(tokens, lambda ids: "".join(words[i] for i in ids), (r"\bIndia\b", r"Indiana"))
    np.testing.assert_array_equal(counts, [4, 3])
