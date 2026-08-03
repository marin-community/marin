# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the fast-transformer quality scorer's two algorithmic contracts:

- ``scorer.score_bme`` — whole-doc (begin/middle/end) window coverage + mean-pooling,
  the fix for scoring long docs on a truncated lead / prefix-degenerate sources.
- ``calibrate.fit_cutpoints`` / ``calibration_knots`` — the monotonic cutpoint remap
  that makes the fixed 0.2-bucket quantization recover the oracle quality level.

Both use a deterministic fake scorer / synthetic labels, so no model or I/O is needed.
"""

from itertools import pairwise
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pytest
from scipy.special import logsumexp

from experiments.datakit.cluster.quality.fast_transformer.artifact import BUCKET_EDGES
from experiments.datakit.cluster.quality.fast_transformer.calibrate import calibration_knots, fit_cutpoints
from experiments.datakit.cluster.quality.fast_transformer.embedding import (
    contrastive_embedding_loss,
    cross_source_teacher_neighbor_loss,
    direct_cosine_embedding_loss,
    embedding_distillation_loss,
    embedding_spread_loss,
    pack_remapped_windows,
    predict_embeddings,
    projected_embedding_distillation_loss,
    source_balanced_token_remap,
    source_conditioned_geometry_loss,
)
from experiments.datakit.cluster.quality.fast_transformer.model import (
    FastEmbeddingTransformer,
    FastTransformerConfig,
)
from experiments.datakit.cluster.quality.fast_transformer.score import _systematic_take
from experiments.datakit.cluster.quality.fast_transformer.scorer import CHUNK_CHARS, PooledScorer, score_bme


class _FakeScorer:
    """Deterministic stand-in for ``PooledScorer``: ``score(texts)`` returns a value
    per text keyed on its first character (default otherwise), and records the exact
    chunk lists it was called with so tests can assert which windows were scored."""

    def __init__(self, by_first_char: dict[str, float] | None = None, default: float = 0.0) -> None:
        self._map = by_first_char or {}
        self._default = default
        self.calls: list[list[str]] = []

    def score(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        self.calls.append(list(texts))
        return np.array([self._map.get(t[:1], self._default) for t in texts], dtype=float)


def _as_scorer(fake: _FakeScorer) -> PooledScorer:
    return cast(PooledScorer, fake)


# ---------- _score_bme: whole-doc window coverage + pooling ----------


def test_bme_short_doc_scores_as_single_window():
    fake = _FakeScorer({"x": 0.3})
    doc = "x" * 100  # <= CHUNK_CHARS
    out = score_bme(_as_scorer(fake), [doc])
    assert fake.calls == [[doc]]  # exactly one chunk = the whole doc
    assert out.tolist() == pytest.approx([0.3])


def test_bme_long_doc_covers_begin_middle_end_and_mean_pools():
    fake = _FakeScorer({"A": 0.0, "B": 0.6, "C": 0.9})
    # begin -> A block, middle -> B block, end -> C block (each exactly one chunk)
    doc = "A" * CHUNK_CHARS + "B" * CHUNK_CHARS + "C" * CHUNK_CHARS
    out = score_bme(_as_scorer(fake), [doc])

    chunks = fake.calls[0]
    assert len(chunks) == 3
    assert all(len(c) == CHUNK_CHARS for c in chunks)
    # the three windows are begin / middle / end of the whole doc -- not just the lead
    assert (chunks[0][0], chunks[1][0], chunks[2][0]) == ("A", "B", "C")
    assert out.tolist() == pytest.approx([(0.0 + 0.6 + 0.9) / 3])  # mean-pooled


def test_bme_batch_pools_each_doc_independently():
    fake = _FakeScorer({"x": 0.3, "A": 0.0, "B": 0.6, "C": 0.9})
    short = "x" * 100
    long = "A" * CHUNK_CHARS + "B" * CHUNK_CHARS + "C" * CHUNK_CHARS
    out = score_bme(_as_scorer(fake), [short, long])
    # all 1 + 3 chunks scored in a single batched call; spans map back per doc
    assert len(fake.calls) == 1 and len(fake.calls[0]) == 4
    assert out.tolist() == pytest.approx([0.3, (0.0 + 0.6 + 0.9) / 3])


def test_bme_window_count_switches_at_chunk_boundary():
    fake = _FakeScorer(default=0.5)
    score_bme(_as_scorer(fake), ["y" * CHUNK_CHARS])  # == threshold
    score_bme(_as_scorer(fake), ["y" * (CHUNK_CHARS + 1)])  # one char over
    assert len(fake.calls[0]) == 1  # <= CHUNK_CHARS -> single window
    assert len(fake.calls[1]) == 3  # > CHUNK_CHARS  -> begin/middle/end


# ---------- calibrate: monotonic cutpoint remap ----------


def test_fit_cutpoints_are_midpoints_of_adjacent_level_medians():
    # level L docs all have raw = L/10 -> medians {1:.1, ..., 5:.5}
    levels = np.repeat([1, 2, 3, 4, 5], 4).astype(float)
    raw = levels / 10.0
    med, cuts = fit_cutpoints(raw, levels)
    assert med == pytest.approx({1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.5})
    assert cuts == pytest.approx([0.15, 0.25, 0.35, 0.45])


def test_fit_cutpoints_enforced_non_decreasing():
    # medians whose raw midpoints would dip (0.55 -> 0.35); accumulate must fix it
    raw = np.array([0.2, 0.8, 0.3, 0.4, 0.5])
    levels = np.array([1, 2, 3, 4, 5], dtype=float)
    _, cuts = fit_cutpoints(raw, levels)
    assert cuts == pytest.approx([0.5, 0.55, 0.55, 0.55])
    assert all(b >= a for a, b in pairwise(cuts))


def test_calibration_knots_are_strictly_increasing_and_recover_levels():
    levels = np.repeat([1, 2, 3, 4, 5], 4).astype(float)
    raw = levels / 10.0
    knots = calibration_knots(raw, levels)
    xk, yk = knots["xk"], knots["yk"]

    assert yk == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    assert len(xk) == 6
    # np.interp requires strictly increasing knots
    assert all(b > a for a, b in pairwise(xk))
    # each oracle level's median maps into (within one of) the matching bucket
    for level in (1, 2, 3, 4, 5):
        bucket = int(np.digitize(np.interp(level / 10.0, xk, yk), BUCKET_EDGES))
        assert abs(bucket - (level - 1)) <= 1


# ---------- score: deterministic non-hashing sample ----------


def test_systematic_sample_is_deterministic_and_hits_target_fraction():
    for pct in (0.1, 0.25, 0.5):
        kept = [i for i in range(1000) if _systematic_take(i, pct)]
        # deterministic: no RNG / no hashing -> identical across calls
        assert kept == [i for i in range(1000) if _systematic_take(i, pct)]
        # ~pct of records, evenly spaced
        assert abs(len(kept) / 1000 - pct) < 0.01


def _embedding_model() -> FastEmbeddingTransformer:
    config = FastTransformerConfig(
        vocab_size=32,
        max_tokens=8,
        pool_window=4,
        pool_kind="meanmaxmin",
        embed_dim=8,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
    )
    return FastEmbeddingTransformer(config, output_dim=6, key=jr.PRNGKey(7))


def test_embedding_transformer_returns_distinct_unit_vectors():
    model = _embedding_model()
    ids = jnp.asarray(
        [
            [2, 3, 4, 5, 0, 0, 0, 0],
            [6, 7, 8, 9, 10, 11, 0, 0],
        ],
        dtype=jnp.int32,
    )

    vectors = np.asarray(model(ids))

    assert vectors.shape == (2, 6)
    assert np.isfinite(vectors).all()
    assert np.linalg.norm(vectors, axis=1) == pytest.approx([1.0, 1.0], abs=1e-6)
    assert not np.allclose(vectors[0], vectors[1])


def test_contrastive_embedding_loss_matches_numpy_reference():
    student = np.asarray(
        [[1.0, 0.2, -0.3], [0.1, 0.9, 0.4], [-0.2, 0.3, 1.1], [0.7, -0.4, 0.2]],
        dtype=np.float32,
    )
    teacher = np.asarray(
        [[0.8, 0.1, -0.1], [0.0, 1.0, 0.2], [-0.1, 0.4, 0.9], [0.6, -0.2, 0.5]],
        dtype=np.float32,
    )
    temperature = 3.0

    def off_diagonal_gram(vectors: np.ndarray) -> np.ndarray:
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        gram = vectors @ vectors.T
        return gram[~np.eye(len(vectors), dtype=bool)].reshape(len(vectors), len(vectors) - 1)

    student_logits = off_diagonal_gram(student) / temperature
    teacher_logits = off_diagonal_gram(teacher) / temperature
    student_log_probabilities = student_logits - logsumexp(student_logits, axis=1, keepdims=True)
    teacher_log_probabilities = teacher_logits - logsumexp(teacher_logits, axis=1, keepdims=True)
    expected = temperature**2 * np.mean(
        np.sum(
            np.exp(teacher_log_probabilities) * (teacher_log_probabilities - student_log_probabilities),
            axis=1,
        )
    )

    actual = float(contrastive_embedding_loss(jnp.asarray(student), jnp.asarray(teacher), temperature))

    assert actual == pytest.approx(expected, rel=1e-5, abs=1e-6)


def test_contrastive_embedding_loss_compiles():
    student = jnp.eye(4, dtype=jnp.float32)
    teacher = jnp.asarray(
        [[1.0, 0.1, 0.0, 0.0], [0.1, 1.0, 0.1, 0.0], [0.0, 0.1, 1.0, 0.1], [0.0, 0.0, 0.1, 1.0]],
        dtype=jnp.float32,
    )

    loss = jax.jit(contrastive_embedding_loss, static_argnums=2)(student, teacher, 3.0)

    assert np.isfinite(float(loss))


def test_direct_cosine_embedding_loss_aligns_matching_rows():
    teacher = jnp.eye(4, dtype=jnp.float32)
    reversed_teacher = teacher[::-1]

    matching_loss = float(direct_cosine_embedding_loss(teacher, teacher))
    reversed_loss = float(direct_cosine_embedding_loss(reversed_teacher, teacher))

    assert matching_loss == pytest.approx(0.0, abs=1e-7)
    assert reversed_loss == pytest.approx(1.0, abs=1e-7)


def test_source_conditioned_geometry_loss_matches_same_source_numpy_pairs():
    student = np.asarray(
        [[1.0, 0.0, 0.0], [0.8, 0.6, 0.0], [0.0, 1.0, 0.0], [0.0, 0.6, 0.8]],
        dtype=np.float32,
    )
    teacher = np.asarray(
        [[1.0, 0.0, 0.0], [0.6, 0.8, 0.0], [0.0, 1.0, 0.0], [0.0, 0.8, 0.6]],
        dtype=np.float32,
    )
    source_ids = np.asarray([0, 0, 1, 1], dtype=np.int32)
    normalized_student = student / np.linalg.norm(student, axis=1, keepdims=True)
    normalized_teacher = teacher / np.linalg.norm(teacher, axis=1, keepdims=True)
    student_cosine = normalized_student @ normalized_student.T
    teacher_cosine = normalized_teacher @ normalized_teacher.T
    selected = (source_ids[:, None] == source_ids[None, :]) & ~np.eye(len(source_ids), dtype=bool)
    expected = np.square(student_cosine - teacher_cosine)[selected].mean()

    actual = source_conditioned_geometry_loss(
        jnp.asarray(student),
        jnp.asarray(teacher),
        jnp.asarray(source_ids),
    )

    assert float(actual) == pytest.approx(float(expected), abs=1e-7)


def test_embedding_prediction_padding_preserves_rows_and_values():
    model = _embedding_model()
    ids = np.asarray(
        [
            [2, 3, 0, 0, 0, 0, 0, 0],
            [4, 5, 6, 0, 0, 0, 0, 0],
            [7, 8, 9, 10, 0, 0, 0, 0],
            [11, 12, 13, 14, 15, 0, 0, 0],
            [16, 17, 18, 19, 20, 21, 0, 0],
        ],
        dtype=np.int32,
    )

    expected = np.asarray(eqx.filter_jit(model)(jnp.asarray(ids)))
    actual = predict_embeddings(model, ids, batch_size=4)

    assert actual.shape == (5, 6)
    assert actual == pytest.approx(expected, abs=1e-5)


def test_embedding_window_packing_remaps_truncates_and_pads():
    remap = np.asarray([1, 1, 8, 9, 10, 11, 12], dtype=np.int32)
    raw_windows = [
        [[2, 3, 4], [5], [6, 2]],
        [[3], [], [4, 5, 6]],
    ]

    packed = pack_remapped_windows(raw_windows, remap, max_tokens=8, tokens_per_window=2)

    assert packed.tolist() == [
        [8, 9, 11, 0, 12, 8, 0, 0],
        [9, 0, 0, 0, 10, 11, 0, 0],
    ]


def test_source_balanced_remap_gives_each_source_equal_weight():
    first_source = np.asarray([0, 0, 900, 100, 0], dtype=np.int64)
    second_source = np.asarray([0, 0, 0, 0, 1], dtype=np.int64)

    remap = source_balanced_token_remap([first_source, second_source], compact_vocab_size=4)

    assert remap.tolist() == [1, 1, 3, 1, 2]


def test_embedding_transformer_takes_contrastive_gradient_step():
    model = _embedding_model()
    ids = jnp.asarray(
        [
            [2, 3, 4, 0, 0, 0, 0, 0],
            [5, 6, 7, 8, 0, 0, 0, 0],
            [9, 10, 11, 12, 13, 0, 0, 0],
            [14, 15, 16, 17, 18, 19, 0, 0],
        ],
        dtype=jnp.int32,
    )
    teacher = jr.normal(jr.PRNGKey(8), (4, 6))
    teacher /= jnp.linalg.norm(teacher, axis=1, keepdims=True)
    optimizer = optax.adam(1e-3)
    optimizer_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    def loss_function(candidate):
        student = candidate(ids, key=jr.PRNGKey(9), inference=False)
        return embedding_distillation_loss(student, teacher, temperature=3.0, direct_cosine_weight=1.0)

    initial_head = np.asarray(model.embedding_head)
    loss, gradients = eqx.filter_value_and_grad(loss_function)(model)
    updates, _ = optimizer.update(gradients, optimizer_state, eqx.filter(model, eqx.is_inexact_array))
    updated = eqx.apply_updates(model, updates)

    assert np.isfinite(float(loss))
    assert not np.array_equal(np.asarray(updated.embedding_head), initial_head)


def test_cross_source_teacher_neighbor_loss_prefers_matching_geometry():
    teacher = jnp.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.9, 0.1],
            [0.1, 0.9],
        ]
    )
    source_ids = jnp.asarray([0, 0, 1, 1])
    matching = teacher
    reversed_cross_source_neighbors = teacher[jnp.asarray([0, 1, 3, 2])]

    matching_loss = cross_source_teacher_neighbor_loss(matching, teacher, source_ids, 1, 0.1)
    reversed_loss = cross_source_teacher_neighbor_loss(reversed_cross_source_neighbors, teacher, source_ids, 1, 0.1)

    assert float(matching_loss) < float(reversed_loss)


def test_embedding_spread_loss_rejects_constant_geometry():
    collapsed = jnp.ones((8, 4))
    varied = jnp.concatenate([jnp.eye(4), -jnp.eye(4)], axis=0)

    collapsed_loss = embedding_spread_loss(collapsed, standard_deviation_target=0.04, covariance_weight=0.1)
    varied_loss = embedding_spread_loss(varied, standard_deviation_target=0.04, covariance_weight=0.1)

    assert float(varied_loss) < float(collapsed_loss)


def test_cross_dimension_distillation_matches_component_contract():
    student = jnp.asarray([[1.0, 0.2], [0.1, 0.9], [-0.4, 0.7], [0.8, -0.3]])
    teacher = jnp.asarray([[0.8, 0.1, 0.4], [0.0, 1.0, -0.2], [-0.3, 0.6, 0.9], [0.7, -0.1, 0.5]])
    projection = jnp.asarray([[0.9, 0.2, -0.1], [-0.2, 0.7, 0.6]])
    temperature = 2.5
    direct_cosine_weight = 0.7

    actual = projected_embedding_distillation_loss(
        student,
        teacher,
        projection,
        temperature,
        direct_cosine_weight,
    )
    expected = contrastive_embedding_loss(student, teacher, temperature) + direct_cosine_weight * (
        direct_cosine_embedding_loss(student @ projection, teacher)
    )

    assert float(actual) == pytest.approx(float(expected), abs=1e-7)


def test_cross_dimension_distillation_gradient_step_reduces_loss():
    student = jnp.asarray([[1.0, 0.2], [0.1, 0.9], [-0.4, 0.7], [0.8, -0.3]])
    teacher = jnp.asarray([[0.8, 0.1, 0.4], [0.0, 1.0, -0.2], [-0.3, 0.6, 0.9], [0.7, -0.1, 0.5]])
    projection = jnp.asarray([[0.9, 0.2, -0.1], [-0.2, 0.7, 0.6]])

    def loss_function(candidate_student, candidate_projection):
        return projected_embedding_distillation_loss(
            candidate_student,
            teacher,
            candidate_projection,
            temperature=3.0,
            direct_cosine_weight=1.0,
        )

    loss, gradients = jax.value_and_grad(loss_function, argnums=(0, 1))(student, projection)
    updated_student = student - 0.01 * gradients[0]
    updated_projection = projection - 0.01 * gradients[1]
    updated_loss = loss_function(updated_student, updated_projection)

    assert np.isfinite(float(loss))
    assert float(updated_loss) < float(loss)


def test_cross_dimension_distillation_rejects_wrong_projection_shape():
    student = jnp.ones((4, 2))
    teacher = jnp.ones((4, 3))
    projection = jnp.ones((3, 2))

    with pytest.raises(ValueError, match="Projection shape"):
        projected_embedding_distillation_loss(student, teacher, projection, 3.0, 1.0)


def test_cross_dimension_distillation_rejects_non_matrix_input():
    student = jnp.ones((4,))
    teacher = jnp.ones((4, 3))
    projection = jnp.ones((2, 3))

    with pytest.raises(ValueError, match="Student rows"):
        projected_embedding_distillation_loss(student, teacher, projection, 3.0, 1.0)
