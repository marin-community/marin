# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import numpy as np

from experiments.grug.fast_track.quality import (
    EmbeddingHeadScorer,
    LabelledEmbedding,
    LabelSplit,
    QualityScoringBatch,
    RidgeHeadConfig,
    fit_quality_head,
    label_split,
)


def test_ridge_head_uses_fixed_training_groups_and_keeps_audit_labels_private():
    rows = [LabelledEmbedding("source", str(i), str(i), (float(i),), 2.0 * i + 3.0) for i in range(100)]
    fitted = fit_quality_head(rows, head=RidgeHeadConfig(0.001), split_seed=9)
    altered = [
        replace(row, label=-1e9) if label_split(row.duplicate_group, 9) is LabelSplit.AUDIT else row for row in rows
    ]
    other_fitted = fit_quality_head(altered, head=RidgeHeadConfig(0.001), split_seed=9)

    np.testing.assert_allclose(fitted.scorer.scores(np.array([[10.0], [20.0]])), [23.0, 43.0], rtol=1e-4)
    assert other_fitted.scorer == fitted.scorer
    assert other_fitted.development == fitted.development
    assert fitted.development.mean_squared_error < 1e-7
    assert fitted.training_documents == sum(label_split(row.duplicate_group, 9) is LabelSplit.TRAIN for row in rows)

    np.testing.assert_allclose(
        EmbeddingHeadScorer(fitted.scorer).scores(
            QualityScoringBatch(
                texts=["10", "20"],
                document_ids=["doc-10", "doc-20"],
                embeddings=np.asarray([[10.0], [20.0]], dtype=np.float32),
            )
        ),
        [23.0, 43.0],
        rtol=1e-4,
    )


def test_dual_ridge_solution_matches_primal_for_small_label_tables():
    vectors = np.asarray([[1.0, 2.0, 0.0, 1.0], [0.0, 1.0, 3.0, 2.0], [2.0, 0.0, 1.0, 4.0]])
    labels = np.asarray([0.2, 0.7, 0.4])
    rows = [
        LabelledEmbedding("source", str(index), str(index), vector, float(label))
        for index, (vector, label) in enumerate(zip(vectors, labels, strict=True))
    ]
    regularization = 0.03

    fitted = RidgeHeadConfig(regularization).fit(rows)
    centered_vectors = vectors - vectors.mean(axis=0)
    centered_labels = labels - labels.mean()
    primal_gram = centered_vectors.T @ centered_vectors / len(rows)
    primal_gram.flat[:: vectors.shape[1] + 1] += regularization
    expected = np.linalg.solve(primal_gram, centered_vectors.T @ centered_labels / len(rows))

    np.testing.assert_allclose(fitted.coefficients, expected, rtol=1e-9, atol=1e-10)
