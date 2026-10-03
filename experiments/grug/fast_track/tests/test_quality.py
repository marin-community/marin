# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import numpy as np
import pytest

from experiments.grug.fast_track.quality import (
    LabelledEmbedding,
    LabelSplit,
    PoolDocument,
    PoolRequirements,
    RidgeHeadConfig,
    audit_pool,
    development_metrics,
    label_split,
    select_top_tokens,
    selection_token_overlap,
    split_labelled_embeddings,
)


def _pool() -> list[PoolDocument]:
    return [
        PoolDocument("web", "short", "g1", 10, "low", "web", "en"),
        PoolDocument("web", "long", "g2", 30, "high", "web", "en"),
        PoolDocument("code", "short", "g3", 20, "low", "code", "en"),
        PoolDocument("code", "long", "g4", 40, "high", "code", "en"),
    ]


def _requirements() -> PoolRequirements:
    return PoolRequirements(
        source_token_shares={"web": 0.4, "code": 0.6},
        source_share_tolerance=0.01,
        quality_bins=("low", "high"),
        min_documents_per_quality_bin=2,
        min_duplicate_groups=4,
        max_duplicate_token_share=0.5,
    )


def test_ridge_head_uses_fixed_training_groups_and_keeps_audit_labels_private():
    rows = [LabelledEmbedding("source", str(i), str(i), (float(i),), 2.0 * i + 3.0) for i in range(100)]
    training, development = split_labelled_embeddings(rows, split_seed=9)
    head = RidgeHeadConfig(0.001).fit(training)
    metrics = development_metrics(head, development)
    altered = [
        replace(row, label=-1e9) if label_split(row.duplicate_group, 9) is LabelSplit.AUDIT else row for row in rows
    ]
    other_training, other_development = split_labelled_embeddings(altered, split_seed=9)
    other_head = RidgeHeadConfig(0.001).fit(other_training)
    other_metrics = development_metrics(other_head, other_development)

    np.testing.assert_allclose(head.scores(np.array([[10.0], [20.0]])), [23.0, 43.0], rtol=1e-4)
    assert other_head == head
    assert other_metrics == metrics
    assert metrics.mean_squared_error < 1e-7


def test_pool_audit_detects_uniform_quality_despite_sufficient_token_volume():
    documents = _pool()
    audit = audit_pool(documents, requirements=_requirements(), labelled_groups=set())

    assert audit.tokens == 100
    assert audit.source_token_shares == {"web": 0.4, "code": 0.6}
    assert audit.effective_token_documents == pytest.approx(10000 / 3000)
    with pytest.raises(ValueError, match="score coverage"):
        audit_pool(
            [replace(row, quality_bin="high") for row in documents],
            requirements=_requirements(),
            labelled_groups=set(),
        )


def test_pool_audit_rejects_cross_source_duplicate_label_leakage():
    with pytest.raises(ValueError, match="overlaps the labelled duplicate groups"):
        audit_pool(_pool(), requirements=_requirements(), labelled_groups={"g3"})


def test_token_selection_records_document_overshoot_and_matched_overlap():
    pool = _pool()
    candidate = select_top_tokens(pool, [1, 4, 3, 2], fraction=0.4, tie_seed=0)
    incumbent = select_top_tokens(pool, [1, 3, 2, 4], fraction=0.4, tie_seed=0)

    assert candidate.indices == (1, 2)
    assert candidate.requested_tokens == 40
    assert candidate.selected_tokens == 50
    assert candidate.source_token_shares == {"web": 0.6, "code": 0.4}
    assert incumbent.indices == (3,)
    assert selection_token_overlap(pool, candidate, incumbent) == 0


def test_tied_selection_is_independent_of_pool_input_order():
    pool = _pool()
    forward = select_top_tokens(pool, [1] * 4, fraction=0.4, tie_seed=7)
    reversed_pool = list(reversed(pool))
    backward = select_top_tokens(reversed_pool, [1] * 4, fraction=0.4, tie_seed=7)

    assert [(pool[i].source, pool[i].document_id) for i in forward.indices] == [
        (reversed_pool[i].source, reversed_pool[i].document_id) for i in backward.indices
    ]
    assert forward.cutoff_ties == 4
