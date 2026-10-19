# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib

import numpy as np
import pyarrow.parquet as pq
import pytest
from levanter.store.cache import CacheMetadata, TreeCache
from zephyr.stage_io import ZephyrWorkerError

from experiments.grug.fast_track.label_exclusion import LabelExclusion
from experiments.grug.fast_track.quality_features import QualityFeatureSource, prepare_quality_features
from experiments.grug.fast_track.quality_pipeline import (
    ScorerSpec,
    SelectedQualityPool,
    SelectionMethod,
    _write_quality_token_cache,
    materialize_selected_pool,
    score_raw_pool,
    select_scored_pool,
)

NO_LABELS = LabelExclusion(label_revision="no-labels-v1", duplicate_groups=frozenset())


class _TextScorer:
    def __init__(self, direction: float):
        self.direction = direction

    def scores(self, batch):
        return np.asarray([self.direction * float(text.rsplit("-", 1)[1]) for text in batch.texts])

    def __call__(self):
        return _TextScorer(self.direction)


def _text_scorer(direction: float, revision: str) -> ScorerSpec:
    return ScorerSpec(_TextScorer(direction), {"implementation": "text-test", "revision": revision})


@pytest.fixture
def variable_token_pool(tmp_path, write_raw_pool):
    rows = []
    for index in range(12):
        token_count = 2 + index % 4
        rows.append(
            {
                "source": "source",
                "id": str(index),
                "sample_rank": f"{index:064x}",
                "duplicate_group": hashlib.sha256(f"text-{index}".encode()).hexdigest(),
                "text": f"text-{index}",
                "input_ids": [index + 1] * token_count,
                "token_count": token_count,
            }
        )
    return write_raw_pool(rows, name="variable")


def test_embedding_scorer_receives_id_aligned_cached_features(raw_pool, zephyr_context, tmp_path, monkeypatch):
    monkeypatch.setattr("experiments.grug.fast_track.quality_features.hero_data.MANIFEST_PREFIX", str(tmp_path))
    monkeypatch.setattr(
        "experiments.grug.fast_track.quality_features.hero_data.pinned_map", lambda _stage: {"source": "harrier"}
    )

    class EmbeddingScorer:
        def scores(self, batch):
            assert batch.embeddings is not None
            assert batch.embeddings.dtype == np.float32
            np.testing.assert_allclose(np.linalg.norm(batch.embeddings, axis=1), 1.0, atol=1e-6)
            return batch.embeddings[:, 0]

        def __call__(self):
            return self

    feature_pool = prepare_quality_features(
        raw_pool,
        ctx=zephyr_context,
        output_path=str(tmp_path / "prepared-features"),
        token_budget=40,
        sources=(QualityFeatureSource("source", raw_pool.sources[0].normalized_path, str(tmp_path / "harrier")),),
    )
    scored = score_raw_pool(
        raw_pool,
        ctx=zephyr_context,
        output_path=str(tmp_path / "embedding-scored"),
        candidate=ScorerSpec(EmbeddingScorer(), {"implementation": "embedding-test", "revision": "v1"}),
        label_exclusion=NO_LABELS,
        token_budget=40,
        prepared_features=feature_pool,
    )

    assert scored.documents == feature_pool.documents
    assert scored.feature_identity == feature_pool.feature_identity
    rows = [row for path in scored.shards for row in pq.read_table(path).to_pylist()]
    assert len(rows) == feature_pool.documents
    assert all("embedding" not in row for row in rows)


def test_random_selection_reuses_the_same_raw_prefix_across_scores(raw_pool, zephyr_context, tmp_path):
    scored = score_raw_pool(
        raw_pool,
        ctx=zephyr_context,
        output_path=str(tmp_path / "scored"),
        candidate=_text_scorer(1, "test-v1"),
        label_exclusion=NO_LABELS,
        token_budget=40,
    )
    inverted = score_raw_pool(
        raw_pool,
        ctx=zephyr_context,
        output_path=str(tmp_path / "inverted"),
        candidate=_text_scorer(-1, "inverted-v1"),
        label_exclusion=NO_LABELS,
        token_budget=40,
    )
    random_a = select_scored_pool(
        scored,
        ctx=zephyr_context,
        output_path=str(tmp_path / "random-a"),
        training_tokens=4,
        selection_method=SelectionMethod.RANDOM,
        tie_seed=11,
        num_ranges=8,
    )
    random_b = select_scored_pool(
        inverted,
        ctx=zephyr_context,
        output_path=str(tmp_path / "random-b"),
        training_tokens=4,
        selection_method=SelectionMethod.RANDOM,
        tie_seed=11,
        num_ranges=8,
    )

    random_a_ids = [row["id"] for path in random_a.shards for row in pq.read_table(path).to_pylist()]
    random_b_ids = [row["id"] for path in random_b.shards for row in pq.read_table(path).to_pylist()]
    assert random_a_ids == random_b_ids
    assert random_a.raw_prefix_tokens == random_b.raw_prefix_tokens == 40


def test_constant_candidate_scores_fail_but_random_control_can_select(raw_pool, zephyr_context, tmp_path):
    class ConstantScorer:
        def scores(self, batch):
            return np.ones(len(batch.texts))

        def __call__(self):
            return self

    scored = score_raw_pool(
        raw_pool,
        ctx=zephyr_context,
        output_path=str(tmp_path / "constant"),
        candidate=ScorerSpec(ConstantScorer(), {"implementation": "constant-test", "revision": "v1"}),
        label_exclusion=NO_LABELS,
        token_budget=40,
    )

    with pytest.raises(ValueError, match="candidate scores are constant"):
        select_scored_pool(
            scored,
            ctx=zephyr_context,
            output_path=str(tmp_path / "constant-candidate"),
            training_tokens=4,
            selection_method=SelectionMethod.CANDIDATE,
            tie_seed=5,
            num_ranges=8,
        )
    random = select_scored_pool(
        scored,
        ctx=zephyr_context,
        output_path=str(tmp_path / "constant-random"),
        training_tokens=4,
        selection_method=SelectionMethod.RANDOM,
        tie_seed=5,
        num_ranges=8,
    )

    assert (random.documents, random.selected_tokens) == (1, 5)


def test_max_scored_pool_matches_exact_rung_pools_through_raw_cache(variable_token_pool, zephyr_context, tmp_path):
    pool = variable_token_pool
    total_tokens = pool.actual_tokens
    classifier = _text_scorer(1, "variable-v1")
    max_scored = score_raw_pool(
        pool,
        ctx=zephyr_context,
        output_path=str(tmp_path / "max-scored"),
        candidate=classifier,
        label_exclusion=NO_LABELS,
        token_budget=total_tokens,
    )
    exact_small = score_raw_pool(
        pool,
        ctx=zephyr_context,
        output_path=str(tmp_path / "exact-small"),
        candidate=classifier,
        label_exclusion=NO_LABELS,
        token_budget=10,
    )
    exact_large = score_raw_pool(
        pool,
        ctx=zephyr_context,
        output_path=str(tmp_path / "exact-large"),
        candidate=classifier,
        label_exclusion=NO_LABELS,
        token_budget=20,
    )
    with pytest.raises(ValueError, match="scored raw prefix budget 10 is below the 20 token rung prefix"):
        select_scored_pool(
            exact_small,
            ctx=zephyr_context,
            output_path=str(tmp_path / "undersized-scored-pool"),
            training_tokens=2,
            selection_method=SelectionMethod.CANDIDATE,
            tie_seed=5,
            num_ranges=32,
        )

    selections = []
    for label, scored, training_tokens in (
        ("max-small", max_scored, 1),
        ("exact-small", exact_small, 1),
        ("max-large", max_scored, 2),
        ("exact-large", exact_large, 2),
    ):
        selected = select_scored_pool(
            scored,
            ctx=zephyr_context,
            output_path=str(tmp_path / f"{label}-selection"),
            training_tokens=training_tokens,
            selection_method=SelectionMethod.CANDIDATE,
            tie_seed=5,
            num_ranges=32,
        )
        assert selected.raw_source_shards == scored.raw_prefix_shards
        assert all(
            "input_ids" in pq.read_schema(path).names for path in selected.raw_source_shards
        ), selected.raw_source_shards
        materialized = materialize_selected_pool(
            selected,
            ctx=zephyr_context,
            output_path=str(tmp_path / f"{label}-materialized"),
        )
        cache_dir = str(tmp_path / f"{label}-cache")
        _write_quality_token_cache(
            materialized,
            cache_dir,
            metadata=CacheMetadata({"preprocessor": "max-pool-rung-test"}),
        )
        cache = TreeCache.load(
            cache_dir,
            {"input_ids": np.zeros(0, dtype=np.int32)},
            CacheMetadata({"preprocessor": "max-pool-rung-test"}),
        )
        cache_rows = cache.get_batch_sync(list(range(len(cache))))
        selected_rows = [row for path in materialized for row in pq.read_table(path).to_pylist()]
        selections.append(
            (
                selected,
                [(row["id"], row["input_ids"]) for row in selected_rows],
                [row["input_ids"].tolist() for row in cache_rows],
            )
        )

    for max_index, exact_index, expected_tokens in ((0, 1, 10), (2, 3, 20)):
        max_selected, max_rows, max_cache = selections[max_index]
        exact_selected, exact_rows, exact_cache = selections[exact_index]
        assert max_rows == exact_rows
        assert max_cache == exact_cache
        assert max_selected.raw_prefix_documents == exact_selected.raw_prefix_documents
        assert max_selected.raw_prefix_tokens > expected_tokens
        assert max_selected.raw_prefix_tokens == exact_selected.raw_prefix_tokens
        assert max_selected.raw_source_shards == max_scored.raw_prefix_shards
        assert len(max_selected.shards) < 32


def test_max_scored_pool_checks_constant_scores_on_each_rung_prefix(variable_token_pool, zephyr_context, tmp_path):
    class PrefixConstantScorer:
        def scores(self, batch):
            return np.asarray(
                [0.0 if int(document_id) < 4 else float(document_id) for document_id in batch.document_ids]
            )

        def __call__(self):
            return self

    pool = variable_token_pool
    total_tokens = pool.actual_tokens
    scored = score_raw_pool(
        pool,
        ctx=zephyr_context,
        output_path=str(tmp_path / "constant-prefix-scored"),
        candidate=ScorerSpec(PrefixConstantScorer(), {"implementation": "text-test", "revision": "prefix-constant-v1"}),
        label_exclusion=NO_LABELS,
        token_budget=total_tokens,
    )

    with pytest.raises(ValueError, match="candidate scores are constant on the rung raw prefix"):
        select_scored_pool(
            scored,
            ctx=zephyr_context,
            output_path=str(tmp_path / "constant-prefix-small"),
            training_tokens=1,
            selection_method=SelectionMethod.CANDIDATE,
            tie_seed=5,
            num_ranges=8,
        )
    selected = select_scored_pool(
        scored,
        ctx=zephyr_context,
        output_path=str(tmp_path / "constant-prefix-large"),
        training_tokens=2,
        selection_method=SelectionMethod.CANDIDATE,
        tie_seed=5,
        num_ranges=8,
    )

    assert selected.raw_prefix_tokens > 20


def test_training_order_does_not_follow_score_rank(raw_pool, zephyr_context, tmp_path):
    positive = score_raw_pool(
        raw_pool,
        ctx=zephyr_context,
        output_path=str(tmp_path / "positive"),
        candidate=_text_scorer(1, "positive-v1"),
        label_exclusion=NO_LABELS,
        token_budget=60,
    )
    inverted = score_raw_pool(
        raw_pool,
        ctx=zephyr_context,
        output_path=str(tmp_path / "inverted"),
        candidate=_text_scorer(-1, "inverted-v1"),
        label_exclusion=NO_LABELS,
        token_budget=60,
    )
    selections = [
        SelectedQualityPool(
            tokenizer=pool.tokenizer,
            tokenizer_hash=pool.tokenizer_hash,
            classifier_identity=pool.classifier_identity,
            selection_method=SelectionMethod.CANDIDATE,
            tie_seed=13,
            requested_tokens=pool.tokens,
            selected_tokens=pool.tokens,
            documents=pool.documents,
            raw_prefix_tokens=pool.tokens,
            raw_prefix_documents=pool.documents,
            raw_prefix_shards=pool.shards,
            raw_source_shards=pool.raw_prefix_shards,
            shards=pool.shards,
            report_path=pool.raw_manifest_path,
        )
        for pool in (positive, inverted)
    ]
    ordered_ids = []
    ordered_rows = []
    for index, selected in enumerate(selections):
        ordered_shards = materialize_selected_pool(
            selected,
            ctx=zephyr_context,
            output_path=str(tmp_path / f"training-order-{index}"),
        )
        rows = [row for path in ordered_shards for row in pq.read_table(path).to_pylist()]
        ordered_rows.append(rows)
        ordered_ids.append([row["id"] for row in rows])

    assert ordered_ids[0] == ordered_ids[1]
    assert set(ordered_ids[0]) == {str(index) for index in range(12)}
    assert [row["quality_score"] for row in ordered_rows[0]] != sorted(
        (row["quality_score"] for row in ordered_rows[0]), reverse=True
    )
    assert [row["quality_score"] for row in ordered_rows[1]] != sorted(
        (row["quality_score"] for row in ordered_rows[1]), reverse=True
    )


def test_frozen_label_duplicate_group_overlap_fails_across_sources(raw_pool, write_raw_pool, zephyr_context, tmp_path):
    rows = [row for path in raw_pool.shards for row in pq.read_table(path).to_pylist()]
    rows[0]["source"] = "another-source"
    overlapping_pool = write_raw_pool(rows, name="overlap", sources=raw_pool.sources)

    with pytest.raises(ZephyrWorkerError, match="overlaps a frozen train, development, or audit duplicate group"):
        score_raw_pool(
            overlapping_pool,
            ctx=zephyr_context,
            output_path=str(tmp_path / "overlapping-score"),
            candidate=_text_scorer(1, "positive-v1"),
            label_exclusion=LabelExclusion(
                label_revision="frozen-v1", duplicate_groups=frozenset({rows[0]["duplicate_group"]})
            ),
            token_budget=60,
        )
