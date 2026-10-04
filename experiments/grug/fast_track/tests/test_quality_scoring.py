# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray.local_backend import LocalClient
from fray.types import ResourceConfig
from zephyr.context import ZephyrContext
from zephyr.stage_io import ZephyrWorkerError

from experiments.grug.fast_track.corpus_sample import CorpusSource, RawCorpusPool
from experiments.grug.fast_track.label_exclusion import LabelExclusion
from experiments.grug.fast_track.quality_features import QualityFeatureSource, prepare_quality_features
from experiments.grug.fast_track.quality_pipeline import (
    SelectedQualityPool,
    SelectionMethod,
    materialize_selected_pool,
    score_raw_pool,
    select_scored_pool,
)
from experiments.grug.fast_track.ranked_pool import RangeTokenTotal


class _TextScorer:
    def __init__(self, direction: float):
        self.direction = direction

    def scores(self, batch):
        return np.asarray([self.direction * float(text.rsplit("-", 1)[1]) for text in batch.texts])

    def __call__(self):
        return _TextScorer(self.direction)


@pytest.fixture
def zephyr_context(tmp_path):
    client = LocalClient()
    context = ZephyrContext(
        client=client,
        max_workers=2,
        resources=ResourceConfig(cpu=1, ram="512m"),
        chunk_storage_prefix=str(tmp_path / "chunks"),
        name="quality-score-test",
    )
    yield context
    context.shutdown()
    client.shutdown(wait=True)


@pytest.fixture
def raw_pool(tmp_path):
    rows = []
    normalized_rows = []
    for index in range(12):
        tokens = [index + 1] * 5
        normalized_rows.append({"id": str(index), "text": f"text-{index}"})
        rows.append(
            {
                "source": "source",
                "id": str(index),
                "sample_rank": hashlib.sha256(f"seed:{index}".encode()).hexdigest(),
                "duplicate_group": hashlib.sha256(f"text-{index}".encode()).hexdigest(),
                "text": f"text-{index}",
                "normalized_shard": str(tmp_path / "normalized.parquet"),
                "normalized_row": index,
                "input_ids": tokens,
                "token_count": len(tokens),
            }
        )
    rows.sort(key=lambda row: (row["sample_rank"], row["source"], row["id"]))
    shard_a, shard_b = tmp_path / "raw-a.parquet", tmp_path / "raw-b.parquet"
    pq.write_table(pa.Table.from_pylist(normalized_rows), tmp_path / "normalized.parquet")
    embedding_path = tmp_path / "harrier" / "normalized.parquet"
    embedding_path.parent.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [{"id": str(index), "embedding": np.full(1024, index + 1, dtype=np.int8)} for index in range(12)],
            schema=pa.schema([("id", pa.string()), ("embedding", pa.list_(pa.int8(), 1024))]),
        ),
        embedding_path,
    )
    pq.write_table(pa.Table.from_pylist(rows[:6]), shard_a)
    pq.write_table(pa.Table.from_pylist(rows[6:]), shard_b)
    return RawCorpusPool(
        tokenizer="test-tokenizer",
        tokenizer_hash="a" * 64,
        sources=(CorpusSource("source", str(tmp_path), 60),),
        seed=19,
        requested_tokens=60,
        actual_tokens=60,
        documents=12,
        shards=(str(shard_a), str(shard_b)),
        range_totals=(
            RangeTokenTotal(0, str(shard_a), 6, 30),
            RangeTokenTotal(1, str(shard_b), 6, 30),
        ),
        source_tokens={"source": 60},
        manifest_path=str(tmp_path / "raw-manifest.json"),
    )


def test_text_scorer_selects_only_after_the_deterministic_raw_prefix(raw_pool, zephyr_context, tmp_path):
    positive = score_raw_pool(
        raw_pool,
        ctx=zephyr_context,
        output_path=str(tmp_path / "positive"),
        scorer_factory=_TextScorer(1),
        classifier_identity={"implementation": "text-test", "revision": "positive-v1"},
        label_exclusion=LabelExclusion(label_revision="no-labels-v1", duplicate_groups=frozenset()),
        token_budget=40,
        incumbent_scorer_factory=_TextScorer(-1),
        incumbent_identity={"implementation": "text-test", "revision": "incumbent-v1"},
    )
    negative = score_raw_pool(
        raw_pool,
        ctx=zephyr_context,
        output_path=str(tmp_path / "negative"),
        scorer_factory=_TextScorer(-1),
        classifier_identity={"implementation": "text-test", "revision": "negative-v1"},
        label_exclusion=LabelExclusion(label_revision="no-labels-v1", duplicate_groups=frozenset()),
        token_budget=40,
    )
    selected_positive = select_scored_pool(
        positive,
        ctx=zephyr_context,
        output_path=str(tmp_path / "positive-selection"),
        training_tokens=4,
        selection_method=SelectionMethod.CANDIDATE,
        tie_seed=3,
        num_ranges=8,
    )
    selected_negative = select_scored_pool(
        negative,
        ctx=zephyr_context,
        output_path=str(tmp_path / "negative-selection"),
        training_tokens=4,
        selection_method=SelectionMethod.CANDIDATE,
        tie_seed=3,
        num_ranges=8,
    )
    selected_incumbent = select_scored_pool(
        positive,
        ctx=zephyr_context,
        output_path=str(tmp_path / "incumbent-selection"),
        training_tokens=4,
        selection_method=SelectionMethod.INCUMBENT,
        tie_seed=3,
        num_ranges=8,
    )
    positive_ids = [row["id"] for path in selected_positive.shards for row in pq.read_table(path).to_pylist()]
    negative_ids = [row["id"] for path in selected_negative.shards for row in pq.read_table(path).to_pylist()]
    incumbent_ids = [row["id"] for path in selected_incumbent.shards for row in pq.read_table(path).to_pylist()]
    raw_prefix_ids = [
        row["id"] for path in selected_positive.raw_prefix_shards for row in pq.read_table(path).to_pylist()
    ]

    assert positive_ids != negative_ids
    assert positive_ids == [max(raw_prefix_ids, key=int)]
    assert negative_ids == [min(raw_prefix_ids, key=int)]
    assert incumbent_ids == negative_ids
    assert selected_positive.raw_prefix_tokens == selected_negative.raw_prefix_tokens == 40
    assert selected_positive.requested_tokens == selected_negative.requested_tokens == 4
    assert selected_positive.usable_tokens == selected_negative.usable_tokens == 4
    assert positive.classifier_identity["revision"] == "positive-v1"
    assert "text" not in pq.read_table(positive.shards[0]).column_names
    assert "input_ids" not in pq.read_table(positive.shards[0]).column_names


def test_embedding_scorer_receives_id_aligned_cached_features(raw_pool, zephyr_context, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "experiments.grug.fast_track.quality_features.hero_data.harrier", lambda _source: str(tmp_path / "harrier")
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
        scorer_factory=EmbeddingScorer(),
        classifier_identity={"implementation": "embedding-test", "revision": "v1"},
        label_exclusion=LabelExclusion(label_revision="no-labels-v1", duplicate_groups=frozenset()),
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
        scorer_factory=_TextScorer(1),
        classifier_identity={"implementation": "text-test", "revision": "test-v1"},
        label_exclusion=LabelExclusion(label_revision="no-labels-v1", duplicate_groups=frozenset()),
        token_budget=40,
    )
    inverted = score_raw_pool(
        raw_pool,
        ctx=zephyr_context,
        output_path=str(tmp_path / "inverted"),
        scorer_factory=_TextScorer(-1),
        classifier_identity={"implementation": "text-test", "revision": "inverted-v1"},
        label_exclusion=LabelExclusion(label_revision="no-labels-v1", duplicate_groups=frozenset()),
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
    assert random_a.usable_tokens == random_b.usable_tokens == 4


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
        scorer_factory=ConstantScorer(),
        classifier_identity={"implementation": "constant-test", "revision": "v1"},
        label_exclusion=LabelExclusion(label_revision="no-labels-v1", duplicate_groups=frozenset()),
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

    assert random.usable_tokens == 4


def test_larger_training_budget_expands_the_raw_candidate_prefix(raw_pool, zephyr_context, tmp_path):
    small_scored = score_raw_pool(
        raw_pool,
        ctx=zephyr_context,
        output_path=str(tmp_path / "scored-small"),
        scorer_factory=_TextScorer(1),
        classifier_identity={"implementation": "text-test", "revision": "v1"},
        label_exclusion=LabelExclusion(label_revision="no-labels-v1", duplicate_groups=frozenset()),
        token_budget=40,
    )
    large_scored = score_raw_pool(
        raw_pool,
        ctx=zephyr_context,
        output_path=str(tmp_path / "scored-large"),
        scorer_factory=_TextScorer(1),
        classifier_identity={"implementation": "text-test", "revision": "v1"},
        label_exclusion=LabelExclusion(label_revision="no-labels-v1", duplicate_groups=frozenset()),
        token_budget=60,
    )
    small = select_scored_pool(
        small_scored,
        ctx=zephyr_context,
        output_path=str(tmp_path / "small"),
        training_tokens=4,
        selection_method=SelectionMethod.CANDIDATE,
        tie_seed=5,
        num_ranges=8,
    )
    large = select_scored_pool(
        large_scored,
        ctx=zephyr_context,
        output_path=str(tmp_path / "large"),
        training_tokens=6,
        selection_method=SelectionMethod.CANDIDATE,
        tie_seed=5,
        num_ranges=8,
    )
    small_candidates = {row["id"] for path in small.raw_prefix_shards for row in pq.read_table(path).to_pylist()}
    large_candidates = {row["id"] for path in large.raw_prefix_shards for row in pq.read_table(path).to_pylist()}

    assert len(small_candidates) == 8
    assert len(large_candidates) == 12
    assert small_candidates < large_candidates


def test_training_order_does_not_follow_score_rank(raw_pool, zephyr_context, tmp_path):
    positive = score_raw_pool(
        raw_pool,
        ctx=zephyr_context,
        output_path=str(tmp_path / "positive"),
        scorer_factory=_TextScorer(1),
        classifier_identity={"implementation": "text-test", "revision": "positive-v1"},
        label_exclusion=LabelExclusion(label_revision="no-labels-v1", duplicate_groups=frozenset()),
        token_budget=60,
    )
    inverted = score_raw_pool(
        raw_pool,
        ctx=zephyr_context,
        output_path=str(tmp_path / "inverted"),
        scorer_factory=_TextScorer(-1),
        classifier_identity={"implementation": "text-test", "revision": "inverted-v1"},
        label_exclusion=LabelExclusion(label_revision="no-labels-v1", duplicate_groups=frozenset()),
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
            usable_tokens=pool.tokens,
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


def test_frozen_label_duplicate_group_overlap_fails_across_sources(raw_pool, zephyr_context, tmp_path):
    first = pq.read_table(raw_pool.shards[0]).to_pylist()
    first[0]["source"] = "another-source"
    overlap_path = tmp_path / "overlap.parquet"
    pq.write_table(pa.Table.from_pylist(first), overlap_path)
    overlapping_pool = RawCorpusPool(
        tokenizer=raw_pool.tokenizer,
        tokenizer_hash=raw_pool.tokenizer_hash,
        sources=raw_pool.sources,
        seed=raw_pool.seed,
        requested_tokens=raw_pool.requested_tokens,
        actual_tokens=raw_pool.actual_tokens,
        documents=raw_pool.documents,
        shards=(str(overlap_path), *raw_pool.shards[1:]),
        range_totals=(
            RangeTokenTotal(0, str(overlap_path), 6, 30),
            raw_pool.range_totals[1],
        ),
        source_tokens=raw_pool.source_tokens,
        manifest_path=raw_pool.manifest_path,
    )

    with pytest.raises(ZephyrWorkerError, match="overlaps a frozen train, development, or audit duplicate group"):
        score_raw_pool(
            overlapping_pool,
            ctx=zephyr_context,
            output_path=str(tmp_path / "overlapping-score"),
            scorer_factory=_TextScorer(1),
            classifier_identity={"implementation": "text-test", "revision": "positive-v1"},
            label_exclusion=LabelExclusion(
                label_revision="frozen-v1", duplicate_groups=frozenset({first[0]["duplicate_group"]})
            ),
            token_budget=60,
        )
