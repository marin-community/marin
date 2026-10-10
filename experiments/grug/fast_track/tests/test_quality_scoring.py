# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray.local_backend import LocalClient
from fray.types import ResourceConfig
from levanter.store.cache import CacheMetadata, TreeCache
from zephyr.context import ZephyrContext
from zephyr.stage_io import ZephyrWorkerError

from experiments.grug.fast_track.corpus_sample import CorpusSource, RawCorpusPool
from experiments.grug.fast_track.label_exclusion import LabelExclusion
from experiments.grug.fast_track.quality_features import QualityFeatureSource, prepare_quality_features
from experiments.grug.fast_track.quality_pipeline import (
    SelectedQualityPool,
    SelectionMethod,
    _write_quality_token_cache,
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


def _variable_token_pool(raw_pool, tmp_path, name):
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
                "normalized_shard": str(tmp_path / "normalized.parquet"),
                "normalized_row": index,
                "input_ids": [index + 1] * token_count,
                "token_count": token_count,
            }
        )
    paths = (tmp_path / f"{name}-a.parquet", tmp_path / f"{name}-b.parquet")
    pq.write_table(pa.Table.from_pylist(rows[:6]), paths[0])
    pq.write_table(pa.Table.from_pylist(rows[6:]), paths[1])
    range_totals = tuple(
        RangeTokenTotal(
            range_key,
            str(path),
            len(range_rows),
            sum(row["token_count"] for row in range_rows),
        )
        for range_key, path, range_rows in (
            (0, paths[0], rows[:6]),
            (1, paths[1], rows[6:]),
        )
    )
    total_tokens = sum(item.tokens for item in range_totals)
    return RawCorpusPool(
        tokenizer=raw_pool.tokenizer,
        tokenizer_hash=raw_pool.tokenizer_hash,
        sources=raw_pool.sources,
        seed=raw_pool.seed,
        requested_tokens=total_tokens,
        actual_tokens=total_tokens,
        documents=len(rows),
        shards=tuple(str(path) for path in paths),
        range_totals=range_totals,
        source_tokens={"source": total_tokens},
        manifest_path=str(tmp_path / f"{name}-manifest.json"),
    )


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
    report = json.loads(Path(selected_positive.report_path).read_text())
    assert report["rung_prefix_shards"] == list(selected_positive.raw_prefix_shards)
    assert "scored_pool_shards" not in report


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


def test_scoring_accepts_explicit_feature_output_pin_and_rejects_mismatches(
    raw_pool, zephyr_context, tmp_path, monkeypatch
):
    default_harrier_path = str(tmp_path / "harrier")
    monkeypatch.setattr(
        "experiments.grug.fast_track.quality_features.hero_data.harrier", lambda _source: default_harrier_path
    )
    custom_harrier_path = tmp_path / "fresh-harrier"
    custom_harrier_path.mkdir()
    pq.write_table(
        pq.read_table(tmp_path / "harrier" / "normalized.parquet"),
        custom_harrier_path / "normalized.parquet",
    )
    custom_source = QualityFeatureSource("source", raw_pool.sources[0].normalized_path, str(custom_harrier_path))
    prepared = prepare_quality_features(
        raw_pool,
        ctx=zephyr_context,
        output_path=str(tmp_path / "explicit-feature-pins"),
        token_budget=40,
        sources=(custom_source,),
    )

    class EmbeddingScorer:
        def scores(self, batch):
            return batch.embeddings[:, 0]

        def __call__(self):
            return self

    score_args = {
        "ctx": zephyr_context,
        "scorer_factory": EmbeddingScorer(),
        "classifier_identity": {"implementation": "embedding-test", "revision": "fresh-v1"},
        "label_exclusion": LabelExclusion(label_revision="no-labels-v1", duplicate_groups=frozenset()),
        "token_budget": 40,
        "prepared_features": prepared,
    }
    scored = score_raw_pool(
        raw_pool,
        output_path=str(tmp_path / "explicit-feature-pin-score"),
        expected_feature_sources=(custom_source,),
        **score_args,
    )
    assert scored.feature_identity == prepared.feature_identity
    with pytest.raises(ValueError, match="prepared feature pool"):
        score_raw_pool(raw_pool, output_path=str(tmp_path / "implicit-feature-pin-score"), **score_args)
    with pytest.raises(ValueError, match="expected feature normalized paths"):
        score_raw_pool(
            raw_pool,
            output_path=str(tmp_path / "wrong-feature-pin-score"),
            expected_feature_sources=(
                QualityFeatureSource("source", "wrong-normalized-path", str(custom_harrier_path)),
            ),
            **score_args,
        )
    with pytest.raises(ValueError, match="require a prepared feature pool"):
        score_raw_pool(
            raw_pool,
            ctx=zephyr_context,
            output_path=str(tmp_path / "pins-without-features"),
            scorer_factory=EmbeddingScorer(),
            classifier_identity={"implementation": "embedding-test", "revision": "raw-only"},
            label_exclusion=LabelExclusion(label_revision="no-labels-v1", duplicate_groups=frozenset()),
            token_budget=40,
            expected_feature_sources=(custom_source,),
        )


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


def test_max_scored_pool_matches_exact_rung_pools_through_raw_cache(raw_pool, zephyr_context, tmp_path):
    pool = _variable_token_pool(raw_pool, tmp_path, "variable")
    total_tokens = pool.actual_tokens
    exclusion = LabelExclusion(label_revision="no-labels-v1", duplicate_groups=frozenset())
    classifier = {"implementation": "text-test", "revision": "variable-v1"}
    max_scored = score_raw_pool(
        pool,
        ctx=zephyr_context,
        output_path=str(tmp_path / "max-scored"),
        scorer_factory=_TextScorer(1),
        classifier_identity=classifier,
        label_exclusion=exclusion,
        token_budget=total_tokens,
    )
    exact_small = score_raw_pool(
        pool,
        ctx=zephyr_context,
        output_path=str(tmp_path / "exact-small"),
        scorer_factory=_TextScorer(1),
        classifier_identity=classifier,
        label_exclusion=exclusion,
        token_budget=10,
    )
    exact_large = score_raw_pool(
        pool,
        ctx=zephyr_context,
        output_path=str(tmp_path / "exact-large"),
        scorer_factory=_TextScorer(1),
        classifier_identity=classifier,
        label_exclusion=exclusion,
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


def test_max_scored_pool_checks_constant_scores_on_each_rung_prefix(raw_pool, zephyr_context, tmp_path):
    class PrefixConstantScorer:
        def scores(self, batch):
            return np.asarray(
                [0.0 if int(document_id) < 4 else float(document_id) for document_id in batch.document_ids]
            )

        def __call__(self):
            return self

    pool = _variable_token_pool(raw_pool, tmp_path, "constant-prefix")
    total_tokens = pool.actual_tokens
    scored = score_raw_pool(
        pool,
        ctx=zephyr_context,
        output_path=str(tmp_path / "constant-prefix-scored"),
        scorer_factory=PrefixConstantScorer(),
        classifier_identity={"implementation": "text-test", "revision": "prefix-constant-v1"},
        label_exclusion=LabelExclusion(label_revision="no-labels-v1", duplicate_groups=frozenset()),
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
