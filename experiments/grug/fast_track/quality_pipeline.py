# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score a raw corpus pool and materialize fixed-budget quality selections."""

import hashlib
import json
import logging
import math
import struct
from bisect import bisect_right
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from functools import partial

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from levanter.data.text.datasets import LmDataConfig
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.store.cache import CacheMetadata, SerialCacheWriter
from levanter.tokenizers import load_tokenizer, tokenizer_content_hash
from marin.execution.artifact import Artifact, read_artifact
from marin.execution.build_context import resolve_version
from marin.execution.fingerprint import canonical_json
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.namespacing import user_namespaced_name
from marin.processing.tokenize.tokenize import TokenizedCache
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset, ShardInfo
from zephyr.readers import load_parquet
from zephyr.writers import ensure_parent_dir

from experiments.grug.fast_track.contracts import FrozenBaselineComponent, FrozenBaselineManifest, ResolvedTrainingBudget
from experiments.grug.fast_track.corpus_sample import RawCorpusPool
from experiments.grug.fast_track.label_exclusion import LabelExclusion
from experiments.grug.fast_track.launch import FlatCacheTrainingSource
from experiments.grug.fast_track.quality import (
    DocumentQualityScorer,
    EmbeddingHeadScorer,
    QualityScoringBatch,
    quality_classifier_identity,
)
from experiments.grug.fast_track.quality_features import (
    HARRIER_FEATURE_IDENTITY,
    PreparedQualityPool,
    QualityFeatureSource,
    normalize_harrier_embeddings,
    pinned_quality_feature_sources,
    prepare_quality_features,
)
from experiments.grug.fast_track.quality_labels import FittedRidgeQualityHead
from experiments.grug.fast_track.ranked_pool import (
    PrefixResult,
    RangeTokenTotal,
    RankedPool,
    ranked_pool,
    take_token_prefix,
)

logger = logging.getLogger(__name__)

QUALITY_CACHE_MAX_PENDING_BYTES = 64 * 1024 * 1024
QUALITY_CACHE_MAX_PENDING_DOCUMENTS = 16_384
TOKEN_CACHE_NAME = "tokens"
QUALITY_FRACTION = 0.1
SCORER_BATCH_ROWS = 128
SCORE_COLUMN = "quality_score"
SCORE_SAMPLE_ROWS = 512
QUALITY_WORKER_RESOURCES = ResourceConfig(cpu=2, ram="8g", disk="8g")
QUALITY_POOL_WORKERS = 8


class SelectionMethod(StrEnum):
    CANDIDATE = "candidate"
    INCUMBENT = "incumbent"
    RANDOM = "random"


@dataclass(frozen=True)
class SelectedQualityPool:
    """Selected document locators and provenance for one rung."""

    tokenizer: str
    tokenizer_hash: str
    classifier_identity: dict[str, str | int | float | bool]
    selection_method: SelectionMethod
    tie_seed: int
    requested_tokens: int
    selected_tokens: int
    usable_tokens: int
    documents: int
    raw_prefix_tokens: int
    raw_prefix_documents: int
    raw_prefix_shards: tuple[str, ...]
    raw_source_shards: tuple[str, ...]
    shards: tuple[str, ...]
    report_path: str


class QualityData(Artifact):
    tokenizer: str
    tokenizer_hash: str
    cache_dir: str
    requested_tokens: int
    actual_tokens: int
    report_path: str


class ScoredPool(Artifact):
    """A bounded raw token pool scored once by a frozen document classifier."""

    tokenizer: str
    tokenizer_hash: str
    raw_manifest_path: str
    raw_seed: int
    classifier_identity: dict[str, str | int | float | bool]
    incumbent_identity: dict[str, str | int | float | bool] | None
    label_revision: str
    label_exclusion_fingerprint: str
    documents: int
    tokens: int
    requested_tokens: int
    raw_prefix_shards: tuple[str, ...]
    shards: tuple[str, ...]
    range_totals: tuple[RangeTokenTotal, ...]
    feature_identity: dict[str, str | int | float] | None


@dataclass(frozen=True)
class _ScoreColumnSummary:
    minimum: float
    maximum: float
    rank_sample: tuple[tuple[float, str, str], ...]


@dataclass(frozen=True)
class _ScoringPrefix:
    raw_prefix: PrefixResult
    feature_paths: tuple[str, ...] | None
    feature_identity: dict[str, str | int | float] | None


@dataclass(frozen=True)
class QualitySpec:
    scored_pool: ArtifactStep[ScoredPool]
    training_tokens: int
    selection_method: SelectionMethod
    tie_seed: int


@dataclass(frozen=True)
class QualityConfig:
    training_tokens: int
    selection_method: SelectionMethod
    tie_seed: int
    scored_pool: ScoredPool
    output_path: str


@dataclass(frozen=True)
class QualityBuildConfig:
    scored_pool_path: str
    training_tokens: int
    selection_method: SelectionMethod
    tie_seed: int
    output_path: str
    context_name: str
    resources: ResourceConfig


@dataclass(frozen=True)
class ScoredPoolBuildConfig:
    raw_pool_path: str
    prepared_features_path: str | None
    token_budget: int
    output_path: str
    context_name: str
    resources: ResourceConfig


@dataclass(frozen=True)
class QualityFeatureBuildConfig:
    raw_pool_path: str
    token_budget: int
    output_path: str
    context_name: str
    resources: ResourceConfig


def _has_pinned_harrier_identity(identity: dict[str, str | int | float]) -> bool:
    return all(identity.get(key) == value for key, value in HARRIER_FEATURE_IDENTITY.items())


def _prepare_scoring_prefix(
    pool: RawCorpusPool,
    *,
    ctx: ZephyrContext,
    output_path: str,
    token_budget: int,
    prepared_features: PreparedQualityPool | None,
    expected_feature_sources: Sequence[QualityFeatureSource] | None,
) -> _ScoringPrefix:
    if prepared_features is None:
        raw_prefix = take_token_prefix(
            RankedPool(pool.shards, pool.range_totals, pool.documents, pool.actual_tokens),
            ctx=ctx,
            output_path=prefix_join(output_path, "raw_prefix"),
            token_budget=token_budget,
        )
        return _ScoringPrefix(raw_prefix, None, None)

    expected_sources = tuple(
        sorted(
            (
                pinned_quality_feature_sources(pool.sources)
                if expected_feature_sources is None
                else expected_feature_sources
            ),
            key=lambda item: item.source,
        )
    )
    if not expected_sources or len({source.source for source in expected_sources}) != len(expected_sources):
        raise ValueError("expected feature source pins must be non-empty and unique")
    source_paths = {source.name: source.normalized_path for source in pool.sources}
    if {source.source for source in expected_sources} != source_paths.keys():
        raise ValueError("expected feature source pins must match the raw pool sources")
    if any(source.normalized_path != source_paths[source.source] for source in expected_sources):
        raise ValueError("expected feature normalized paths must match the raw pool source pins")
    if (
        prepared_features.raw_manifest_path != pool.manifest_path
        or prepared_features.raw_seed != pool.seed
        or prepared_features.tokenizer_hash != pool.tokenizer_hash
        or prepared_features.requested_tokens != token_budget
        or prepared_features.sources != expected_sources
    ):
        raise ValueError("prepared feature pool does not match the raw scoring pool and exact token budget")
    raw_prefix = PrefixResult(
        prepared_features.raw_prefix_shards,
        token_budget,
        prepared_features.documents,
        prepared_features.actual_tokens,
        prepared_features.actual_tokens - token_budget,
    )
    feature_paths = tuple(item.path for item in prepared_features.feature_shards)
    if len(feature_paths) != len(raw_prefix.shards):
        raise ValueError("prepared features do not match the raw prefix shard count")
    feature_identity = prepared_features.feature_identity
    if not _has_pinned_harrier_identity(feature_identity):
        raise ValueError("prepared quality pool has different Harrier feature pins")
    return _ScoringPrefix(raw_prefix, feature_paths, feature_identity)


def _prepare_scoring_batch(
    rows: list[dict],
    *,
    feature_batches: Iterator[pa.RecordBatch] | None,
    row_offset: int,
    label_exclusion: LabelExclusion,
) -> tuple[QualityScoringBatch, list[str]]:
    texts = [row["text"] for row in rows]
    if any(not isinstance(text, str) for text in texts):
        raise ValueError("raw corpus records require string text")

    token_hashes = []
    for row in rows:
        if row["duplicate_group"] in label_exclusion.duplicate_groups:
            raise ValueError(
                "raw quality pool overlaps a frozen train, development, or audit duplicate group "
                f"({row['source']}/{row['id']})"
            )
        if row["id"] in label_exclusion.normalized_document_ids:
            raise ValueError(f"raw quality pool overlaps a frozen normalized label ID ({row['source']}/{row['id']})")
        tokens = np.asarray(row["input_ids"], dtype="<i4")
        if len(tokens) != row["token_count"]:
            raise ValueError("raw corpus token_count differs from input_ids length")
        token_hashes.append(hashlib.sha256(tokens.tobytes()).hexdigest())

    embeddings = None
    if feature_batches is not None:
        try:
            feature_batch = next(feature_batches)
        except StopIteration as exc:
            raise ValueError("prepared feature shard ended before its raw prefix shard") from exc
        feature_rows = feature_batch.to_pylist()
        if len(feature_rows) != len(rows):
            raise ValueError("prepared feature and raw prefix batch lengths differ")
        for row_index, (raw_row, feature_row) in enumerate(zip(rows, feature_rows, strict=True)):
            if raw_row["id"] != feature_row["id"] or feature_row["raw_row_index"] != row_offset + row_index:
                raise ValueError("prepared feature ID or row position differs from the raw prefix")
        embeddings = normalize_harrier_embeddings(np.asarray([row["embedding"] for row in feature_rows], dtype=np.int8))
    return QualityScoringBatch(texts, [row["id"] for row in rows], embeddings), token_hashes


def _score_raw_file(
    file: dict[str, str | int],
    *,
    output_path: str,
    scorer: DocumentQualityScorer,
    incumbent_scorer: DocumentQualityScorer | None,
    label_exclusion: LabelExclusion,
) -> RangeTokenTotal:
    path = prefix_join(output_path, f"part-{int(file['index']):05d}.parquet")
    ensure_parent_dir(path)
    documents = 0
    tokens_total = 0
    writer = None
    with ExitStack() as stack:
        source = stack.enter_context(StoragePath(str(file["path"])).open("rb"))
        target = stack.enter_context(StoragePath(path).open("wb"))
        feature_batches = None
        if "feature_path" in file:
            feature_stream = stack.enter_context(StoragePath(str(file["feature_path"])).open("rb"))
            feature_batches = iter(pq.ParquetFile(feature_stream).iter_batches(batch_size=SCORER_BATCH_ROWS))
        for batch in pq.ParquetFile(source).iter_batches(batch_size=SCORER_BATCH_ROWS):
            rows = batch.to_pylist()
            scoring_batch, token_hashes = _prepare_scoring_batch(
                rows,
                feature_batches=feature_batches,
                row_offset=documents,
                label_exclusion=label_exclusion,
            )
            scores = np.asarray(scorer.scores(scoring_batch), dtype=np.float64)
            if scores.shape != (len(rows),) or not np.isfinite(scores).all():
                raise ValueError("text scorer must return one finite score per document")
            incumbent_scores = None
            if incumbent_scorer is not None:
                incumbent_scores = np.asarray(incumbent_scorer.scores(scoring_batch), dtype=np.float64)
                if incumbent_scores.shape != (len(rows),) or not np.isfinite(incumbent_scores).all():
                    raise ValueError("incumbent scorer must return one finite score per document")
            output = []
            for index, (row, score) in enumerate(zip(rows, scores, strict=True)):
                scored_row = {key: value for key, value in row.items() if key not in {"text", "input_ids"}}
                scored_row.update(
                    {
                        "raw_shard_index": int(file["index"]),
                        "raw_row_index": documents + index,
                        "token_sha256": token_hashes[index],
                        SCORE_COLUMN: float(score),
                    }
                )
                if incumbent_scores is not None:
                    scored_row["incumbent_score"] = float(incumbent_scores[index])
                output.append(scored_row)
            table = pa.Table.from_pylist(output)
            if writer is None:
                writer = stack.enter_context(pq.ParquetWriter(target, table.schema))
            writer.write_table(table)
            documents += len(rows)
            tokens_total += sum(row["token_count"] for row in rows)
        if feature_batches is not None:
            try:
                next(feature_batches)
            except StopIteration:
                pass
            else:
                raise ValueError("prepared feature shard has rows beyond its raw prefix shard")
    if documents == 0:
        raise ValueError("raw scoring shard must contain at least one document")
    return RangeTokenTotal(int(file["index"]), path, documents, tokens_total)


def score_raw_pool(
    pool: RawCorpusPool,
    *,
    ctx: ZephyrContext,
    output_path: str,
    scorer_factory: Callable[[], DocumentQualityScorer],
    classifier_identity: dict[str, str | int | float | bool],
    label_exclusion: LabelExclusion,
    token_budget: int,
    prepared_features: PreparedQualityPool | None = None,
    expected_feature_sources: Sequence[QualityFeatureSource] | None = None,
    incumbent_scorer_factory: Callable[[], DocumentQualityScorer] | None = None,
    incumbent_identity: dict[str, str | int | float | bool] | None = None,
) -> ScoredPool:
    """Score one bounded raw token prefix and write document locators with scores."""
    classifier_identity = quality_classifier_identity(classifier_identity)
    if token_budget <= 0:
        raise ValueError("scoring token budget must be positive")
    if token_budget > pool.requested_tokens:
        raise ValueError(
            f"raw corpus pool was limited to {pool.requested_tokens} requested tokens, below the "
            f"{token_budget} scoring budget"
        )
    if prepared_features is None and expected_feature_sources is not None:
        raise ValueError("expected feature source pins require a prepared feature pool")
    if (incumbent_scorer_factory is None) != (incumbent_identity is None):
        raise ValueError("incumbent scorer and identity must be supplied together")
    if incumbent_identity is not None:
        incumbent_identity = quality_classifier_identity(incumbent_identity)
    label_exclusion_fingerprint = hashlib.sha256(canonical_json(label_exclusion.identity()).encode()).hexdigest()
    scoring_prefix = _prepare_scoring_prefix(
        pool,
        ctx=ctx,
        output_path=output_path,
        token_budget=token_budget,
        prepared_features=prepared_features,
        expected_feature_sources=expected_feature_sources,
    )
    raw_prefix = scoring_prefix.raw_prefix
    feature_paths = scoring_prefix.feature_paths
    feature_identity = scoring_prefix.feature_identity

    def score_shard(files: Iterator[dict[str, str | int]], _: ShardInfo) -> Iterator[RangeTokenTotal]:
        scorer = scorer_factory()
        incumbent_scorer = incumbent_scorer_factory() if incumbent_scorer_factory is not None else None
        for file in files:
            yield _score_raw_file(
                file,
                output_path=output_path,
                scorer=scorer,
                incumbent_scorer=incumbent_scorer,
                label_exclusion=label_exclusion,
            )

    file_records = []
    for index, path in enumerate(raw_prefix.shards):
        file_record: dict[str, str | int] = {"index": index, "path": path}
        if feature_paths is not None:
            file_record["feature_path"] = feature_paths[index]
        file_records.append(file_record)
    files = Dataset.from_list(file_records).reshard(len(raw_prefix.shards))
    summaries = ctx.execute(files.map_shard(score_shard)).results
    ranges = tuple(sorted(summaries, key=lambda item: item.range_key))
    if (
        sum(item.documents for item in ranges) != raw_prefix.total_documents
        or sum(item.tokens for item in ranges) != raw_prefix.total_tokens
    ):
        raise ValueError("scored pool totals differ from the raw prefix manifest")
    paths = [item.path for item in ranges]
    return ScoredPool(
        tokenizer=pool.tokenizer,
        tokenizer_hash=pool.tokenizer_hash,
        raw_manifest_path=pool.manifest_path,
        raw_seed=pool.seed,
        classifier_identity=classifier_identity,
        incumbent_identity=incumbent_identity,
        label_revision=label_exclusion.label_revision,
        label_exclusion_fingerprint=label_exclusion_fingerprint,
        documents=raw_prefix.total_documents,
        tokens=raw_prefix.total_tokens,
        requested_tokens=token_budget,
        raw_prefix_shards=raw_prefix.shards,
        shards=tuple(paths),
        range_totals=ranges,
        feature_identity=feature_identity,
    )


def select_scored_pool(
    pool: ScoredPool,
    *,
    ctx: ZephyrContext,
    output_path: str,
    training_tokens: int,
    selection_method: SelectionMethod,
    tie_seed: int,
    num_ranges: int = 256,
) -> SelectedQualityPool:
    """Take a raw random prefix, then rank only that rung pool by score."""
    if training_tokens <= 0:
        raise ValueError("training token budget must be positive")
    score_column = SCORE_COLUMN
    classifier_identity = pool.classifier_identity
    if selection_method is SelectionMethod.INCUMBENT:
        if pool.incumbent_identity is None:
            raise ValueError("scored pool does not contain incumbent scores")
        score_column = "incumbent_score"
        classifier_identity = pool.incumbent_identity
    raw_token_budget = math.ceil(training_tokens / QUALITY_FRACTION)
    if raw_token_budget > pool.requested_tokens:
        raise ValueError(
            f"scored raw prefix budget {pool.requested_tokens} is below the {raw_token_budget} token rung prefix"
        )
    raw_prefix = take_token_prefix(
        RankedPool(pool.shards, pool.range_totals, pool.documents, pool.tokens),
        ctx=ctx,
        output_path=prefix_join(output_path, "raw_prefix"),
        token_budget=raw_token_budget,
    )

    def summarize(files: Iterator[dict[str, str | int]], _: ShardInfo) -> Iterator[_ScoreColumnSummary]:
        for file in files:
            minimum = math.inf
            maximum = -math.inf
            rank_sample = []
            with StoragePath(str(file["path"])).open("rb") as stream:
                for batch in pq.ParquetFile(stream).iter_batches(
                    batch_size=SCORER_BATCH_ROWS, columns=[score_column, "source", "id"]
                ):
                    rows = batch.to_pylist()
                    scores = [float(row[score_column]) for row in rows]
                    minimum = min(minimum, min(scores))
                    maximum = max(maximum, max(scores))
                    if len(rank_sample) < SCORE_SAMPLE_ROWS:
                        rank_sample.extend(
                            (score, row["source"], row["id"]) for row, score in zip(rows, scores, strict=True)
                        )
                        rank_sample = rank_sample[:SCORE_SAMPLE_ROWS]
            yield _ScoreColumnSummary(minimum, maximum, tuple(rank_sample))

    prefix_files = Dataset.from_list([{"path": path} for path in raw_prefix.shards]).reshard(len(raw_prefix.shards))
    summaries = ctx.execute(prefix_files.map_shard(summarize)).results
    minimum = min(item.minimum for item in summaries)
    maximum = max(item.maximum for item in summaries)
    rank_sample = tuple(sample for item in summaries for sample in item.rank_sample)
    if selection_method is not SelectionMethod.RANDOM and minimum == maximum:
        raise ValueError(f"{selection_method.value} scores are constant on the rung raw prefix")

    prefix_records = Dataset.from_list(list(raw_prefix.shards)).flat_map(load_parquet)

    def tie_hash(source: str, document_id: str) -> str:
        return hashlib.sha256(f"{tie_seed}:{source}:{document_id}".encode()).hexdigest()

    def descending_score_key(score: float) -> str:
        bits = int.from_bytes(struct.pack(">d", score), "big")
        ordered = (~bits & ((1 << 64) - 1)) if bits >> 63 else bits ^ (1 << 63)
        return f"{(~ordered) & ((1 << 64) - 1):016x}"

    def row_rank(row: dict) -> str:
        tie = tie_hash(row["source"], row["id"])
        if selection_method is SelectionMethod.RANDOM:
            return f"{tie}:{row['source']}:{row['id']}"
        return f"{descending_score_key(float(row[score_column]))}:{tie}:{row['source']}:{row['id']}"

    if selection_method is SelectionMethod.RANDOM:
        rank_boundaries = ()
    else:
        sample_ranks = sorted(
            f"{descending_score_key(score)}:{tie_hash(source, document_id)}:{source}:{document_id}"
            for score, source, document_id in rank_sample
        )
        if not sample_ranks:
            raise ValueError("scored pool has no rank sample")
        rank_boundaries = tuple(
            sample_ranks[min(len(sample_ranks) - 1, (len(sample_ranks) * index) // num_ranges)]
            for index in range(1, num_ranges)
        )

    def score_range(row: dict) -> int:
        if selection_method is SelectionMethod.RANDOM:
            return int(row_rank(row)[:8], 16) * num_ranges // (1 << 32)
        return bisect_right(rank_boundaries, row_rank(row))

    ranked = ranked_pool(
        prefix_records,
        ctx=ctx,
        output_path=prefix_join(output_path, "score_order"),
        rank_key=row_rank,
        token_count=lambda row: row["token_count"],
        range_key=score_range,
        num_ranges=num_ranges,
    )
    selection = take_token_prefix(
        ranked,
        ctx=ctx,
        output_path=prefix_join(output_path, "selected"),
        token_budget=training_tokens,
    )
    report_path = prefix_join(output_path, "selection.json")
    report = {
        "raw_manifest_path": pool.raw_manifest_path,
        "rung_prefix_shards": raw_prefix.shards,
        "classifier_identity": classifier_identity,
        "raw_prefix_requested_tokens": raw_token_budget,
        "raw_prefix_usable_tokens": raw_prefix.usable_tokens,
        "raw_prefix_selected_tokens": raw_prefix.total_tokens,
        "raw_prefix_documents": raw_prefix.total_documents,
        "selection_method": selection_method.value,
        "tie_seed": tie_seed,
        "requested_training_tokens": training_tokens,
        "selected_tokens": selection.total_tokens,
        "usable_tokens": selection.usable_tokens,
        "selected_documents": selection.total_documents,
        "score_bounds": {"minimum": minimum, "maximum": maximum},
        "selected_shards": selection.shards,
    }
    with StoragePath(report_path).open("w") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return SelectedQualityPool(
        tokenizer=pool.tokenizer,
        tokenizer_hash=pool.tokenizer_hash,
        classifier_identity=classifier_identity,
        selection_method=selection_method,
        tie_seed=tie_seed,
        requested_tokens=training_tokens,
        selected_tokens=selection.total_tokens,
        usable_tokens=selection.usable_tokens,
        documents=selection.total_documents,
        raw_prefix_tokens=raw_prefix.total_tokens,
        raw_prefix_documents=raw_prefix.total_documents,
        raw_prefix_shards=raw_prefix.shards,
        raw_source_shards=pool.raw_prefix_shards,
        shards=selection.shards,
        report_path=report_path,
    )


def materialize_selected_pool(selected: SelectedQualityPool, *, ctx: ZephyrContext, output_path: str) -> tuple[str, ...]:
    """Fetch selected tokens and write them in the score-independent raw sample order."""
    raw_paths = selected.raw_source_shards

    def write_selected_range(raw_shard_index: int, locators: Iterator[dict]) -> RangeTokenTotal:
        source_path = raw_paths[raw_shard_index]
        output_path_for_shard = prefix_join(output_path, f"range-{raw_shard_index:05d}.parquet")
        ensure_parent_dir(output_path_for_shard)
        locator = next(locators, None)
        row_index = 0
        documents = 0
        tokens = 0
        writer = None
        pending = []
        with ExitStack() as stack:
            source = stack.enter_context(StoragePath(source_path).open("rb"))
            target = stack.enter_context(StoragePath(output_path_for_shard).open("wb"))

            def write_pending() -> None:
                nonlocal writer
                table = pa.Table.from_pylist(pending)
                if writer is None:
                    writer = stack.enter_context(pq.ParquetWriter(target, table.schema))
                writer.write_table(table)
                pending.clear()

            for batch in pq.ParquetFile(source).iter_batches(batch_size=SCORER_BATCH_ROWS):
                for raw_row in batch.to_pylist():
                    if locator is not None and locator["raw_row_index"] < row_index:
                        raise ValueError("selected raw row locators are not unique and ordered")
                    if locator is not None and locator["raw_row_index"] == row_index:
                        if (
                            raw_row["source"] != locator["source"]
                            or raw_row["id"] != locator["id"]
                            or raw_row["sample_rank"] != locator["sample_rank"]
                            or raw_row["token_count"] != locator["token_count"]
                        ):
                            raise ValueError("selected raw row locator does not match its source document")
                        token_hash = hashlib.sha256(np.asarray(raw_row["input_ids"], dtype="<i4").tobytes()).hexdigest()
                        if token_hash != locator["token_sha256"]:
                            raise ValueError(f"selected token checksum differs for {raw_row['source']}/{raw_row['id']}")
                        output_row = {key: value for key, value in raw_row.items() if key != "text"}
                        output_row["token_sha256"] = token_hash
                        output_row[SCORE_COLUMN] = locator[SCORE_COLUMN]
                        if "incumbent_score" in locator:
                            output_row["incumbent_score"] = locator["incumbent_score"]
                        pending.append(output_row)
                        if len(pending) == SCORER_BATCH_ROWS:
                            write_pending()
                        documents += 1
                        tokens += len(raw_row["input_ids"])
                        locator = next(locators, None)
                    row_index += 1
            if pending:
                write_pending()
        if locator is not None:
            raise ValueError("selected raw row locator is outside its source shard")
        return RangeTokenTotal(raw_shard_index, output_path_for_shard, documents, tokens)

    locator_records = Dataset.from_list(list(selected.shards)).flat_map(load_parquet)
    ranges = ctx.execute(
        locator_records.group_by(
            key=lambda row: row["raw_shard_index"],
            reducer=write_selected_range,
            sort_by=lambda row: row["raw_row_index"],
            num_output_shards=len(raw_paths),
        )
    ).results
    return tuple(item.path for item in sorted(ranges, key=lambda item: item.range_key))


def prepare_quality_data(config: QualityConfig, *, ctx: ZephyrContext) -> QualityData:
    """Select one rung from a scored pool and stream its tokens into a cache."""
    selected = select_scored_pool(
        config.scored_pool,
        ctx=ctx,
        output_path=config.output_path,
        training_tokens=config.training_tokens,
        selection_method=config.selection_method,
        tie_seed=config.tie_seed,
    )
    training_shards = materialize_selected_pool(
        selected,
        ctx=ctx,
        output_path=prefix_join(config.output_path, "training_order"),
    )
    cache_dir = prefix_join(config.output_path, TOKEN_CACHE_NAME)
    cache_metadata = CacheMetadata(
        TextLmDatasetFormat().build_preprocessor(load_tokenizer(config.scored_pool.tokenizer)).metadata
    )
    _write_quality_token_cache(training_shards, cache_dir, metadata=cache_metadata)
    return QualityData(
        tokenizer=config.scored_pool.tokenizer,
        tokenizer_hash=config.scored_pool.tokenizer_hash,
        cache_dir=cache_dir,
        requested_tokens=config.training_tokens,
        actual_tokens=selected.selected_tokens,
        report_path=selected.report_path,
    )


def _write_quality_token_cache(training_shards: Sequence[str], cache_dir: str, *, metadata: CacheMetadata) -> None:
    exemplar = {"input_ids": np.zeros(0, dtype=np.int32)}
    with SerialCacheWriter(cache_dir, exemplar, metadata=metadata) as writer:
        pending: list[dict[str, np.ndarray]] = []
        pending_bytes = 0
        pending_tokens = 0
        written_documents = 0
        written_tokens = 0

        def flush_pending() -> None:
            nonlocal pending_bytes, pending_tokens, written_documents, written_tokens
            if not pending:
                return
            writer.write_batch(pending)
            written_documents += len(pending)
            written_tokens += pending_tokens
            pending.clear()
            pending_bytes = 0
            pending_tokens = 0
            logger.info(
                "Quality token cache progress: %d documents, %d tokens",
                written_documents,
                written_tokens,
            )

        for path in training_shards:
            with StoragePath(path).open("rb") as stream:
                for batch in pq.ParquetFile(stream).iter_batches(
                    batch_size=1,
                    columns=["input_ids", "token_count", "token_sha256", "source", "id"],
                ):
                    input_ids = batch.column("input_ids")[0].values
                    tokens = np.array(input_ids.to_numpy(zero_copy_only=False), dtype="<i4", copy=True)
                    token_count = batch.column("token_count")[0].as_py()
                    source = batch.column("source")[0].as_py()
                    document_id = batch.column("id")[0].as_py()
                    token_sha256 = batch.column("token_sha256")[0].as_py()
                    if len(tokens) != token_count:
                        raise ValueError(f"selected token count differs for {source}/{document_id}")
                    if hashlib.sha256(tokens.tobytes()).hexdigest() != token_sha256:
                        raise ValueError(f"selected token checksum differs for {source}/{document_id}")

                    token_bytes = tokens.nbytes
                    if pending and pending_bytes + token_bytes > QUALITY_CACHE_MAX_PENDING_BYTES:
                        flush_pending()
                    pending.append({"input_ids": tokens})
                    pending_bytes += token_bytes
                    pending_tokens += len(tokens)
                    if (
                        pending_bytes >= QUALITY_CACHE_MAX_PENDING_BYTES
                        or len(pending) >= QUALITY_CACHE_MAX_PENDING_DOCUMENTS
                    ):
                        flush_pending()
        flush_pending()


@dataclass(frozen=True)
class QualityTrainingSource:
    selection: ArtifactStep[QualityData]
    training_tokens: int

    def dependencies(self) -> tuple[ArtifactStep, ...]:
        return (self.selection,)

    def data_config(
        self,
        *,
        ctx: StepContext,
        validation: Sequence[ArtifactStep[TokenizedCache]],
        tokenizer: str,
        budget: ResolvedTrainingBudget,
    ) -> LmDataConfig:
        if self.training_tokens != budget.token_count:
            raise ValueError("quality selection token budget must equal the resolved training rung budget")
        if ctx.is_fingerprint:
            cache_dir = prefix_join(ctx.artifact_path(self.selection), TOKEN_CACHE_NAME)
        else:
            data = ctx.resolved(self.selection)
            if data.tokenizer != tokenizer:
                raise ValueError("quality selection tokenizer differs from the training tokenizer")
            if data.tokenizer_hash != tokenizer_content_hash(tokenizer):
                raise ValueError("quality selection tokenizer content differs from the training tokenizer")
            # Do not count whole-document cutoff overshoot as pool capacity.
            if data.requested_tokens != self.training_tokens:
                raise ValueError("quality selection token budget must equal the resolved training rung budget")
            cache_dir = data.cache_dir
        training_data = FlatCacheTrainingSource(
            FrozenBaselineManifest(tokenizer, (FrozenBaselineComponent("quality", cache_dir, 1.0),))
        ).data_config(ctx=ctx, validation=validation, tokenizer=tokenizer, budget=budget)
        # The rung must end on a complete mixture block so the loader cannot wrap the cache tail.
        complete_mixture_block = math.gcd(budget.token_count // budget.sequence_length, training_data.mixture_block_size)
        return replace(training_data, mixture_block_size=complete_mixture_block)


def build_scored_pool(
    raw_pool: ArtifactStep[RawCorpusPool],
    *,
    scorer_factory: Callable[[], DocumentQualityScorer],
    classifier_identity: dict[str, str | int | float | bool],
    label_exclusion: LabelExclusion,
    token_budget: int,
    prepared_features: ArtifactStep[PreparedQualityPool] | None = None,
    expected_feature_sources: Sequence[QualityFeatureSource] | None = None,
    extra_dependencies: Sequence[ArtifactStep] = (),
    incumbent_scorer_factory: Callable[[], DocumentQualityScorer] | None = None,
    incumbent_identity: dict[str, str | int | float | bool] | None = None,
    context_name: str = "fast-track-quality-score",
    worker_resources: ResourceConfig = QUALITY_WORKER_RESOURCES,
    version: str | None = None,
) -> ArtifactStep[ScoredPool]:
    """Bind one worker-loaded classifier to a fixed raw token prefix."""
    classifier_identity = quality_classifier_identity(classifier_identity)
    if token_budget <= 0:
        raise ValueError("scoring token budget must be positive")
    normalized_expected_feature_sources = (
        None
        if expected_feature_sources is None
        else tuple(sorted(tuple(expected_feature_sources), key=lambda item: item.source))
    )
    if prepared_features is None and normalized_expected_feature_sources is not None:
        raise ValueError("expected feature source pins require a prepared feature pool")
    if incumbent_identity is not None:
        incumbent_identity = quality_classifier_identity(incumbent_identity)
    label_exclusion_identity = label_exclusion.identity()
    identity = {
        "raw_pool": raw_pool.name,
        "raw_pool_version": raw_pool.version,
        "classifier_identity": classifier_identity,
        "incumbent_identity": incumbent_identity,
        "label_exclusion": label_exclusion_identity,
        "token_budget": token_budget,
        "prepared_features": (
            None if prepared_features is None else {"name": prepared_features.name, "version": prepared_features.version}
        ),
        "expected_feature_sources": (
            None
            if normalized_expected_feature_sources is None
            else [asdict(item) for item in normalized_expected_feature_sources]
        ),
        "extra_dependencies": [(dependency.name, dependency.version) for dependency in extra_dependencies],
    }
    digest = hashlib.sha256(canonical_json(identity).encode()).hexdigest()[:20]
    name = f"fast-track/scored-pool/{digest}"
    version = resolve_version(name, version)

    def build_config(ctx: StepContext) -> ScoredPoolBuildConfig:
        return ScoredPoolBuildConfig(
            raw_pool_path=ctx.artifact_path(raw_pool),
            prepared_features_path=None if prepared_features is None else ctx.artifact_path(prepared_features),
            token_budget=token_budget,
            output_path=ctx.output_path,
            context_name=context_name,
            resources=worker_resources,
        )

    def run(config: ScoredPoolBuildConfig) -> ScoredPool:
        pool = RawCorpusPool.raw_load(config.raw_pool_path)
        features = (
            None
            if config.prepared_features_path is None
            else PreparedQualityPool.raw_load(config.prepared_features_path)
        )
        with ZephyrContext(
            name=config.context_name,
            resources=config.resources,
            max_workers=QUALITY_POOL_WORKERS,
        ) as ctx:
            return score_raw_pool(
                pool,
                ctx=ctx,
                output_path=config.output_path,
                scorer_factory=scorer_factory,
                classifier_identity=classifier_identity,
                incumbent_scorer_factory=incumbent_scorer_factory,
                incumbent_identity=incumbent_identity,
                label_exclusion=label_exclusion,
                token_budget=config.token_budget,
                prepared_features=features,
                expected_feature_sources=normalized_expected_feature_sources,
            )

    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=ScoredPool,
        run=run,
        build_config=build_config,
        deps=(raw_pool, *(() if prepared_features is None else (prepared_features,)), *extra_dependencies),
    )


def build_quality_features(
    raw_pool: ArtifactStep[RawCorpusPool],
    *,
    token_budget: int,
    context_name: str = "fast-track-quality-features",
    worker_resources: ResourceConfig = QUALITY_WORKER_RESOURCES,
    version: str | None = None,
) -> ArtifactStep[PreparedQualityPool]:
    """Bind a pinned, reusable Harrier feature prefix to a raw corpus pool."""
    if token_budget <= 0:
        raise ValueError("quality feature token budget must be positive")
    identity = {
        "raw_pool": raw_pool.name,
        "raw_pool_version": raw_pool.version,
        "token_budget": token_budget,
        "harrier": HARRIER_FEATURE_IDENTITY,
    }
    digest = hashlib.sha256(canonical_json(identity).encode()).hexdigest()[:20]
    name = f"fast-track/quality-features/{digest}"
    version = resolve_version(name, version)

    def build_config(ctx: StepContext) -> QualityFeatureBuildConfig:
        return QualityFeatureBuildConfig(
            raw_pool_path=ctx.artifact_path(raw_pool),
            token_budget=token_budget,
            output_path=ctx.output_path,
            context_name=context_name,
            resources=worker_resources,
        )

    def run(config: QualityFeatureBuildConfig) -> PreparedQualityPool:
        pool = RawCorpusPool.raw_load(config.raw_pool_path)
        sources = pinned_quality_feature_sources(pool.sources)
        with ZephyrContext(
            name=config.context_name,
            resources=config.resources,
            max_workers=QUALITY_POOL_WORKERS,
        ) as ctx:
            return prepare_quality_features(
                pool,
                ctx=ctx,
                output_path=config.output_path,
                token_budget=config.token_budget,
                sources=sources,
            )

    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=PreparedQualityPool,
        run=run,
        build_config=build_config,
        deps=(raw_pool,),
    )


def build_ridge_scored_pool(
    raw_pool: ArtifactStep[RawCorpusPool],
    *,
    head_artifact_path: str,
    token_budget: int,
    context_name: str = "fast-track-quality-ridge-score",
    worker_resources: ResourceConfig = QUALITY_WORKER_RESOURCES,
) -> ArtifactStep[ScoredPool]:
    """Bind one fitted ridge head to its frozen labels and aligned Harrier features."""
    fitted = read_artifact(head_artifact_path, FittedRidgeQualityHead)
    if not _has_pinned_harrier_identity(fitted.feature_identity):
        raise ValueError("fitted ridge head and scorer require different Harrier feature pins")
    model_identity = {
        "head": asdict(fitted.head),
        "feature_identity": fitted.feature_identity,
        "label_revision": fitted.label_exclusion.label_revision,
    }
    classifier_identity = {
        "implementation": "ridge",
        "revision": hashlib.sha256(canonical_json(model_identity).encode()).hexdigest(),
    }
    head_digest = hashlib.sha256(head_artifact_path.encode()).hexdigest()[:20]
    head_name = f"fast-track/fitted-ridge/{head_digest}"
    head_version = resolve_version(head_name, None)
    head_step = ArtifactStep.adopt(
        user_namespaced_name(head_name, head_version),
        head_version,
        head_artifact_path,
        kind=FittedRidgeQualityHead,
    )
    features = build_quality_features(raw_pool, token_budget=token_budget)
    return build_scored_pool(
        raw_pool,
        scorer_factory=partial(EmbeddingHeadScorer, fitted.head),
        classifier_identity=classifier_identity,
        label_exclusion=fitted.label_exclusion,
        token_budget=token_budget,
        prepared_features=features,
        extra_dependencies=(head_step,),
        context_name=context_name,
        worker_resources=worker_resources,
    )


def build_quality_data(spec: QualitySpec, *, version: str | None = None) -> ArtifactStep[QualityData]:
    """Build one fixed-budget selection from a reusable scored pool."""
    identity = {
        "scored_pool": spec.scored_pool.name,
        "scored_pool_version": spec.scored_pool.version,
        "training_tokens": spec.training_tokens,
        "method": spec.selection_method.value,
        "tie_seed": spec.tie_seed,
    }
    digest = hashlib.sha256(canonical_json(identity).encode()).hexdigest()[:20]
    name = f"fast-track/quality/{digest}"
    version = resolve_version(name, version)

    def build_config(ctx: StepContext) -> QualityBuildConfig:
        return QualityBuildConfig(
            scored_pool_path=ctx.artifact_path(spec.scored_pool),
            training_tokens=spec.training_tokens,
            selection_method=spec.selection_method,
            tie_seed=spec.tie_seed,
            output_path=ctx.output_path,
            context_name=f"fast-track-quality-{digest}",
            resources=QUALITY_WORKER_RESOURCES,
        )

    def run(config: QualityBuildConfig) -> QualityData:
        scored_pool = ScoredPool.raw_load(config.scored_pool_path)
        quality_config = QualityConfig(
            spec.training_tokens, spec.selection_method, spec.tie_seed, scored_pool, config.output_path
        )
        with ZephyrContext(
            name=config.context_name,
            resources=config.resources,
            max_workers=QUALITY_POOL_WORKERS,
        ) as ctx:
            return prepare_quality_data(quality_config, ctx=ctx)

    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=QualityData,
        run=run,
        build_config=build_config,
        deps=(spec.scored_pool,),
    )
