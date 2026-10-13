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
from itertools import islice

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from levanter.data.text.datasets import LmDataConfig
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.store.cache import CacheMetadata, SerialCacheWriter
from levanter.tokenizers import load_tokenizer
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

from experiments.grug.fast_track.batching import bounded_batches
from experiments.grug.fast_track.contracts import (
    FrozenBaselineComponent,
    FrozenBaselineManifest,
    ResolvedTrainingBudget,
    adopt_artifact_path,
    require_tokenizer_content,
)
from experiments.grug.fast_track.corpus_sample import QUALITY_POOL_MULTIPLIER, RawCorpusPool
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
    write_parquet_tables,
)

logger = logging.getLogger(__name__)

QUALITY_CACHE_MAX_PENDING_BYTES = 64 * 1024 * 1024
QUALITY_CACHE_MAX_PENDING_DOCUMENTS = 16_384
TOKEN_CACHE_NAME = "tokens"
SCORER_BATCH_ROWS = 128
SCORE_COLUMN = "quality_score"
INCUMBENT_SCORE_COLUMN = "incumbent_score"
SCORE_SAMPLE_ROWS = 512
QUALITY_WORKER_RESOURCES = ResourceConfig(cpu=2, ram="8g", disk="8g")
QUALITY_POOL_WORKERS = 8
QUALITY_FEATURES_CONTEXT = "fast-track-quality-features"
QUALITY_SCORE_CONTEXT = "fast-track-quality-score"


class SelectionMethod(StrEnum):
    CANDIDATE = "candidate"
    INCUMBENT = "incumbent"
    RANDOM = "random"


@dataclass(frozen=True)
class ScorerSpec:
    """A worker-loaded document scorer factory and its stable classifier identity."""

    factory: Callable[[], DocumentQualityScorer]
    identity: dict[str, str | int | float | bool]

    def __post_init__(self) -> None:
        quality_classifier_identity(self.identity)


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
class _ScoreInput:
    index: int
    path: str
    feature_path: str | None


@dataclass(frozen=True)
class QualitySpec:
    scored_pool: ArtifactStep[ScoredPool]
    training_tokens: int
    selection_method: SelectionMethod
    tie_seed: int


@dataclass(frozen=True)
class QualityBuildConfig:
    scored_pool_path: str
    training_tokens: int
    selection_method: SelectionMethod
    tie_seed: int
    output_path: str
    resources: ResourceConfig


@dataclass(frozen=True)
class ScoredPoolBuildConfig:
    raw_pool_path: str
    prepared_features_path: str | None
    token_budget: int
    output_path: str
    resources: ResourceConfig


@dataclass(frozen=True)
class QualityFeatureBuildConfig:
    raw_pool_path: str
    token_budget: int
    output_path: str
    resources: ResourceConfig


def _has_pinned_harrier_identity(identity: dict[str, str | int | float]) -> bool:
    return all(identity.get(key) == value for key, value in HARRIER_FEATURE_IDENTITY.items())


def _token_sha256(input_ids: Sequence[int] | np.ndarray) -> str:
    return hashlib.sha256(np.asarray(input_ids, dtype="<i4").tobytes()).hexdigest()


def _prepare_scoring_prefix(
    pool: RawCorpusPool,
    *,
    ctx: ZephyrContext,
    output_path: str,
    token_budget: int,
    prepared_features: PreparedQualityPool | None,
) -> _ScoringPrefix:
    if prepared_features is None:
        raw_prefix = take_token_prefix(
            RankedPool(pool.shards, pool.range_totals, pool.documents, pool.actual_tokens),
            ctx=ctx,
            output_path=prefix_join(output_path, "raw_prefix"),
            token_budget=token_budget,
        )
        return _ScoringPrefix(raw_prefix, None, None)

    if (
        prepared_features.raw_manifest_path != pool.manifest_path
        or prepared_features.raw_seed != pool.seed
        or prepared_features.tokenizer_hash != pool.tokenizer_hash
        or prepared_features.requested_tokens != token_budget
        or prepared_features.sources != pinned_quality_feature_sources(pool.sources)
    ):
        raise ValueError("prepared feature pool does not match the raw scoring pool and exact token budget")
    if not _has_pinned_harrier_identity(prepared_features.feature_identity):
        raise ValueError("prepared quality pool has different Harrier feature pins")
    raw_prefix = PrefixResult(
        prepared_features.raw_prefix_shards, prepared_features.documents, prepared_features.actual_tokens
    )
    feature_paths = tuple(item.path for item in prepared_features.feature_shards)
    return _ScoringPrefix(raw_prefix, feature_paths, prepared_features.feature_identity)


def _prepare_scoring_batch(
    rows: list[dict],
    *,
    feature_batches: Iterator[pa.RecordBatch] | None,
    row_offset: int,
    label_exclusion: LabelExclusion,
) -> tuple[QualityScoringBatch, list[str]]:
    token_hashes = []
    for row in rows:
        if row["duplicate_group"] in label_exclusion.duplicate_groups:
            raise ValueError(
                "raw quality pool overlaps a frozen train, development, or audit duplicate group "
                f"({row['source']}/{row['id']})"
            )
        if row["id"] in label_exclusion.normalized_document_ids:
            raise ValueError(f"raw quality pool overlaps a frozen normalized label ID ({row['source']}/{row['id']})")
        if len(row["input_ids"]) != row["token_count"]:
            raise ValueError("raw corpus token_count differs from input_ids length")
        token_hashes.append(_token_sha256(row["input_ids"]))

    embeddings = None
    if feature_batches is not None:
        feature_batch = next(feature_batches, None)
        if feature_batch is None:
            raise ValueError("prepared feature shard ended before its raw prefix shard")
        feature_rows = feature_batch.to_pylist()
        if len(feature_rows) != len(rows):
            raise ValueError("prepared feature and raw prefix batch lengths differ")
        for row_index, (raw_row, feature_row) in enumerate(zip(rows, feature_rows, strict=True)):
            if raw_row["id"] != feature_row["id"] or feature_row["raw_row_index"] != row_offset + row_index:
                raise ValueError("prepared feature ID or row position differs from the raw prefix")
        embeddings = normalize_harrier_embeddings(np.asarray([row["embedding"] for row in feature_rows], dtype=np.int8))
    return QualityScoringBatch([row["text"] for row in rows], [row["id"] for row in rows], embeddings), token_hashes


def _checked_scores(scorer: DocumentQualityScorer, batch: QualityScoringBatch) -> np.ndarray:
    scores = np.asarray(scorer.scores(batch), dtype=np.float64)
    if scores.shape != (len(batch.texts),) or not np.isfinite(scores).all():
        raise ValueError("document scorer must return one finite score per document")
    return scores


def _score_raw_file(
    item: _ScoreInput,
    *,
    output_path: str,
    scorer: DocumentQualityScorer,
    incumbent_scorer: DocumentQualityScorer | None,
    label_exclusion: LabelExclusion,
) -> RangeTokenTotal:
    path = prefix_join(output_path, f"part-{item.index:05d}.parquet")
    documents = 0
    tokens_total = 0
    with ExitStack() as stack:
        source = stack.enter_context(StoragePath(item.path).open("rb"))
        feature_batches = None
        if item.feature_path is not None:
            feature_stream = stack.enter_context(StoragePath(item.feature_path).open("rb"))
            feature_batches = iter(pq.ParquetFile(feature_stream).iter_batches(batch_size=SCORER_BATCH_ROWS))

        def scored_tables() -> Iterator[pa.Table]:
            nonlocal documents, tokens_total
            for batch in pq.ParquetFile(source).iter_batches(batch_size=SCORER_BATCH_ROWS):
                rows = batch.to_pylist()
                scoring_batch, token_hashes = _prepare_scoring_batch(
                    rows,
                    feature_batches=feature_batches,
                    row_offset=documents,
                    label_exclusion=label_exclusion,
                )
                scores = _checked_scores(scorer, scoring_batch)
                incumbent_scores = None if incumbent_scorer is None else _checked_scores(incumbent_scorer, scoring_batch)
                output = []
                for index, row in enumerate(rows):
                    scored_row = {key: value for key, value in row.items() if key not in {"text", "input_ids"}}
                    scored_row.update(
                        {
                            "raw_shard_index": item.index,
                            "raw_row_index": documents + index,
                            "token_sha256": token_hashes[index],
                            SCORE_COLUMN: float(scores[index]),
                        }
                    )
                    if incumbent_scores is not None:
                        scored_row[INCUMBENT_SCORE_COLUMN] = float(incumbent_scores[index])
                    output.append(scored_row)
                documents += len(rows)
                tokens_total += sum(row["token_count"] for row in rows)
                yield pa.Table.from_pylist(output)

        write_parquet_tables(path, scored_tables())
        if feature_batches is not None and next(feature_batches, None) is not None:
            raise ValueError("prepared feature shard has rows beyond its raw prefix shard")
    if documents == 0:
        raise ValueError("raw scoring shard must contain at least one document")
    return RangeTokenTotal(item.index, path, documents, tokens_total)


def score_raw_pool(
    pool: RawCorpusPool,
    *,
    ctx: ZephyrContext,
    output_path: str,
    candidate: ScorerSpec,
    label_exclusion: LabelExclusion,
    token_budget: int,
    prepared_features: PreparedQualityPool | None = None,
    incumbent: ScorerSpec | None = None,
) -> ScoredPool:
    """Score one bounded raw token prefix and write document locators with scores."""
    if token_budget > pool.requested_tokens:
        raise ValueError(
            f"raw corpus pool was limited to {pool.requested_tokens} requested tokens, below the "
            f"{token_budget} scoring budget"
        )
    label_exclusion_fingerprint = hashlib.sha256(canonical_json(label_exclusion.identity()).encode()).hexdigest()
    scoring_prefix = _prepare_scoring_prefix(
        pool,
        ctx=ctx,
        output_path=output_path,
        token_budget=token_budget,
        prepared_features=prepared_features,
    )
    raw_prefix = scoring_prefix.raw_prefix
    feature_paths = scoring_prefix.feature_paths

    def score_shard(items: Iterator[_ScoreInput], _: ShardInfo) -> Iterator[RangeTokenTotal]:
        scorer = candidate.factory()
        incumbent_scorer = None if incumbent is None else incumbent.factory()
        for item in items:
            yield _score_raw_file(
                item,
                output_path=output_path,
                scorer=scorer,
                incumbent_scorer=incumbent_scorer,
                label_exclusion=label_exclusion,
            )

    inputs = [
        _ScoreInput(index, path, None if feature_paths is None else feature_paths[index])
        for index, path in enumerate(raw_prefix.shards)
    ]
    files = Dataset.from_list(inputs).reshard(len(inputs))
    summaries = ctx.execute(files.map_shard(score_shard)).results
    ranges = tuple(sorted(summaries, key=lambda item: item.range_key))
    if (
        sum(item.documents for item in ranges) != raw_prefix.total_documents
        or sum(item.tokens for item in ranges) != raw_prefix.total_tokens
    ):
        raise ValueError("scored pool totals differ from the raw prefix manifest")
    return ScoredPool(
        tokenizer=pool.tokenizer,
        tokenizer_hash=pool.tokenizer_hash,
        raw_manifest_path=pool.manifest_path,
        raw_seed=pool.seed,
        classifier_identity=candidate.identity,
        incumbent_identity=None if incumbent is None else incumbent.identity,
        label_revision=label_exclusion.label_revision,
        label_exclusion_fingerprint=label_exclusion_fingerprint,
        documents=raw_prefix.total_documents,
        tokens=raw_prefix.total_tokens,
        requested_tokens=token_budget,
        raw_prefix_shards=raw_prefix.shards,
        shards=tuple(item.path for item in ranges),
        range_totals=ranges,
        feature_identity=scoring_prefix.feature_identity,
    )


def _tie_hash(tie_seed: int, source: str, document_id: str) -> str:
    return hashlib.sha256(f"{tie_seed}:{source}:{document_id}".encode()).hexdigest()


def _descending_score_key(score: float) -> str:
    bits = int.from_bytes(struct.pack(">d", score), "big")
    ordered = (~bits & ((1 << 64) - 1)) if bits >> 63 else bits ^ (1 << 63)
    return f"{(~ordered) & ((1 << 64) - 1):016x}"


def _score_rank_key(score: float, source: str, document_id: str, tie_seed: int) -> str:
    return f"{_descending_score_key(score)}:{_tie_hash(tie_seed, source, document_id)}:{source}:{document_id}"


def _score_rank(row: dict, *, score_column: str, tie_seed: int) -> str:
    return _score_rank_key(float(row[score_column]), row["source"], row["id"], tie_seed)


def _score_range(row: dict, *, score_column: str, tie_seed: int, boundaries: tuple[str, ...]) -> int:
    return bisect_right(boundaries, _score_rank(row, score_column=score_column, tie_seed=tie_seed))


def _random_rank(row: dict, *, tie_seed: int) -> str:
    return f"{_tie_hash(tie_seed, row['source'], row['id'])}:{row['source']}:{row['id']}"


def _random_range(row: dict, *, tie_seed: int, num_ranges: int) -> int:
    return int(_tie_hash(tie_seed, row["source"], row["id"])[:8], 16) * num_ranges // (1 << 32)


def _summarize_scores(shards: tuple[str, ...], *, ctx: ZephyrContext, score_column: str) -> _ScoreColumnSummary:
    """Return score bounds and a bounded rank sample from each shard."""

    def summarize(paths: Iterator[str], _: ShardInfo) -> Iterator[_ScoreColumnSummary]:
        for path in paths:
            minimum = math.inf
            maximum = -math.inf
            rank_sample = []
            with StoragePath(path).open("rb") as stream:
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

    summaries = ctx.execute(Dataset.from_list(list(shards)).reshard(len(shards)).map_shard(summarize)).results
    return _ScoreColumnSummary(
        min(item.minimum for item in summaries),
        max(item.maximum for item in summaries),
        tuple(sample for item in summaries for sample in item.rank_sample),
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
    score_column = SCORE_COLUMN
    classifier_identity = pool.classifier_identity
    if selection_method is SelectionMethod.INCUMBENT:
        if pool.incumbent_identity is None:
            raise ValueError("scored pool does not contain incumbent scores")
        score_column = INCUMBENT_SCORE_COLUMN
        classifier_identity = pool.incumbent_identity
    raw_token_budget = training_tokens * QUALITY_POOL_MULTIPLIER
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

    if selection_method is SelectionMethod.RANDOM:
        score_bounds = None
        rank_key = partial(_random_rank, tie_seed=tie_seed)
        range_key = partial(_random_range, tie_seed=tie_seed, num_ranges=num_ranges)
    else:
        summary = _summarize_scores(raw_prefix.shards, ctx=ctx, score_column=score_column)
        if summary.minimum == summary.maximum:
            raise ValueError(f"{selection_method.value} scores are constant on the rung raw prefix")
        score_bounds = {"minimum": summary.minimum, "maximum": summary.maximum}
        sample_ranks = sorted(
            _score_rank_key(score, source, document_id, tie_seed) for score, source, document_id in summary.rank_sample
        )
        boundaries = tuple(
            sample_ranks[min(len(sample_ranks) - 1, (len(sample_ranks) * index) // num_ranges)]
            for index in range(1, num_ranges)
        )
        rank_key = partial(_score_rank, score_column=score_column, tie_seed=tie_seed)
        range_key = partial(_score_range, score_column=score_column, tie_seed=tie_seed, boundaries=boundaries)

    ranked = ranked_pool(
        Dataset.from_list(list(raw_prefix.shards)).flat_map(load_parquet),
        ctx=ctx,
        output_path=prefix_join(output_path, "score_order"),
        rank_key=rank_key,
        range_key=range_key,
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
        "raw_prefix_selected_tokens": raw_prefix.total_tokens,
        "raw_prefix_documents": raw_prefix.total_documents,
        "selection_method": selection_method.value,
        "tie_seed": tie_seed,
        "requested_training_tokens": training_tokens,
        "selected_tokens": selection.total_tokens,
        "selected_documents": selection.total_documents,
        "score_bounds": score_bounds,
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
        path = prefix_join(output_path, f"range-{raw_shard_index:05d}.parquet")
        locator = next(locators, None)
        tokens = 0

        def selected_rows() -> Iterator[dict]:
            nonlocal locator, tokens
            row_index = 0
            with StoragePath(source_path).open("rb") as source:
                for batch in pq.ParquetFile(source).iter_batches(batch_size=SCORER_BATCH_ROWS):
                    if locator is None:
                        return
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
                            token_hash = _token_sha256(raw_row["input_ids"])
                            if token_hash != locator["token_sha256"]:
                                raise ValueError(
                                    f"selected token checksum differs for {raw_row['source']}/{raw_row['id']}"
                                )
                            output_row = {key: value for key, value in raw_row.items() if key != "text"}
                            output_row["token_sha256"] = token_hash
                            output_row[SCORE_COLUMN] = locator[SCORE_COLUMN]
                            if INCUMBENT_SCORE_COLUMN in locator:
                                output_row[INCUMBENT_SCORE_COLUMN] = locator[INCUMBENT_SCORE_COLUMN]
                            tokens += len(raw_row["input_ids"])
                            yield output_row
                            locator = next(locators, None)
                        row_index += 1

        def tables() -> Iterator[pa.Table]:
            rows = selected_rows()
            while batch := list(islice(rows, SCORER_BATCH_ROWS)):
                yield pa.Table.from_pylist(batch)

        documents = write_parquet_tables(path, tables())
        if locator is not None:
            raise ValueError("selected raw row locator is outside its source shard")
        return RangeTokenTotal(raw_shard_index, path, documents, tokens)

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


def prepare_quality_data(
    scored_pool: ScoredPool,
    *,
    ctx: ZephyrContext,
    output_path: str,
    training_tokens: int,
    selection_method: SelectionMethod,
    tie_seed: int,
) -> QualityData:
    """Select one rung from a scored pool and stream its tokens into a cache."""
    selected = select_scored_pool(
        scored_pool,
        ctx=ctx,
        output_path=output_path,
        training_tokens=training_tokens,
        selection_method=selection_method,
        tie_seed=tie_seed,
    )
    training_shards = materialize_selected_pool(
        selected,
        ctx=ctx,
        output_path=prefix_join(output_path, "training_order"),
    )
    cache_dir = prefix_join(output_path, TOKEN_CACHE_NAME)
    cache_metadata = CacheMetadata(
        TextLmDatasetFormat().build_preprocessor(load_tokenizer(scored_pool.tokenizer)).metadata
    )
    _write_quality_token_cache(training_shards, cache_dir, metadata=cache_metadata)
    return QualityData(
        tokenizer=scored_pool.tokenizer,
        tokenizer_hash=scored_pool.tokenizer_hash,
        cache_dir=cache_dir,
        requested_tokens=training_tokens,
        actual_tokens=selected.selected_tokens,
        report_path=selected.report_path,
    )


def _selected_token_rows(paths: Sequence[str]) -> Iterator[np.ndarray]:
    """Yield each selected document's tokens, in order, after a length check."""
    for path in paths:
        with StoragePath(path).open("rb") as stream:
            for batch in pq.ParquetFile(stream).iter_batches(
                batch_size=SCORER_BATCH_ROWS, columns=["input_ids", "token_count", "source", "id"]
            ):
                input_ids = batch.column("input_ids")
                lengths = input_ids.value_lengths().to_numpy()
                mismatched = np.flatnonzero(lengths != batch.column("token_count").to_numpy())
                if len(mismatched):
                    index = int(mismatched[0])
                    raise ValueError(
                        f"selected token count differs for "
                        f"{batch.column('source')[index].as_py()}/{batch.column('id')[index].as_py()}"
                    )
                tokens = input_ids.flatten().to_numpy().astype("<i4")
                ends = np.cumsum(lengths)
                for start, end in zip(ends - lengths, ends, strict=True):
                    yield tokens[start:end]


def _write_quality_token_cache(training_shards: Sequence[str], cache_dir: str, *, metadata: CacheMetadata) -> None:
    exemplar = {"input_ids": np.zeros(0, dtype=np.int32)}
    written_documents = 0
    written_tokens = 0
    with SerialCacheWriter(cache_dir, exemplar, metadata=metadata) as writer:
        for batch in bounded_batches(
            _selected_token_rows(training_shards),
            max_rows=QUALITY_CACHE_MAX_PENDING_DOCUMENTS,
            max_bytes=QUALITY_CACHE_MAX_PENDING_BYTES,
            byte_size=lambda tokens: tokens.nbytes,
        ):
            writer.write_batch([{"input_ids": tokens} for tokens in batch])
            written_documents += len(batch)
            written_tokens += sum(len(tokens) for tokens in batch)
            logger.info("Quality token cache progress: %d documents, %d tokens", written_documents, written_tokens)


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
            require_tokenizer_content(tokenizer, data.tokenizer_hash, owner="quality selection")
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
    candidate: ScorerSpec,
    label_exclusion: LabelExclusion,
    token_budget: int,
    prepared_features: ArtifactStep[PreparedQualityPool] | None = None,
    extra_dependencies: Sequence[ArtifactStep] = (),
    incumbent: ScorerSpec | None = None,
    worker_resources: ResourceConfig = QUALITY_WORKER_RESOURCES,
    version: str | None = None,
) -> ArtifactStep[ScoredPool]:
    """Bind one worker-loaded classifier to a fixed raw token prefix."""
    identity = {
        "raw_pool": raw_pool.name,
        "raw_pool_version": raw_pool.version,
        "classifier_identity": candidate.identity,
        "incumbent_identity": None if incumbent is None else incumbent.identity,
        "label_exclusion": label_exclusion.identity(),
        "token_budget": token_budget,
        "prepared_features": (
            None if prepared_features is None else {"name": prepared_features.name, "version": prepared_features.version}
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
            name=QUALITY_SCORE_CONTEXT,
            resources=config.resources,
            max_workers=QUALITY_POOL_WORKERS,
        ) as ctx:
            return score_raw_pool(
                pool,
                ctx=ctx,
                output_path=config.output_path,
                candidate=candidate,
                incumbent=incumbent,
                label_exclusion=label_exclusion,
                token_budget=config.token_budget,
                prepared_features=features,
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
    worker_resources: ResourceConfig = QUALITY_WORKER_RESOURCES,
    version: str | None = None,
) -> ArtifactStep[PreparedQualityPool]:
    """Bind a pinned, reusable Harrier feature prefix to a raw corpus pool."""
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
            resources=worker_resources,
        )

    def run(config: QualityFeatureBuildConfig) -> PreparedQualityPool:
        pool = RawCorpusPool.raw_load(config.raw_pool_path)
        with ZephyrContext(
            name=QUALITY_FEATURES_CONTEXT,
            resources=config.resources,
            max_workers=QUALITY_POOL_WORKERS,
        ) as ctx:
            return prepare_quality_features(
                pool,
                ctx=ctx,
                output_path=config.output_path,
                token_budget=config.token_budget,
                sources=pinned_quality_feature_sources(pool.sources),
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
    return build_scored_pool(
        raw_pool,
        candidate=ScorerSpec(partial(EmbeddingHeadScorer, fitted.head), classifier_identity),
        label_exclusion=fitted.label_exclusion,
        token_budget=token_budget,
        prepared_features=build_quality_features(raw_pool, token_budget=token_budget),
        extra_dependencies=(adopt_artifact_path(head_artifact_path, kind=FittedRidgeQualityHead),),
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
            resources=QUALITY_WORKER_RESOURCES,
        )

    def run(config: QualityBuildConfig) -> QualityData:
        with ZephyrContext(
            name=f"fast-track-quality-{digest}",
            resources=config.resources,
            max_workers=QUALITY_POOL_WORKERS,
        ) as ctx:
            return prepare_quality_data(
                ScoredPool.raw_load(config.scored_pool_path),
                ctx=ctx,
                output_path=config.output_path,
                training_tokens=config.training_tokens,
                selection_method=config.selection_method,
                tie_seed=config.tie_seed,
            )

    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=QualityData,
        run=run,
        build_config=build_config,
        deps=(spec.scored_pool,),
    )
