# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare a corpus-proportional text pool without topic or quality selection."""

import hashlib
import json
import logging
import math
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from functools import partial

import pyarrow.parquet as pq
from fray.types import ResourceConfig
from levanter.data._preprocessor import BatchProcessor
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.tokenizers import MarinTokenizer, load_tokenizer, tokenizer_content_hash
from marin.datakit.normalize import NormalizedData
from marin.datakit.sources import all_sources
from marin.execution.artifact import Artifact, read_artifact
from marin.execution.build_context import resolve_version
from marin.execution.fingerprint import canonical_json
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.namespacing import user_namespaced_name
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from experiments.datakit import hero_data
from experiments.datakit.reference_pipeline import TokenizerSpec
from experiments.grug.fast_track.label_exclusion import LabelExclusion
from experiments.grug.fast_track.ranked_pool import RangeTokenTotal, ranked_pool, take_token_prefix

QUALITY_FRACTION = 0.1
TOKENIZE_BATCH_ROWS = 128
PARQUET_BATCH_ROWS = 1024
SAMPLE_HEADROOM = 1.25
SAMPLE_RANGES = 64
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CorpusSource:
    """One fixed normalized artifact and its token-count estimate."""

    name: str
    normalized_path: str
    estimated_tokens: int


@dataclass(frozen=True)
class CorpusSampleSpec:
    """Inputs and maximum capacity for a classifier-independent pool."""

    sources: tuple[CorpusSource, ...]
    tokenizer: TokenizerSpec
    token_budget: int
    seed: int
    label_exclusion: LabelExclusion | None = None

    def __post_init__(self) -> None:
        if not self.sources or len({source.name for source in self.sources}) != len(self.sources):
            raise ValueError("corpus sources must be non-empty and have distinct names")
        if any(source.estimated_tokens <= 0 for source in self.sources) or self.token_budget <= 0:
            raise ValueError("corpus estimates and requested token count must be positive")


@dataclass(frozen=True)
class CorpusSampleConfig:
    spec: CorpusSampleSpec
    output_path: str


class RawCorpusPool(Artifact):
    """A fixed random-order pool before topic or quality selection."""

    tokenizer: str
    tokenizer_hash: str
    sources: tuple[CorpusSource, ...]
    seed: int
    requested_tokens: int
    actual_tokens: int
    documents: int
    shards: tuple[str, ...]
    range_totals: tuple[RangeTokenTotal, ...]
    source_tokens: dict[str, int]
    manifest_path: str


@dataclass(frozen=True)
class _CorpusShard:
    source: str
    path: str


@dataclass(frozen=True)
class _CorpusShardSummary:
    range_total: RangeTokenTotal
    source_tokens: dict[str, int]


def corpus_sources() -> tuple[CorpusSource, ...]:
    """Resolve the checked-in normalized artifact pins for the available corpus."""
    paths = json.loads(hero_data.manifest_path().read_text())
    registry = all_sources()
    return tuple(
        CorpusSource(
            name=name.removeprefix("normalized/"),
            normalized_path=prefix_join(hero_data.MANIFEST_PREFIX, path),
            estimated_tokens=math.ceil(registry[name.removeprefix("normalized/")].rough_token_count_b * 1e9),
        )
        for name, path in sorted(paths.items())
        if name.startswith("normalized/")
    )


def sample_rank(source: str, document_id: str, seed: int) -> str:
    """Return a document priority that does not depend on worker or shard order."""
    return hashlib.sha256(canonical_json([seed, source, document_id]).encode()).hexdigest()


def _sample_records(
    shard: _CorpusShard,
    *,
    probability: float,
    seed: int,
    tokenizer: MarinTokenizer,
    excluded_groups: frozenset[str],
    excluded_document_ids: frozenset[str],
) -> Iterator[dict]:
    threshold = min(1 << 256, math.ceil(probability * (1 << 256)))
    preprocessor = TextLmDatasetFormat().build_preprocessor(tokenizer)
    pending = []
    normalized_row = 0
    with StoragePath(shard.path).open("rb") as stream:
        for batch in pq.ParquetFile(stream).iter_batches(batch_size=PARQUET_BATCH_ROWS, columns=["id", "text"]):
            for row in batch.to_pylist():
                row_offset = normalized_row
                normalized_row += 1
                document_id, text = row["id"], row["text"]
                if not isinstance(document_id, str) or not isinstance(text, str):
                    raise ValueError(f"{shard.path}: normalized id and text must be strings")
                rank = sample_rank(shard.source, document_id, seed)
                if int(rank, 16) >= threshold:
                    continue
                if document_id in excluded_document_ids:
                    continue
                duplicate_group = hashlib.sha256(text.encode()).hexdigest()
                if duplicate_group in excluded_groups:
                    continue
                pending.append(
                    {
                        "source": shard.source,
                        "id": document_id,
                        "sample_rank": rank,
                        "duplicate_group": duplicate_group,
                        "normalized_shard": shard.path,
                        "normalized_row": row_offset,
                        "text": text,
                    }
                )
                if len(pending) == TOKENIZE_BATCH_ROWS:
                    yield from _encode_records(pending, preprocessor)
                    pending.clear()
        if pending:
            yield from _encode_records(pending, preprocessor)


def _encode_records(records: list[dict], preprocessor: BatchProcessor[dict, dict]) -> Iterator[dict]:
    encoded = preprocessor([{"text": row["text"]} for row in records])
    for row, tokens in zip(records, encoded, strict=True):
        input_ids = list(tokens["input_ids"])
        if not input_ids:
            raise ValueError(f"tokenizer produced no tokens for {row['source']}/{row['id']}")
        yield {**row, "input_ids": input_ids, "token_count": len(input_ids)}


def _read_sample(shard: _CorpusShard, *, probability: float, spec: CorpusSampleSpec) -> Iterator[dict]:
    actual_hash = tokenizer_content_hash(spec.tokenizer.name)
    if actual_hash != spec.tokenizer.identity:
        raise ValueError(f"tokenizer content differs from the corpus specification: {actual_hash}")
    yield from _sample_records(
        shard,
        probability=probability,
        seed=spec.seed,
        tokenizer=load_tokenizer(spec.tokenizer.name),
        excluded_groups=spec.label_exclusion.duplicate_groups if spec.label_exclusion else frozenset(),
        excluded_document_ids=spec.label_exclusion.normalized_document_ids if spec.label_exclusion else frozenset(),
    )


def _source_summary(shard: dict) -> _CorpusShardSummary:
    path = shard["path"]
    counts: Counter[str] = Counter()
    documents = 0
    with StoragePath(path).open("rb") as stream:
        for batch in pq.ParquetFile(stream).iter_batches(columns=["source", "token_count"]):
            for row in batch.to_pylist():
                counts[row["source"]] += row["token_count"]
                documents += 1
    return _CorpusShardSummary(RangeTokenTotal(shard["index"], path, documents, sum(counts.values())), dict(counts))


def prepare_corpus_pool(config: CorpusSampleConfig, *, ctx: ZephyrContext) -> RawCorpusPool:
    """Sample all sources at one inclusion probability and measure actual tokens.

    Source estimates only size the first pass. If it is too small, increase the
    same hash cutoff. The final token prefix therefore does not depend on those
    estimates. Each pass streams input shards and tokenizes selected rows only.
    """
    spec = config.spec
    shards = []
    for source in spec.sources:
        normalized = read_artifact(source.normalized_path, NormalizedData)
        paths = sorted(str(path) for path in StoragePath(prefix_join(normalized.main_output_dir, "*.parquet")).glob())
        if not paths:
            raise ValueError(f"no normalized corpus shards at {normalized.main_output_dir}")
        shards.extend(_CorpusShard(source.name, path) for path in paths)
    estimate = sum(source.estimated_tokens for source in spec.sources)
    probability = min(1.0, SAMPLE_HEADROOM * spec.token_budget / estimate)
    attempt = 0
    while True:
        attempt_path = prefix_join(config.output_path, f"sample-{attempt:03d}")
        dataset = Dataset.from_list(shards).flat_map(partial(_read_sample, probability=probability, spec=spec))
        ranked = ranked_pool(
            dataset,
            ctx=ctx,
            output_path=attempt_path,
            rank_key=lambda row: f"{row['sample_rank']}:{row['source']}:{row['id']}",
            token_count=lambda row: row["token_count"],
            range_key=lambda row: int(row["sample_rank"][:8], 16) * SAMPLE_RANGES // (1 << 32),
            num_ranges=SAMPLE_RANGES,
        )
        if ranked.total_tokens >= spec.token_budget:
            break
        if probability == 1:
            raise ValueError(f"corpus has {ranked.total_tokens:,} tokens, but the pool requires {spec.token_budget:,}")
        probability = min(1.0, probability * 2)
        attempt += 1
    selected = take_token_prefix(
        ranked, ctx=ctx, output_path=prefix_join(config.output_path, "pool"), token_budget=spec.token_budget
    )
    counts: Counter[str] = Counter()
    selected_files = [{"index": index, "path": path} for index, path in enumerate(selected.shards)]
    summaries = ctx.execute(Dataset.from_list(selected_files).map(_source_summary)).results
    ranges = tuple(sorted((item.range_total for item in summaries), key=lambda item: item.range_key))
    for result in summaries:
        counts.update(result.source_tokens)
    manifest_path = prefix_join(config.output_path, "corpus.json")
    report = {
        "sampling_method": "corpus-proportional-hash",
        "spec": _spec_identity(spec),
        "input_shards": [asdict(shard) for shard in shards],
        "inclusion_probability": probability,
        "source_tokens": dict(counts),
        "shards": list(selected.shards),
        "range_totals": [asdict(item) for item in ranges],
        "requested_tokens": spec.token_budget,
        "actual_tokens": selected.total_tokens,
        "documents": selected.total_documents,
        "document_overshoot_tokens": selected.total_tokens - spec.token_budget,
    }
    StoragePath(manifest_path).write_text(json.dumps(report, indent=2))
    logger.info(
        "Prepared corpus pool: %d documents, %d tokens, source tokens=%s, manifest=%s",
        selected.total_documents,
        selected.total_tokens,
        dict(counts),
        manifest_path,
    )
    return RawCorpusPool(
        tokenizer=spec.tokenizer.name,
        tokenizer_hash=spec.tokenizer.identity,
        sources=spec.sources,
        seed=spec.seed,
        requested_tokens=spec.token_budget,
        actual_tokens=selected.total_tokens,
        documents=selected.total_documents,
        shards=selected.shards,
        range_totals=ranges,
        source_tokens=dict(counts),
        manifest_path=manifest_path,
    )


def build_corpus_pool(spec: CorpusSampleSpec, *, version: str | None = None) -> ArtifactStep[RawCorpusPool]:
    """Bind corpus sampling to fixed source, tokenizer, seed, and capacity inputs."""
    digest = hashlib.sha256(canonical_json(_spec_identity(spec)).encode()).hexdigest()[:20]
    name = f"fast-track/corpus-pool/{digest}"
    version = resolve_version(name, version)

    def build_config(ctx: StepContext) -> CorpusSampleConfig:
        return CorpusSampleConfig(spec, ctx.output_path)

    def run(config: CorpusSampleConfig) -> RawCorpusPool:
        with ZephyrContext(resources=ResourceConfig(cpu=2, ram="8g"), max_workers=8) as ctx:
            return prepare_corpus_pool(config, ctx=ctx)

    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=RawCorpusPool,
        build_config=build_config,
        run=run,
    )


def _spec_identity(spec: CorpusSampleSpec) -> dict:
    return {**asdict(spec), "label_exclusion": spec.label_exclusion.identity() if spec.label_exclusion else None}
