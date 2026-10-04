# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare a corpus-proportional text pool without topic or quality selection."""

import hashlib
import json
import logging
import math
import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from fractions import Fraction
from functools import partial

import pyarrow as pa
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from levanter.data._preprocessor import BatchProcessor
from levanter.data.text._batch_tokenizer import BatchTokenizer
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
from experiments.grug.fast_track.batching import bounded_batches
from experiments.grug.fast_track.contracts import (
    TOKENIZATION_CHUNK_CHARS,
    TOKENIZATION_MAX_DOCUMENT_BYTES,
    TOKENIZATION_POLICY,
)
from experiments.grug.fast_track.label_exclusion import LabelExclusion
from experiments.grug.fast_track.ranked_pool import RangeTokenTotal, ranked_pool, take_token_prefix

QUALITY_FRACTION = 0.1
TOKENIZE_BATCH_ROWS = 128
TOKENIZE_BATCH_MAX_BYTES = 256 * 1024
PARQUET_BATCH_ROWS = 32
SAMPLE_RANGES = 1024
SAMPLE_HEADROOM = Fraction(5, 4)
SAMPLING_POLICY = "corpus-id-interval-v1"
_ID_SPACE = 1 << 128
_NORMALIZED_ID = re.compile(r"[0-9a-f]{32}\Z")
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
    normalized_paths = {
        name.removeprefix("normalized/"): path for name, path in paths.items() if name.startswith("normalized/")
    }
    missing = sorted(name for name in registry if not normalized_paths.get(name))
    if missing:
        raise ValueError(f"hero data manifest is missing normalized pins for registered sources: {missing}")
    return tuple(
        CorpusSource(
            name=name,
            normalized_path=prefix_join(hero_data.MANIFEST_PREFIX, normalized_paths[name]),
            estimated_tokens=math.ceil(registry[name].rough_token_count_b * 1e9),
        )
        for name in sorted(registry)
    )


def _rank_salt(source: str, seed: int) -> int:
    salt = hashlib.sha256(canonical_json([SAMPLING_POLICY, seed, source]).encode()).digest()[:16]
    return int.from_bytes(salt, "big")


def _sampling_cutoff(token_budget: int, estimate: int) -> int:
    numerator = SAMPLE_HEADROOM.numerator * token_budget * _ID_SPACE
    denominator = SAMPLE_HEADROOM.denominator * estimate
    return min(_ID_SPACE, (numerator + denominator - 1) // denominator)


def _id_intervals(source: str, seed: int, cutoff: int) -> tuple[tuple[int, int], ...]:
    """Return the half-open ID intervals that map below the cyclic rank cutoff."""
    if cutoff <= 0:
        return ()
    if cutoff >= _ID_SPACE:
        return ((0, _ID_SPACE),)
    start = (-_rank_salt(source, seed)) % _ID_SPACE
    end = (start + cutoff) % _ID_SPACE
    if end > start:
        return ((start, end),)
    return tuple(interval for interval in ((0, end), (start, _ID_SPACE)) if interval[0] < interval[1])


def _parquet_id_int(value: object) -> int | None:
    if not isinstance(value, str) or not _NORMALIZED_ID.fullmatch(value):
        return None
    return int(value, 16)


# Fixed-width lowercase hex strings sort in the same order as their integer IDs.
def _row_group_may_match(row_group: pq.RowGroupMetaData, intervals: tuple[tuple[int, int], ...]) -> bool:
    id_column = next(
        (
            row_group.column(index)
            for index in range(row_group.num_columns)
            if row_group.column(index).path_in_schema == "id"
        ),
        None,
    )
    if id_column is None:
        return True
    statistics = id_column.statistics
    if statistics is None or not statistics.has_min_max or not statistics.has_null_count or statistics.null_count:
        return True
    minimum = _parquet_id_int(statistics.min)
    maximum = _parquet_id_int(statistics.max)
    if minimum is None or maximum is None or minimum > maximum:
        return True
    return any(maximum >= start and minimum < end for start, end in intervals)


def _row_groups_to_read(parquet: pq.ParquetFile, intervals: tuple[tuple[int, int], ...]) -> set[int]:
    # Normalized shards sort rows by ID, so row-group bounds permit sparse reads.
    return {
        index
        for index in range(parquet.metadata.num_row_groups)
        if _row_group_may_match(parquet.metadata.row_group(index), intervals)
    }


def _iter_normalized_rows(
    parquet: pq.ParquetFile, shard_path: str, intervals: tuple[tuple[int, int], ...]
) -> Iterator[tuple[str, str, int]]:
    schema = parquet.schema_arrow
    if schema.get_field_index("id") < 0 or schema.get_field_index("text") < 0:
        raise ValueError(f"{shard_path}: normalized parquet must contain id and text fields")
    for name in ("id", "text"):
        field_type = schema.field(name).type
        if not (pa.types.is_string(field_type) or pa.types.is_large_string(field_type)):
            raise ValueError(f"{shard_path}: normalized {name} field must use an Arrow string type")

    normalized_row = 0
    row_groups = _row_groups_to_read(parquet, intervals)
    for row_group_index in range(parquet.metadata.num_row_groups):
        row_group = parquet.metadata.row_group(row_group_index)
        if row_group_index not in row_groups:
            normalized_row += row_group.num_rows
            continue
        batches = parquet.iter_batches(
            batch_size=PARQUET_BATCH_ROWS,
            columns=["id", "text"],
            row_groups=[row_group_index],
        )
        for batch in batches:
            ids, texts = batch.column("id"), batch.column("text")
            for index in range(batch.num_rows):
                row_offset = normalized_row
                normalized_row += 1
                document_id, text = ids[index].as_py(), texts[index].as_py()
                if not isinstance(document_id, str) or not isinstance(text, str):
                    raise ValueError(f"{shard_path}: normalized id and text must be strings at row {row_offset}")
                if not _NORMALIZED_ID.fullmatch(document_id):
                    raise ValueError(
                        f"{shard_path}: normalized id at row {row_offset} must be 32 lowercase hexadecimal characters"
                    )
                yield document_id, text, row_offset


def _sample_records(
    shard: _CorpusShard,
    *,
    cutoff: int,
    seed: int,
    tokenizer: MarinTokenizer,
    excluded_groups: frozenset[str],
    excluded_document_ids: frozenset[str],
) -> Iterator[dict]:
    intervals = _id_intervals(shard.source, seed, cutoff)
    rank_salt = _rank_salt(shard.source, seed)
    preprocessor = BatchTokenizer(
        tokenizer,
        enforce_bos=True,
        enforce_eos=True,
        text_field="text",
        _workaround_len=TOKENIZATION_CHUNK_CHARS,
        long_string_workaround=True,
    )

    def records() -> Iterator[tuple[dict, int]]:
        with StoragePath(shard.path).open("rb") as stream:
            parquet = pq.ParquetFile(stream)
            for document_id, text, row_offset in _iter_normalized_rows(parquet, shard.path, intervals):
                rank_value = (int(document_id, 16) + rank_salt) % _ID_SPACE
                if rank_value >= cutoff:
                    continue
                rank = f"{rank_value:032x}"
                if document_id in excluded_document_ids:
                    continue
                text_bytes_data = text.encode("utf-8")
                text_bytes = len(text_bytes_data)
                duplicate_group = hashlib.sha256(text_bytes_data).hexdigest()
                if duplicate_group in excluded_groups:
                    continue
                if text_bytes > TOKENIZATION_MAX_DOCUMENT_BYTES:
                    raise ValueError(
                        f"document exceeds tokenization size limit: source={shard.source!r}, "
                        f"id={document_id!r}, bytes={text_bytes}, "
                        f"limit={TOKENIZATION_MAX_DOCUMENT_BYTES}"
                    )
                yield (
                    {
                        "source": shard.source,
                        "id": document_id,
                        "sample_rank": rank,
                        "duplicate_group": duplicate_group,
                        "normalized_shard": shard.path,
                        "normalized_row": row_offset,
                        "text": text,
                    },
                    text_bytes,
                )

    for batch in bounded_batches(
        records(),
        max_rows=TOKENIZE_BATCH_ROWS,
        max_bytes=TOKENIZE_BATCH_MAX_BYTES,
        byte_size=lambda record: record[1],
    ):
        yield from _encode_records([item[0] for item in batch], preprocessor)


def _encode_records(records: list[dict], preprocessor: BatchProcessor[dict, dict]) -> Iterator[dict]:
    encoded = preprocessor([{"text": row["text"]} for row in records])
    for row, tokens in zip(records, encoded, strict=True):
        input_ids = list(tokens["input_ids"])
        if not input_ids:
            raise ValueError(f"tokenizer produced no tokens for {row['source']}/{row['id']}")
        yield {**row, "input_ids": input_ids, "token_count": len(input_ids)}


def _read_sample(shard: _CorpusShard, *, cutoff: int, spec: CorpusSampleSpec) -> Iterator[dict]:
    actual_hash = tokenizer_content_hash(spec.tokenizer.name)
    if actual_hash != spec.tokenizer.identity:
        raise ValueError(f"tokenizer content differs from the corpus specification: {actual_hash}")
    yield from _sample_records(
        shard,
        cutoff=cutoff,
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


def prepare_corpus_pool(
    config: CorpusSampleConfig, *, ctx: ZephyrContext, num_ranges: int = SAMPLE_RANGES
) -> RawCorpusPool:
    """Sample all sources at one inclusion probability and measure actual tokens.

    Source estimates only size the first pass. If it is too small, increase the
    same ID-rank cutoff. The final token prefix therefore does not depend on
    those estimates. Each pass reads ID-selected row groups and tokenizes rows.
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
    cutoff = _sampling_cutoff(spec.token_budget, estimate)
    attempt = 0
    while True:
        attempt_path = prefix_join(config.output_path, f"sample-{attempt:03d}")
        dataset = Dataset.from_list(shards).flat_map(partial(_read_sample, cutoff=cutoff, spec=spec))
        ranked = ranked_pool(
            dataset,
            ctx=ctx,
            output_path=attempt_path,
            rank_key=lambda row: f"{row['sample_rank']}:{row['source']}:{row['id']}",
            token_count=lambda row: row["token_count"],
            range_key=lambda row, sampling_cutoff=cutoff: int(row["sample_rank"], 16) * num_ranges // sampling_cutoff,
            num_ranges=num_ranges,
        )
        if ranked.total_tokens >= spec.token_budget:
            break
        if cutoff == _ID_SPACE:
            raise ValueError(f"corpus has {ranked.total_tokens:,} tokens, but the pool requires {spec.token_budget:,}")
        cutoff = min(_ID_SPACE, cutoff * 2)
        attempt += 1
    selected = take_token_prefix(
        ranked, ctx=ctx, output_path=prefix_join(config.output_path, "pool"), token_budget=spec.token_budget
    )
    counts: Counter[str] = Counter({source.name: 0 for source in spec.sources})
    selected_files = [{"index": index, "path": path} for index, path in enumerate(selected.shards)]
    summaries = ctx.execute(Dataset.from_list(selected_files).map(_source_summary)).results
    ranges = tuple(sorted((item.range_total for item in summaries), key=lambda item: item.range_key))
    for result in summaries:
        counts.update(result.source_tokens)
    manifest_path = prefix_join(config.output_path, "corpus.json")
    report = {
        "sampling_method": SAMPLING_POLICY,
        "spec": _spec_identity(spec),
        "input_shards": [asdict(shard) for shard in shards],
        "sampling_cutoff_hex": f"{cutoff:033x}",
        "inclusion_probability": cutoff / _ID_SPACE,
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
    return {
        **asdict(spec),
        "label_exclusion": spec.label_exclusion.identity() if spec.label_exclusion else None,
        "sampling_policy": SAMPLING_POLICY,
        "tokenization_policy": TOKENIZATION_POLICY,
        "tokenization_chunk_chars": TOKENIZATION_CHUNK_CHARS,
        "tokenization_max_document_bytes": TOKENIZATION_MAX_DOCUMENT_BYTES,
    }
