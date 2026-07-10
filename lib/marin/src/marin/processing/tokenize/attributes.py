# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage A of the split tokenize pipeline: NormalizedData → datakit attribute parquet.

Reads one or more :class:`~marin.datakit.normalize.NormalizedData` sources, runs
the shared tokenization core, and writes co-partitioned attribute parquet shards
with columns ``id`` and ``input_ids`` (one row per input document).

Each output shard mirrors the basename of its source shard, which ensures the
datakit invariants hold by construction:

* Same shard count as source (co-partitioned).
* Sorted by ``id`` within each shard (sources are already sorted; the
  tokenize pipeline preserves order).

Downstream:

* :func:`marin.processing.tokenize.store_builder.build_levanter_store` consumes
  one or more :class:`TokenizedAttrData` artifacts to produce a Levanter cache
  per split (sharded layout).
* Other datakit attribute consumers (joins, mixing) can use the ``id`` column to
  align tokens with quality scores, dedup flags, etc.
"""
import dataclasses
import logging
import os

import pyarrow as pa
from fray.types import ResourceConfig
from levanter.data.text.formats import LmDatasetFormatBase, TextLmDatasetFormat
from levanter.tokenizers import TokenizerBackend
from pydantic import BaseModel
from rigging.filesystem import StoragePath, prefix_join
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.readers import load_file

from marin.datakit.normalize import NormalizedData
from marin.execution.artifact import read_artifact
from marin.execution.step_spec import StepSpec
from marin.processing.tokenize._core import tokenize_pipeline

logger = logging.getLogger(__name__)


class TokenizedAttrData(BaseModel):
    """Per-split datakit attribute datasets produced by :func:`tokenize_attributes`.

    Each split's attribute parquet shards live under ``output_dirs[split]/`` with
    columns ``id: string`` and ``input_ids: list<int>``. Shards mirror their source
    :class:`~marin.datakit.normalize.NormalizedData` partitions 1:1 (same basename,
    same row order, same id range), so the dataset is sorted by ``id`` per partition
    and co-partitioned with the source — both datakit invariants.

    Persisted as the step's ``.artifact``. Load via
    ``Artifact.from_path(step, TokenizedAttrData)``.

    Attributes:
        version: Schema version.
        output_dirs: Map from split name (e.g. ``"train"``, ``"validation"``) to the
            directory containing that split's attribute parquet shards.
        source_main_dirs: Map from split name to the source ``NormalizedData.main_output_dir``
            whose shards this dataset mirrors. Used by consumers to verify
            co-partitioning.
        tokenizer: Tokenizer name/path used (informational; consumers should re-verify
            against any other inputs they combine this with).
        tokenizer_backend: Tokenizer backend, as ``TokenizerBackend.value``.
        counters: Aggregated zephyr counters per split.
    """

    version: str = "v1"
    output_dirs: dict[str, str]
    source_main_dirs: dict[str, str]
    tokenizer: str
    tokenizer_backend: str
    counters: dict[str, dict[str, int | float]]

    def shard_paths(self, split: str) -> list[str]:
        """Return parquet shard paths for ``split`` in order, or ``[]`` if absent."""
        d = self.output_dirs.get(split)
        if d is None:
            return []
        return sorted(str(m) for m in StoragePath(prefix_join(d, "*.parquet")).glob())


@dataclasses.dataclass(frozen=True, kw_only=True)
class TokenizeAttributesConfig:
    """Config for :func:`tokenize_attributes`.

    At least one of ``train_source`` or ``validation_source`` must be provided.
    Each is an independent :class:`NormalizedData` artifact whose shards become
    the corresponding split's co-partitioned attribute parquet output.
    """

    train_source: NormalizedData | None = None
    validation_source: NormalizedData | None = None
    output_path: str
    tokenizer: str
    tokenizer_backend: TokenizerBackend = TokenizerBackend.HF
    format: LmDatasetFormatBase = TextLmDatasetFormat()  # noqa: RUF009
    sample_count: int | None = None
    text_field: str = "text"
    max_workers: int = 4096
    worker_resources: ResourceConfig = dataclasses.field(default_factory=lambda: ResourceConfig(ram="10g", disk="5g"))

    def __post_init__(self):
        if self.train_source is None and self.validation_source is None:
            raise ValueError("at least one of train_source / validation_source must be provided")


_ATTRIBUTE_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("input_ids", pa.list_(pa.int32())),
    ]
)


def _process_split(
    *,
    source: NormalizedData,
    split: str,
    config: TokenizeAttributesConfig,
) -> tuple[str, dict[str, int | float]]:
    """Tokenize one split's NormalizedData into co-partitioned attribute parquet.

    Returns ``(split_output_dir, counters)``.
    """
    source_shards = sorted(str(m) for m in StoragePath(prefix_join(source.main_output_dir, "*.parquet")).glob())
    if not source_shards:
        raise FileNotFoundError(f"No parquet shards found under {source.main_output_dir}")

    split_dir = prefix_join(config.output_path, split)
    output_basenames = tuple(os.path.basename(p) for p in source_shards)

    def _output_path(shard_idx: int, total_shards: int, sd: str = split_dir, bn: tuple = output_basenames) -> str:
        return prefix_join(sd, bn[shard_idx])

    logger.info(
        "Tokenizing %s (split=%s): %d source shards → %s",
        source.main_output_dir,
        split,
        len(source_shards),
        split_dir,
    )

    ds = Dataset.from_list(source_shards).flat_map(load_file)
    tokenized_ds, _ = tokenize_pipeline(
        ds,
        data_format=config.format,
        text_field=config.text_field,
        sample_count=config.sample_count,
        # The first source shard is parquet by construction; pass it for the
        # row-group-aware window sizing in the shared core.
        sample_parquet_path=source_shards[0],
        levanter_batch_size=None,
    )

    pipeline = tokenized_ds.write_parquet(
        _output_path,
        schema=_attribute_schema(config.format),
        skip_existing=True,
    )

    ctx = ZephyrContext(
        resources=config.worker_resources,
        max_workers=min(config.max_workers, len(source_shards)),
        name=f"tokenize-attributes-{split}",
    )
    ctx.put("tokenizer_name", config.tokenizer)
    ctx.put("tokenizer_backend", config.tokenizer_backend)

    outcome = ctx.execute(pipeline, verbose=True)
    return split_dir, dict(outcome.counters)


def _attribute_schema(data_format: LmDatasetFormatBase) -> pa.Schema | None:
    """Return the parquet schema for attribute output, or ``None`` to let zephyr infer.

    For text formats we pin ``id: string`` and ``input_ids: list<int32>`` to keep
    files compact and stable across workers. For chat or other multi-output formats,
    we let zephyr infer from the first record so additional columns like
    ``assistant_masks`` flow through unchanged.
    """
    if isinstance(data_format, TextLmDatasetFormat):
        return _ATTRIBUTE_SCHEMA
    return None


def tokenize_attributes(config: TokenizeAttributesConfig) -> TokenizedAttrData:
    """Tokenize :class:`NormalizedData` source(s) into datakit attribute parquet.

    Each split's source shards become co-partitioned attribute parquet files
    sharing basenames with the source. Output records carry ``{id, input_ids}``
    (plus any extra fields produced by the format processor for non-text formats).

    Args:
        config: See :class:`TokenizeAttributesConfig`.

    Returns:
        A :class:`TokenizedAttrData` describing the per-split output directories,
        source linkage, tokenizer config, and counters.
    """
    output_dirs: dict[str, str] = {}
    source_main_dirs: dict[str, str] = {}
    counters: dict[str, dict[str, int | float]] = {}

    splits: list[tuple[str, NormalizedData]] = []
    if config.train_source is not None:
        splits.append(("train", config.train_source))
    if config.validation_source is not None:
        splits.append(("validation", config.validation_source))

    for split, source in splits:
        split_dir, split_counters = _process_split(source=source, split=split, config=config)
        output_dirs[split] = split_dir
        source_main_dirs[split] = source.main_output_dir
        counters[split] = split_counters

    return TokenizedAttrData(
        output_dirs=output_dirs,
        source_main_dirs=source_main_dirs,
        tokenizer=config.tokenizer,
        tokenizer_backend=config.tokenizer_backend.value,
        counters=counters,
    )


def tokenize_attributes_step(
    *,
    name: str,
    train_normalize: StepSpec | None = None,
    validation_normalize: StepSpec | None = None,
    tokenizer: str,
    tokenizer_backend: TokenizerBackend = TokenizerBackend.HF,
    data_format: LmDatasetFormatBase | None = None,
    sample_count: int | None = None,
    text_field: str = "text",
    max_workers: int = 4096,
    worker_resources: ResourceConfig | None = None,
    override_output_path: str | None = None,
) -> StepSpec:
    """Create a :class:`StepSpec` that tokenizes :class:`NormalizedData` source(s) into attribute parquet.

    At least one of ``train_normalize`` or ``validation_normalize`` must be provided;
    each upstream step's output is loaded as :class:`NormalizedData` and routed to
    the corresponding split. The artifact persisted at the step's output path is
    a :class:`TokenizedAttrData`.

    Args:
        name: Step name (e.g. ``"fineweb/tokenize"``).
        train_normalize: Upstream normalize step whose output feeds the train split.
        validation_normalize: Upstream normalize step whose output feeds the validation split.
        tokenizer: Tokenizer name/path forwarded to :class:`TokenizeAttributesConfig`.
        tokenizer_backend: Tokenizer backend.
        data_format: Levanter :class:`LmDatasetFormatBase`. Defaults to ``TextLmDatasetFormat()``.
        sample_count: Per-shard sample cap, or ``None`` for full data.
        text_field: Record field used for id fallback when input lacks ``id``.
            Datakit-normalized inputs always carry ``id`` so this only matters
            on non-normalized paths (not used here, but mirrored to the config).
        max_workers: Zephyr worker cap.
        worker_resources: Per-worker resources; defaults inside the config.
        override_output_path: Optional explicit output path.
    """
    if train_normalize is None and validation_normalize is None:
        raise ValueError("tokenize_attributes_step: at least one of train_normalize / validation_normalize required")

    fmt = data_format or TextLmDatasetFormat()
    deps: list[StepSpec] = [s for s in (train_normalize, validation_normalize) if s is not None]

    def _fn(output_path: str) -> TokenizedAttrData:
        kwargs: dict = {
            "output_path": output_path,
            "tokenizer": tokenizer,
            "tokenizer_backend": tokenizer_backend,
            "format": fmt,
            "sample_count": sample_count,
            "text_field": text_field,
            "max_workers": max_workers,
        }
        if train_normalize is not None:
            kwargs["train_source"] = read_artifact(train_normalize.output_path, NormalizedData)
        if validation_normalize is not None:
            kwargs["validation_source"] = read_artifact(validation_normalize.output_path, NormalizedData)
        if worker_resources is not None:
            kwargs["worker_resources"] = worker_resources
        return tokenize_attributes(TokenizeAttributesConfig(**kwargs))

    return StepSpec(
        name=name,
        deps=deps,
        fn=_fn,
        hash_attrs={
            "tokenizer": tokenizer,
            "tokenizer_backend": tokenizer_backend.value,
            "format": repr(fmt),
            "sample_count": sample_count,
            "text_field": text_field,
        },
        override_output_path=override_output_path,
    )
