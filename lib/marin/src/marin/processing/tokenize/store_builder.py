# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage B of the split tokenize pipeline: tokenized records → Levanter cache.

Two entry points:

* :func:`build_from_datasets` — modular core. Takes a caller-prepared zephyr
  ``Dataset`` of ``{id, input_ids, ...}`` records and writes a Levanter cache
  with a top-level sharded ledger referencing per-shard subdirectories under
  ``output_path``. Used by :func:`marin.processing.tokenize.tokenize.tokenize`
  on the legacy hot path (no parquet round-trip) and by :func:`build_levanter_store`.

* :func:`build_levanter_store` — convenience wrapper. Reads attribute parquet
  partitions from one or more :class:`TokenizedAttrData` artifacts and builds a
  Levanter cache per split.

The output layout is the "sharded" layout introduced by Levanter ``#5430``:
``cache_dir/shard_ledger.json`` aggregates the per-shard ledgers, while the
shard data itself lives under ``cache_dir/part-NNNNN-of-MMMMM/`` and must NOT
be removed — those subdirectories are the cache. Downstream readers load via
``TreeCache.load`` (or ``UrlDatasetSourceConfig``), which transparently
dispatches on the sharded layout.
"""
import dataclasses
import json
import logging
import time

import numpy as np
from fray import ResourceConfig
from levanter.store.cache import (
    CacheLedger,
    ShardedCacheLayout,
    consolidate_shard_cache_ledgers,
    write_levanter_cache,
)
from pydantic import BaseModel
from rigging.filesystem import StoragePath, prefix_join
from zephyr import Dataset, ZephyrContext
from zephyr.dataset import format_shard_path
from zephyr.readers import load_file

from marin.execution.artifact import read_artifact
from marin.execution.step_spec import StepSpec
from marin.processing.tokenize.attributes import TokenizedAttrData

logger = logging.getLogger(__name__)


class LevanterSplitStats(BaseModel):
    """Per-split summary of a built Levanter store.

    Attributes:
        path: Cache directory for this split. Contains a top-level
            ``shard_ledger.json`` plus the per-shard subdirectories it references;
            load via ``TreeCache.load``.
        total_elements: Document count, mirrors ``CacheLedger.total_num_rows``.
        total_tokens: Total tokens across all documents (``input_ids`` field count).
    """

    path: str
    total_elements: int
    total_tokens: int


class LevanterStoreData(BaseModel):
    """Outcome of :func:`build_levanter_store`: a Levanter cache per split.

    Persisted as the step's ``.artifact``. Load via
    ``Artifact.from_path(step, LevanterStoreData)``.

    Attributes:
        version: Schema version.
        cache_path: Base directory; each split's Levanter cache lives in
            ``cache_path/<split>``.
        splits: Map from split name to per-split stats.
        source_dirs: For provenance — the ``output_dirs`` of each source
            :class:`TokenizedAttrData` flattened in input order.
        tokenizer: Tokenizer name copied from the first source (informational).
    """

    version: str = "v1"
    cache_path: str
    splits: dict[str, LevanterSplitStats]
    source_dirs: list[dict[str, str]]
    tokenizer: str


def _strip_id(record: dict) -> dict:
    """Drop ``id`` from a record. Levanter ``TreeStore`` is positional, not keyed."""
    if "id" not in record:
        return record
    return {k: v for k, v in record.items() if k != "id"}


def _structural_exemplar(record: dict) -> dict:
    """Shrink a record to a structural template for cache consolidation.

    :func:`consolidate_shard_cache_ledgers` only needs the exemplar's pytree
    structure and leaf dtypes to open a ``TreeStore``, never its values. Slicing
    sequence leaves to a single element keeps the per-shard payload returned to
    the coordinator tiny — a full document can be megabytes for long-context
    data, and we return one per shard.

    This is an optimization to avoid materializing too much data on the
    entrypoint: every shard ships its exemplar back, but only the structure is
    needed, so we never pull whole documents onto the coordinator.
    """
    return {k: v[:1] if isinstance(v, np.ndarray | list | tuple) else v for k, v in record.items()}


def build_from_datasets(
    *,
    ctx: ZephyrContext,
    dataset: Dataset,
    output_path: str,
    batch_size: int | None = None,
    skip_existing: bool = True,
) -> CacheLedger:
    """Write a Levanter cache from a tokenized records dataset.

    The dataset is expected to yield per-doc records shaped like
    ``{id, input_ids, ...}``. ``id`` is stripped before writing because
    ``TreeStore`` stores positional concatenated arrays, not keyed rows.

    The dataset's shard structure determines the number of per-shard Levanter
    caches written under ``{output_path}/part-NNNNN-of-MMMMM``. Those shard
    directories *are* the cache and must remain in place; the consolidation
    step only writes a top-level ``shard_ledger.json`` that references them by
    relative path (sharded layout, see Levanter ``#5430``).

    Args:
        ctx: Zephyr context. The caller is responsible for setting any tokenizer
            shared values (``ctx.put('tokenizer_name', ...)`` etc.) before calling
            if the upstream dataset relies on them.
        dataset: Zephyr ``Dataset`` of tokenized records.
        output_path: Destination cache directory.
        batch_size: Per-shard Levanter cache flush size. ``None`` keeps Levanter's
            default (16384). Lower values reduce peak memory for datasets with
            very large documents.
        skip_existing: Skip writing intermediate shards whose output already exists.

    Returns:
        The merged sharded ``CacheLedger``.
    """
    pipeline_start = time.monotonic()

    output_pattern = prefix_join(output_path, "part-{shard:05d}-of-{total:05d}")
    write_kwargs: dict = {"metadata": {}}
    if batch_size is not None:
        write_kwargs["batch_size"] = batch_size

    def _write_shard(records, shard_info):
        """Write one shard and emit ``(path, exemplar)``.

        ``exemplar`` is ``None`` for empty shards and for skipped (already-written)
        shards; the coordinator picks the first non-``None`` one for consolidation.
        """
        shard_path = format_shard_path(output_pattern, shard_info.shard_idx, shard_info.total_shards)
        if skip_existing:
            if StoragePath(prefix_join(shard_path, ".success")).exists():
                logger.info("Skipping write, output exists: %s", shard_path)
                yield (shard_path, None)
                return
        result = write_levanter_cache(records, shard_path, **write_kwargs)
        exemplar = result["exemplar"]
        yield (shard_path, _structural_exemplar(exemplar) if exemplar is not None else None)

    temp_shards = dataset.map(_strip_id).map_shard(_write_shard)

    tokenize_start = time.monotonic()
    shard_results = ctx.execute(temp_shards).results
    tokenize_elapsed = time.monotonic() - tokenize_start

    shard_paths = [path for path, _ in shard_results]
    exemplar = next((ex for _, ex in shard_results if ex is not None), None)

    consolidate_start = time.monotonic()
    logger.info("Consolidating %d shards into %s", len(shard_paths), output_path)
    ledger = consolidate_shard_cache_ledgers(
        shard_cache_paths=shard_paths,
        output_path=output_path,
        exemplar=exemplar,
    )
    consolidate_elapsed = time.monotonic() - consolidate_start

    pipeline_elapsed = time.monotonic() - pipeline_start
    logger.info(
        "build_from_datasets done: %s in %.1fs (write: %.1fs, consolidate: %.1fs)",
        output_path,
        pipeline_elapsed,
        tokenize_elapsed,
        consolidate_elapsed,
    )

    return ledger


def write_stats_json(output_path: str, ledger: CacheLedger) -> tuple[str, dict[str, int]]:
    """Write a ``.stats.json`` summary next to a Levanter cache.

    Returns ``(stats_path, stats_dict)`` where ``stats_dict`` carries
    ``total_tokens`` and ``total_elements``.
    """
    total_tokens = ledger.field_counts.get("input_ids", 0)
    stats = {"total_tokens": total_tokens, "total_elements": ledger.total_num_rows}
    stats_path = prefix_join(output_path, ".stats.json")
    StoragePath(stats_path).write_text(json.dumps(stats))
    return stats_path, stats


@dataclasses.dataclass(frozen=True, kw_only=True)
class BuildLevanterStoreConfig:
    """Config for assembling a Levanter cache from one or more :class:`TokenizedAttrData` sources.

    All sources must agree on tokenizer/format — that's the caller's responsibility.
    The store builder concatenates attribute records across sources in the order
    provided and writes one Levanter cache per split (sharded layout).
    """

    sources: list[TokenizedAttrData]
    cache_path: str
    max_workers: int = 4096
    levanter_batch_size: int | None = None
    worker_resources: ResourceConfig = dataclasses.field(default_factory=lambda: ResourceConfig(ram="10g", disk="5g"))

    def __post_init__(self):
        if not self.sources:
            raise ValueError("BuildLevanterStoreConfig: at least one source required")


def _split_shard_paths(sources: list[TokenizedAttrData], split: str) -> list[str]:
    """Collect the parquet shard paths for a given split across all sources."""
    paths: list[str] = []
    for source in sources:
        split_dir = source.output_dirs.get(split)
        if split_dir is None:
            continue
        paths.extend(sorted(source.shard_paths(split)))
    return paths


def build_levanter_store(config: BuildLevanterStoreConfig) -> LevanterStoreData:
    """Build one Levanter cache per split from :class:`TokenizedAttrData` sources.

    For each split that exists in any source, this:

    1. Collects parquet shard paths across all sources.
    2. Runs :func:`build_from_datasets` to write per-shard caches and a
       top-level sharded ledger.
    3. Writes ``.stats.json`` next to the ledger.

    Splits with no shards are skipped. If a split's ledger already exists,
    this loads the existing stats rather than rebuilding.

    Returns:
        :class:`LevanterStoreData` describing the cache layout and per-split counts.
    """
    splits = sorted({s for src in config.sources for s in src.output_dirs.keys()})
    if not splits:
        raise ValueError("No splits found across sources; nothing to build")

    split_stats: dict[str, LevanterSplitStats] = {}

    for split in splits:
        shard_paths = _split_shard_paths(config.sources, split)
        if not shard_paths:
            logger.info("No shards for split %s; skipping", split)
            continue

        split_output = prefix_join(config.cache_path, split)

        if _ledger_exists(split_output):
            logger.info("Shard ledger already exists for %s at %s; loading", split, split_output)
            ledger = CacheLedger.load(split_output)
        else:
            ctx = ZephyrContext(
                resources=config.worker_resources,
                max_workers=min(config.max_workers, len(shard_paths)),
                name=f"build-levanter-store-{split}",
            )

            dataset = Dataset.from_list(shard_paths).flat_map(load_file)
            ledger = build_from_datasets(
                ctx=ctx,
                dataset=dataset,
                output_path=split_output,
                batch_size=config.levanter_batch_size,
            )

        stats_path, stats = write_stats_json(split_output, ledger)
        split_stats[split] = LevanterSplitStats(
            path=split_output,
            total_elements=stats["total_elements"],
            total_tokens=stats["total_tokens"],
        )
        logger.info(
            "build_levanter_store: split=%s docs=%d tokens=%d shards=%d → %s (stats: %s)",
            split,
            stats["total_elements"],
            stats["total_tokens"],
            len(shard_paths),
            split_output,
            stats_path,
        )

    return LevanterStoreData(
        cache_path=config.cache_path,
        splits=split_stats,
        source_dirs=[dict(src.output_dirs) for src in config.sources],
        tokenizer=config.sources[0].tokenizer,
    )


def _ledger_exists(cache_path: str) -> bool:
    """Return whether a Levanter cache ledger already exists at ``cache_path``."""
    return StoragePath(ShardedCacheLayout.parse(cache_path).ledger).exists()


def build_levanter_store_step(
    *,
    name: str,
    tokenize_steps: list[StepSpec],
    max_workers: int = 4096,
    levanter_batch_size: int | None = None,
    worker_resources: ResourceConfig | None = None,
    override_output_path: str | None = None,
) -> StepSpec:
    """Create a :class:`StepSpec` that assembles a Levanter store from one or more
    :class:`TokenizedAttrData` step outputs.

    The step's output path becomes ``LevanterStoreData.cache_path``; each
    split's Levanter cache lives at ``cache_path/<split>``.

    Args:
        name: Step name.
        tokenize_steps: Upstream :func:`tokenize_attributes_step` results to combine.
            Records are concatenated across sources in the given order; sources
            must agree on tokenizer/format (caller's responsibility).
        max_workers: Zephyr worker cap.
        levanter_batch_size: Per-shard Levanter cache flush size.
        worker_resources: Per-worker resources; defaults inside the config.
        override_output_path: Optional explicit output path.
    """
    if not tokenize_steps:
        raise ValueError("build_levanter_store_step: at least one tokenize step required")

    def _fn(output_path: str) -> LevanterStoreData:
        kwargs: dict = {
            "sources": [read_artifact(s.output_path, TokenizedAttrData) for s in tokenize_steps],
            "cache_path": output_path,
            "max_workers": max_workers,
            "levanter_batch_size": levanter_batch_size,
        }
        if worker_resources is not None:
            kwargs["worker_resources"] = worker_resources
        return build_levanter_store(BuildLevanterStoreConfig(**kwargs))

    return StepSpec(
        name=name,
        deps=list(tokenize_steps),
        fn=_fn,
        hash_attrs={
            "levanter_batch_size": levanter_batch_size,
        },
        override_output_path=override_output_path,
    )
