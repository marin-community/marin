# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compute MinHash bucket attributes for a normalized datakit dataset.

Reads a :class:`~marin.datakit.normalize.NormalizedData` (Parquet shards under
``main_output_dir``), runs the dupekit MinHash + LSH pipeline per shard, and
writes a co-partitioned attribute dataset whose Parquet files share their
basenames with the source shards.

The output is a :class:`MinHashAttrData` artifact recording the MinHash params
and the attr directory. Downstream :func:`~marin.processing.classification.\
deduplication.fuzzy_dups.compute_fuzzy_dups_attrs` consumes one or more of
these artifacts to produce duplicate markers.
"""

from __future__ import annotations

import logging
import os

import dupekit
import pyarrow as pa
from fray import ResourceConfig
from pydantic import BaseModel
from zephyr import Dataset, ZephyrContext, counters, write_parquet_file

from marin.datakit.normalize import NormalizedData
from marin.execution.artifact import Artifact
from marin.execution.step_spec import StepSpec
from marin.processing.classification.deduplication.dedup_commons import _load_batches
from marin.utils import fsspec_glob

logger = logging.getLogger(__name__)


class MinHashParams(BaseModel):
    """MinHash + LSH parameters that downstream fuzzy-dup consumers must agree on.

    Two ``MinHashAttrData`` artifacts can only be combined in
    :func:`compute_fuzzy_dups_attrs` if their params are equal.
    """

    num_perms: int
    num_bands: int
    ngram_size: int
    seed: int


class MinHashAttrData(BaseModel):
    """Co-partitioned MinHash bucket attrs computed for one ``NormalizedData``.

    Persisted as the step's ``.artifact``. Load via
    ``Artifact.load(step, MinHashAttrData)``.

    Attributes:
        version: Schema version of this artifact.
        params: MinHash params; downstream jobs require these to match.
        source_main_dir: Source ``NormalizedData.main_output_dir`` whose shards
            this dataset mirrors 1:1.
        attr_dir: Directory containing per-shard attr Parquet files. Filenames
            mirror the source shards. Each row has ``id: str`` and
            ``buckets: list[str]``.
        counters: Aggregated zephyr counters.
    """

    version: str = "v1"
    params: MinHashParams
    source_main_dir: str
    attr_dir: str
    counters: dict[str, int]


def _attr_records(batch: pa.RecordBatch, params: MinHashParams) -> list[dict]:
    """Run the dupekit MinHash+LSH pipeline on *batch* and yield attr records.

    Yields one ``{id, buckets}`` record per input document with at least one
    bucket. Documents whose signature column is null (empty/whitespace text
    after cleaning) are dropped and counted via ``minhash/empty_signatures``.
    """
    pipeline = [
        dupekit.Transformation.CleanText(input_col="text", output_col="clean_text"),
        dupekit.Transformation.MinHash(
            input_col="clean_text",
            output_col="signature",
            num_perms=params.num_perms,
            ngram_size=params.ngram_size,
            seed=params.seed,
        ),
        dupekit.Transformation.MinHashLSH(input_col="signature", output_col="buckets", num_bands=params.num_bands),
        dupekit.Transformation.SelectColumns(columns=["id", "buckets"]),
    ]
    result_batch = dupekit.transform(batch, pipeline)
    ids = result_batch["id"]
    buckets_col = result_batch["buckets"]

    out: list[dict] = []
    for doc_id, doc_buckets in zip(ids, buckets_col, strict=True):
        if not doc_buckets.is_valid:
            counters.increment("minhash/empty_signatures")
            continue
        bucket_strs = [str(b) for b in doc_buckets.as_py()]
        counters.increment("minhash/documents")
        counters.increment("minhash/buckets", len(bucket_strs))
        out.append({"id": doc_id.as_py(), "buckets": bucket_strs})
    return out


def _make_shard_processor(attr_dir: str, params: MinHashParams):
    """Return a per-shard map fn that loads one source parquet, runs MinHash, writes attr parquet."""

    def process(shard_path: str) -> dict:
        basename = os.path.basename(shard_path)
        attr_path = f"{attr_dir}/{basename}"

        def records():
            for batch in _load_batches(shard_path, columns=["id", "text"]):
                yield from _attr_records(batch, params)

        result = write_parquet_file(records(), attr_path)
        return {"source_path": shard_path, "attr_path": attr_path, "count": result["count"]}

    return process


def compute_minhash_attrs(
    *,
    source: NormalizedData,
    output_path: str,
    num_perms: int = 286,
    num_bands: int = 26,
    ngram_size: int = 5,
    seed: int = 42,
    worker_resources: ResourceConfig | None = None,
    max_workers: int | None = None,
) -> MinHashAttrData:
    """Compute MinHash bucket attributes for *source* and persist as Parquet.

    Each source shard under ``source.main_output_dir`` produces a same-named
    attr file under ``<output_path>/outputs/`` with columns ``id`` and
    ``buckets`` (``list[str]``). The output dataset is co-partitioned with the
    source per the datakit invariant.

    Args:
        source: The normalized source dataset to read from.
        output_path: Output root. Attr files land under ``<output_path>/outputs/``.
        num_perms: Number of MinHash permutations. Must be divisible by
            ``num_bands``.
        num_bands: Number of LSH bands.
        ngram_size: Word n-gram size for shingling.
        seed: MinHash seed.
        worker_resources: Per-worker resource request. Sized similarly to the
            old ``dedup_fuzzy_document``: dupekit's Rust MinHash pipeline uses
            a native thread pool and may consume up to ~2 cores beyond the
            Python thread.
        max_workers: Max Zephyr workers. Defaults to Zephyr's own default.

    Returns:
        :class:`MinHashAttrData` describing the attr directory and counters.
    """
    if num_perms % num_bands != 0:
        raise ValueError(f"num_perms ({num_perms}) must be divisible by num_bands ({num_bands})")

    params = MinHashParams(num_perms=num_perms, num_bands=num_bands, ngram_size=ngram_size, seed=seed)
    attr_dir = os.path.join(output_path, "outputs")

    source_shards = sorted(fsspec_glob(f"{source.main_output_dir.rstrip('/')}/*.parquet"))
    if not source_shards:
        raise FileNotFoundError(f"No parquet shards found under {source.main_output_dir}")

    logger.info(
        "Computing MinHash attrs for %s → %s: %d shards, params=%s",
        source.main_output_dir,
        attr_dir,
        len(source_shards),
        params,
    )

    ctx_kwargs: dict = {
        "name": "minhash-attrs",
        "resources": worker_resources or ResourceConfig(cpu=5, ram="32g", disk="5g"),
    }
    if max_workers is not None:
        ctx_kwargs["max_workers"] = max_workers
    ctx = ZephyrContext(**ctx_kwargs)

    pipeline = Dataset.from_list(source_shards).map(_make_shard_processor(attr_dir, params))
    outcome = ctx.execute(pipeline, verbose=True)

    return MinHashAttrData(
        params=params,
        source_main_dir=source.main_output_dir,
        attr_dir=attr_dir,
        counters=dict(outcome.counters),
    )


def compute_minhash_attrs_step(
    *,
    name: str,
    normalize: StepSpec,
    num_perms: int = 286,
    num_bands: int = 26,
    ngram_size: int = 5,
    seed: int = 42,
    worker_resources: ResourceConfig | None = None,
    max_workers: int | None = None,
    override_output_path: str | None = None,
) -> StepSpec:
    """Create a StepSpec that computes MinHash attrs from a normalize step."""
    return StepSpec(
        name=name,
        deps=[normalize],
        fn=lambda output_path: compute_minhash_attrs(
            source=Artifact.load(normalize, NormalizedData),
            output_path=output_path,
            num_perms=num_perms,
            num_bands=num_bands,
            ngram_size=ngram_size,
            seed=seed,
            worker_resources=worker_resources,
            max_workers=max_workers,
        ),
        hash_attrs={
            "num_perms": num_perms,
            "num_bands": num_bands,
            "ngram_size": ngram_size,
            "seed": seed,
        },
        override_output_path=override_output_path,
    )
