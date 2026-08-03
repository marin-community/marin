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

import logging
import os

import dupekit
import pyarrow as pa
import pyarrow.compute as pc
from fray.types import ResourceConfig
from pydantic import BaseModel
from rigging.filesystem import StoragePath, prefix_join
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext

from marin.datakit.normalize import NormalizedData
from marin.datakit.source_key import DatakitArtifactPath, datakit_source_key
from marin.execution.artifact import read_artifact
from marin.execution.step_spec import StepSpec

logger = logging.getLogger(__name__)
MINHASH_ATTR_DATA_VERSION = 3
_MAX_MINHASH_TASKS_PER_WORKER = 16


class MinHashParams(BaseModel):
    """MinHash + LSH parameters that downstream fuzzy-dup consumers must agree on.

    Two ``MinHashAttrData`` artifacts can only be combined in
    :func:`compute_fuzzy_dups_attrs` if their params are equal.

    ``text_cap_chars`` (added v2) caps each document's character length
    before shingling/MinHash. Documents with O(10M+) shingles produce
    saturated MinHash signatures that band-collide with arbitrary other
    documents, creating large CC false-positive clusters. Capping bounds
    signature density so a single mega-doc cannot link unrelated
    documents into one cluster. ``None`` preserves pre-v2 behavior.
    """

    num_perms: int
    num_bands: int
    ngram_size: int
    seed: int
    text_cap_chars: int | None = None


class MinHashAttrData(BaseModel):
    """Co-partitioned MinHash bucket attrs computed for one ``NormalizedData``.

    Persisted as the step's ``.artifact``. Load via
    ``Artifact.from_path(step, MinHashAttrData)``.

    Attributes:
        version: Schema version of this artifact.
        params: MinHash params; downstream jobs require these to match.
        source_key: Prefix-relative identity of the ``NormalizedData.main_output_dir``
            whose shards this dataset mirrors 1:1.
        attr_dir: Directory containing per-shard attr Parquet files. Filenames
            mirror the source shards. Each row has ``id: str`` and
            ``buckets: list[str]``.
        counters: Aggregated zephyr counters.
    """

    version: str = f"v{MINHASH_ATTR_DATA_VERSION}"
    params: MinHashParams
    source_key: str
    attr_dir: DatakitArtifactPath
    counters: dict[str, int | float]


def _minhash_batch(batch: pa.RecordBatch, params: MinHashParams) -> pa.RecordBatch:
    """Return MinHash bucket attributes for non-empty documents in one batch.

    Returns one ``{id, buckets}`` row per input document with at least one
    bucket. Documents whose signature column is null are dropped and counted
    via ``minhash/empty_signatures``.
    """
    if params.text_cap_chars is not None:
        # Truncate the text column to cap shingle count per doc. Mega-docs
        # otherwise produce saturated MinHash signatures that LSH-collide
        # with arbitrary content; see CROSS_SOURCE_REPORT.md § Root cause.
        #
        # TODO(rav): measure impact -- compare CC-size distribution and
        # cross-source false-positive rate (mega-doc-seeded blobs) against
        # the v1 (uncapped) run, and sweep the cap to find the
        # recall/precision knee. See `minhash/text_truncated` counter for
        # the per-job cap rate.
        text = pc.fill_null(batch["text"], "")
        truncated_mask = pc.greater(pc.utf8_length(text), params.text_cap_chars)
        n_truncated = pc.sum(pc.cast(truncated_mask, pa.int64())).as_py() or 0
        if n_truncated:
            counters.pipeline.update_counter("minhash/text_truncated", n_truncated)
        batch = batch.set_column(
            batch.schema.get_field_index("text"),
            "text",
            pc.utf8_slice_codeunits(text, 0, params.text_cap_chars),
        )

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
    valid_signatures = pc.is_valid(result_batch["buckets"])
    documents = pc.sum(pc.cast(valid_signatures, pa.int64())).as_py() or 0
    empty_signatures = result_batch.num_rows - documents
    if empty_signatures:
        counters.pipeline.update_counter("minhash/empty_signatures", empty_signatures)

    result_batch = result_batch.filter(valid_signatures)
    bucket_strings = pc.cast(result_batch["buckets"], pa.list_(pa.string()))
    bucket_count = pc.sum(pc.list_value_length(bucket_strings)).as_py() or 0
    counters.pipeline.update_counter("minhash/documents", documents)
    counters.pipeline.update_counter("minhash/buckets", bucket_count)
    return result_batch.set_column(result_batch.schema.get_field_index("buckets"), "buckets", bucket_strings)


def compute_minhash_attrs(
    *,
    source: NormalizedData,
    output_path: str,
    num_perms: int = 286,
    num_bands: int = 26,
    ngram_size: int = 5,
    text_cap_chars: int | None = 500_000,
    seed: int = 42,
    worker_resources: ResourceConfig | None = None,
    max_workers: int | None = None,
    map_task_resources: ResourceConfig | None = None,
    reduce_task_resources: ResourceConfig | None = None,
    zephyr_context: ZephyrContext | None = None,
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
        ngram_size: Character n-gram size for shingling. Applied to text
            after dupekit's CleanText (lowercase, strip punctuation,
            collapse whitespace).
        text_cap_chars: If set, truncate each document to this many chars
            before MinHash so mega-docs cannot saturate the signature
            space and cause LSH false-positive blobs. Default 500,000
            chars (~100K shingles at ngram=5). Pass ``None`` to disable
            (pre-v2 behavior).
        seed: MinHash seed.
        worker_resources: Per-Iris-worker resource request. Unless
            ``map_task_resources`` is explicit, each whole worker CPU admits
            one concurrent MinHash subprocess, capped at 16 per worker.
        max_workers: Max Zephyr workers. Defaults to Zephyr's own default.
        map_task_resources: ResourceConfig for map-stage tasks. Defaults to a
            single-CPU, proportional RAM/disk share of ``worker_resources``,
            with at most 16 shares per worker.
        reduce_task_resources: ResourceConfig for reduce-stage tasks.
        zephyr_context: Optional shared Zephyr context.

    Returns:
        :class:`MinHashAttrData` describing the attr directory and counters.
    """
    if num_perms % num_bands != 0:
        raise ValueError(f"num_perms ({num_perms}) must be divisible by num_bands ({num_bands})")

    params = MinHashParams(
        num_perms=num_perms,
        num_bands=num_bands,
        ngram_size=ngram_size,
        seed=seed,
        text_cap_chars=text_cap_chars,
    )
    attr_dir = prefix_join(output_path, "outputs")

    source_shards = sorted(str(m) for m in StoragePath(prefix_join(source.main_output_dir, "*.parquet")).glob())
    if not source_shards:
        raise FileNotFoundError(f"No parquet shards found under {source.main_output_dir}")

    logger.info(
        "Computing MinHash attrs for %s → %s: %d shards, params=%s",
        source.main_output_dir,
        attr_dir,
        len(source_shards),
        params,
    )

    resources = worker_resources or ResourceConfig(cpu=5, ram="32g", disk="5g")
    if map_task_resources is None:
        tasks_per_worker = max(1, min(_MAX_MINHASH_TASKS_PER_WORKER, int(resources.cpu)))
        task_cpu = min(1.0, resources.cpu)
        map_task_resources = resources.scale(
            cpu=task_cpu / resources.cpu,
            ram=1 / tasks_per_worker,
            disk=1 / tasks_per_worker,
        )

    ctx_kwargs: dict = {
        "name": "minhash-attrs",
        "resources": resources,
    }
    if max_workers is not None:
        ctx_kwargs["max_workers"] = max_workers
    ctx = zephyr_context or ZephyrContext(**ctx_kwargs)

    # Preserve source basenames; zephyr's `{basename}` placeholder is synthetic.
    output_basenames = tuple(os.path.basename(p) for p in source_shards)

    def _output_path(shard_idx: int, total_shards: int, ad: str = attr_dir, bn: tuple = output_basenames) -> str:
        return prefix_join(ad, bn[shard_idx])

    pipeline = (
        Dataset.from_list(source_shards)
        .load_parquet(columns=["id", "text"], batch_mode=True)
        .map_batches(lambda batch, p=params: _minhash_batch(batch, p))
        .write_parquet(_output_path, skip_existing=True)
    )
    outcome = ctx.execute(
        pipeline,
        verbose=True,
        map_task_resources=map_task_resources,
        reduce_task_resources=reduce_task_resources,
    )

    return MinHashAttrData(
        params=params,
        source_key=datakit_source_key(source.main_output_dir),
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
    text_cap_chars: int | None = 500_000,
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
            source=read_artifact(normalize.output_path, NormalizedData),
            output_path=output_path,
            num_perms=num_perms,
            num_bands=num_bands,
            ngram_size=ngram_size,
            text_cap_chars=text_cap_chars,
            seed=seed,
            worker_resources=worker_resources,
            max_workers=max_workers,
        ),
        hash_attrs={
            "artifact_version": MINHASH_ATTR_DATA_VERSION,
            "num_perms": num_perms,
            "num_bands": num_bands,
            "ngram_size": ngram_size,
            "text_cap_chars": text_cap_chars,
            "seed": seed,
        },
        override_output_path=override_output_path,
    )
