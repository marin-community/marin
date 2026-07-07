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
from collections.abc import Iterator

import dupekit
import pyarrow as pa
from fray import ResourceConfig
from pydantic import BaseModel
from rigging.filesystem import prefix_join
from zephyr import Dataset, ZephyrContext, counters

from marin.datakit.normalize import NormalizedData
from marin.execution.artifact import read_artifact
from marin.execution.step_spec import StepSpec
from marin.processing.classification.deduplication.dedup_commons import _load_batches
from marin.utils import fsspec_glob

logger = logging.getLogger(__name__)


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
        source_main_dir: Source ``NormalizedData.main_output_dir`` whose shards
            this dataset mirrors 1:1.
        attr_dir: Directory containing per-shard attr Parquet files. Filenames
            mirror the source shards. Each row has ``id: str`` and
            ``buckets: list[str]``.
        counters: Aggregated zephyr counters.
    """

    version: str = "v2"
    params: MinHashParams
    source_main_dir: str
    attr_dir: str
    counters: dict[str, int | float]


def _attr_records(batch: pa.RecordBatch, params: MinHashParams) -> list[dict]:
    """Run the dupekit MinHash+LSH pipeline on *batch* and yield attr records.

    Yields one ``{id, buckets}`` record per input document with at least one
    bucket. Documents whose signature column is null (empty/whitespace text
    after cleaning) are dropped and counted via ``minhash/empty_signatures``.
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
        cap = params.text_cap_chars
        n_truncated = 0
        truncated: list[str] = []
        for t in batch["text"]:
            text = t.as_py() or ""
            if len(text) > cap:
                truncated.append(text[:cap])
                n_truncated += 1
            else:
                truncated.append(text)
        if n_truncated:
            counters.pipeline.update_counter("minhash/text_truncated", n_truncated)
        batch = batch.set_column(
            batch.schema.get_field_index("text"),
            "text",
            pa.array(truncated, type=pa.string()),
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
    ids = result_batch["id"]
    buckets_col = result_batch["buckets"]

    out: list[dict] = []
    for doc_id, doc_buckets in zip(ids, buckets_col, strict=True):
        if not doc_buckets.is_valid:
            counters.pipeline.update_counter("minhash/empty_signatures", 1)
            continue
        bucket_strs = [str(b) for b in doc_buckets.as_py()]
        counters.pipeline.update_counter("minhash/documents", 1)
        counters.pipeline.update_counter("minhash/buckets", len(bucket_strs))
        out.append({"id": doc_id.as_py(), "buckets": bucket_strs})
    return out


def _shard_attr_records(shard_path: str, params: MinHashParams) -> Iterator[dict]:
    """Stream ``{id, buckets}`` attr records for one source parquet shard."""
    for batch in _load_batches(shard_path, columns=["id", "text"]):
        yield from _attr_records(batch, params)


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
    map_workers_per_actor: int | None = None,
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

    params = MinHashParams(
        num_perms=num_perms,
        num_bands=num_bands,
        ngram_size=ngram_size,
        seed=seed,
        text_cap_chars=text_cap_chars,
    )
    attr_dir = prefix_join(output_path, "outputs")

    source_shards = sorted(fsspec_glob(prefix_join(source.main_output_dir, "*.parquet")))
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
    if map_workers_per_actor is not None:
        ctx_kwargs["map_workers_per_actor"] = map_workers_per_actor
    ctx = ZephyrContext(**ctx_kwargs)

    # Preserve source basenames; zephyr's `{basename}` placeholder is synthetic.
    output_basenames = tuple(os.path.basename(p) for p in source_shards)

    def _output_path(shard_idx: int, total_shards: int, ad: str = attr_dir, bn: tuple = output_basenames) -> str:
        return prefix_join(ad, bn[shard_idx])

    pipeline = (
        Dataset.from_list(source_shards)
        .flat_map(lambda path, p=params: _shard_attr_records(path, p))
        .write_parquet(_output_path, skip_existing=True)
    )
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
            "num_perms": num_perms,
            "num_bands": num_bands,
            "ngram_size": ngram_size,
            "text_cap_chars": text_cap_chars,
            "seed": seed,
        },
        override_output_path=override_output_path,
    )
