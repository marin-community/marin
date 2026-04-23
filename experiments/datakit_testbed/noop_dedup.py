# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Metadata-only dedup step for the Datakit Testbed no-dedup baseline.

Produces a ``FuzzyDupsAttrData`` shaped identically to the
``compute_fuzzy_dups_attrs`` output, but every per-shard attr parquet is empty.
Combined with ``consolidate(filter=KEEP_DOC, name="is_cluster_canonical",
keep_if_missing=True)`` every document in each input passes through, so the
DAG shape matches the real-dedup variant without touching any data.

The step **never reads** input parquet rows — it only enumerates shard
filenames via :func:`fsspec_glob` and writes one 0-row attr parquet per
input shard. Consolidate's 1:1 attr-file invariant (see
``consolidate._attribute_paths_for_filter``) requires the files to exist
even though no row inside them will ever match a document id.
"""

from __future__ import annotations

import logging
import os

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem import url_to_fs

from marin.datakit.normalize import NormalizedData
from marin.execution.artifact import Artifact
from marin.execution.step_spec import StepSpec
from marin.processing.classification.deduplication.fuzzy_dups import (
    FuzzyDupsAttrData,
    FuzzyDupsPerSource,
)
from marin.processing.classification.deduplication.fuzzy_minhash import MinHashParams
from marin.utils import fsspec_glob, fsspec_mkdirs

logger = logging.getLogger(__name__)


# Schema consolidate's `load_parquet(columns=["id", "attributes"])` will read.
# Only the top-level column names strictly matter in the 0-row case — the
# filter combiner short-circuits on `right is None` before touching the
# `attributes` struct — but we declare the real struct fields so the file is
# also interpretable by any downstream consumer that assumes fuzzy_dups shape.
_NOOP_ATTR_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field(
            "attributes",
            pa.struct(
                [
                    pa.field("is_cluster_canonical", pa.bool_()),
                    pa.field("dup_cluster_id", pa.string()),
                ]
            ),
        ),
    ]
)

# Sentinel MinHashParams. The noop artifact carries no real MinHash state, but
# ``FuzzyDupsAttrData`` requires a params value. Zero-filled sentinel is safe
# because the noop artifact is consumed directly by consolidate — never by
# another dedup step that would cross-validate params.
_NOOP_PARAMS = MinHashParams(num_perms=0, num_bands=0, ngram_size=0, seed=0)


def _write_empty_attr_parquet(path: str) -> None:
    """Write a 0-row parquet with the attr schema at *path* (local or GCS)."""
    fs, fs_path = url_to_fs(path)
    parent = os.path.dirname(fs_path)
    if parent:
        fsspec_mkdirs(parent, exist_ok=True)
    with fs.open(fs_path, "wb") as f:
        pq.write_table(_NOOP_ATTR_SCHEMA.empty_table(), f)


def compute_noop_dedup_attrs(
    *,
    inputs: list[NormalizedData],
    output_path: str,
) -> FuzzyDupsAttrData:
    """Emit empty-attr ``FuzzyDupsAttrData`` for *inputs* without reading them.

    Each source gets a ``source_NNN`` attr directory under ``output_path``
    mirroring the source's normalized shard layout — one 0-row parquet per
    input parquet.

    Args:
        inputs: Normalized source artifacts. Ordered deterministically by
            ``main_output_dir`` so ``source_NNN`` tags are stable across runs.
        output_path: Root under which per-source attr trees are written.

    Raises:
        ValueError: If ``inputs`` is empty.
        FileNotFoundError: If any source's ``main_output_dir`` has no shards.
    """
    if not inputs:
        raise ValueError("compute_noop_dedup_attrs requires at least one input")

    ordered = sorted(inputs, key=lambda d: d.main_output_dir)
    sources: dict[str, FuzzyDupsPerSource] = {}
    total_shards = 0

    for i, data in enumerate(ordered):
        tag = f"source_{i:03d}"
        attr_dir = f"{output_path}/outputs/{tag}"
        input_base = data.main_output_dir.rstrip("/")

        shards = sorted(fsspec_glob(f"{input_base}/**/*.parquet"))
        if not shards:
            raise FileNotFoundError(f"No parquet shards under {input_base}")

        for shard in shards:
            rel = os.path.relpath(shard, input_base)
            _write_empty_attr_parquet(f"{attr_dir}/{rel}")

        total_shards += len(shards)
        sources[data.main_output_dir] = FuzzyDupsPerSource(attr_dir=attr_dir)
        logger.info("noop-dedup: wrote %d empty attr shards under %s", len(shards), attr_dir)

    logger.info("noop-dedup: total %d empty attr shards across %d sources", total_shards, len(ordered))

    return FuzzyDupsAttrData(
        params=_NOOP_PARAMS,
        sources=sources,
        counters={"noop_dedup/empty_shards": total_shards},
    )


def compute_noop_dedup_attrs_step(
    *,
    name: str,
    normalized_steps: list[StepSpec],
    override_output_path: str | None = None,
) -> StepSpec:
    """Create a StepSpec that emits noop fuzzy-dup attrs for *normalized_steps*."""
    return StepSpec(
        name=name,
        deps=list(normalized_steps),
        fn=lambda output_path: compute_noop_dedup_attrs(
            inputs=[Artifact.load(s, NormalizedData) for s in normalized_steps],
            output_path=output_path,
        ),
        override_output_path=override_output_path,
    )
