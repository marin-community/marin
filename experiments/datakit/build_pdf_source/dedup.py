# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact-dedup the combined corpus into a datakit source.

The one stage here is existing datakit tooling: the standard normalize pass with
``DedupMode.EXACT``, wired over the combined corpus the same way the reference pipeline wires it
over its sources (``experiments/datakit/reference_pipeline.py``).

Extraction deliberately keeps byte-identical duplicates in place (the crawl holds ~9.8%
exact-duplicate PDFs, and each route writes one unsorted shard per fetched shard), so this pass is
the linear scan that splits them into ``outputs/dups``. It also applies the standard whitespace-run
capping the extractors skipped, and it is the pipeline's one shuffle: the global sort by
content-hash ``id`` into partitions sized by bytes (``target_partition_bytes``), which is what makes
the output the :class:`~marin.datakit.normalize.NormalizedData` source the reference pipeline's
cross-source stages consume like any other. Because it runs over the union, byte-identical text
recovered from PDFs on either side of the router collapses here.

Decontamination and fuzzy dedup are not run here. Both decide across every source and against the
eval sets -- the drop sets, the shared bloom, the minhash clusters -- and the reference pipeline
runs them over this source alongside the others.
"""

import pyarrow as pa
from fray.types import ResourceConfig
from marin.datakit.normalize import DedupMode, normalize_step
from marin.execution.step_spec import StepSpec

from experiments.datakit.build_pdf_source.common import CORPUS, MAIN_OUTPUT_SUBDIR

# Below the library default of 32 GB, which is sized for multi-TB sources; this corpus is far
# smaller.
_NORMALIZE_WORKER_RESOURCES = ResourceConfig(cpu=2, ram="16g", disk="10g")
_MAX_WORKERS = 32


def exact_dedup_step(combined: StepSpec, schema: pa.Schema) -> StepSpec:
    """Exact-dedup the combined corpus with the standard datakit normalize pass.

    ``group_by`` on ``id`` co-locates byte-identical text from both routes in one reducer, so the
    dedup is global across the union rather than per-route. ``id_field="source_id"`` round-trips
    the extraction record unchanged: normalize copies every column, re-derives ``id`` from the
    text (the same content hash extraction wrote, unless whitespace capping changed the text), and
    restores ``source_id`` as-is. The default ``id_field="id"`` would clobber ``source_id`` with
    the content hash instead.

    Dedup keeps the first record of each id group and sends the rest to ``outputs/dups``, so on
    the documents whose text both routes recovered identically, ``needs_ocr`` -- like
    ``source_id`` and ``url`` -- names one of the PDFs that produced the text rather than all of
    them. The others are in the dups output, not discarded.
    """
    return normalize_step(
        name=f"data/datakit/normalize/{CORPUS}",
        download=combined,
        relative_input_path=MAIN_OUTPUT_SUBDIR,
        text_field="text",
        id_field="source_id",
        file_extensions=(".parquet",),
        dedup_mode=DedupMode.EXACT,
        output_schema=schema,
        worker_resources=_NORMALIZE_WORKER_RESOURCES,
        max_workers=_MAX_WORKERS,
    )
