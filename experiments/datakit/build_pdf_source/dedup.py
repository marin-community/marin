# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact-dedup and decontaminate the combined corpus (#7620).

Every stage here is existing datakit tooling, wired over the combined corpus the same way the
reference pipeline wires it over its sources (``experiments/datakit/reference_pipeline.py``, the
testbed dedup variants, and the testbed decon arm). Nothing is built new; this module only chooses
inputs, names, and sizes.

1. **Exact dedup** -- the standard datakit normalize pass with ``DedupMode.EXACT``. Extraction
   deliberately kept byte-identical duplicates in its main output (see ``extract.keep_all``);
   this pass is the linear scan that docstring promised, splitting them into ``outputs/dups``.
   It also applies the standard whitespace-run capping the extractors skipped. It must run
   *before* the attribute joins below: ``consolidate`` joins attributes by ``id``, and a corpus
   whose shards still contain repeated ids would drop every copy of a marked id, canonical
   included. Because it runs over the union, byte-identical text recovered from PDFs on either
   side of the router collapses here.
2. **Decontamination** -- the shared eval bloom (same name and parameters as the reference
   pipeline, so an already-built bloom under this prefix is reused as-is), corpus-side drop
   sets, and a mark; ``consolidate`` drops marked documents.

Fuzzy dedup is deliberately deferred: it has to elect one canonical member per near-duplicate
cluster, and that election should use the quality signal (#7619), which does not exist until the
quality step has run. It lands in :mod:`~experiments.datakit.build_pdf_source.fuzzy_dedup`, after
:mod:`~experiments.datakit.build_pdf_source.quality_label`. Decontamination has no such problem --
``contaminated`` is a per-document predicate over paragraph overlap against the eval bloom, so it
drops every marked document and nothing depends on the order they are visited or on which copy
survived an earlier stage.

The cleaned dataset lands as standard :class:`~marin.datakit.normalize.NormalizedData`
(``outputs/main``; ``outputs/dups`` is part of the format and stays empty here -- the exact
duplicates live in step 1's dups output).

Prerequisite: the combined eval corpus must be staged at ``<MARIN_PREFIX>/datakit/decontam/evals``
(it is on ``s3://marin-us-east-02a/marin``, where this pipeline runs).
"""

import logging
from functools import partial

import pyarrow as pa
from fray.types import ResourceConfig
from marin.datakit.decon import (
    DeconAttributes,
    DropSetSource,
    all_source_drop_sets_step,
    build_eval_bloom_step,
    decon_step,
)
from marin.datakit.normalize import DedupMode, NormalizedData, normalize_step
from marin.execution.artifact import read_artifact
from marin.execution.step_spec import StepSpec
from marin.processing.classification.consolidate import FilterConfig, FilterType, consolidate
from rigging.filesystem import prefix_join

from experiments.datakit.decontam.config import (
    BLOOM_STEP_NAME,
    ESTIMATED_DOC_COUNT,
    EVAL_ROOT,
    FALSE_POSITIVE_RATE,
    FLAGGED_SAMPLE_SIZE,
    GLOBAL_DF_COMMON_MIN_ABS,
    GLOBAL_DF_COMMON_MIN_SOURCES,
    GLOBAL_DF_SAMPLE_DOCS,
    NGRAM_LENGTH,
    OVERLAP_THRESHOLD,
    SOURCE_DF_COMMON_FRAC,
    SOURCE_DF_COMMON_MIN_ABS,
    SOURCE_DF_SAMPLE_DOCS,
)
from experiments.datakit.decontam.prepare_eval_corpus import DECON_EXCLUDED_EVAL_TASKS

logger = logging.getLogger(__name__)

_CORPUS = "common_crawl_focus_2026_22_pdf"
# The drop-set subdir this corpus's source-local common-ngram filter lands under.
_DROP_SET_SOURCE = "combined"

# Decontamination policy comes from the shared decontam config: the bloom step keeps the
# reference pipeline's exact name and parameters on purpose, so the ~270 MB bloom already built
# under this prefix is a cache hit rather than a rebuild.

# Normalize workers re-shuffle whole documents rather than attributes, so they get more RAM
# (the library default of 32 GB is sized for multi-TB sources; this corpus is far smaller).
_NORMALIZE_WORKER_RESOURCES = ResourceConfig(cpu=2, ram="16g", disk="10g")
_MAX_WORKERS = 32
# Decon and consolidate stream documents past an attribute join rather than reshuffling them, so
# they get a modest shape.
_DECON_WORKER_RESOURCES = ResourceConfig(cpu=2, ram="8g")
# Not Zephyr's 1 GB default, at any stage: that default is OOM-killed (exit 137) at run end
# across this pipeline family, after the stage's work is already on disk. Normalize gets the most
# because its ``group_by`` holds shuffle metadata on top of per-task state.
_NORMALIZE_COORDINATOR_RESOURCES = ResourceConfig(cpu=1, ram="32g", preemptible=False)
_DECON_COORDINATOR_RESOURCES = ResourceConfig(cpu=1, ram="8g", preemptible=False)
# Exact dedup collapses the corpus to ~23 shards, so every stage downstream of it has ~23 tasks.
# Asking for the 32 workers the sharded stages need would queue for capacity the stage cannot use:
# a prior run waited over seven hours for workers it then finished with in under three minutes.
_DECON_MAX_WORKERS = 8


def _normalize_combined_step(combined: StepSpec, schema: pa.Schema) -> StepSpec:
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
        name=f"data/datakit/normalize/{_CORPUS}",
        download=combined,
        relative_input_path="outputs/main",
        text_field="text",
        id_field="source_id",
        file_extensions=(".parquet",),
        dedup_mode=DedupMode.EXACT,
        output_schema=schema,
        worker_resources=_NORMALIZE_WORKER_RESOURCES,
        max_workers=_MAX_WORKERS,
        coordinator_resources=_NORMALIZE_COORDINATOR_RESOURCES,
    )


def _decontaminate_steps(normalized: StepSpec) -> tuple[StepSpec, StepSpec, StepSpec]:
    """Build the eval bloom, this corpus's drop sets, and its decontamination attributes.

    With a single source the cross-source drop set is inert -- ``GLOBAL_DF_COMMON_MIN_SOURCES``
    of 3 can never be met -- but the source-local set still catches this corpus's own
    high-frequency boilerplate, which is what keeps common PDF chrome from reading as eval
    overlap. The thresholds stay the shared policy values rather than being retuned for one
    pipeline.
    """
    bloom = build_eval_bloom_step(
        name=BLOOM_STEP_NAME,
        eval_data_sources=[EVAL_ROOT],
        ngram_length=NGRAM_LENGTH,
        overlap_threshold=OVERLAP_THRESHOLD,
        estimated_doc_count=ESTIMATED_DOC_COUNT,
        false_positive_rate=FALSE_POSITIVE_RATE,
        exclude_eval_dirs=DECON_EXCLUDED_EVAL_TASKS,
    )
    drop_sets = all_source_drop_sets_step(
        name=f"data/datakit/decon_drop/{_CORPUS}",
        sources=[
            DropSetSource(
                name=_DROP_SET_SOURCE,
                data_path=prefix_join(normalized.output_path, "outputs/main"),
                dependency=normalized,
            )
        ],
        prebuilt_bloom=bloom,
        ngram_length=NGRAM_LENGTH,
        sample_docs=SOURCE_DF_SAMPLE_DOCS,
        common_frac=SOURCE_DF_COMMON_FRAC,
        common_min_abs=SOURCE_DF_COMMON_MIN_ABS,
        global_sample_docs=GLOBAL_DF_SAMPLE_DOCS,
        global_common_min_abs=GLOBAL_DF_COMMON_MIN_ABS,
        global_common_min_sources=GLOBAL_DF_COMMON_MIN_SOURCES,
        worker_resources=_DECON_WORKER_RESOURCES,
        max_workers=_DECON_MAX_WORKERS,
        coordinator_resources=_DECON_COORDINATOR_RESOURCES,
    )
    decontam = decon_step(
        name=f"data/datakit/decontam/{_CORPUS}",
        normalized=normalized,
        prebuilt_bloom=bloom,
        drop_sets=drop_sets,
        drop_set_source=_DROP_SET_SOURCE,
        ngram_length=NGRAM_LENGTH,
        overlap_threshold=OVERLAP_THRESHOLD,
        flagged_sample_size=FLAGGED_SAMPLE_SIZE,
        worker_resources=_DECON_WORKER_RESOURCES,
        max_workers=_DECON_MAX_WORKERS,
        coordinator_resources=_DECON_COORDINATOR_RESOURCES,
    )
    return bloom, drop_sets, decontam


def consolidate_decontaminated(
    output_path: str, normalized_output_path: str, decontam_output_path: str
) -> NormalizedData:
    """Drop every contaminated document from the deduplicated corpus.

    Decon attributes are dense -- one row per document -- so a missing row means the attribute
    dataset is no longer co-partitioned with the corpus, and ``consolidate`` drops the document
    rather than guessing. That is the contract the producer promises, so no ``keep_if_missing``.
    """
    normalized = read_artifact(normalized_output_path, NormalizedData)
    decontam = read_artifact(decontam_output_path, DeconAttributes)
    outcome = consolidate(
        input_path=normalized.main_output_dir,
        output_path=prefix_join(output_path, "outputs/main"),
        filetype="parquet",
        filters=[
            FilterConfig(
                type=FilterType.REMOVE_DOC,
                attribute_path=decontam.main_output_dir,
                name="contaminated",
                attribute_filetype="parquet",
            )
        ],
        worker_resources=_DECON_WORKER_RESOURCES,
        max_workers=_DECON_MAX_WORKERS,
        coordinator_resources=_DECON_COORDINATOR_RESOURCES,
    )
    return NormalizedData(
        main_output_dir=prefix_join(output_path, "outputs/main"),
        dup_output_dir=prefix_join(output_path, "outputs/dups"),
        counters=dict(outcome.counters),
    )


def _clean_step(normalized: StepSpec, decontam: StepSpec) -> StepSpec:
    """The dataset a downstream consumer reads: combined, exact-deduplicated, decontaminated."""
    return StepSpec(
        name=f"data/datakit/clean/{_CORPUS}",
        deps=[normalized, decontam],
        hash_attrs={"filters": ("remove_contaminated",), "v": 1},
        fn=partial(
            consolidate_decontaminated,
            normalized_output_path=normalized.output_path,
            decontam_output_path=decontam.output_path,
        ),
    )


def dedup_steps(combined: StepSpec, schema: pa.Schema) -> list[StepSpec]:
    """Build the exact dedup + decontamination DAG over the combined corpus.

    Returns every step the runner needs, the cleaned output last:
    ``data/datakit/clean/common_crawl_focus_2026_22_pdf`` is the
    :class:`~marin.datakit.normalize.NormalizedData` the quality step reads.
    """
    normalized = _normalize_combined_step(combined, schema)
    bloom, drop_sets, decontam = _decontaminate_steps(normalized)
    return [normalized, bloom, drop_sets, decontam, _clean_step(normalized, decontam)]
