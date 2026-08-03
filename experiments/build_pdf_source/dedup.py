# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deduplicate and decontaminate the extracted documents (#7620).

Every stage here is existing datakit tooling, wired over the two extraction routes the same way
the reference pipeline wires it over its sources (``experiments/datakit/reference_pipeline.py``,
the testbed dedup variants, and the testbed decon arm). Nothing is built new; this module only
chooses inputs, names, and sizes.

Per route (text and OCR are kept as two sources end to end, like reference-pipeline sources,
because their schemas differ by the OCR diagnostics columns):

1. **Exact dedup** -- the standard datakit normalize pass with ``DedupMode.EXACT``. Extraction
   deliberately kept byte-identical duplicates in its main output (see ``extract._keep_all``);
   this pass is the linear scan that docstring promised, splitting them into ``outputs/dups``.
   It also applies the standard whitespace-run capping the extractors skipped. It must run
   *before* the attribute joins below: ``consolidate`` joins attributes by ``id``, and a corpus
   whose shards still contain repeated ids would drop every copy of a marked id, canonical
   included.
2. **Fuzzy dedup** -- MinHash per route, connected components across both routes, then
   ``consolidate`` keeps one canonical member per duplicate cluster. Cross-route near-dups
   (the same document OCR'd once and text-extracted once elsewhere) collapse here.
3. **Decontamination** -- the shared eval bloom (same name and parameters as the reference
   pipeline, so an already-built bloom under this prefix is reused as-is), corpus-side
   drop sets, and a per-route mark; ``consolidate`` drops marked documents.

The cleaned dataset lands per route as standard :class:`~marin.datakit.normalize.NormalizedData`
(``outputs/main``; ``outputs/dups`` is part of the format and stays empty here -- the exact
duplicates live in step 1's dups output), ready for tokenization and the #7621 training run.

Prerequisite: the combined eval corpus must be staged at ``<MARIN_PREFIX>/datakit/decontam/evals``
(it is on ``s3://marin-us-east-02a/marin``, where this pipeline runs). The quality filter (#7619)
is not built yet; when it lands it slots between extraction and this module, and re-keying is
automatic through the step hashes.
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
from marin.processing.classification.deduplication.fuzzy_dups import (
    FuzzyDupsAttrData,
    compute_fuzzy_dups_attrs_step,
)
from marin.processing.classification.deduplication.fuzzy_minhash import compute_minhash_attrs_step
from rigging.filesystem import marin_prefix, prefix_join

from experiments.build_pdf_source import extract_ocr
from experiments.build_pdf_source.document_record import PDF_DOCUMENT_FIELDS
from experiments.datakit.decontam.config import (
    GLOBAL_DF_COMMON_MIN_ABS,
    GLOBAL_DF_COMMON_MIN_SOURCES,
    GLOBAL_DF_SAMPLE_DOCS,
    SOURCE_DF_COMMON_FRAC,
    SOURCE_DF_COMMON_MIN_ABS,
    SOURCE_DF_SAMPLE_DOCS,
)
from experiments.datakit.decontam.prepare_eval_corpus import DECON_EXCLUDED_EVAL_TASKS

logger = logging.getLogger(__name__)

_CORPUS = "common_crawl_focus_2026_22_pdf"

# Decontamination policy, verbatim from the reference pipeline. The bloom step keeps the
# reference's exact name and parameters on purpose: step identity is name + params, so the
# ~270 MB bloom already built under this prefix is a cache hit rather than a rebuild.
EVAL_ROOT = f"{marin_prefix()}/datakit/decontam/evals"
ESTIMATED_DOC_COUNT = 50_000_000
FALSE_POSITIVE_RATE = 1e-9
NGRAM_LENGTH = 13
OVERLAP_THRESHOLD = 0.5
FLAGGED_SAMPLE_SIZE = 8

# MinHash parameters are the compute_minhash_attrs_step defaults, which equal the reference
# pipeline's MinhashConfig; they are restated in that step's hash_attrs by the factory.
_FUZZY_CC_MAX_ITERATIONS = 10
_FUZZY_MAX_PARALLELISM = 64

_WORKER_RESOURCES = ResourceConfig(cpu=2, ram="8g")
# Normalize workers re-shuffle whole documents rather than attributes, so they get more RAM
# (the library default of 32 GB is sized for multi-TB sources; this corpus is far smaller).
_NORMALIZE_WORKER_RESOURCES = ResourceConfig(cpu=2, ram="16g", disk="10g")
_MAX_WORKERS = 64

_TEXT_ROUTE_SCHEMA = pa.schema(PDF_DOCUMENT_FIELDS)


def _normalize_route_step(route: str, extracted: StepSpec, schema: pa.Schema) -> StepSpec:
    """Exact-dedup one route's extraction output with the standard datakit normalize pass.

    ``id_field="source_id"`` round-trips the extraction record unchanged: normalize copies every
    column, re-derives ``id`` from the text (the same content hash extraction wrote, unless
    whitespace capping changed the text), and restores ``source_id`` as-is. The default
    ``id_field="id"`` would clobber ``source_id`` with the content hash instead.
    """
    return normalize_step(
        name=f"data/datakit/normalize/{_CORPUS}_{route}",
        download=extracted,
        relative_input_path="outputs/main",
        text_field="text",
        id_field="source_id",
        file_extensions=(".parquet",),
        dedup_mode=DedupMode.EXACT,
        output_schema=schema,
        worker_resources=_NORMALIZE_WORKER_RESOURCES,
        max_workers=_MAX_WORKERS,
    )


def consolidate_route(
    output_path: str,
    normalized_output_path: str,
    decontam_output_path: str,
    fuzzy_dups_output_path: str,
) -> NormalizedData:
    """Apply the decontamination and fuzzy-dedup attributes to one route's normalized output.

    Both filters follow the attribute producers' contracts: decon attributes are dense (one row
    per document, so a missing row means broken co-partitioning and the document is dropped),
    fuzzy-dup attributes are sparse (singleton clusters get no row and are kept).
    """
    normalized = read_artifact(normalized_output_path, NormalizedData)
    decontam = read_artifact(decontam_output_path, DeconAttributes)
    fuzzy_dups = read_artifact(fuzzy_dups_output_path, FuzzyDupsAttrData)
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
            ),
            FilterConfig(
                type=FilterType.KEEP_DOC,
                attribute_path=fuzzy_dups.attr_dir_for_source(normalized.main_output_dir),
                name="is_cluster_canonical",
                attribute_filetype="parquet",
                keep_if_missing=True,
            ),
        ],
        worker_resources=_WORKER_RESOURCES,
        max_workers=_MAX_WORKERS,
    )
    return NormalizedData(
        main_output_dir=prefix_join(output_path, "outputs/main"),
        dup_output_dir=prefix_join(output_path, "outputs/dups"),
        counters=dict(outcome.counters),
    )


def _clean_route_step(route: str, normalized: StepSpec, decontam: StepSpec, fuzzy_dups: StepSpec) -> StepSpec:
    return StepSpec(
        name=f"data/datakit/clean/{_CORPUS}_{route}",
        deps=[normalized, decontam, fuzzy_dups],
        hash_attrs={"filters": ("remove_contaminated", "keep_cluster_canonical"), "v": 1},
        fn=partial(
            consolidate_route,
            normalized_output_path=normalized.output_path,
            decontam_output_path=decontam.output_path,
            fuzzy_dups_output_path=fuzzy_dups.output_path,
        ),
    )


def dedup_steps(text_extraction: StepSpec, ocr_extraction: StepSpec) -> list[StepSpec]:
    """Build the dedup + decontamination DAG over the two extraction routes.

    Returns every step the runner needs, cleaned outputs last:
    ``data/datakit/clean/{corpus}_{text,ocr}`` are the datasets #7621 trains on.
    """
    normalized = {
        "text": _normalize_route_step("text", text_extraction, _TEXT_ROUTE_SCHEMA),
        "ocr": _normalize_route_step("ocr", ocr_extraction, extract_ocr._OUTPUT_SCHEMA),
    }

    minhash = {
        route: compute_minhash_attrs_step(
            name=f"data/datakit/minhash/{_CORPUS}_{route}",
            normalize=step,
            worker_resources=_WORKER_RESOURCES,
            max_workers=_MAX_WORKERS,
        )
        for route, step in normalized.items()
    }
    fuzzy_dups = compute_fuzzy_dups_attrs_step(
        name=f"data/datakit/fuzzy_dups/{_CORPUS}",
        minhash_steps=list(minhash.values()),
        cc_max_iterations=_FUZZY_CC_MAX_ITERATIONS,
        max_parallelism=_FUZZY_MAX_PARALLELISM,
        worker_resources=_WORKER_RESOURCES,
    )

    bloom = build_eval_bloom_step(
        name="datakit/bloom/_combined_fixed",
        eval_data_sources=[EVAL_ROOT],
        ngram_length=NGRAM_LENGTH,
        overlap_threshold=OVERLAP_THRESHOLD,
        estimated_doc_count=ESTIMATED_DOC_COUNT,
        false_positive_rate=FALSE_POSITIVE_RATE,
        exclude_eval_dirs=DECON_EXCLUDED_EVAL_TASKS,
    )
    # With two sources the global drop set is inert (GLOBAL_DF_COMMON_MIN_SOURCES=3 can never be
    # met); the source-local sets still catch this corpus's own high-frequency boilerplate. The
    # thresholds stay the shared policy values rather than being retuned for one pipeline.
    drop_sets = all_source_drop_sets_step(
        name=f"data/datakit/decon_drop/{_CORPUS}",
        sources=[
            DropSetSource(
                name=route,
                data_path=f"{step.output_path.rstrip('/')}/outputs/main",
                dependency=step,
            )
            for route, step in normalized.items()
        ],
        prebuilt_bloom=bloom,
        ngram_length=NGRAM_LENGTH,
        sample_docs=SOURCE_DF_SAMPLE_DOCS,
        common_frac=SOURCE_DF_COMMON_FRAC,
        common_min_abs=SOURCE_DF_COMMON_MIN_ABS,
        global_sample_docs=GLOBAL_DF_SAMPLE_DOCS,
        global_common_min_abs=GLOBAL_DF_COMMON_MIN_ABS,
        global_common_min_sources=GLOBAL_DF_COMMON_MIN_SOURCES,
        worker_resources=_WORKER_RESOURCES,
        max_workers=_MAX_WORKERS,
    )
    decontam = {
        route: decon_step(
            name=f"data/datakit/decontam/{_CORPUS}_{route}",
            normalized=step,
            prebuilt_bloom=bloom,
            drop_sets=drop_sets,
            drop_set_source=route,
            ngram_length=NGRAM_LENGTH,
            overlap_threshold=OVERLAP_THRESHOLD,
            flagged_sample_size=FLAGGED_SAMPLE_SIZE,
            worker_resources=_WORKER_RESOURCES,
            max_workers=_MAX_WORKERS,
        )
        for route, step in normalized.items()
    }

    clean = [_clean_route_step(route, normalized[route], decontam[route], fuzzy_dups) for route in normalized]
    return [*normalized.values(), *minhash.values(), fuzzy_dups, bloom, drop_sets, *decontam.values(), *clean]
