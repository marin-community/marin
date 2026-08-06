# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Join the two docling passes over the 10% sample into one corpus, then dedup and decontaminate it.

Docling converted the whole classified sample, but in two runs, because the router had already
split it: :func:`~experiments.build_pdf_source.extract_fleet.fleet_extract_step` converted the
text-extractable route as a pipeline step, and
:func:`~experiments.build_pdf_source.extract_fleet.fleet_backfill_step` converted the OCR route
afterwards as a one-off. Both used the same converter, the same options and the same boilerplate
pass, so the split is bookkeeping rather than a difference in the data. This module puts the two
back together and takes the result through the rest of #7620's chain, so that the docling corpus
and :mod:`~experiments.build_pdf_source.repair_ocr_all`'s OCR corpus are the same 10% sample
processed two ways, matched document for document on ``source_id``.

The union is a real step rather than a pair of paths handed to normalize because the two runs are
separate datasets: exact dedup has to see them together (the same text can be recovered from
different PDFs on either side of the router, and two copies of it are one document), and the router
decision that split them is worth keeping. The combine writes it as ``needs_ocr``, the classifier's
own column name -- after the union nothing else records which pass a document came from, and it is
the axis the two corpora exist to be compared along.

Nothing is re-extracted and nothing is re-post-processed. Running headers, footers and page numbers
were stripped by :mod:`~experiments.build_pdf_source.boilerplate` inside both conversion runs, and
a second pass is not idempotent -- the page-separator newline is itself a repeated edge pattern.
Loop repair, which the OCR corpus needed, has nothing to do here: repetition loops are a decoding
failure of an autoregressive transcriber, and docling is a parser.

From there the chain is the OCR corpus's, for the same reasons: exact dedup, then decontamination
against the shared eval bloom, then a consolidate that drops every marked document. Fuzzy dedup is
left out because it has to elect one canonical member per near-duplicate cluster and the quality
signal that election should use (#7619) does not exist yet. Every decontamination parameter is
imported from :mod:`~experiments.build_pdf_source.dedup` rather than restated, so this corpus is
deconned under exactly the policy the production routes are and the ~270 MB bloom is a cache hit.

All pure CPU over 5.9 GB and 309,891 documents:

    uv run iris --cluster=marin job run --target-cluster cw-us-east-08a \\
        --job-name docling-all-combine \\
        --cpu 2 --memory 8GB --enable-extra-resources \\
        -- python -m experiments.build_pdf_source.combine_docling_all
"""

import logging
from collections.abc import Iterator
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
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.processing.classification.consolidate import FilterConfig, FilterType, consolidate
from rigging.filesystem import marin_prefix, prefix_join, url_to_fs
from rigging.log_setup import configure_logging
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.runners import SubprocessRunner

from experiments.build_pdf_source.dedup import (
    ESTIMATED_DOC_COUNT,
    EVAL_ROOT,
    FALSE_POSITIVE_RATE,
    FLAGGED_SAMPLE_SIZE,
    NGRAM_LENGTH,
    OVERLAP_THRESHOLD,
)
from experiments.build_pdf_source.extract_fleet import _FLEET_OUTPUT_SCHEMA
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

_COUNTER_PREFIX = "focus_crawl_pdf_docling_combine"

# The two docling passes, by resolved output path. Named literally rather than rebuilt through
# their step factories: reconstructing those specs would make this step depend on them, and any
# drift in their hash_attrs or in the fetch and classify steps beneath them would present a
# multi-day CPU conversion as a cache miss. These paths are content-addressed and complete.
_ROUTE_DIRS: tuple[tuple[bool, str], ...] = (
    (False, "data/datakit/extract/common_crawl_focus_2026_22_pdf_text_84cbb532"),
    (True, "data/datakit/extract/common_crawl_focus_2026_22_pdf_docling_ocr_route_98f8b74a"),
)

_CORPUS = "common_crawl_focus_2026_22_pdf_docling_all"
_COMBINED_NAME = f"data/datakit/extract/{_CORPUS}"
_NORMALIZE_NAME = f"data/datakit/normalize/{_CORPUS}"
# The drop-set subdir this corpus's source-local common-ngram filter lands under.
_DROP_SET_SOURCE = "docling_all"

# The router decision that split the corpus in two, under the classifier's own column name.
_COMBINED_SCHEMA = pa.schema([*_FLEET_OUTPUT_SCHEMA, pa.field("needs_ocr", pa.bool_(), nullable=False)])

# The column the reader injects so a batch knows which pass it came from. Dropped before writing.
_SOURCE_FILE_COLUMN = "_source_file"

_DRIVER_RESOURCES = ResourceConfig(cpu=2, ram="16g", disk="8g")
# One task per input shard, each a ~2 MB parquet file whose records it holds as dicts while tagging.
_MAP_TASK_RESOURCES = ResourceConfig(cpu=1, ram="4g", disk="4g")
_TASKS_PER_WORKER = 8
_WORKER_RESOURCES = ResourceConfig(cpu=_TASKS_PER_WORKER, ram="32g", disk="16g")
_MAX_WORKERS = 32
_NORMALIZE_WORKER_RESOURCES = ResourceConfig(cpu=2, ram="16g", disk="10g")
# Decon and consolidate stream documents past an attribute join rather than reshuffling them, so
# they get the same modest shape the production routes' decon steps use.
_DECON_WORKER_RESOURCES = ResourceConfig(cpu=2, ram="8g")
# Not Zephyr's 1 GB default, at any stage. The OCR campaign was OOM-killed (exit 137) on that
# default twice, at 1,771 of 1,773 tasks and again at 22 of 23 -- two orders of magnitude apart,
# same failure, both after the stage's work was already on disk. Normalize gets the most because
# its ``group_by`` holds shuffle metadata on top of per-task state.
_COORDINATOR_RESOURCES = ResourceConfig(cpu=1, ram="16g", preemptible=False)
_NORMALIZE_COORDINATOR_RESOURCES = ResourceConfig(cpu=1, ram="32g", preemptible=False)
_DECON_COORDINATOR_RESOURCES = ResourceConfig(cpu=1, ram="8g", preemptible=False)
# Exact dedup collapses the corpus to ~23 shards, so every stage downstream of it has ~23 tasks.
# Asking for the 32 workers the sharded stages need would queue for capacity the stage cannot use:
# the OCR run waited over seven hours for workers it then finished with in under three minutes.
_DECON_MAX_WORKERS = 8


def route_of(shard: str, route_dirs: tuple[tuple[bool, str], ...]) -> bool:
    """The router decision for the pass that wrote ``shard``, by the directory it lives under.

    A shard under neither pass raises: the driver listed these directories itself, so a path that
    does not belong to one is a bug in that listing rather than a document to guess about.
    """
    for needs_ocr, directory in route_dirs:
        if shard.startswith(directory):
            return needs_ocr
    raise ValueError(f"Shard {shard} belongs to neither docling pass: {[d for _, d in route_dirs]}")


def tag_batch(batch: pa.RecordBatch, route_dirs: tuple[tuple[bool, str], ...]) -> Iterator[dict]:
    """Emit one Parquet row group's records, tagged with the route the pass that wrote it converted.

    Every row of a batch comes from one file, so the route is resolved once and the injected path
    column is dropped -- it is scaffolding for this step, not part of the corpus.
    """
    if not batch.num_rows:
        return
    needs_ocr = route_of(batch.column(_SOURCE_FILE_COLUMN)[0].as_py(), route_dirs)
    tally = "from_ocr_route" if needs_ocr else "from_text_route"
    for row in batch.to_pylist():
        del row[_SOURCE_FILE_COLUMN]
        counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/documents_out", 1)
        counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/{tally}", 1)
        yield {**row, "needs_ocr": needs_ocr}


def combine_docling_all(output_path: str) -> NormalizedData:
    """Write both docling passes into one directory.

    A pure 1:1 shard map -- no shuffle. The global sort and the exact dedup both belong to the
    normalize step that consumes this, so doing either here would be paid for twice.
    """
    prefix = marin_prefix()
    shards: list[str] = []
    route_dirs: list[tuple[bool, str]] = []
    for needs_ocr, directory in _ROUTE_DIRS:
        main = prefix_join(prefix_join(prefix, directory), "outputs/main")
        filesystem, path = url_to_fs(main)
        found = sorted(filesystem.unstrip_protocol(shard) for shard in filesystem.glob(f"{path}/*.parquet"))
        if not found:
            raise RuntimeError(f"No extracted shards under {main}")
        # The resolved directory, taken from a shard rather than from ``main``, so the prefix the
        # tasks match against is the one the reader will hand them back.
        route_dirs.append((needs_ocr, found[0].rsplit("/", 1)[0] + "/"))
        shards.extend(found)
    logger.info("Combining %d shards across %d docling passes", len(shards), len(_ROUTE_DIRS))

    main_output_dir = prefix_join(output_path, "outputs/main")
    pipeline = (
        Dataset.from_list(shards)
        .load_parquet(batch_mode=True, include_file_paths=True, file_path_column=_SOURCE_FILE_COLUMN)
        .flat_map(partial(tag_batch, route_dirs=tuple(route_dirs)))
        .write_parquet(
            prefix_join(main_output_dir, "part-{shard:05d}-of-{total:05d}.parquet"),
            schema=_COMBINED_SCHEMA,
            skip_existing=True,
        )
    )
    with ZephyrContext(
        name="focus-crawl-pdf-docling-combine",
        resources=_WORKER_RESOURCES,
        max_workers=_MAX_WORKERS,
        stage_runner_factory=SubprocessRunner,
        coordinator_resources=_COORDINATOR_RESOURCES,
    ) as pool:
        outcome = pool.execute(pipeline, map_task_resources=_MAP_TASK_RESOURCES)

    return NormalizedData(
        main_output_dir=main_output_dir,
        dup_output_dir=prefix_join(output_path, "outputs/dups"),
        counters=dict(outcome.counters),
    )


def combine_step() -> StepSpec:
    """Union the two docling passes into one corpus, tagged with the route each came from."""
    return StepSpec(
        name=_COMBINED_NAME,
        deps=[],
        hash_attrs={"passes": tuple(directory for _, directory in _ROUTE_DIRS), "route_column": "needs_ocr"},
        fn=remote(combine_docling_all, resources=_DRIVER_RESOURCES, pip_dependency_groups=["datakit"]),
    )


def normalize_combined_step(combined: StepSpec) -> StepSpec:
    """Exact-dedup the combined corpus.

    ``group_by`` on ``id`` co-locates byte-identical text from both passes in one reducer, so the
    dedup is global across the union rather than per-pass. ``id_field="source_id"`` round-trips the
    extraction record: normalize re-derives ``id`` from the text and keeps ``source_id`` as the
    crawl identity, as the text route's normalize does.

    Dedup keeps the first record of each id group and sends the rest to ``outputs/dups``, so on the
    documents whose text both passes recovered identically, ``needs_ocr`` -- like ``source_id`` and
    ``url`` -- names one of the PDFs that produced the text rather than all of them. The others are
    in the dups output, not discarded.
    """
    return normalize_step(
        name=_NORMALIZE_NAME,
        download=combined,
        relative_input_path="outputs/main",
        text_field="text",
        id_field="source_id",
        file_extensions=(".parquet",),
        dedup_mode=DedupMode.EXACT,
        output_schema=_COMBINED_SCHEMA,
        worker_resources=_NORMALIZE_WORKER_RESOURCES,
        max_workers=_MAX_WORKERS,
        coordinator_resources=_NORMALIZE_COORDINATOR_RESOURCES,
    )


def decontaminate_steps(normalized: StepSpec) -> tuple[StepSpec, StepSpec, StepSpec]:
    """Build the eval bloom, this corpus's drop sets, and its decontamination attributes.

    The bloom keeps :mod:`~experiments.build_pdf_source.dedup`'s exact name and parameters, so the
    one already built under this prefix is a cache hit. With a single source the cross-source drop
    set is inert -- ``GLOBAL_DF_COMMON_MIN_SOURCES`` of 3 can never be met -- but the source-local
    set still catches this corpus's own high-frequency boilerplate, which is what keeps common PDF
    chrome from reading as eval overlap.
    """
    bloom = build_eval_bloom_step(
        name="datakit/bloom/_combined_fixed",
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


def clean_step(normalized: StepSpec, decontam: StepSpec) -> StepSpec:
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


def combine_docling_all_steps() -> list[StepSpec]:
    """Every step from the two stored docling passes to the cleaned dataset."""
    combined = combine_step()
    normalized = normalize_combined_step(combined)
    bloom, drop_sets, decontam = decontaminate_steps(normalized)
    return [combined, normalized, bloom, drop_sets, decontam, clean_step(normalized, decontam)]


def main() -> None:
    configure_logging(logging.INFO)
    StepRunner().run(combine_docling_all_steps())


if __name__ == "__main__":
    main()
