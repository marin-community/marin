# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bring the all-routes OCR corpus up to the current post-processing, then exact-dedup it.

The all-routes corpus (:mod:`~experiments.build_pdf_source.extract_ocr_all`) was produced on
2026-08-03 at ``schema_version`` 1, before loop repair existed. Everything else in its
post-processing is already current: :mod:`~experiments.build_pdf_source.boilerplate` has not
changed since the commit that introduced it, and the run's recorded ``boilerplate_*`` settings are
the defaults the pipeline still uses, so running headers, footers and page numbers were stripped by
exactly the code that would strip them today. Loop repair is the whole delta.

**The stored corpus is post-boilerplate, and that is irreversible.** The ``raw`` directory beside
``outputs/main`` is raw only in the sense of *pre-shuffle*: phase 1 of ``ocr_pdf_text`` writes fully
built records there and phase 2 only sorts them by ``id``. Both directories hold the same
already-stripped, already-page-joined text, and neither keeps the model's original response. So
re-post-processing here means applying loop repair to what survives, not rebuilding from the model
output.

Pages are recovered exactly: ``record`` gives every page a trailing newline and sets
``page_offsets`` to the cumulative lengths of those pages, so slicing ``text`` at the offsets
returns the stored pages byte for byte.

**Loop repair runs without its per-page truncation gate**, because the flag is gone -- the corpus
keeps only a per-document ``pages_truncated`` count, and reconstructing which pages it referred to
from page lengths is unreliable (a token-budget check reproduced the recorded count on 10% of
truncated documents, chars-per-token spanning 2.3 to 4.8 across the corpus). Dropping the gate
costs almost nothing here. Measured over 71,845 pages of the corpus, the detector fires on 2 of
51,963 pages that are *known* non-truncated -- every page of a document whose ``pages_truncated``
is 0 -- destroying 0.003% of characters, against the 2.89% it removes overall, which matches the
~3% the gated detector was calibrated to find. The size and counter-score thresholds in
:mod:`~experiments.build_pdf_source.loop_repair`, not the truncation flag, are what hold precision
on this corpus.

Boilerplate is deliberately **not** re-run after repair. A second pass is not idempotent: the
page-separator newline that ``record`` appends is itself a repeated edge pattern, so it would strip
blank lines from ~49% of documents it has no business touching.

After repair the corpus goes through exact dedup and decontamination, which is #7620's chain minus
its fuzzy stage. Fuzzy dedup is deliberately left out: it has to elect one canonical member per
near-duplicate cluster, and the quality signal that election should use (#7619) does not exist yet,
so running it now would freeze an arbitrary winner. Decontamination has no such problem --
``contaminated`` is a per-document predicate over paragraph overlap against the eval bloom, so it
drops every marked document and nothing depends on the order they are visited or on which copy
survived an earlier stage.

Every decontamination parameter is imported from :mod:`~experiments.build_pdf_source.dedup` rather
than restated, so this corpus is deconned under exactly the policy the two production routes are,
and the ~270 MB shared eval bloom is a cache hit rather than a rebuild.

All pure CPU over 6.2 GB and 315,276 documents -- no GPUs, nothing re-OCR'd:

    uv run iris --cluster=marin job run --target-cluster cw-us-east-08a \\
        --job-name ocr-all-repair \\
        --cpu 2 --memory 8GB --enable-extra-resources \\
        -- python -m experiments.build_pdf_source.repair_ocr_all
"""

import logging
from collections.abc import Iterator
from functools import partial
from itertools import accumulate

import pyarrow as pa
from fray.types import ResourceConfig
from marin.datakit.decon import (
    DeconAttributes,
    DropSetSource,
    all_source_drop_sets_step,
    build_eval_bloom_step,
    decon_step,
)
from marin.datakit.normalize import DedupMode, NormalizedData, generate_id, normalize_step
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
from experiments.build_pdf_source.extract_ocr import _OUTPUT_SCHEMA, LOOP_OPTIONS, OcrStatus
from experiments.build_pdf_source.loop_repair import LoopOptions, repair_page
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

_COUNTER_PREFIX = "focus_crawl_pdf_ocr_repair"

# The eleven partitions of the all-routes run, by resolved output path. Named literally rather than
# rebuilt through ``ocr_all_partition_step``: reconstructing those specs would make this step depend
# on them, and any drift in their hash_attrs or in the fetch step beneath them would present an
# 8-node-hour GPU re-run as a cache miss. These paths are content-addressed and complete.
_PARTITION_DIRS: tuple[str, ...] = (
    "data/datakit/extract/common_crawl_focus_2026_22_pdf_ocr_all_p00_d1ba3da0",
    "data/datakit/extract/common_crawl_focus_2026_22_pdf_ocr_all_p01_3f32dcf6",
    "data/datakit/extract/common_crawl_focus_2026_22_pdf_ocr_all_p02_ed059be5",
    "data/datakit/extract/common_crawl_focus_2026_22_pdf_ocr_all_p03_48527e3d",
    "data/datakit/extract/common_crawl_focus_2026_22_pdf_ocr_all_p04_ead6340f",
    "data/datakit/extract/common_crawl_focus_2026_22_pdf_ocr_all_p05_d6b7ef12",
    "data/datakit/extract/common_crawl_focus_2026_22_pdf_ocr_all_p06_f7ad19ec",
    "data/datakit/extract/common_crawl_focus_2026_22_pdf_ocr_all_p07_c32dbc34",
    "data/datakit/extract/common_crawl_focus_2026_22_pdf_ocr_all_p08_e0c436f3",
    "data/datakit/extract/common_crawl_focus_2026_22_pdf_ocr_all_p09_5b5f2a92",
    "data/datakit/extract/common_crawl_focus_2026_22_pdf_ocr_all_p10_baa57ebd",
)

_CORPUS = "common_crawl_focus_2026_22_pdf_ocr_all"
_REPAIR_NAME = f"data/datakit/extract/{_CORPUS}_repaired"
_NORMALIZE_NAME = f"data/datakit/normalize/{_CORPUS}"
# The drop-set subdir this corpus's source-local common-ngram filter lands under.
_DROP_SET_SOURCE = "ocr_all"

_DRIVER_RESOURCES = ResourceConfig(cpu=2, ram="16g", disk="8g")
# One task per input shard, each a ~3.5 MB parquet file whose text it holds twice while repairing.
_MAP_TASK_RESOURCES = ResourceConfig(cpu=1, ram="4g", disk="4g")
_TASKS_PER_WORKER = 8
_WORKER_RESOURCES = ResourceConfig(cpu=_TASKS_PER_WORKER, ram="32g", disk="16g")
_MAX_WORKERS = 32
_NORMALIZE_WORKER_RESOURCES = ResourceConfig(cpu=2, ram="16g", disk="10g")
# Decon and consolidate stream documents past an attribute join rather than reshuffling them, so
# they get the same modest shape the production routes' decon steps use.
_DECON_WORKER_RESOURCES = ResourceConfig(cpu=2, ram="8g")
# Not Zephyr's 1 GB default, at any stage. That default was OOM-killed (exit 137) twice here, and
# shard count is not what drives it: the repair stage died at 1,771 of 1,773 tasks complete, and
# the decontamination stage died at 22 of 23 -- two orders of magnitude apart, same failure, both
# after their work was already on disk. ``extract_ocr`` reached the same conclusion at ~380 tasks
# and set 8 GB. Normalize gets the most because its ``group_by`` holds shuffle metadata on top of
# per-task state.
_COORDINATOR_RESOURCES = ResourceConfig(cpu=1, ram="16g", preemptible=False)
_NORMALIZE_COORDINATOR_RESOURCES = ResourceConfig(cpu=1, ram="32g", preemptible=False)
_DECON_COORDINATOR_RESOURCES = ResourceConfig(cpu=1, ram="8g", preemptible=False)
# Exact dedup collapses the corpus to ~23 shards, so every stage downstream of it has ~23 tasks.
# Asking for the 32 workers the 1,773-shard stages need would queue for capacity the stage cannot
# use: the first run waited over seven hours between the drop sets and decontamination for workers
# it then finished with in under three minutes.
_DECON_MAX_WORKERS = 8


def split_pages(text: str, page_offsets: list[int]) -> list[str]:
    """Recover the stored pages from a document's text.

    ``page_offsets`` is the cumulative character count of each page, newline included, so the
    offsets partition ``text`` exactly.
    """
    pages = []
    start = 0
    for offset in page_offsets:
        pages.append(text[start:offset])
        start = offset
    return pages


def repair_document(row: dict, loop: LoopOptions) -> dict | None:
    """Apply loop repair to one stored document, or drop it if nothing survives.

    Every page is examined. The per-page truncation flag the pipeline gates on was never stored,
    and on this corpus the gate is not what holds precision -- see the module docstring.
    """
    pages = split_pages(row["text"], row["page_offsets"])
    repairs = [repair_page(page, True, loop) for page in pages]
    if not any(repair.looped for repair in repairs):
        return row

    looped_pages = [index for index, repair in enumerate(repairs, start=1) if repair.looped]
    dropped = sum(repair.dropped_chars for repair in repairs)
    # Restore the trailing newline the page separator relies on, exactly as ``record`` does: salvage
    # right-strips, and without this the last line of a repaired page fuses with the next page.
    kept = [page if not page or page.endswith("\n") else page + "\n" for page in (r.text for r in repairs)]
    text = "".join(kept)

    counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/documents_repaired", 1)
    counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/pages_looped", len(looped_pages))
    counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/loop_chars_dropped", dropped)
    if not text.strip():
        counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/emptied_by_repair", 1)
        return None

    error = row["extraction_error"]
    clause = (
        f"{len(looped_pages)} of {row['num_pages']} pages repeated themselves and were cut back, "
        f"dropping {dropped} characters"
    )
    return {
        **row,
        "id": generate_id(text),
        "text": text,
        "page_offsets": list(accumulate(len(page) for page in kept)),
        "extraction_status": str(OcrStatus.PARTIAL),
        "extraction_error": f"{error}; {clause}" if error else clause,
        "looped_pages": looped_pages,
        "loop_chars_dropped": dropped,
    }


def repair_batch(batch: pa.RecordBatch, loop: LoopOptions) -> Iterator[dict]:
    """Repair one Parquet row group, adding the schema-2 loop columns to every record."""
    for row in batch.to_pylist():
        row.setdefault("looped_pages", [])
        row.setdefault("loop_chars_dropped", 0)
        repaired = repair_document(row, loop)
        if repaired is not None:
            counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/documents_out", 1)
            yield repaired


def repair_ocr_all(output_path: str) -> NormalizedData:
    """Apply loop repair across every partition of the all-routes corpus.

    A pure 1:1 shard map -- no shuffle. The global sort and the exact dedup both belong to the
    normalize step that consumes this, so doing either here would be paid for twice.
    """
    prefix = marin_prefix()
    shards: list[str] = []
    for partition in _PARTITION_DIRS:
        main = prefix_join(prefix_join(prefix, partition), "outputs/main")
        filesystem, path = url_to_fs(main)
        found = sorted(filesystem.unstrip_protocol(shard) for shard in filesystem.glob(f"{path}/*.parquet"))
        if not found:
            raise RuntimeError(f"No extracted shards under {main}")
        shards.extend(found)
    logger.info("Repairing %d shards across %d partitions", len(shards), len(_PARTITION_DIRS))

    main_output_dir = prefix_join(output_path, "outputs/main")
    pipeline = (
        Dataset.from_list(shards)
        .load_parquet(batch_mode=True)
        .flat_map(partial(repair_batch, loop=LOOP_OPTIONS))
        .write_parquet(
            prefix_join(main_output_dir, "part-{shard:05d}-of-{total:05d}.parquet"),
            schema=_OUTPUT_SCHEMA,
            skip_existing=True,
        )
    )
    with ZephyrContext(
        name="focus-crawl-pdf-ocr-repair",
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


def repair_step() -> StepSpec:
    """Loop-repair the all-routes corpus, bringing it to the current extraction schema."""
    return StepSpec(
        name=_REPAIR_NAME,
        deps=[],
        hash_attrs={
            "partitions": _PARTITION_DIRS,
            "loop_min_page_chars": LOOP_OPTIONS.min_page_chars,
            "loop_min_loop_chars": LOOP_OPTIONS.min_loop_chars,
            "loop_min_loop_fraction": LOOP_OPTIONS.min_loop_fraction,
            "loop_max_trailing_chars": LOOP_OPTIONS.max_trailing_chars,
            "loop_min_counter_score": LOOP_OPTIONS.min_counter_score,
            "loop_min_salvage_prefix": LOOP_OPTIONS.min_salvage_prefix,
            # The truncation gate repair_page applies in the pipeline is off here; the flag it reads
            # was never stored. Part of the identity of this output, not an implementation detail.
            "truncation_gated": False,
            "schema_version": 2,
        },
        fn=remote(repair_ocr_all, resources=_DRIVER_RESOURCES, pip_dependency_groups=["datakit"]),
    )


def normalize_repaired_step(repaired: StepSpec) -> StepSpec:
    """Exact-dedup the repaired corpus.

    ``group_by`` on ``id`` co-locates byte-identical text from every partition in one reducer, so
    the dedup is global across all eleven rather than per-partition. ``id_field="source_id"``
    round-trips the extraction record: normalize re-derives ``id`` from the text and keeps
    ``source_id`` as the crawl identity, as the text route's normalize does.
    """
    return normalize_step(
        name=_NORMALIZE_NAME,
        download=repaired,
        relative_input_path="outputs/main",
        text_field="text",
        id_field="source_id",
        file_extensions=(".parquet",),
        dedup_mode=DedupMode.EXACT,
        output_schema=_OUTPUT_SCHEMA,
        worker_resources=_NORMALIZE_WORKER_RESOURCES,
        max_workers=_MAX_WORKERS,
        coordinator_resources=_NORMALIZE_COORDINATOR_RESOURCES,
    )


def decontaminate_steps(normalized: StepSpec) -> tuple[StepSpec, StepSpec, StepSpec]:
    """Build the eval bloom, this corpus's drop sets, and its decontamination attributes.

    The bloom keeps :mod:`~experiments.build_pdf_source.dedup`'s exact name and parameters, so the
    one already built under this prefix is a cache hit. With a single source the cross-source drop
    set is inert -- ``GLOBAL_DF_COMMON_MIN_SOURCES`` of 3 can never be met -- but the source-local
    set still catches this corpus's own high-frequency boilerplate, which is what keeps common
    PDF chrome from reading as eval overlap.
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
    """The dataset a downstream consumer reads: repaired, exact-deduplicated, decontaminated."""
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


def repair_ocr_all_steps() -> list[StepSpec]:
    """Every step from the stored all-routes corpus to the cleaned dataset."""
    repaired = repair_step()
    normalized = normalize_repaired_step(repaired)
    bloom, drop_sets, decontam = decontaminate_steps(normalized)
    return [repaired, normalized, bloom, drop_sets, decontam, clean_step(normalized, decontam)]


def main() -> None:
    configure_logging(logging.INFO)
    StepRunner().run(repair_ocr_all_steps())


if __name__ == "__main__":
    main()
