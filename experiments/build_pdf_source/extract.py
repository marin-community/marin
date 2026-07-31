# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 5: extract text from the PDFs the router sent to embedded-text extraction.

Only the text-extractable route runs here. In the 10% sample that is 214,444 of 315,776 classified
documents and 3.6M of 5.6M pages; the OCR route is a separate and much more expensive pipeline.

The routing decision is a join, and the cheap side is broadcast. The classify step wrote a narrow
table -- about 30 MB for the whole sample -- precisely so extraction could hold it in memory and
filter as it reads, rather than paying to have 411 GiB of PDF bytes shuffled into two datasets. The
driver reads that table once, keeps the ``(warc_filename, warc_record_offset)`` keys of the
text-extractable documents, and hands the set to every map task.

Cost is dominated by docling's layout model, one pass per page. Tasks are costed at one CPU and the
converter is built once per worker process, because building it compiles the layout graph and that
must not happen per document.
"""

import logging
from collections.abc import Iterator
from functools import cache, partial

import pyarrow as pa
from fray.types import ResourceConfig
from marin.datakit.normalize import (
    MainOutput,
    NormalizedData,
    generate_id,
    make_split_writer,
)
from marin.execution.artifact import read_artifact
from marin.execution.remote import remote
from marin.execution.step_spec import StepSpec
from rigging.filesystem import prefix_join, url_to_fs
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.runners import SubprocessRunner

from experiments.build_pdf_source.boilerplate import BoilerplateOptions, strip_document_boilerplate
from experiments.build_pdf_source.classify import routing_keys
from experiments.build_pdf_source.common import (
    FOCUS_CRAWL,
    LayoutModelData,
    PdfClassificationData,
    PdfSourceData,
)
from experiments.build_pdf_source.docling_extract.model_spec import LayoutBackend, TableBackend
from experiments.build_pdf_source.document_record import PDF_DOCUMENT_FIELDS, source_id

logger = logging.getLogger(__name__)

TABLE_BACKEND = TableBackend.PYMUPDF
# FP32, not the quantized graph. Measured over 600 documents on cw-us-east-02a, INT8 reproduces only
# 19% of them to within 1% of their tokens and yields 4.1% less text on average -- a one-sided loss,
# so it drops content rather than reordering it. An FP32-against-FP32 control over the same 600 puts
# the extractor's own noise floor at mean token F1 0.99995, three orders of magnitude smaller, so
# that gap is quantization and not run-to-run variance. It buys 1.21x on VNNI, which is not a trade
# worth making for a corpus being built to train on. The INT8 path stays selectable for the day
# throughput matters more.
LAYOUT_BACKEND = LayoutBackend.TORCH_HERON
PICTURE_ALPHA_RATIO = 0.4
# Bounds the tail. The page distribution is long -- p50 is 5 pages on this route but the maximum is
# over 3,000 -- and a task holds several documents, so one pathological document must not stall it.
DOCUMENT_TIMEOUT = 10 * 60
# Running headers and footers are stripped before the text is stored, so the id is computed over
# the text a consumer actually reads. See :mod:`experiments.build_pdf_source.boilerplate`.
BOILERPLATE_OPTIONS = BoilerplateOptions()

_SOURCE_COLUMNS = ["pdf", "warc_filename", "warc_record_offset", "content_digest", "url"]

# This route adds nothing of its own: everything docling reports about a document is already part
# of the record both routes share. See :mod:`experiments.build_pdf_source.document_record`.
_OUTPUT_SCHEMA = pa.schema(PDF_DOCUMENT_FIELDS)

_DRIVER_RESOURCES = ResourceConfig(cpu=2, ram="16g", disk="8g")
_WORKER_RESOURCES = ResourceConfig(cpu=8, ram="64g", disk="64g")
# Layout inference is CPU-bound and single-threaded per document, so tasks cost one CPU and
# multiplex eight-deep per worker. RAM is generous because each task holds its own converter, its
# own layout graph, and the broadcast routing set.
_MAP_TASK_RESOURCES = ResourceConfig(cpu=1, ram="7g", disk="2g")
_MAX_WORKERS = 28
_HEARTBEAT_TIMEOUT = 30 * 60


@cache
def _converter(options: "ExtractionOptions"):  # noqa: F821
    """Build the docling converter once per worker process.

    Cached on the options, which are frozen and hashable, so every task in a process shares one
    compiled layout graph rather than compiling its own.
    """
    from experiments.build_pdf_source.docling_extract.converter import build_converter  # noqa: PLC0415

    logger.info("Building docling converter (%s, %s)", options.table_backend, options.layout_backend)
    return build_converter(options)


def extract_batch(
    batch: pa.RecordBatch,
    keys: frozenset[tuple[str, int]],
    options: "ExtractionOptions",  # noqa: F821
    boilerplate: BoilerplateOptions,
) -> Iterator[dict]:
    """Extract text from the text-extractable documents in one Parquet row group.

    A document that cannot be converted, or that yields no text once its running headers are
    removed, is counted and dropped rather than emitted: the output of this step is a normalized
    text dataset, and a row with no text has no ``id`` and nothing for a downstream consumer to do.
    """
    from experiments.build_pdf_source.docling_extract.converter import extract_text  # noqa: PLC0415

    converter = _converter(options)
    for row in batch.to_pylist():
        key = (row["warc_filename"], row["warc_record_offset"])
        if key not in keys:
            counters.pipeline.update_counter("focus_crawl_pdf/skipped_ocr_route", 1)
            continue

        try:
            extracted = extract_text(converter, row["pdf"], options, name=row["url"] or "document.pdf")
        except Exception as error:
            # Docling raises a wide range of types on damaged input, and a document we cannot
            # convert is data, not a pipeline failure: it is counted and the shard carries on.
            counters.pipeline.update_counter("focus_crawl_pdf/extraction_failed", 1)
            logger.warning("Could not extract %s: %s", row["url"], error)
            continue

        stripped = strip_document_boilerplate(extracted.text, extracted.page_offsets, boilerplate)
        text = stripped.text
        counters.pipeline.update_counter("focus_crawl_pdf/extracted", 1)
        counters.pipeline.update_counter("focus_crawl_pdf/extracted_pages", extracted.num_pages)
        counters.pipeline.update_counter("focus_crawl_pdf/boilerplate_lines_removed", stripped.lines_removed)
        counters.pipeline.update_counter("focus_crawl_pdf/boilerplate_pages_stripped", stripped.pages_stripped)
        if extracted.extraction_error:
            counters.pipeline.update_counter("focus_crawl_pdf/partial_extraction", 1)

        if not text.strip():
            counters.pipeline.update_counter("focus_crawl_pdf/empty_text_filtered", 1)
            continue
        counters.pipeline.update_counter("focus_crawl_pdf/extracted_characters", len(text))

        yield {
            "id": generate_id(text),
            "text": text,
            "source_id": source_id(row["warc_filename"], row["warc_record_offset"]),
            "source": FOCUS_CRAWL,
            "warc_filename": row["warc_filename"],
            "warc_record_offset": row["warc_record_offset"],
            "content_digest": row["content_digest"],
            "url": row["url"],
            "num_pages": extracted.num_pages,
            "page_offsets": stripped.page_offsets,
            "extraction_status": extracted.status,
            "extraction_error": extracted.extraction_error,
            "boilerplate_lines_removed": stripped.lines_removed,
        }


def _keep_all(_key: str, records: Iterator[dict]) -> Iterator[MainOutput]:
    """Emit every record to the main output.

    Extraction deliberately does not deduplicate. The crawl holds roughly 9.8% exact-duplicate
    PDFs and extraction turns those into byte-identical text, so it is tempting to collapse them
    here -- but deduplication and decontamination are #7620, which has to make that decision across
    every source and against the eval sets, not just within this one. Extraction's job is to
    produce documents with a content-derived ``id``; #7620 is what uses it.

    The grouping this reducer runs under is still worth its cost: it sorts records by ``id``, which
    is part of the datakit normalized format and is what makes a later dedup pass a linear scan.
    """
    yield from (MainOutput(data=record) for record in records)


def extract_pdf_text(
    output_path: str,
    source_output_path: str,
    classification_output_path: str,
    layout_model_output_path: str,
) -> NormalizedData:
    """Extract the text-extractable route into the datakit normalized format.

    The output is :class:`~marin.datakit.normalize.NormalizedData` rather than a bespoke artifact,
    because this is the point at which PDFs become documents: text with a content-derived ``id``,
    sorted by it, in ``outputs/main``. That is the same contract the OCR route will produce, so a
    downstream consumer joins the two routes without knowing which extractor produced a document.
    ``outputs/dups`` is part of the format and is written empty: deduplication is #7620.
    """
    from experiments.build_pdf_source.docling_extract.converter import ExtractionOptions  # noqa: PLC0415

    source = read_artifact(source_output_path, PdfSourceData)
    classification = read_artifact(classification_output_path, PdfClassificationData)
    layout_model = read_artifact(layout_model_output_path, LayoutModelData)

    options = ExtractionOptions(
        table_backend=TABLE_BACKEND,
        layout_backend=LAYOUT_BACKEND,
        layout_model_path=layout_model.model_path,
        layout_label_map=layout_model.label_map,
        picture_alpha_ratio=PICTURE_ALPHA_RATIO,
        document_timeout=DOCUMENT_TIMEOUT,
    )
    keys = routing_keys(classification.main_output_dir, needs_ocr=False)

    filesystem, path = url_to_fs(source.main_output_dir)
    num_shards = len(filesystem.glob(f"{path}/*.parquet"))
    if not num_shards:
        raise RuntimeError(f"No fetched PDFs under {source.main_output_dir}")

    pipeline = (
        Dataset.from_files(prefix_join(source.main_output_dir, "*.parquet"))
        .load_parquet(columns=_SOURCE_COLUMNS, batch_mode=True)
        .flat_map(partial(extract_batch, keys=keys, options=options, boilerplate=BOILERPLATE_OPTIONS))
        # Sorts by ``id`` within each output shard, which is part of the normalized format. The
        # duplicates this surfaces are deliberately left in place -- see :func:`_keep_all`.
        .group_by(
            key=lambda record: record["id"],
            reducer=_keep_all,
            sort_by=lambda record: record["id"],
            num_output_shards=num_shards,
        )
        .map_shard(make_split_writer(output_path, output_schema=_OUTPUT_SCHEMA))
    )
    outcome = ZephyrContext(
        name="focus-crawl-pdf-extract",
        resources=_WORKER_RESOURCES,
        max_workers=_MAX_WORKERS,
        stage_runner_factory=SubprocessRunner,
        map_task_resources=_MAP_TASK_RESOURCES,
        heartbeat_timeout=_HEARTBEAT_TIMEOUT,
    ).execute(pipeline)
    return NormalizedData(
        main_output_dir=prefix_join(output_path, "outputs/main"),
        dup_output_dir=prefix_join(output_path, "outputs/dups"),
        counters=dict(outcome.counters),
    )


def extract_step(source: StepSpec, classification: StepSpec, layout_model: StepSpec) -> StepSpec:
    """Build the text extraction step for the router's text-extractable route."""
    return StepSpec(
        name="data/datakit/extract/common_crawl_focus_2026_22_pdf_text",
        deps=[source, classification, layout_model],
        hash_attrs={
            "table_backend": str(TABLE_BACKEND),
            "layout_backend": str(LAYOUT_BACKEND),
            "picture_alpha_ratio": PICTURE_ALPHA_RATIO,
            "document_timeout": DOCUMENT_TIMEOUT,
            "boilerplate_min_pages": BOILERPLATE_OPTIONS.min_pages,
            "boilerplate_min_page_fraction": BOILERPLATE_OPTIONS.min_page_fraction,
            "boilerplate_max_page_fraction": BOILERPLATE_OPTIONS.max_page_fraction,
            "boilerplate_max_edge_lines": BOILERPLATE_OPTIONS.max_edge_lines,
            "schema_version": 2,
        },
        fn=remote(
            partial(
                extract_pdf_text,
                source_output_path=source.output_path,
                classification_output_path=classification.output_path,
                layout_model_output_path=layout_model.output_path,
            ),
            resources=_DRIVER_RESOURCES,
            pip_dependency_groups=["datakit"],
        ),
    )
