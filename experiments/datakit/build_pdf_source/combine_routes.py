# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Choose one extraction per document and union the two routes into one corpus.

Documents the router kept come from pdf-inspector; documents it escalated come from the VLM and
their pdf-inspector row is dropped. The pdf-inspector side is filtered shard by shard: each of its
shards is named after the fetched shard it was read from, and so is the routing shard holding its
decisions, so a task opens the routing shard of the same name
(:func:`~experiments.datakit.build_pdf_source.classify.shard_routing`) and never the corpus-wide
table. The VLM side needs no filter -- it only ever read the escalated subset.

The union is a real step rather than a pair of paths handed to normalize because the two routes are
separate datasets: exact dedup has to see them together (the same text can be recovered from
different PDFs on either side of the router, and two copies of it are one document), and the router
decision is worth keeping. The combine writes it as ``needs_ocr``, the router's own column name --
after the union nothing else records which route a document came from, and it is the axis the two
halves exist to be compared along.

The two routes' schemas differ by their route-specific columns -- the OCR page accounting on one
side, pdf-inspector's classification signals on the other -- so the combined schema carries all of
them as nullable, null on every row of the other route. Carrying rather than dropping is deliberate:
these are per-document provenance and routing inputs that nothing downstream can recompute, so a
routing decision stays auditable from the corpus alone. Where both routes declare the same column --
``mean_render_dpi`` and ``pages_below_legibility_floor``, which one route measured while rendering
and the other computed from page geometry before deciding not to -- the column is shared rather than
duplicated, because it means the same thing on both sides. Nothing is re-extracted and nothing is
re-post-processed: running headers and footers were stripped by
:mod:`~experiments.datakit.build_pdf_source.boilerplate` inside both extraction runs, and a second
pass is not idempotent (the page-separator newline is itself a repeated edge pattern).

The output is one shard per input shard across both routes, unsorted: the global sort and the exact
dedup belong to the normalize step that consumes this
(:mod:`~experiments.datakit.build_pdf_source.dedup`), and doing either here would be paid for twice.
"""

import logging
import os
from collections.abc import Iterator
from functools import partial

import pyarrow as pa
from fray.types import ResourceConfig
from marin.execution.artifact import read_artifact
from marin.execution.remote import remote
from marin.execution.step_spec import StepSpec
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.runners import SubprocessRunner

from experiments.datakit.build_pdf_source.classify import shard_routing
from experiments.datakit.build_pdf_source.common import (
    CORPUS,
    MAIN_OUTPUT_SUBDIR,
    SHARD_PATTERN,
    SOURCE_FILE_COLUMN,
    PdfClassificationData,
    PdfDocumentsData,
)
from experiments.datakit.build_pdf_source.document_record import PDF_DOCUMENT_FIELDS
from experiments.datakit.build_pdf_source.extract_inspector import INSPECTOR_FIELDS
from experiments.datakit.build_pdf_source.extract_ocr import OCR_FIELDS

logger = logging.getLogger(__name__)

_COUNTER_PREFIX = "focus_crawl_pdf_combine"

_COMBINED_NAME = f"data/datakit/combine/{CORPUS}"


def _route_fields(*routes: tuple[pa.Field, ...]) -> tuple[pa.Field, ...]:
    """Every route-specific column once, nullable, in declaration order.

    A column two routes both declare is carried once. That is only sound if they agree on its type,
    so disagreement raises here rather than surfacing as an unreadable Parquet file at corpus scale.
    """
    declared: dict[str, pa.Field] = {}
    for field in (field for route in routes for field in route):
        existing = declared.get(field.name)
        if existing is not None and existing.type != field.type:
            raise ValueError(f"routes declare {field.name} as both {existing.type} and {field.type}")
        declared.setdefault(field.name, pa.field(field.name, field.type, nullable=True))
    return tuple(declared.values())


# The shared document record, the router's decision (under its own column name), and each route's
# own columns made nullable -- null on every row of a route that never wrote them.
COMBINED_SCHEMA = pa.schema(
    [
        *PDF_DOCUMENT_FIELDS,
        pa.field("needs_ocr", pa.bool_(), nullable=False),
        *_route_fields(OCR_FIELDS, INSPECTOR_FIELDS),
    ]
)
_OCR_NAMES: frozenset[str] = frozenset(field.name for field in OCR_FIELDS)
_INSPECTOR_NAMES: frozenset[str] = frozenset(field.name for field in INSPECTOR_FIELDS)
# What each route does *not* write and therefore has to be null-filled with. A column both routes
# write is on neither list, so it survives the fill on both sides.
_OCR_ONLY_NAMES: tuple[str, ...] = tuple(sorted(_OCR_NAMES - _INSPECTOR_NAMES))
_INSPECTOR_ONLY_NAMES: tuple[str, ...] = tuple(sorted(_INSPECTOR_NAMES - _OCR_NAMES))

_DRIVER_RESOURCES = ResourceConfig(cpu=2, ram="16g", disk="8g")
# One task per input shard, each a ~2 MB parquet file whose records it holds as dicts while tagging.
_MAP_TASK_RESOURCES = ResourceConfig(cpu=1, ram="4g", disk="4g")
_TASKS_PER_WORKER = 8
_WORKER_RESOURCES = ResourceConfig(cpu=_TASKS_PER_WORKER, ram="32g", disk="16g")
_MAX_WORKERS = 32


def route_of(shard: str, route_dirs: tuple[tuple[bool, str], ...]) -> bool:
    """The router decision for the route that wrote ``shard``, by the directory it lives under.

    A shard under neither route raises: the driver listed these directories itself, so a path that
    does not belong to one is a bug in that listing rather than a document to guess about.
    """
    for needs_ocr, directory in route_dirs:
        if shard.startswith(directory):
            return needs_ocr
    raise ValueError(f"Shard {shard} belongs to neither extraction route: {[d for _, d in route_dirs]}")


def tag_batch(batch: pa.RecordBatch, route_dirs: tuple[tuple[bool, str], ...], routing_dir: str) -> Iterator[dict]:
    """Emit one Parquet row group's records, tagged with the route the corpus takes them from.

    Every row of a batch comes from one file, so the route is resolved once and the injected path
    column is dropped -- it is scaffolding for this step, not part of the corpus. Each route's rows
    lack the other route's columns entirely; they are filled with nulls so both land in one schema.

    The pdf-inspector side is filtered against the routing shard named after the batch's own shard:
    it holds every fetched document, including the ones the router escalated and the ones it
    produced no text for, and only the kept documents belong in the corpus. A document the routing
    shard does not know is an error rather than a drop, because the table is total over the
    extraction and a missing key means the two are not the co-partitioned pair this pipeline was
    built from. The VLM side needs no filter -- it only ever read the escalated subset -- so its
    routing shard is never opened.
    """
    if not batch.num_rows:
        return
    source_file = batch.column(SOURCE_FILE_COLUMN)[0].as_py()
    needs_ocr = route_of(source_file, route_dirs)
    routing = None if needs_ocr else shard_routing(routing_dir, os.path.basename(source_file))
    tally = "from_ocr_route" if needs_ocr else "from_inspector_route"
    absent = _INSPECTOR_ONLY_NAMES if needs_ocr else _OCR_ONLY_NAMES
    for row in batch.to_pylist():
        del row[SOURCE_FILE_COLUMN]
        if routing is not None:
            key = (row["warc_filename"], row["warc_record_offset"])
            decision = routing.get(key)
            if decision is None:
                raise ValueError(f"{row['url']} ({key[0]}:{key[1]}) has no routing decision under {routing_dir}")
            if decision.needs_ocr:
                counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/escalated_inspector_row_dropped", 1)
                continue
        row.update(dict.fromkeys(absent))
        counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/documents_out", 1)
        counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/{tally}", 1)
        yield {**row, "needs_ocr": needs_ocr}


def combine_routes(
    output_path: str, inspector_output_path: str, ocr_output_path: str, classification_output_path: str
) -> PdfDocumentsData:
    """Write the chosen extraction of every document into one directory, one shard per input shard."""
    routing_dir = read_artifact(classification_output_path, PdfClassificationData).main_output_dir

    shards: list[str] = []
    route_dirs: list[tuple[bool, str]] = []
    for needs_ocr, extraction_path in ((False, inspector_output_path), (True, ocr_output_path)):
        main = read_artifact(extraction_path, PdfDocumentsData).main_output_dir
        found = sorted(str(shard) for shard in StoragePath(prefix_join(main, "*.parquet")).glob())
        if not found:
            raise RuntimeError(f"No extracted shards under {main}")
        # The resolved directory, taken from a shard rather than from ``main``, so the prefix the
        # tasks match against is the one the reader will hand them back.
        route_dirs.append((needs_ocr, found[0].rsplit("/", 1)[0] + "/"))
        shards.extend(found)
    logger.info("Combining %d shards across %d extraction routes", len(shards), len(route_dirs))

    main_output_dir = prefix_join(output_path, MAIN_OUTPUT_SUBDIR)
    pipeline = (
        Dataset.from_list(shards)
        .load_parquet(batch_mode=True, include_file_paths=True, file_path_column=SOURCE_FILE_COLUMN)
        .flat_map(partial(tag_batch, route_dirs=tuple(route_dirs), routing_dir=routing_dir))
        .write_parquet(prefix_join(main_output_dir, SHARD_PATTERN), schema=COMBINED_SCHEMA, skip_existing=True)
    )
    outcome = ZephyrContext(
        name="focus-crawl-pdf-combine",
        resources=_WORKER_RESOURCES,
        max_workers=_MAX_WORKERS,
        stage_runner_factory=SubprocessRunner,
    ).execute(pipeline, map_task_resources=_MAP_TASK_RESOURCES)
    return PdfDocumentsData(main_output_dir=main_output_dir, counters=dict(outcome.counters))


def combine_step(inspector_extraction: StepSpec, ocr_extraction: StepSpec, classification: StepSpec) -> StepSpec:
    """Union the two extraction routes into one corpus, tagged with the route each came from."""
    return StepSpec(
        name=_COMBINED_NAME,
        deps=[inspector_extraction, ocr_extraction, classification],
        hash_attrs={"route_column": "needs_ocr", "schema_version": 2},
        fn=remote(
            partial(
                combine_routes,
                inspector_output_path=inspector_extraction.output_path,
                ocr_output_path=ocr_extraction.output_path,
                classification_output_path=classification.output_path,
            ),
            resources=_DRIVER_RESOURCES,
            pip_dependency_groups=["datakit"],
        ),
    )
