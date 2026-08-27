# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Union the two extraction routes into one corpus, tagged with the router's decision.

The classifier split the fetched sample in two and each half was extracted by a different engine:
:func:`~experiments.datakit.build_pdf_source.extract_fleet.fleet_extract_step` ran the text-extractable
route through docling, and :func:`~experiments.datakit.build_pdf_source.extract_ocr.ocr_extract_step` ran
the rest through a vision model. Both emit :data:`~experiments.datakit.build_pdf_source.document_record
.PDF_DOCUMENT_FIELDS`; each route appends its own diagnostic columns. This module puts the two
back together so the rest of the chain -- dedup, decontamination, quality, fuzzy, LID -- runs once
over the whole corpus.

The union is a real step rather than a pair of paths handed to normalize because the two routes are
separate datasets: exact dedup has to see them together (the same text can be recovered from
different PDFs on either side of the router, and two copies of it are one document), and the router
decision that split them is worth keeping. The combine writes it as ``needs_ocr``, the classifier's
own column name -- after the union nothing else records which route a document came from, and it is
the axis the two halves exist to be compared along.

The two routes' schemas differ by their route-specific columns -- the OCR diagnostics on one side,
the fleet's ``layout_backend`` provenance on the other -- so the combined schema carries all of
them as nullable, null on every row of the other route. Carrying rather than dropping is
deliberate: ``layout_backend`` is per-document provenance placement decided and nothing can
recompute, exactly like the OCR page accounting, and downstream studies already read it. Nothing
is re-extracted and nothing is re-post-processed -- running headers, footers and page numbers were
stripped by :mod:`~experiments.datakit.build_pdf_source.boilerplate` inside both extraction runs, and a
second pass is not idempotent (the page-separator newline is itself a repeated edge pattern).

``outputs/dups`` is part of the :class:`~marin.datakit.normalize.NormalizedData` format and stays
empty here: exact dedup is the normalize step downstream (see :mod:`~experiments.datakit.build_pdf_source
.dedup`), and this union deliberately splits nothing.
"""

import logging
from collections.abc import Iterator
from functools import partial

import pyarrow as pa
from fray.types import ResourceConfig
from marin.datakit.normalize import NormalizedData
from marin.execution.remote import remote
from marin.execution.step_spec import StepSpec
from rigging.filesystem import StoragePath, prefix_join
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.runners import SubprocessRunner

from experiments.datakit.build_pdf_source.document_record import PDF_DOCUMENT_FIELDS
from experiments.datakit.build_pdf_source.extract_fleet import FLEET_FIELDS
from experiments.datakit.build_pdf_source.extract_ocr import OCR_FIELDS

logger = logging.getLogger(__name__)

_COUNTER_PREFIX = "focus_crawl_pdf_combine"

_CORPUS = "common_crawl_focus_2026_22_pdf"
_COMBINED_NAME = f"data/datakit/combine/{_CORPUS}"

# The shared document record, the router decision that split the corpus in two (under the
# classifier's own column name), and each route's own columns made nullable -- null on every row
# the other route produced, since that route never wrote them.
COMBINED_SCHEMA = pa.schema(
    [
        *PDF_DOCUMENT_FIELDS,
        pa.field("needs_ocr", pa.bool_(), nullable=False),
        *(pa.field(field.name, field.type, nullable=True) for field in (*OCR_FIELDS, *FLEET_FIELDS)),
    ]
)
_OCR_DIAGNOSTIC_NAMES: tuple[str, ...] = tuple(field.name for field in OCR_FIELDS)
_FLEET_COLUMN_NAMES: tuple[str, ...] = tuple(field.name for field in FLEET_FIELDS)

# The column the reader injects so a batch knows which route it came from. Dropped before writing.
_SOURCE_FILE_COLUMN = "_source_file"

_DRIVER_RESOURCES = ResourceConfig(cpu=2, ram="16g", disk="8g")
# One task per input shard, each a ~2 MB parquet file whose records it holds as dicts while tagging.
_MAP_TASK_RESOURCES = ResourceConfig(cpu=1, ram="4g", disk="4g")
_TASKS_PER_WORKER = 8
_WORKER_RESOURCES = ResourceConfig(cpu=_TASKS_PER_WORKER, ram="32g", disk="16g")
_MAX_WORKERS = 32
# Not Zephyr's 1 GB default: that default is OOM-killed (exit 137) at run end across this pipeline
# family, after the stage's work is already on disk.
_COORDINATOR_RESOURCES = ResourceConfig(cpu=1, ram="16g", preemptible=False)


def route_of(shard: str, route_dirs: tuple[tuple[bool, str], ...]) -> bool:
    """The router decision for the route that wrote ``shard``, by the directory it lives under.

    A shard under neither route raises: the driver listed these directories itself, so a path that
    does not belong to one is a bug in that listing rather than a document to guess about.
    """
    for needs_ocr, directory in route_dirs:
        if shard.startswith(directory):
            return needs_ocr
    raise ValueError(f"Shard {shard} belongs to neither extraction route: {[d for _, d in route_dirs]}")


def tag_batch(batch: pa.RecordBatch, route_dirs: tuple[tuple[bool, str], ...]) -> Iterator[dict]:
    """Emit one Parquet row group's records, tagged with the route that extracted it.

    Every row of a batch comes from one file, so the route is resolved once and the injected path
    column is dropped -- it is scaffolding for this step, not part of the corpus. Each route's rows
    lack the other route's columns entirely; they are filled with nulls so both land in one schema.
    """
    if not batch.num_rows:
        return
    needs_ocr = route_of(batch.column(_SOURCE_FILE_COLUMN)[0].as_py(), route_dirs)
    tally = "from_ocr_route" if needs_ocr else "from_text_route"
    for row in batch.to_pylist():
        del row[_SOURCE_FILE_COLUMN]
        row.update(dict.fromkeys(_FLEET_COLUMN_NAMES if needs_ocr else _OCR_DIAGNOSTIC_NAMES))
        counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/documents_out", 1)
        counters.pipeline.update_counter(f"{_COUNTER_PREFIX}/{tally}", 1)
        yield {**row, "needs_ocr": needs_ocr}


def combine_routes(output_path: str, text_output_path: str, ocr_output_path: str) -> NormalizedData:
    """Write both extraction routes into one directory.

    A pure 1:1 shard map -- no shuffle. The global sort and the exact dedup both belong to the
    normalize step that consumes this, so doing either here would be paid for twice.
    """
    shards: list[str] = []
    route_dirs: list[tuple[bool, str]] = []
    for needs_ocr, extraction_path in ((False, text_output_path), (True, ocr_output_path)):
        main = prefix_join(extraction_path, "outputs/main")
        found = sorted(str(shard) for shard in StoragePath(prefix_join(main, "*.parquet")).glob())
        if not found:
            raise RuntimeError(f"No extracted shards under {main}")
        # The resolved directory, taken from a shard rather than from ``main``, so the prefix the
        # tasks match against is the one the reader will hand them back.
        route_dirs.append((needs_ocr, found[0].rsplit("/", 1)[0] + "/"))
        shards.extend(found)
    logger.info("Combining %d shards across %d extraction routes", len(shards), len(route_dirs))

    main_output_dir = prefix_join(output_path, "outputs/main")
    pipeline = (
        Dataset.from_list(shards)
        .load_parquet(batch_mode=True, include_file_paths=True, file_path_column=_SOURCE_FILE_COLUMN)
        .flat_map(partial(tag_batch, route_dirs=tuple(route_dirs)))
        .write_parquet(
            prefix_join(main_output_dir, "part-{shard:05d}-of-{total:05d}.parquet"),
            schema=COMBINED_SCHEMA,
            skip_existing=True,
        )
    )
    with ZephyrContext(
        name="focus-crawl-pdf-combine",
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


def combine_step(text_extraction: StepSpec, ocr_extraction: StepSpec) -> StepSpec:
    """Union the two extraction routes into one corpus, tagged with the route each came from."""
    return StepSpec(
        name=_COMBINED_NAME,
        deps=[text_extraction, ocr_extraction],
        hash_attrs={"route_column": "needs_ocr"},
        fn=remote(
            partial(
                combine_routes,
                text_output_path=text_extraction.output_path,
                ocr_output_path=ocr_extraction.output_path,
            ),
            resources=_DRIVER_RESOURCES,
            pip_dependency_groups=["datakit"],
        ),
    )
