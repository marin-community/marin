# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TEMPORARY -- measure PyMuPDF ruling-line tables against TableFormer, on one layout backend.

DELETE once the result is recorded in
``.agents/ops/2026-07-30-layout-quantization-recipe.md``. Nothing in the pipeline imports this.

The INT8-versus-FP32 comparison showed the text loss was almost entirely emptied tables, and a
local perturbation test showed why: ``extract_table_data`` clips the page to the layout model's
box, runs PyMuPDF ``find_tables``, and declines unless it finds exactly one -- and shrinking a box
by a single pixel per side is enough to break most tables in the sample. Because ``fill()`` leaves a
declined table untouched and ``TableBackend.PYMUPDF`` disables TableFormer, a decline is not a
degraded table, it is an empty one.

That is a property of the reader, not of quantization, so **the FP32 default is exposed to it too**
and the layout comparison could not see it: both of its arms shared this table reader. This run
holds the layout backend fixed at :attr:`LayoutBackend.TORCH_HERON` and varies only the table
reader, which is the comparison the switchable ``TableBackend`` was built for.

TableFormer infers a grid from text positions instead of requiring ruling lines inside an exact
clip, so the question is whether it recovers the tables PyMuPDF drops -- and at what cost, since it
is a second neural model on the per-page path.

The headline numbers are table-content ones, not token F1: the two readers are *supposed* to
disagree on table text, so agreement statistics say little. What matters is how much table content
each recovers and how many tables come back empty.

Run on the same x86 cluster the other comparisons used::

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \\
        --job-name compare-table-backends \\
        -- python -m experiments.build_pdf_source._compare_table_backends
"""

import logging
import re
import time
from collections import Counter
from collections.abc import Iterator
from functools import partial

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from marin.execution.artifact import read_artifact
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from pydantic import BaseModel
from rigging.filesystem import prefix_join, url_to_fs
from rigging.log_setup import configure_logging
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.runners import SubprocessRunner

from experiments.build_pdf_source.boilerplate import BoilerplateOptions, strip_document_boilerplate
from experiments.build_pdf_source.classify import classify_step, model_step
from experiments.build_pdf_source.common import PdfClassificationData, PdfSourceData
from experiments.build_pdf_source.docling_extract.model_spec import LayoutBackend, TableBackend
from experiments.build_pdf_source.extract import text_extractable_keys
from experiments.build_pdf_source.fetch import fetch_step
from experiments.build_pdf_source.layout_model import layout_model_step
from experiments.build_pdf_source.plan import plan_step

logger = logging.getLogger(__name__)

# Matched to the layout comparison so the runs describe the same corpus.
SOURCE_SHARDS = 24
DOCUMENTS_PER_SHARD = 25
_SOURCE_COLUMNS = ["pdf", "warc_filename", "warc_record_offset", "url"]

# The layout backend is held fixed; only the table reader varies.
_LAYOUT_BACKEND = LayoutBackend.TORCH_HERON
_ARMS = (str(TableBackend.PYMUPDF), str(TableBackend.DOCLING))

_TABLE_BLOCK = re.compile(r"<docling_table>(.*?)</docling_table>", re.DOTALL)

_TEXT_SCHEMA = pa.schema(
    [
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("arm", pa.string(), nullable=False),
        pa.field("url", pa.string(), nullable=False),
        pa.field("text", pa.string(), nullable=True),
        pa.field("error", pa.string(), nullable=True),
        pa.field("seconds", pa.float64(), nullable=False),
        pa.field("tables", pa.int64(), nullable=False),
        pa.field("table_chars", pa.int64(), nullable=False),
        pa.field("empty_tables", pa.int64(), nullable=False),
    ]
)

_DRIVER_RESOURCES = ResourceConfig(cpu=4, ram="32g", disk="16g")
_WORKER_RESOURCES = ResourceConfig(cpu=8, ram="64g", disk="64g")
# TableFormer is a second neural model on the per-page path, so give a task more headroom than the
# layout comparison did.
_MAP_TASK_RESOURCES = ResourceConfig(cpu=1, ram="10g", disk="6g")
_MAX_WORKERS = 28
_HEARTBEAT_TIMEOUT = 45 * 60


class ArmStats(BaseModel):
    """What one table reader recovered."""

    documents: int
    total_tables: int
    total_table_chars: int
    total_empty_tables: int
    documents_with_empty_table: int
    mean_chars_per_document: float
    mean_table_chars_per_document: float
    seconds_per_document: float


class TableComparisonReport(BaseModel):
    """PyMuPDF ruling-line tables against TableFormer, layout held fixed."""

    version: str = "v1"
    documents: int
    both_extracted: int
    failures: dict[str, int]
    pymupdf: ArmStats
    docling: ArmStats
    # Positive means TableFormer recovered more table text than PyMuPDF on that document.
    docling_table_chars_ratio: float
    documents_where_docling_recovered_more: int
    documents_where_pymupdf_recovered_more: int
    documents_pymupdf_empty_docling_not: int
    documents_docling_empty_pymupdf_not: int
    mean_char_ratio: float


def _table_stats(text: str) -> tuple[int, int, int]:
    """Return (table count, characters inside tables, tables that came back empty)."""
    blocks = _TABLE_BLOCK.findall(text)
    return len(blocks), sum(len(b) for b in blocks), sum(1 for b in blocks if not b.strip())


def extract_shard(work: dict, keys: frozenset[tuple[str, int]]) -> Iterator[dict]:
    """Extract one shard's text-extractable documents with one table reader."""
    from experiments.build_pdf_source.docling_extract.converter import (  # noqa: PLC0415
        ExtractionOptions,
        build_converter,
        extract_text,
    )

    arm = work["arm"]
    options = ExtractionOptions(
        table_backend=TableBackend(arm),
        layout_backend=_LAYOUT_BACKEND,
        layout_model_path=None,
        layout_label_map={},
        picture_alpha_ratio=0.4,
    )
    boilerplate = BoilerplateOptions()

    filesystem, _ = url_to_fs(work["shard"])
    with filesystem.open(work["shard"], "rb") as stream:
        table = pq.read_table(stream, columns=_SOURCE_COLUMNS)

    converter = build_converter(options)
    emitted = 0
    for row in table.to_pylist():
        if emitted >= DOCUMENTS_PER_SHARD:
            break
        if (row["warc_filename"], row["warc_record_offset"]) not in keys:
            continue
        emitted += 1

        output = {
            "source_id": f"{row['warc_filename']}:{row['warc_record_offset']}",
            "arm": arm,
            "url": row["url"],
            "text": None,
            "error": None,
            "tables": 0,
            "table_chars": 0,
            "empty_tables": 0,
        }
        started = time.perf_counter()
        try:
            extracted = extract_text(converter, row["pdf"], options, name=row["url"] or "doc.pdf")
        except Exception as error:
            counters.pipeline.update_counter(f"table_compare/{arm}_failed", 1)
            logger.warning("arm %s extraction failed for %s: %s", arm, row["url"], error, exc_info=True)
            yield output | {"error": f"{type(error).__name__}: {error}", "seconds": time.perf_counter() - started}
            continue

        stripped = strip_document_boilerplate(extracted.text, extracted.page_offsets, boilerplate)
        tables, table_chars, empty = _table_stats(stripped.text)
        counters.pipeline.update_counter(f"table_compare/{arm}_extracted", 1)
        yield output | {
            "text": stripped.text,
            "seconds": time.perf_counter() - started,
            "tables": tables,
            "table_chars": table_chars,
            "empty_tables": empty,
        }


def _arm_stats(rows: list[dict]) -> ArmStats:
    return ArmStats(
        documents=len(rows),
        total_tables=sum(r["tables"] for r in rows),
        total_table_chars=sum(r["table_chars"] for r in rows),
        total_empty_tables=sum(r["empty_tables"] for r in rows),
        documents_with_empty_table=sum(1 for r in rows if r["empty_tables"]),
        mean_chars_per_document=float(np.mean([len(r["text"]) for r in rows])),
        mean_table_chars_per_document=float(np.mean([r["table_chars"] for r in rows])),
        seconds_per_document=float(np.mean([r["seconds"] for r in rows])),
    )


def _compare(texts: dict[str, dict[str, dict]]) -> TableComparisonReport:
    """Score TableFormer against the PyMuPDF ruling-line reader."""
    pymupdf_arm, docling_arm = _ARMS
    paired = [
        (rows[pymupdf_arm], rows[docling_arm])
        for rows in texts.values()
        if rows.get(pymupdf_arm, {}).get("text") is not None and rows.get(docling_arm, {}).get("text") is not None
    ]
    failures: Counter = Counter()
    for rows in texts.values():
        for row in rows.values():
            if row["error"]:
                failures[f"{row['arm']}:{row['error'].partition(':')[0]}"] += 1

    if not paired:
        raise RuntimeError(f"No document extracted on both table backends. failures={dict(failures)}")

    left = [p[0] for p in paired]
    right = [p[1] for p in paired]
    left_chars = sum(r["table_chars"] for r in left)
    right_chars = sum(r["table_chars"] for r in right)

    return TableComparisonReport(
        documents=len(texts),
        both_extracted=len(paired),
        failures=dict(failures),
        pymupdf=_arm_stats(left),
        docling=_arm_stats(right),
        docling_table_chars_ratio=right_chars / left_chars if left_chars else float("inf"),
        documents_where_docling_recovered_more=sum(1 for a, b in paired if b["table_chars"] > a["table_chars"]),
        documents_where_pymupdf_recovered_more=sum(1 for a, b in paired if a["table_chars"] > b["table_chars"]),
        documents_pymupdf_empty_docling_not=sum(1 for a, b in paired if a["empty_tables"] and not b["empty_tables"]),
        documents_docling_empty_pymupdf_not=sum(1 for a, b in paired if b["empty_tables"] and not a["empty_tables"]),
        mean_char_ratio=float(np.mean([len(b["text"]) / max(len(a["text"]), 1) for a, b in paired])),
    )


def compare(output_path: str, source_output_path: str, classification_output_path: str) -> TableComparisonReport:
    """Extract the same documents with both table readers, then compare what the tables hold."""
    source = read_artifact(source_output_path, PdfSourceData)
    classification = read_artifact(classification_output_path, PdfClassificationData)

    filesystem, path = url_to_fs(source.main_output_dir)
    protocol = source.main_output_dir.split("://")[0] if "://" in source.main_output_dir else ""
    shards = sorted(filesystem.glob(f"{path}/*.parquet"))[:SOURCE_SHARDS]
    if not shards:
        raise RuntimeError(f"No fetched PDFs under {source.main_output_dir}")
    shards = [f"{protocol}://{shard}" if protocol else shard for shard in shards]

    work = [{"arm": arm, "shard": shard} for arm in _ARMS for shard in shards]
    logger.info("Extracting %d shards on %d table backends as %d tasks", len(shards), len(_ARMS), len(work))

    texts_dir = prefix_join(output_path, "outputs/texts")
    pipeline = (
        Dataset.from_list(work)
        .flat_map(partial(extract_shard, keys=text_extractable_keys(classification.main_output_dir)))
        .write_parquet(
            prefix_join(texts_dir, "part-{shard:05d}-of-{total:05d}.parquet"),
            schema=_TEXT_SCHEMA,
            skip_existing=True,
        )
    )
    ZephyrContext(
        name="table-backend-compare",
        resources=_WORKER_RESOURCES,
        max_workers=min(_MAX_WORKERS, len(work)),
        stage_runner_factory=SubprocessRunner,
        heartbeat_timeout=_HEARTBEAT_TIMEOUT,
    ).execute(pipeline, map_task_resources=_MAP_TASK_RESOURCES)

    texts: dict[str, dict[str, dict]] = {}
    text_fs, text_path = url_to_fs(texts_dir)
    for written in sorted(text_fs.glob(f"{text_path}/*.parquet")):
        with text_fs.open(written, "rb") as stream:
            for row in pq.read_table(stream).to_pylist():
                texts.setdefault(row["source_id"], {})[row["arm"]] = row

    report = _compare(texts)
    logger.info("=== TABLE BACKEND COMPARISON (layout fixed at %s) ===", _LAYOUT_BACKEND)
    for key, value in report.model_dump().items():
        logger.info("  %s: %s", key, value)
    return report


def comparison_step(source: StepSpec, classification: StepSpec, layout_model: StepSpec) -> StepSpec:
    """Build the table-backend comparison step."""
    return StepSpec(
        name="data/datakit/validate/table_backend_comparison",
        deps=[layout_model],
        hash_attrs={
            "source_output_path": source.output_path,
            "classification_output_path": classification.output_path,
            "source_shards": SOURCE_SHARDS,
            "documents_per_shard": DOCUMENTS_PER_SHARD,
            "layout_backend": str(_LAYOUT_BACKEND),
            # Bump after any dependency or extractor change: the pipeline writes with
            # skip_existing=True and this hash says nothing about either, so a previous run's rows
            # are silently inherited. 2: attempt 1 crashed the DOCLING arm on a page whose document
            # backend had already closed it, and left 14 clean-but-pre-fix shards behind.
            "attempt": 2,
        },
        fn=remote(
            partial(
                compare,
                source_output_path=source.output_path,
                classification_output_path=classification.output_path,
            ),
            resources=_DRIVER_RESOURCES,
            pip_dependency_groups=["datakit"],
        ),
    )


def main() -> None:
    configure_logging(logging.INFO)
    plan = plan_step()
    fetch = fetch_step(plan)
    classify = classify_step(fetch, model_step())
    layout_model = layout_model_step(fetch)
    StepRunner().run([layout_model, comparison_step(fetch, classify, layout_model)])


if __name__ == "__main__":
    main()
