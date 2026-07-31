# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TEMPORARY -- the full layout x table backend matrix, measured in one run.

DELETE once the result is recorded in
`.agents/ops/2026-07-30-layout-quantization-recipe.md`. Nothing in the pipeline imports this.

Three of the four cells have been measured, in three separate runs:

* FP32 + PyMuPDF is the current default.
* INT8 + PyMuPDF lost 4% of the corpus text, almost all of it emptied tables.
* FP32 + TableFormer recovered 33% more table words than PyMuPDF and emptied 1,256 fewer tables.

The missing cell is the interesting one. INT8's damage was not diffuse: it was
`extract_table_data` clipping to a slightly-wrong box, finding zero or two ruled tables, and
declining -- which under `TableBackend.PYMUPDF` leaves the table with no cells at all. TableFormer
infers a grid from text positions inside the region instead of requiring ruling lines within an
exact clip, so it should be much less sensitive to a box being a few pixels off. If that holds,
INT8 + TableFormer recovers most of what INT8 + PyMuPDF lost, and INT8's 1.21x speedup roughly
cancels TableFormer's 1.36x cost.

All four cells run here rather than being compared across the earlier runs: those ran under
different fleet conditions and their wall-clock numbers are not comparable to each other.

Two structural choices, both about latency rather than total work:

*One document per task.* 4 arms x 600 documents = 2,400 tasks, sized so the fleet can clear them in
a few waves. The cost is a converter build per document -- the layout model and, on TableFormer
arms, a second network -- which is why every row records `build_seconds` apart from `seconds`.
**Compare arms on `seconds`, never on wall clock**, or model loading swamps the signal being
measured.

*Documents are materialised first.* The fetched shards are ~250 MB each because they carry PDF
bytes, so 2,400 tasks reading a shard apiece would move ~600 GB to extract 2,400 documents. The
driver instead reads the 24 shards once and writes each selected document as its own small object,
after which a task reads only its own.

Run::

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \\
        --job-name compare-backend-matrix \\
        -- python -m experiments.build_pdf_source._compare_backend_matrix
"""

import logging
import re
import time
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
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
from experiments.build_pdf_source.common import LayoutModelData, PdfClassificationData, PdfSourceData
from experiments.build_pdf_source.docling_extract.model_spec import LayoutBackend, TableBackend
from experiments.build_pdf_source.extract import text_extractable_keys
from experiments.build_pdf_source.fetch import fetch_step
from experiments.build_pdf_source.layout_model import layout_model_step
from experiments.build_pdf_source.plan import plan_step

logger = logging.getLogger(__name__)

SOURCE_SHARDS = 24
DOCUMENTS_PER_SHARD = 25
_SOURCE_COLUMNS = ["pdf", "warc_filename", "warc_record_offset", "url"]
_MATERIALISE_THREADS = 32

BASELINE = "fp32_pymupdf"
_ARMS: dict[str, tuple[LayoutBackend, TableBackend]] = {
    BASELINE: (LayoutBackend.TORCH_HERON, TableBackend.PYMUPDF),
    "fp32_tableformer": (LayoutBackend.TORCH_HERON, TableBackend.DOCLING),
    "int8_pymupdf": (LayoutBackend.INT8, TableBackend.PYMUPDF),
    "int8_tableformer": (LayoutBackend.INT8, TableBackend.DOCLING),
}

_TABLE_BLOCK = re.compile(r"<docling_table>(.*?)</docling_table>", re.DOTALL)
_WORD = re.compile(r"\w+")

_DOCUMENT_SCHEMA = pa.schema(
    [
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("url", pa.string(), nullable=True),
        pa.field("pdf", pa.large_binary(), nullable=False),
    ]
)

_TEXT_SCHEMA = pa.schema(
    [
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("arm", pa.string(), nullable=False),
        pa.field("url", pa.string(), nullable=True),
        pa.field("text", pa.string(), nullable=True),
        pa.field("error", pa.string(), nullable=True),
        pa.field("seconds", pa.float64(), nullable=False),
        pa.field("build_seconds", pa.float64(), nullable=False),
        pa.field("tables", pa.int64(), nullable=False),
        pa.field("table_words", pa.int64(), nullable=False),
        pa.field("empty_tables", pa.int64(), nullable=False),
    ]
)

# 125 worker pods x 8 slots = 1,000 concurrent tasks, so 2,400 tasks clear in three waves. RAM is
# sized so eight tasks fit a pod: each task holds the layout model and, on TableFormer arms, a
# second network, in its own subprocess.
_DRIVER_RESOURCES = ResourceConfig(cpu=16, ram="96g", disk="128g")
_WORKER_RESOURCES = ResourceConfig(cpu=8, ram="96g", disk="64g")
_MAP_TASK_RESOURCES = ResourceConfig(cpu=1, ram="10g", disk="6g")
_MAX_WORKERS = 125
_HEARTBEAT_TIMEOUT = 45 * 60


class ArmStats(BaseModel):
    """One cell of the matrix."""

    layout: str
    table: str
    documents: int
    failures: dict[str, int]
    mean_chars_per_document: float
    total_table_words: int
    total_tables: int
    total_empty_tables: int
    documents_with_empty_table: int
    # Extraction only. `build_seconds` is the converter build this harness repeats per document and
    # is not part of the per-document cost a real run would pay.
    seconds_per_document: float
    build_seconds_per_task: float
    # Against the FP32 + PyMuPDF baseline, over documents both arms extracted.
    mean_token_f1_vs_baseline: float
    mean_char_ratio_vs_baseline: float
    table_word_ratio_vs_baseline: float
    documents_losing_all_table_words_vs_baseline: int
    documents_gaining_table_words_vs_baseline: int


class MatrixReport(BaseModel):
    version: str = "v1"
    documents: int
    arms: dict[str, ArmStats]


def _table_stats(text: str) -> tuple[int, int, int]:
    """(tables, word tokens inside tables, tables that came back empty)."""
    blocks = _TABLE_BLOCK.findall(text)
    return (
        len(blocks),
        sum(len(_WORD.findall(b)) for b in blocks),
        sum(1 for b in blocks if not b.strip()),
    )


def _token_f1(first: str, second: str) -> float:
    left, right = Counter(first.split()), Counter(second.split())
    if not left and not right:
        return 1.0
    overlap = sum((left & right).values())
    if not overlap:
        return 0.0
    precision = overlap / sum(right.values())
    recall = overlap / sum(left.values())
    return 2 * precision * recall / (precision + recall)


def _write_document(row: dict, destination: str) -> str:
    """Write one fetched row out as a single-row Parquet file."""
    one = pa.Table.from_pylist(
        [
            {
                "source_id": f"{row['warc_filename']}:{row['warc_record_offset']}",
                "url": row["url"],
                "pdf": row["pdf"],
            }
        ],
        schema=_DOCUMENT_SCHEMA,
    )
    filesystem, path = url_to_fs(destination)
    with filesystem.open(path, "wb") as stream:
        pq.write_table(one, stream)
    return destination


def materialise_documents(source_dir: str, keys: frozenset[tuple[str, int]], documents_dir: str) -> list[str]:
    """Write each selected document to its own object and return their paths.

    The fetched shards carry PDF bytes and run ~250 MB apiece. Splitting them once here is what lets
    a task read one document instead of one shard.
    """
    filesystem, path = url_to_fs(documents_dir)
    existing = sorted(filesystem.glob(f"{path}/*.parquet"))
    if existing:
        protocol = documents_dir.split("://")[0] if "://" in documents_dir else ""
        logger.info("Reusing %d materialised documents under %s", len(existing), documents_dir)
        return [f"{protocol}://{name}" if protocol else name for name in existing]

    source_fs, source_path = url_to_fs(source_dir)
    protocol = source_dir.split("://")[0] if "://" in source_dir else ""
    shards = sorted(source_fs.glob(f"{source_path}/*.parquet"))[:SOURCE_SHARDS]
    if not shards:
        raise RuntimeError(f"No fetched PDFs under {source_dir}")

    written: list[str] = []
    with ThreadPoolExecutor(max_workers=_MATERIALISE_THREADS) as pool:
        for shard_index, shard in enumerate(shards):
            with source_fs.open(f"{protocol}://{shard}" if protocol else shard, "rb") as stream:
                table = pq.read_table(stream, columns=_SOURCE_COLUMNS)
            selected = [row for row in table.to_pylist() if (row["warc_filename"], row["warc_record_offset"]) in keys][
                :DOCUMENTS_PER_SHARD
            ]
            targets = [
                (row, prefix_join(documents_dir, f"doc-{shard_index:03d}-{index:03d}.parquet"))
                for index, row in enumerate(selected)
            ]
            written.extend(pool.map(lambda pair: _write_document(*pair), targets))
            logger.info("Materialised shard %d/%d (%d documents so far)", shard_index + 1, len(shards), len(written))
    return written


def extract_one(
    work: dict,
    layout_model_path: str,
    layout_label_map: dict[int, str],
) -> Iterator[dict]:
    """Extract a single document under one (layout, table) combination."""
    from experiments.build_pdf_source.docling_extract.converter import (  # noqa: PLC0415
        ExtractionOptions,
        build_converter,
        extract_text,
    )

    arm = work["arm"]
    layout, table = LayoutBackend(work["layout"]), TableBackend(work["table"])
    options = ExtractionOptions(
        table_backend=table,
        layout_backend=layout,
        layout_model_path=layout_model_path if layout is LayoutBackend.INT8 else None,
        layout_label_map=layout_label_map if layout is LayoutBackend.INT8 else {},
        picture_alpha_ratio=0.4,
    )

    filesystem, path = url_to_fs(work["document"])
    with filesystem.open(path, "rb") as stream:
        row = pq.read_table(stream).to_pylist()[0]

    build_started = time.perf_counter()
    converter = build_converter(options)
    build_seconds = time.perf_counter() - build_started

    output = {
        "source_id": row["source_id"],
        "arm": arm,
        "url": row["url"],
        "text": None,
        "error": None,
        "build_seconds": build_seconds,
        "tables": 0,
        "table_words": 0,
        "empty_tables": 0,
    }
    started = time.perf_counter()
    try:
        extracted = extract_text(converter, row["pdf"], options, name=row["url"] or "doc.pdf")
    except Exception as error:
        counters.pipeline.update_counter(f"matrix/{arm}_failed", 1)
        logger.warning("arm %s failed for %s: %s", arm, row["url"], error, exc_info=True)
        yield output | {"error": f"{type(error).__name__}: {error}", "seconds": time.perf_counter() - started}
        return

    stripped = strip_document_boilerplate(extracted.text, extracted.page_offsets, BoilerplateOptions())
    tables, table_words, empty = _table_stats(stripped.text)
    counters.pipeline.update_counter(f"matrix/{arm}_extracted", 1)
    yield output | {
        "text": stripped.text,
        "seconds": time.perf_counter() - started,
        "tables": tables,
        "table_words": table_words,
        "empty_tables": empty,
    }


def _arm_stats(arm: str, rows: dict[str, dict], baseline: dict[str, dict]) -> ArmStats:
    layout, table = _ARMS[arm]
    good = [r for r in rows.values() if r["text"] is not None]
    failures: Counter = Counter()
    for row in rows.values():
        if row["error"]:
            failures[row["error"].partition(":")[0]] += 1

    paired = [
        (baseline[sid], row)
        for sid, row in rows.items()
        if row["text"] is not None and baseline.get(sid, {}).get("text") is not None
    ]
    f1 = [_token_f1(b["text"], r["text"]) for b, r in paired]
    ratio = [len(r["text"]) / max(len(b["text"]), 1) for b, r in paired]
    base_words = sum(b["table_words"] for b, _ in paired)
    arm_words = sum(r["table_words"] for _, r in paired)

    return ArmStats(
        layout=str(layout),
        table=str(table),
        documents=len(good),
        failures=dict(failures),
        mean_chars_per_document=float(np.mean([len(r["text"]) for r in good])) if good else 0.0,
        total_table_words=sum(r["table_words"] for r in good),
        total_tables=sum(r["tables"] for r in good),
        total_empty_tables=sum(r["empty_tables"] for r in good),
        documents_with_empty_table=sum(1 for r in good if r["empty_tables"]),
        seconds_per_document=float(np.mean([r["seconds"] for r in good])) if good else 0.0,
        build_seconds_per_task=float(np.mean([r["build_seconds"] for r in rows.values()])) if rows else 0.0,
        mean_token_f1_vs_baseline=float(np.mean(f1)) if f1 else 0.0,
        mean_char_ratio_vs_baseline=float(np.mean(ratio)) if ratio else 0.0,
        table_word_ratio_vs_baseline=(arm_words / base_words) if base_words else 0.0,
        documents_losing_all_table_words_vs_baseline=sum(
            1 for b, r in paired if b["table_words"] and not r["table_words"]
        ),
        documents_gaining_table_words_vs_baseline=sum(1 for b, r in paired if r["table_words"] > b["table_words"]),
    )


def compare(
    output_path: str,
    source_output_path: str,
    classification_output_path: str,
    layout_model_output_path: str,
) -> MatrixReport:
    """Run every (layout, table) combination over the same documents and report the matrix."""
    source = read_artifact(source_output_path, PdfSourceData)
    classification = read_artifact(classification_output_path, PdfClassificationData)
    layout_model = read_artifact(layout_model_output_path, LayoutModelData)

    documents = materialise_documents(
        source.main_output_dir,
        text_extractable_keys(classification.main_output_dir),
        prefix_join(output_path, "documents"),
    )
    work = [
        {"arm": arm, "layout": str(layout), "table": str(table), "document": document}
        for arm, (layout, table) in _ARMS.items()
        for document in documents
    ]
    logger.info("Matrix: %d arms x %d documents = %d tasks", len(_ARMS), len(documents), len(work))

    texts_dir = prefix_join(output_path, "outputs/texts")
    pipeline = (
        Dataset.from_list(work)
        .flat_map(
            partial(
                extract_one,
                layout_model_path=layout_model.model_path,
                layout_label_map=layout_model.label_map,
            )
        )
        .write_parquet(
            prefix_join(texts_dir, "part-{shard:05d}-of-{total:05d}.parquet"),
            schema=_TEXT_SCHEMA,
            skip_existing=True,
        )
    )
    ZephyrContext(
        name="backend-matrix",
        resources=_WORKER_RESOURCES,
        max_workers=min(_MAX_WORKERS, len(work)),
        stage_runner_factory=SubprocessRunner,
        map_task_resources=_MAP_TASK_RESOURCES,
        heartbeat_timeout=_HEARTBEAT_TIMEOUT,
    ).execute(pipeline)

    by_arm: dict[str, dict[str, dict]] = {arm: {} for arm in _ARMS}
    text_fs, text_path = url_to_fs(texts_dir)
    for written in sorted(text_fs.glob(f"{text_path}/*.parquet")):
        with text_fs.open(written, "rb") as stream:
            for row in pq.read_table(stream).to_pylist():
                by_arm.setdefault(row["arm"], {})[row["source_id"]] = row

    baseline = by_arm[BASELINE]
    report = MatrixReport(
        documents=len({sid for rows in by_arm.values() for sid in rows}),
        arms={arm: _arm_stats(arm, rows, baseline) for arm, rows in by_arm.items()},
    )
    logger.info("=== BACKEND MATRIX ===")
    for arm, stats in report.arms.items():
        logger.info("  %s: %s", arm, stats.model_dump_json())
    return report


def comparison_step(source: StepSpec, classification: StepSpec, layout_model: StepSpec) -> StepSpec:
    return StepSpec(
        name="data/datakit/validate/backend_matrix",
        deps=[layout_model],
        hash_attrs={
            "source_output_path": source.output_path,
            "classification_output_path": classification.output_path,
            "source_shards": SOURCE_SHARDS,
            "documents_per_shard": DOCUMENTS_PER_SHARD,
            "arms": sorted(_ARMS),
            # Bump after any dependency or extractor change: the pipeline writes with
            # skip_existing=True and this hash covers neither, so earlier rows are inherited.
            "attempt": 1,
        },
        fn=remote(
            partial(
                compare,
                source_output_path=source.output_path,
                classification_output_path=classification.output_path,
                layout_model_output_path=layout_model.output_path,
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
