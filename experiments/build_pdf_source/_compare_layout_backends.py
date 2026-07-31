# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TEMPORARY -- measure what quantizing the layout model costs in extracted text.

DELETE once the result is recorded in
``.agents/ops/2026-07-30-layout-quantization-recipe.md``. Nothing in the pipeline imports this.

The recipe sweep in :mod:`experiments.build_pdf_source._tune_layout_quantization` compared INT8
against FP32 at the level of *detections*: 98% of FP32 boxes reproduced, but only 78% of them
keeping their label. Whether that matters is not answerable there. Detections are taken at a 0.3
score threshold, so a page's ~31 of them include many low-confidence overlapping boxes where a
label flip costs nothing, and docling runs cluster postprocessing afterwards. What matters is the
text, so this measures the text: the same documents through the whole extractor twice, once on
each layout backend.

Extraction fans out over the cluster rather than running in the driver. A document averages ~17
pages and a page costs about a second, so 600 documents on two backends is roughly nine CPU-hours
-- a day's work serially, a few minutes across the fleet. Every ``(backend, shard)`` pair is one
Zephyr task, so both backends run concurrently and a task builds exactly one converter, which
matters because building one compiles the layout graph. The driver then reads the texts back and
does the comparison, which is cheap.

Reported per document:

``exact``
    Whitespace-normalised texts identical. The strictest reading, and the one that will be low even
    for a harmless difference.
``token_f1``
    Multiset F1 over whitespace tokens. Insensitive to reordering, which is what a changed reading
    order produces, and sensitive to text actually gained or lost.
``char_ratio``
    INT8 length over FP32 length. Catches a backend systematically dropping or duplicating regions
    even when the tokens that survive agree.

Run on an x86 cluster -- ``cw-us-east-02a``, whose CPU pool is Genoa and whose region already holds
the corpus::

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \\
        --job-name compare-layout-backends \\
        -- python -m experiments.build_pdf_source._compare_layout_backends
"""

import logging
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
from experiments.build_pdf_source.common import LayoutModelData, PdfClassificationData, PdfSourceData
from experiments.build_pdf_source.docling_extract.model_spec import LayoutBackend, TableBackend
from experiments.build_pdf_source.extract import text_extractable_keys
from experiments.build_pdf_source.fetch import fetch_step
from experiments.build_pdf_source.layout_model import layout_model_step
from experiments.build_pdf_source.plan import plan_step

logger = logging.getLogger(__name__)

# 24 shards x 25 documents = 600 documents, run as 48 tasks (two backends per shard).
SOURCE_SHARDS = 24
DOCUMENTS_PER_SHARD = 25
_SOURCE_COLUMNS = ["pdf", "warc_filename", "warc_record_offset", "url"]
# Agreement at or above this token F1 is treated as "the same document" for the headline number.
_AGREEMENT_THRESHOLD = 0.99

_TEXT_SCHEMA = pa.schema(
    [
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("backend", pa.string(), nullable=False),
        pa.field("url", pa.string(), nullable=False),
        pa.field("text", pa.string(), nullable=True),
        pa.field("error", pa.string(), nullable=True),
        pa.field("seconds", pa.float64(), nullable=False),
    ]
)

_DRIVER_RESOURCES = ResourceConfig(cpu=4, ram="32g", disk="16g")
_WORKER_RESOURCES = ResourceConfig(cpu=8, ram="64g", disk="64g")
_MAP_TASK_RESOURCES = ResourceConfig(cpu=1, ram="7g", disk="4g")
_MAX_WORKERS = 28
_HEARTBEAT_TIMEOUT = 30 * 60


class ComparisonReport(BaseModel):
    """How far INT8 extraction drifted from FP32 extraction, in text."""

    version: str = "v1"
    documents: int
    both_extracted: int
    fp32_only: int
    int8_only: int
    both_failed: int
    fp32_failures: dict[str, int]
    int8_failures: dict[str, int]
    exact_match_fraction: float
    mean_token_f1: float
    median_token_f1: float
    p10_token_f1: float
    agreeing_fraction: float
    mean_char_ratio: float
    fp32_seconds_per_document: float
    int8_seconds_per_document: float


def _normalized(text: str) -> str:
    return " ".join(text.split())


def _token_f1(first: str, second: str) -> float:
    """Multiset F1 over whitespace tokens; 1.0 when both are empty."""
    left, right = Counter(first.split()), Counter(second.split())
    if not left and not right:
        return 1.0
    overlap = sum((left & right).values())
    if not overlap:
        return 0.0
    precision = overlap / sum(right.values())
    recall = overlap / sum(left.values())
    return 2 * precision * recall / (precision + recall)


def extract_shard(
    work: dict,
    keys: frozenset[tuple[str, int]],
    layout_model_path: str,
    layout_label_map: dict[int, str],
) -> Iterator[dict]:
    """Extract one shard's text-extractable documents on one layout backend.

    One converter per task, built once here rather than per document, because building it compiles
    the layout graph. A per-document failure is recorded on its row rather than raised, so one bad
    PDF does not cost the shard -- but the row carries the error so the driver can tell a real
    disagreement from a backend that could not run at all.
    """
    from experiments.build_pdf_source.docling_extract.converter import (  # noqa: PLC0415
        ExtractionOptions,
        build_converter,
        extract_text,
    )

    backend = LayoutBackend(work["backend"])
    options = ExtractionOptions(
        table_backend=TableBackend.PYMUPDF,
        layout_backend=backend,
        layout_model_path=layout_model_path if backend is LayoutBackend.INT8 else None,
        layout_label_map=layout_label_map if backend is LayoutBackend.INT8 else {},
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
            "backend": str(backend),
            "url": row["url"],
            "text": None,
            "error": None,
        }
        started = time.perf_counter()
        try:
            extracted = extract_text(converter, row["pdf"], options, name=row["url"] or "doc.pdf")
        except Exception as error:
            counters.pipeline.update_counter(f"layout_compare/{backend}_failed", 1)
            logger.warning("%s extraction failed for %s: %s", backend, row["url"], error, exc_info=True)
            yield output | {"error": f"{type(error).__name__}: {error}", "seconds": time.perf_counter() - started}
            continue

        stripped = strip_document_boilerplate(extracted.text, extracted.page_offsets, boilerplate)
        counters.pipeline.update_counter(f"layout_compare/{backend}_extracted", 1)
        yield output | {"text": stripped.text, "seconds": time.perf_counter() - started}


def _compare_texts(texts: dict[str, dict[str, dict]]) -> ComparisonReport:
    """Join the two backends' texts by document and score the agreement."""
    fp32_key, int8_key = str(LayoutBackend.TORCH_HERON), str(LayoutBackend.INT8)
    exact = 0
    scores: list[float] = []
    ratios: list[float] = []
    fp32_only = int8_only = both_failed = 0
    fp32_failures: Counter = Counter()
    int8_failures: Counter = Counter()
    fp32_seconds: list[float] = []
    int8_seconds: list[float] = []

    for rows in texts.values():
        fp32, int8 = rows.get(fp32_key), rows.get(int8_key)
        if fp32 is None or int8 is None:
            # Only one backend reached this document at all; not a comparison.
            continue
        fp32_seconds.append(fp32["seconds"])
        int8_seconds.append(int8["seconds"])
        if fp32["error"]:
            fp32_failures[fp32["error"].partition(":")[0]] += 1
        if int8["error"]:
            int8_failures[int8["error"].partition(":")[0]] += 1

        if fp32["text"] is None and int8["text"] is None:
            both_failed += 1
        elif int8["text"] is None:
            fp32_only += 1
        elif fp32["text"] is None:
            int8_only += 1
        else:
            scores.append(_token_f1(fp32["text"], int8["text"]))
            ratios.append(len(int8["text"]) / len(fp32["text"]) if fp32["text"] else 1.0)
            exact += _normalized(fp32["text"]) == _normalized(int8["text"])

    if not scores:
        raise RuntimeError(
            "No document extracted on both backends -- a harness or backend bug, not a measurement. "
            f"fp32 failures={dict(fp32_failures)}, int8 failures={dict(int8_failures)}, "
            f"fp32_only={fp32_only}, int8_only={int8_only}, both_failed={both_failed}"
        )

    return ComparisonReport(
        documents=len(texts),
        both_extracted=len(scores),
        fp32_only=fp32_only,
        int8_only=int8_only,
        both_failed=both_failed,
        fp32_failures=dict(fp32_failures),
        int8_failures=dict(int8_failures),
        exact_match_fraction=exact / len(scores),
        mean_token_f1=float(np.mean(scores)),
        median_token_f1=float(np.median(scores)),
        p10_token_f1=float(np.percentile(scores, 10)),
        agreeing_fraction=float(np.mean([score >= _AGREEMENT_THRESHOLD for score in scores])),
        mean_char_ratio=float(np.mean(ratios)),
        fp32_seconds_per_document=float(np.mean(fp32_seconds)),
        int8_seconds_per_document=float(np.mean(int8_seconds)),
    )


def compare(
    output_path: str,
    source_output_path: str,
    classification_output_path: str,
    layout_model_output_path: str,
) -> ComparisonReport:
    """Extract the same documents on both layout backends across the fleet, then compare."""
    source = read_artifact(source_output_path, PdfSourceData)
    classification = read_artifact(classification_output_path, PdfClassificationData)
    layout_model = read_artifact(layout_model_output_path, LayoutModelData)

    filesystem, path = url_to_fs(source.main_output_dir)
    protocol = source.main_output_dir.split("://")[0] if "://" in source.main_output_dir else ""
    shards = sorted(filesystem.glob(f"{path}/*.parquet"))[:SOURCE_SHARDS]
    if not shards:
        raise RuntimeError(f"No fetched PDFs under {source.main_output_dir}")
    shards = [f"{protocol}://{shard}" if protocol else shard for shard in shards]

    work = [
        {"backend": str(backend), "shard": shard}
        for backend in (LayoutBackend.TORCH_HERON, LayoutBackend.INT8)
        for shard in shards
    ]
    logger.info("Extracting %d shards on both backends as %d tasks", len(shards), len(work))

    texts_dir = prefix_join(output_path, "outputs/texts")
    pipeline = (
        Dataset.from_list(work)
        .flat_map(
            partial(
                extract_shard,
                keys=text_extractable_keys(classification.main_output_dir),
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
        name="layout-backend-compare",
        resources=_WORKER_RESOURCES,
        max_workers=min(_MAX_WORKERS, len(work)),
        stage_runner_factory=SubprocessRunner,
        map_task_resources=_MAP_TASK_RESOURCES,
        heartbeat_timeout=_HEARTBEAT_TIMEOUT,
    ).execute(pipeline)

    texts: dict[str, dict[str, dict]] = {}
    text_fs, text_path = url_to_fs(texts_dir)
    for written in sorted(text_fs.glob(f"{text_path}/*.parquet")):
        with text_fs.open(written, "rb") as stream:
            for row in pq.read_table(stream).to_pylist():
                texts.setdefault(row["source_id"], {})[row["backend"]] = row

    report = _compare_texts(texts)
    logger.info("=== LAYOUT BACKEND TEXT AGREEMENT ===")
    for key, value in report.model_dump().items():
        logger.info("  %s: %s", key, value)
    return report


def comparison_step(source: StepSpec, classification: StepSpec, layout_model: StepSpec) -> StepSpec:
    """Build the comparison step."""
    return StepSpec(
        name="data/datakit/validate/layout_backend_text_agreement",
        deps=[layout_model],
        hash_attrs={
            "source_output_path": source.output_path,
            "classification_output_path": classification.output_path,
            "source_shards": SOURCE_SHARDS,
            "documents_per_shard": DOCUMENTS_PER_SHARD,
            # 5: rerun on the fixed lock. Attempts 1-4 ran against a uv.lock generated with
            # --prerelease=allow, which gave the datakit extra torch 2.13.0 with no nvidia-nccl at
            # all; that cost ~17% of extractions to an ImportError and, more importantly, ran the
            # FP32 layout model on a different torch than the control does.
            "attempt": 5,
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
    # Only the layout model is a dependency; fetch and classify are named by their resolved paths
    # so a missing artifact fails fast instead of relaunching a 411 GiB download.
    StepRunner().run([layout_model, comparison_step(fetch, classify, layout_model)])


if __name__ == "__main__":
    main()
