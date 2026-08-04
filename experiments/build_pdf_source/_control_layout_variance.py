# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TEMPORARY -- measure the noise floor of the extractor, FP32 against FP32.

DELETE once the result is recorded in
``.agents/ops/2026-07-30-layout-quantization-recipe.md``. Nothing in the pipeline imports this.

:mod:`experiments.build_pdf_source._compare_layout_backends` measured INT8 against FP32 and found a
mean token F1 of 0.935. That number is only attributable to quantization if extracting the same
document twice on the *same* backend gives 1.0. Docling's default pipeline is threaded, so that is
an assumption, not a fact -- and an unmeasured noise floor is exactly the kind of thing that makes a
quantization result look worse than it is.

This is the control. Same documents, same shards, same resources, same code path; both arms are
:data:`LayoutBackend.TORCH_HERON`. The two arms are separate Zephyr tasks and therefore separate
processes, so whatever nondeterminism the pipeline has -- thread interleaving, hash ordering,
allocator-dependent tie-breaks in cluster postprocessing -- is free to show up.

Reading the result:

* mean token F1 of 1.0 means the extractor is deterministic and the entire 0.065 gap in the INT8
  comparison is the cost of quantization.
* anything below 1.0 is a noise floor that must be subtracted before the INT8 number is quoted, and
  the char-ratio spread says whether the noise is symmetric (harmless churn) or one-sided.

Run on the same x86 cluster the comparison used, so the two are comparable::

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \\
        --job-name control-layout-variance \\
        -- python -m experiments.build_pdf_source._control_layout_variance
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
from experiments.build_pdf_source.common import PdfClassificationData, PdfSourceData
from experiments.build_pdf_source.docling_extract.model_spec import LayoutBackend, TableBackend
from experiments.build_pdf_source.extract import text_extractable_keys
from experiments.build_pdf_source.fetch import fetch_step
from experiments.build_pdf_source.layout_model import layout_model_step
from experiments.build_pdf_source.plan import plan_step

logger = logging.getLogger(__name__)

# Matched to the INT8 comparison so the two reports describe the same corpus.
SOURCE_SHARDS = 24
DOCUMENTS_PER_SHARD = 25
_SOURCE_COLUMNS = ["pdf", "warc_filename", "warc_record_offset", "url"]
_AGREEMENT_THRESHOLD = 0.99

# Both arms are the same backend. The labels exist only to keep the two runs' rows apart.
_ARMS = ("a", "b")
_CONTROL_BACKEND = LayoutBackend.TORCH_HERON

_TEXT_SCHEMA = pa.schema(
    [
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("arm", pa.string(), nullable=False),
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


class ControlReport(BaseModel):
    """How far the extractor drifts from itself across two independent runs."""

    version: str = "v1"
    documents: int
    both_extracted: int
    one_arm_only: int
    both_failed: int
    failures: dict[str, int]
    exact_match_fraction: float
    mean_token_f1: float
    median_token_f1: float
    p10_token_f1: float
    min_token_f1: float
    agreeing_fraction: float
    mean_char_ratio: float
    max_char_ratio_deviation: float
    seconds_per_document: float


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


def extract_shard(work: dict, keys: frozenset[tuple[str, int]]) -> Iterator[dict]:
    """Extract one shard's text-extractable documents on the control backend.

    Deliberately identical to the comparison harness's per-shard extraction, minus the backend
    switch: the point of a control is that nothing differs between the arms except which task ran.
    """
    from experiments.build_pdf_source.docling_extract.converter import (  # noqa: PLC0415
        ExtractionOptions,
        build_converter,
        extract_text,
    )

    options = ExtractionOptions(
        table_backend=TableBackend.PYMUPDF,
        layout_backend=_CONTROL_BACKEND,
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
            "arm": work["arm"],
            "url": row["url"],
            "text": None,
            "error": None,
        }
        started = time.perf_counter()
        try:
            extracted = extract_text(converter, row["pdf"], options, name=row["url"] or "doc.pdf")
        except Exception as error:
            counters.pipeline.update_counter(f"layout_control/{work['arm']}_failed", 1)
            logger.warning("arm %s extraction failed for %s: %s", work["arm"], row["url"], error, exc_info=True)
            yield output | {"error": f"{type(error).__name__}: {error}", "seconds": time.perf_counter() - started}
            continue

        stripped = strip_document_boilerplate(extracted.text, extracted.page_offsets, boilerplate)
        counters.pipeline.update_counter(f"layout_control/{work['arm']}_extracted", 1)
        yield output | {"text": stripped.text, "seconds": time.perf_counter() - started}


def _compare_arms(texts: dict[str, dict[str, dict]]) -> ControlReport:
    """Score arm A against arm B, document by document."""
    first_arm, second_arm = _ARMS
    exact = 0
    scores: list[float] = []
    ratios: list[float] = []
    seconds: list[float] = []
    one_arm_only = both_failed = 0
    failures: Counter = Counter()

    for rows in texts.values():
        first, second = rows.get(first_arm), rows.get(second_arm)
        if first is None or second is None:
            continue
        seconds.extend([first["seconds"], second["seconds"]])
        for row in (first, second):
            if row["error"]:
                failures[row["error"].partition(":")[0]] += 1

        if first["text"] is None and second["text"] is None:
            both_failed += 1
        elif first["text"] is None or second["text"] is None:
            one_arm_only += 1
        else:
            scores.append(_token_f1(first["text"], second["text"]))
            ratios.append(len(second["text"]) / len(first["text"]) if first["text"] else 1.0)
            exact += _normalized(first["text"]) == _normalized(second["text"])

    if not scores:
        raise RuntimeError(
            "No document extracted on both arms -- a harness bug, not a measurement. "
            f"failures={dict(failures)}, one_arm_only={one_arm_only}, both_failed={both_failed}"
        )

    return ControlReport(
        documents=len(texts),
        both_extracted=len(scores),
        one_arm_only=one_arm_only,
        both_failed=both_failed,
        failures=dict(failures),
        exact_match_fraction=exact / len(scores),
        mean_token_f1=float(np.mean(scores)),
        median_token_f1=float(np.median(scores)),
        p10_token_f1=float(np.percentile(scores, 10)),
        min_token_f1=float(np.min(scores)),
        agreeing_fraction=float(np.mean([score >= _AGREEMENT_THRESHOLD for score in scores])),
        mean_char_ratio=float(np.mean(ratios)),
        max_char_ratio_deviation=float(np.max(np.abs(np.array(ratios) - 1.0))),
        seconds_per_document=float(np.mean(seconds)),
    )


def control(
    output_path: str,
    source_output_path: str,
    classification_output_path: str,
) -> ControlReport:
    """Extract the same documents twice on the same backend, then score the two runs."""
    source = read_artifact(source_output_path, PdfSourceData)
    classification = read_artifact(classification_output_path, PdfClassificationData)

    filesystem, path = url_to_fs(source.main_output_dir)
    protocol = source.main_output_dir.split("://")[0] if "://" in source.main_output_dir else ""
    shards = sorted(filesystem.glob(f"{path}/*.parquet"))[:SOURCE_SHARDS]
    if not shards:
        raise RuntimeError(f"No fetched PDFs under {source.main_output_dir}")
    shards = [f"{protocol}://{shard}" if protocol else shard for shard in shards]

    work = [{"arm": arm, "shard": shard} for arm in _ARMS for shard in shards]
    logger.info("Extracting %d shards on %d arms as %d tasks", len(shards), len(_ARMS), len(work))

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
        name="layout-control-variance",
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

    report = _compare_arms(texts)
    logger.info("=== EXTRACTOR RUN-TO-RUN VARIANCE (FP32 vs FP32) ===")
    for key, value in report.model_dump().items():
        logger.info("  %s: %s", key, value)
    return report


def control_step(source: StepSpec, classification: StepSpec, layout_model: StepSpec) -> StepSpec:
    """Build the control step.

    Depends on the layout-model step only so the run is scheduled behind it, matching the
    comparison's shape; the control itself never loads the quantized graph.
    """
    return StepSpec(
        name="data/datakit/validate/layout_control_variance",
        deps=[layout_model],
        hash_attrs={
            "source_output_path": source.output_path,
            "classification_output_path": classification.output_path,
            "source_shards": SOURCE_SHARDS,
            "documents_per_shard": DOCUMENTS_PER_SHARD,
            # Bump this after any dependency change. The pipeline writes with skip_existing=True
            # and the step hash says nothing about the environment, so a run that failed on a bad
            # lock leaves error rows in this prefix that every later run silently inherits --
            # which is exactly how attempt 1 reported 600 ImportErrors that a killed predecessor
            # had written eight minutes earlier.
            "attempt": 2,
        },
        fn=remote(
            partial(
                control,
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
    StepRunner().run([layout_model, control_step(fetch, classify, layout_model)])


if __name__ == "__main__":
    main()
