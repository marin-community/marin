# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TEMPORARY -- convert 100 corpus documents with INT8 + TableFormer on a guaranteed-x86 pod.

DELETE once the int8-vs-fp32 TableFormer comparison is recorded. Nothing imports this.

The adaptive fleet converts x86-landed documents with INT8 + TableFormer, but that cell's text has
only ever been compared against the fp32 + PyMuPDF baseline, never head-to-head against the
fp32 + TableFormer text it actually replaces -- and the matrix's INT8 texts came from ARM nodes
with 51 timeout truncations mixed in. This job produces clean INT8 + TableFormer texts for the
first 100 corpus documents on a verified x86 pod (it fails fast on any other arch rather than
report numbers from the wrong instruction set), written as a parquet an offline comparison can
join against the matrix's ``fp32_tableformer`` texts by ``source_id``.

Four converter subprocesses on a cpu=4 pod, mirroring the pool's operating point.

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \\
        --job-name int8-quality-sample \\
        -- python -m experiments.build_pdf_source._int8_quality_sample
"""

import logging
import pickle
import platform
import subprocess
import sys
import tempfile
import time
from functools import partial

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

from experiments.build_pdf_source.classify import classify_step, model_step, routing_keys
from experiments.build_pdf_source.common import LayoutModelData, PdfClassificationData, PdfSourceData

# INT8 explicitly, not the fleet's X86_LAYOUT_BACKEND: the fleet retired INT8 on this sample's
# evidence, and rerunning the sample must keep measuring INT8 rather than silently measuring FP32.
from experiments.build_pdf_source.docling_extract.model_spec import LayoutBackend
from experiments.build_pdf_source.extract_fleet import TABLE_BACKEND
from experiments.build_pdf_source.fetch import fetch_step
from experiments.build_pdf_source.layout_model import layout_model_step
from experiments.build_pdf_source.plan import plan_step

logger = logging.getLogger(__name__)

_DOCUMENTS = 100
_WORKERS = 4
_SOURCE_COLUMNS = ["pdf", "warc_filename", "warc_record_offset", "url"]

_DRIVER_RESOURCES = ResourceConfig(cpu=4, ram="32g", disk="32g")

_TEXT_SCHEMA = pa.schema(
    [
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("url", pa.string(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("error", pa.string(), nullable=True),
        pa.field("seconds", pa.float64(), nullable=False),
        pa.field("text", pa.string(), nullable=True),
    ]
)


class SampleReport(BaseModel):
    arch: str
    documents: int
    converted: int
    failed: int
    mean_seconds: float
    texts_path: str


def _child_main(payload_path: str, results_path: str) -> None:
    """Convert one slice of the sample and dump the rows for the parent to aggregate."""
    from experiments.build_pdf_source.docling_extract.converter import (  # noqa: PLC0415
        build_converter,
        extract_text,
    )

    with open(payload_path, "rb") as stream:
        options, rows = pickle.load(stream)
    converter = build_converter(options)
    results = []
    for row in rows:
        source_id = f"{row['warc_filename']}:{row['warc_record_offset']}"
        started = time.perf_counter()
        try:
            extracted = extract_text(converter, row["pdf"], options, name=row["url"] or "document.pdf")
        except Exception as exc:
            results.append(
                {
                    "source_id": source_id,
                    "url": row["url"],
                    "status": "failure",
                    "error": f"{type(exc).__name__}: {exc}",
                    "seconds": time.perf_counter() - started,
                    "text": None,
                }
            )
            continue
        results.append(
            {
                "source_id": source_id,
                "url": row["url"],
                "status": extracted.status,
                "error": extracted.extraction_error,
                "seconds": time.perf_counter() - started,
                "text": extracted.text,
            }
        )
    with open(results_path, "wb") as stream:
        pickle.dump(results, stream)


def sample(
    output_path: str,
    source_output_path: str,
    classification_output_path: str,
    layout_model_output_path: str,
) -> SampleReport:
    from experiments.build_pdf_source.docling_extract.converter import ExtractionOptions  # noqa: PLC0415

    arch = platform.machine()
    if arch != "x86_64":
        # Numbers from the wrong instruction set are worse than no numbers: OpenVINO has no INT8
        # path on ARM. CPU pods carry no arch constraint, so rerun until placement cooperates.
        raise RuntimeError(f"Landed on {arch}; this sample must run on x86_64 -- rerun the job")

    source = read_artifact(source_output_path, PdfSourceData)
    classification = read_artifact(classification_output_path, PdfClassificationData)
    layout_model = read_artifact(layout_model_output_path, LayoutModelData)
    keys = routing_keys(classification.main_output_dir, needs_ocr=False)

    filesystem, path = url_to_fs(source.main_output_dir)
    rows: list[dict] = []
    for shard in sorted(filesystem.glob(f"{path}/*.parquet")):
        if len(rows) >= _DOCUMENTS:
            break
        with filesystem.open(shard, "rb") as stream:
            table = pq.read_table(stream, columns=_SOURCE_COLUMNS)
        for row in table.to_pylist():
            if len(rows) >= _DOCUMENTS:
                break
            if (row["warc_filename"], row["warc_record_offset"]) in keys:
                rows.append(row)

    options = ExtractionOptions(
        table_backend=TABLE_BACKEND,
        layout_backend=LayoutBackend.INT8,
        layout_model_path=layout_model.model_path,
        layout_label_map=layout_model.label_map,
    )
    slices = [rows[index::_WORKERS] for index in range(_WORKERS)]
    children = []
    for rows_slice in slices:
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as stream:
            pickle.dump((options, rows_slice), stream)
            payload_path = stream.name
        results_path = f"{payload_path}.results"
        children.append(
            (
                subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "experiments.build_pdf_source._int8_quality_sample",
                        payload_path,
                        results_path,
                    ],
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                ),
                results_path,
            )
        )
    results: list[dict] = []
    for process, results_path in children:
        if process.wait() != 0:
            raise RuntimeError(f"Sample worker exited with {process.returncode}")
        with open(results_path, "rb") as stream:
            results.extend(pickle.load(stream))

    destination = prefix_join(output_path, "texts/int8_tableformer_x86.parquet")
    destination_filesystem, destination_path = url_to_fs(destination)
    with destination_filesystem.open(destination_path, "wb") as stream:
        pq.write_table(pa.Table.from_pylist(results, schema=_TEXT_SCHEMA), stream)

    converted = [result for result in results if result["text"] is not None]
    seconds = [result["seconds"] for result in results]
    report = SampleReport(
        arch=arch,
        documents=len(results),
        converted=len(converted),
        failed=len(results) - len(converted),
        mean_seconds=sum(seconds) / len(seconds),
        texts_path=destination,
    )
    logger.info("SAMPLE %s", report.model_dump())
    return report


def sample_step(source: StepSpec, classification: StepSpec, layout_model: StepSpec) -> StepSpec:
    return StepSpec(
        name="data/datakit/validate/int8_quality_sample",
        deps=[layout_model],
        hash_attrs={
            "source_output_path": source.output_path,
            "classification_output_path": classification.output_path,
            "documents": _DOCUMENTS,
            "attempt": 1,
        },
        fn=remote(
            partial(
                sample,
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
    StepRunner().run([layout_model, sample_step(fetch, classify, layout_model)])


if __name__ == "__main__":
    if len(sys.argv) == 3:
        _child_main(sys.argv[1], sys.argv[2])
    else:
        main()
