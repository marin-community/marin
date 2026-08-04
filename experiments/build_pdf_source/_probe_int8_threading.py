# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TEMPORARY -- explain the INT8 slowdown seen in the fleet matrix. DELETE once root-caused.

The matrix run measured INT8 layout at a 139-154s per-document median against FP32's 12-25s --
a ~10x inversion of the 1.21x speedup the zephyr-era tune step measured. Two hypotheses survive
the (failed) node archaeology, and this probe separates them on one pod:

* **Environment**: the converter pool exports ``OMP_NUM_THREADS=1`` (and friends) before spawning
  converters; the zephyr map tasks never did. If OpenVINO's CPU plugin loses its INT8 kernels or
  its threading under that pin, INT8 is slow on any x86 node.
* **Architecture**: CPU-only pods carry no arch constraint (see
  ``.agents/projects/2026-07-30_iris_pool_arch_selection.md``), and OpenVINO's ARM plugin does not
  take the INT8 path at all. The probe job requests the exact pod shape the matrix pools used
  (cpu=4, 32g), prints the arch it lands on, and every timing row carries it.

Six conditions, each in a fresh subprocess so the thread environment is set before any import:
{fp32, int8} x {pinned, unpinned} solo, then {fp32, int8} pinned with four concurrent converter
processes -- the pool's actual operating point. If int8/pinned/solo already shows the slowdown on
x86, the environment did it; if every x86 condition is fast, the matrix pools landed somewhere
these probes do not.

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \\
        --job-name probe-int8-threading \\
        -- python -m experiments.build_pdf_source._probe_int8_threading
"""

import json
import logging
import os
import pickle
import platform
import subprocess
import sys
import tempfile
import time
from functools import partial

import pyarrow.parquet as pq
from fray.types import ResourceConfig
from marin.execution.artifact import read_artifact
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from pydantic import BaseModel
from rigging.filesystem import url_to_fs
from rigging.log_setup import configure_logging

from experiments.build_pdf_source.classify import classify_step, model_step, routing_keys
from experiments.build_pdf_source.common import LayoutModelData, PdfClassificationData, PdfSourceData
from experiments.build_pdf_source.fetch import fetch_step
from experiments.build_pdf_source.layout_model import layout_model_step
from experiments.build_pdf_source.plan import plan_step

logger = logging.getLogger(__name__)

_DOCUMENTS = 8
_SOURCE_COLUMNS = ["pdf", "warc_filename", "warc_record_offset", "url"]

# The matrix pool pod shape, exactly: where this job lands is itself evidence.
_DRIVER_RESOURCES = ResourceConfig(cpu=4, ram="32g", disk="32g")

_PINNED_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
_RESULT_MARKER = "PROBE_RESULT "


class Condition(BaseModel):
    backend: str
    pinned: bool
    concurrency: int


class ConditionResult(BaseModel):
    condition: Condition
    arch: str
    build_seconds: float
    doc_seconds: list[float]
    mean_doc_seconds: float


class ThreadingReport(BaseModel):
    arch: str
    cpu_flags: str
    results: list[ConditionResult]


def _cpu_flags() -> str:
    """The instruction-set facts that decide INT8 throughput, from /proc/cpuinfo."""
    try:
        with open("/proc/cpuinfo") as stream:
            for line in stream:
                if line.startswith("flags") or line.startswith("Features"):
                    interesting = [
                        flag
                        for flag in line.split(":", 1)[1].split()
                        if "vnni" in flag or "amx" in flag or "avx512" in flag or flag in ("asimd", "sve")
                    ]
                    return " ".join(sorted(set(interesting))) or "none-of-interest"
    except OSError:
        pass
    return "unavailable"


def _child_main(payload_path: str) -> None:
    """One condition: build a converter, convert every document, print one JSON result line."""
    from experiments.build_pdf_source.docling_extract.converter import (  # noqa: PLC0415
        build_converter,
        extract_text,
    )

    with open(payload_path, "rb") as stream:
        options, rows = pickle.load(stream)
    started = time.perf_counter()
    converter = build_converter(options)
    build_seconds = time.perf_counter() - started
    doc_seconds = []
    for row in rows:
        started = time.perf_counter()
        extract_text(converter, row["pdf"], options, name=row["url"] or "document.pdf")
        doc_seconds.append(time.perf_counter() - started)
    print(
        _RESULT_MARKER
        + json.dumps({"arch": platform.machine(), "build_seconds": build_seconds, "doc_seconds": doc_seconds})
    )


def _run_condition(condition: Condition, payload_paths: dict[str, str]) -> ConditionResult:
    env = dict(os.environ)
    for key in _PINNED_ENV:
        env.pop(key, None)
    if condition.pinned:
        env.update(_PINNED_ENV)
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-m",
                "experiments.build_pdf_source._probe_int8_threading",
                payload_paths[condition.backend],
            ],
            env=env,
            stdout=subprocess.PIPE,
            text=True,
        )
        for _ in range(condition.concurrency)
    ]
    outputs = [process.communicate()[0] for process in processes]
    for process in processes:
        if process.returncode != 0:
            raise RuntimeError(f"Probe child exited with {process.returncode} for {condition}")
    parsed = []
    for output in outputs:
        [line] = [line for line in output.splitlines() if line.startswith(_RESULT_MARKER)]
        parsed.append(json.loads(line.removeprefix(_RESULT_MARKER)))
    doc_seconds = [second for result in parsed for second in result["doc_seconds"]]
    result = ConditionResult(
        condition=condition,
        arch=parsed[0]["arch"],
        build_seconds=max(result["build_seconds"] for result in parsed),
        doc_seconds=doc_seconds,
        mean_doc_seconds=sum(doc_seconds) / len(doc_seconds),
    )
    logger.info("CONDITION %s", result.model_dump())
    return result


def probe(
    output_path: str,
    source_output_path: str,
    classification_output_path: str,
    layout_model_output_path: str,
) -> ThreadingReport:
    from experiments.build_pdf_source.docling_extract.converter import ExtractionOptions  # noqa: PLC0415
    from experiments.build_pdf_source.docling_extract.model_spec import LayoutBackend, TableBackend  # noqa: PLC0415

    source = read_artifact(source_output_path, PdfSourceData)
    classification = read_artifact(classification_output_path, PdfClassificationData)
    layout_model = read_artifact(layout_model_output_path, LayoutModelData)
    keys = routing_keys(classification.main_output_dir, needs_ocr=False)

    filesystem, path = url_to_fs(source.main_output_dir)
    shard = sorted(filesystem.glob(f"{path}/*.parquet"))[0]
    with filesystem.open(shard, "rb") as stream:
        table = pq.read_table(stream, columns=_SOURCE_COLUMNS)
    rows = [row for row in table.to_pylist() if (row["warc_filename"], row["warc_record_offset"]) in keys][:_DOCUMENTS]

    # The table reader is held at PYMUPDF in every condition so the timings isolate the layout
    # backend, which is where the anomaly lives.
    arm_options = {
        "fp32": ExtractionOptions(table_backend=TableBackend.PYMUPDF, layout_backend=LayoutBackend.TORCH_HERON),
        "int8": ExtractionOptions(
            table_backend=TableBackend.PYMUPDF,
            layout_backend=LayoutBackend.INT8,
            layout_model_path=layout_model.model_path,
            layout_label_map=layout_model.label_map,
        ),
    }
    payload_paths = {}
    for backend, options in arm_options.items():
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as stream:
            pickle.dump((options, rows), stream)
            payload_paths[backend] = stream.name

    conditions = [
        Condition(backend="fp32", pinned=False, concurrency=1),
        Condition(backend="fp32", pinned=True, concurrency=1),
        Condition(backend="int8", pinned=False, concurrency=1),
        Condition(backend="int8", pinned=True, concurrency=1),
        Condition(backend="fp32", pinned=True, concurrency=4),
        Condition(backend="int8", pinned=True, concurrency=4),
    ]
    report = ThreadingReport(
        arch=platform.machine(),
        cpu_flags=_cpu_flags(),
        results=[_run_condition(condition, payload_paths) for condition in conditions],
    )
    logger.info("ARCH %s FLAGS %s", report.arch, report.cpu_flags)
    for result in report.results:
        logger.info(
            "SUMMARY %s/%s/x%d: mean %.1fs build %.1fs",
            result.condition.backend,
            "pinned" if result.condition.pinned else "unpinned",
            result.condition.concurrency,
            result.mean_doc_seconds,
            result.build_seconds,
        )
    return report


def probe_step(source: StepSpec, classification: StepSpec, layout_model: StepSpec) -> StepSpec:
    return StepSpec(
        name="data/datakit/validate/int8_threading_probe",
        deps=[layout_model],
        hash_attrs={
            "source_output_path": source.output_path,
            "classification_output_path": classification.output_path,
            "documents": _DOCUMENTS,
            "attempt": 1,
        },
        fn=remote(
            partial(
                probe,
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
    StepRunner().run([layout_model, probe_step(fetch, classify, layout_model)])


if __name__ == "__main__":
    if len(sys.argv) == 2:
        _child_main(sys.argv[1])
    else:
        main()
