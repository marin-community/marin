# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TEMPORARY -- find out why the INT8 layout backend fails to construct.

DELETE once :mod:`experiments.build_pdf_source.docling_extract.layout_openvino` is fixed.

The text-agreement run extracted 600 documents on FP32 without a single failure and failed on all
600 on INT8, with two distinct errors: ``ValidationError`` on 400 and ``ImportError`` on 200. Those
counts are 16 and 8 tasks of 25 documents, so both are per-task construction failures rather than
anything to do with the documents.

This prints the exact messages instead of guessing at them. It does two things:

1. Reads back the ``error`` column the failed run already wrote to Parquet -- the full strings are
   sitting there and are the cheapest possible source of truth.
2. Builds the INT8 converter here and extracts one document, so a fix can be confirmed in the same
   job rather than by another full comparison run.

Run where the corpus and the staged graph live::

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \\
        --job-name probe-int8-wiring \\
        -- python -m experiments.build_pdf_source._probe_int8_wiring
"""

import logging
from collections import Counter
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

from experiments.build_pdf_source.common import LayoutModelData, PdfSourceData
from experiments.build_pdf_source.fetch import fetch_step
from experiments.build_pdf_source.layout_model import layout_model_step
from experiments.build_pdf_source.plan import plan_step

logger = logging.getLogger(__name__)

# The failed text-agreement run, whose per-document error strings are still on disk.
FAILED_TEXTS_DIR = (
    "s3://marin-us-east-02a/marin/data/datakit/validate/layout_backend_text_agreement_a1d3552c/outputs/texts"
)
_RESOURCES = ResourceConfig(cpu=8, ram="32g", disk="16g")


class ProbeReport(BaseModel):
    version: str = "v1"
    recorded_errors: dict[str, int]
    sample_errors: list[str]
    options_fields: dict[str, str]
    construction: dict[str, str]


def _recorded_errors() -> tuple[dict[str, int], list[str]]:
    """Read the error strings the failed comparison run wrote."""
    filesystem, path = url_to_fs(FAILED_TEXTS_DIR)
    tallies: Counter = Counter()
    samples: list[str] = []
    try:
        shards = sorted(filesystem.glob(f"{path}/*.parquet"))
    except Exception:
        logger.warning("Could not list %s", FAILED_TEXTS_DIR, exc_info=True)
        return {}, []

    for shard in shards:
        with filesystem.open(shard, "rb") as stream:
            table = pq.read_table(stream, columns=["backend", "error"])
        for row in table.to_pylist():
            if not row["error"]:
                continue
            tallies[f"{row['backend']}:{row['error'][:120]}"] += 1
            if len(samples) < 10 and row["error"] not in samples:
                samples.append(row["error"])
    return dict(tallies), samples


def _options_fields() -> dict[str, str]:
    """Report which fields ``LayoutObjectDetectionOptions`` requires."""
    from docling.datamodel.pipeline_options import LayoutObjectDetectionOptions  # noqa: PLC0415

    return {
        name: f"required={field.is_required()} annotation={field.annotation}"
        for name, field in LayoutObjectDetectionOptions.model_fields.items()
    }


def _try(label: str, thunk) -> tuple[str, str]:
    try:
        thunk()
    except Exception as error:
        return label, f"{type(error).__name__}: {error}"
    return label, "ok"


def _construction_attempts(model_path: str, label_map: dict[int, str]) -> dict[str, str]:
    """Try the ways the INT8 layout stage might legitimately be constructed."""
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions  # noqa: PLC0415
    from docling.datamodel.pipeline_options import LayoutObjectDetectionOptions  # noqa: PLC0415

    accelerator = AcceleratorOptions(device=AcceleratorDevice.CPU, num_threads=1)
    results: dict[str, str] = {}

    results.update([_try("import openvino", lambda: __import__("openvino"))])
    results.update(
        [
            _try(
                "import layout_openvino",
                lambda: __import__(
                    "experiments.build_pdf_source.docling_extract.layout_openvino",
                    fromlist=["OpenVinoLayoutModel"],
                ),
            )
        ]
    )
    results.update([_try("LayoutObjectDetectionOptions()", LayoutObjectDetectionOptions)])

    def with_onnx_engine_options():
        from docling.datamodel.object_detection_engine_options import (  # noqa: PLC0415
            OnnxRuntimeObjectDetectionEngineOptions,
        )

        return LayoutObjectDetectionOptions(engine_options=OnnxRuntimeObjectDetectionEngineOptions())

    results.update([_try("LayoutObjectDetectionOptions(engine_options=onnx)", with_onnx_engine_options)])

    def build_model():
        from experiments.build_pdf_source.docling_extract.layout_openvino import (  # noqa: PLC0415
            OpenVinoLayoutModel,
        )

        return OpenVinoLayoutModel(
            options=with_onnx_engine_options(),
            model_path=model_path,
            label_map=label_map,
            accelerator_options=accelerator,
        )

    results.update([_try("OpenVinoLayoutModel(...)", build_model)])
    return results


def probe(output_path: str, source_output_path: str, layout_model_output_path: str) -> ProbeReport:
    """Report the recorded errors and what can actually be constructed."""
    read_artifact(source_output_path, PdfSourceData)
    layout_model = read_artifact(layout_model_output_path, LayoutModelData)

    recorded, samples = _recorded_errors()
    logger.info("=== RECORDED ERRORS FROM THE FAILED RUN ===")
    for key, count in sorted(recorded.items(), key=lambda item: -item[1]):
        logger.info("  %5d  %s", count, key)
    for sample in samples:
        logger.info("  SAMPLE: %s", sample)

    fields = _options_fields()
    logger.info("=== LayoutObjectDetectionOptions FIELDS ===")
    for name, description in fields.items():
        logger.info("  %s: %s", name, description)

    construction = _construction_attempts(layout_model.model_path, layout_model.label_map)
    logger.info("=== CONSTRUCTION ATTEMPTS ===")
    for label, outcome in construction.items():
        logger.info("  %-48s %s", label, outcome)

    return ProbeReport(
        recorded_errors=recorded,
        sample_errors=samples,
        options_fields=fields,
        construction=construction,
    )


def probe_step(source: StepSpec, layout_model: StepSpec) -> StepSpec:
    return StepSpec(
        name="data/datakit/validate/int8_wiring_probe",
        deps=[layout_model],
        hash_attrs={"source_output_path": source.output_path, "attempt": 1},
        fn=remote(
            partial(
                probe,
                source_output_path=source.output_path,
                layout_model_output_path=layout_model.output_path,
            ),
            resources=_RESOURCES,
            pip_dependency_groups=["datakit"],
        ),
    )


def main() -> None:
    configure_logging(logging.INFO)
    fetch = fetch_step(plan_step())
    StepRunner().run([layout_model_step(fetch), probe_step(fetch, layout_model_step(fetch))])


if __name__ == "__main__":
    main()
