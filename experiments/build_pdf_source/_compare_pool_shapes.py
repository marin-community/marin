# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TEMPORARY -- 64 pods x 4 converters against 256 pods x 1, on the production arm.

DELETE once the operating point is chosen. Nothing in the pipeline imports this.

Same corpus, same fp32 + TableFormer configuration, same 256-converter budget -- only the pod
shape varies. Docling's multicore scaling within a pod is suspected sublinear (and the matrix's
INT8 arms showed how much in-pod contention can matter), so the question is whether four
single-converter cores in one 4-cpu pod pay a sharing tax that 256 single-cpu pods do not.
Compare shapes on per-document ``seconds`` for compute cost and on ``fleet_ready_seconds`` for
the scheduling cost of 4x as many pods; ``convert_wall_seconds`` over 600 documents is tail-bound
(~3 waves) and only weakly informative.

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \\
        --job-name compare-pool-shapes \\
        -- python -m experiments.build_pdf_source._compare_pool_shapes
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial

from fray.types import ResourceConfig
from marin.execution.artifact import read_artifact
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.inference.converter_pool import remote_converter_pool
from pydantic import BaseModel
from rigging.log_setup import configure_logging

from experiments.build_pdf_source._compare_matrix_fleet import _arm_options, _load_corpus
from experiments.build_pdf_source.classify import classify_step, model_step
from experiments.build_pdf_source.common import LayoutModelData, PdfClassificationData, PdfSourceData
from experiments.build_pdf_source.docling_extract.service import build_handler
from experiments.build_pdf_source.extract_fleet import MODEL_ID, build_pool_config, convert_document
from experiments.build_pdf_source.fetch import fetch_step
from experiments.build_pdf_source.layout_model import layout_model_step
from experiments.build_pdf_source.plan import plan_step

logger = logging.getLogger(__name__)

# The production arm from the matrix: fp32 layout, TableFormer tables.
_ARM = "fp32_tableformer"
_IN_FLIGHT = 384

# Equal converter budgets, equal aggregate RAM; only the packing differs.
_SHAPES: dict[str, tuple[int, int, ResourceConfig]] = {
    "64x4": (64, 4, ResourceConfig(cpu=4, ram="32g", disk="32g")),
    "256x1": (256, 1, ResourceConfig(cpu=1, ram="8g", disk="16g")),
}

_DRIVER_RESOURCES = ResourceConfig(cpu=8, ram="32g", disk="16g")


class ShapeStats(BaseModel):
    instances: int
    processes_per_instance: int
    documents: int
    converted: int
    failed: int
    lost: int
    mean_seconds: float
    p50_seconds: float
    p90_seconds: float
    fleet_ready_seconds: float
    convert_wall_seconds: float


class ShapeReport(BaseModel):
    arm: str
    shapes: dict[str, ShapeStats]


def _run_shape(name: str, rows: list[dict], layout_model: LayoutModelData) -> ShapeStats:
    instances, processes, resources = _SHAPES[name]
    pool_config = replace(
        build_pool_config(partial(build_handler, _arm_options(_ARM, layout_model))),
        model_id=f"{MODEL_ID}-shape-{name}",
        instances=instances,
        processes_per_instance=processes,
        worker_resources=resources,
    )
    launch_started = time.monotonic()
    outcomes: list[tuple[str, float]] = []
    with remote_converter_pool(pool_config) as session:
        fleet_ready = time.monotonic() - launch_started
        logger.info("[%s] fleet ready in %.0fs at %s", name, fleet_ready, session.endpoint.base_url)
        convert_started = time.monotonic()
        with ThreadPoolExecutor(max_workers=_IN_FLIGHT) as pool:
            futures = [pool.submit(convert_document, session.endpoint.base_url, _IN_FLIGHT, row) for row in rows]
            for future, row in zip(futures, rows, strict=True):
                try:
                    document = future.result()
                except Exception as error:
                    logger.warning("[%s] request lost for %s: %s", name, row["url"], error)
                    outcomes.append(("lost", 0.0))
                    continue
                outcomes.append((document.status, document.seconds))
        wall = time.monotonic() - convert_started
        session.check_alive()

    seconds = sorted(second for status, second in outcomes if status != "lost")
    stats = ShapeStats(
        instances=instances,
        processes_per_instance=processes,
        documents=len(outcomes),
        converted=sum(1 for status, _ in outcomes if status not in ("failure", "lost")),
        failed=sum(1 for status, _ in outcomes if status == "failure"),
        lost=sum(1 for status, _ in outcomes if status == "lost"),
        mean_seconds=sum(seconds) / len(seconds) if seconds else 0.0,
        p50_seconds=seconds[len(seconds) // 2] if seconds else 0.0,
        p90_seconds=seconds[int(len(seconds) * 0.9)] if seconds else 0.0,
        fleet_ready_seconds=fleet_ready,
        convert_wall_seconds=wall,
    )
    logger.info("SHAPE %s: %s", name, stats.model_dump())
    return stats


def shapes(
    output_path: str,
    source_output_path: str,
    classification_output_path: str,
    layout_model_output_path: str,
) -> ShapeReport:
    source = read_artifact(source_output_path, PdfSourceData)
    classification = read_artifact(classification_output_path, PdfClassificationData)
    layout_model = read_artifact(layout_model_output_path, LayoutModelData)
    rows = _load_corpus(source, classification)

    report = ShapeReport(
        arm=_ARM,
        shapes={name: _run_shape(name, rows, layout_model) for name in _SHAPES},
    )
    for name, stats in report.shapes.items():
        logger.info("FINAL %s: %s", name, stats.model_dump())
    return report


def shapes_step(source: StepSpec, classification: StepSpec, layout_model: StepSpec) -> StepSpec:
    return StepSpec(
        name="data/datakit/validate/pool_shape_comparison",
        deps=[layout_model],
        hash_attrs={
            "source_output_path": source.output_path,
            "classification_output_path": classification.output_path,
            "arm": _ARM,
            "shapes": sorted(_SHAPES),
            "attempt": 1,
        },
        fn=remote(
            partial(
                shapes,
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
    StepRunner().run([layout_model, shapes_step(fetch, classify, layout_model)])


if __name__ == "__main__":
    main()
