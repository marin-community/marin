# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TEMPORARY -- pick the NNCF recipe for the layout model, on the hardware that will run it.

DELETE once a recipe is chosen and recorded in :mod:`experiments.build_pdf_source.layout_model`.
Nothing in the pipeline imports this.

Post-training quantization of a DETR-family detector is not a one-liner: the detection head --
sigmoid over class logits, top-k, gather -- has a dynamic range that INT8 cannot hold, and
quantizing it collapses the model to 300 identical boxes. FinePDFs carried an ``--ignore-scope``
flag for exactly this. Which combination of preset, transformer handling, and ignored subgraph is
needed is an empirical question, and it has to be answered on Emerald Rapids: an arm64 laptop shows
no INT8 speedup at all, which means its CPU plugin is not taking the INT8 path, so neither its
timings nor its numerics say anything about what the fleet will do.

Each recipe is quantized on pages rendered from the crawl's own PDFs and then compared against the
FP32 graph on held-out pages from the same corpus: how many detections survive, whether the labels
agree, how far the boxes move, and how long a page takes.

Run it where the fetched corpus lives::

    uv run iris --cluster=marin job run --target-cluster cw-us-east-08a \\
        --job-name tune-layout-quant \\
        -- python -m experiments.build_pdf_source._tune_layout_quantization
"""

import logging
import pathlib
import time
from collections.abc import Iterator
from functools import partial

import numpy as np
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from marin.execution.artifact import read_artifact
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from pydantic import BaseModel
from rigging.filesystem import url_to_fs
from rigging.log_setup import configure_logging

from experiments.build_pdf_source.common import PdfSourceData
from experiments.build_pdf_source.docling_extract.model_spec import (
    HERON_ONNX_FILENAME,
    HERON_ONNX_REPO,
    HERON_ONNX_REVISION,
    LAYOUT_IMAGE_SIZE,
)
from experiments.build_pdf_source.fetch import fetch_step
from experiments.build_pdf_source.plan import plan_step

logger = logging.getLogger(__name__)

CALIBRATION_PAGES = 256
EVALUATION_PAGES = 64
_SOURCE_SHARDS = 4
_SCORE_THRESHOLD = 0.3
# Two boxes describe the same region above this IoU.
_MATCH_IOU = 0.5
_RESOURCES = ResourceConfig(cpu=32, ram="128g", disk="32g")


class RecipeResult(BaseModel):
    """How one quantization recipe compared against the FP32 graph."""

    recipe: str
    milliseconds_per_page: float
    speedup: float
    mean_detections: float
    fp32_mean_detections: float
    matched_fraction: float
    label_agreement: float
    mean_box_shift: float
    collapsed_pages: int


class QuantizationReport(BaseModel):
    version: str = "v1"
    int8_accelerator: str
    calibration_pages: int
    evaluation_pages: int
    fp32_milliseconds_per_page: float
    results: list[RecipeResult]


def _require_x86() -> str:
    """Refuse to report numbers from an ARM node, and say which INT8 path the host actually has.

    ``cw-us-east-08a`` holds two node pools: ``cpu-erapids`` (x86 Emerald Rapids, with AMX-INT8)
    and ``gb200`` (NVIDIA Grace, ARM). Iris has no way to select one -- ``ResourceSpecProto``
    carries only cpu, memory, disk and device, a CPU request converts to no device constraint at
    all, and a gb200 node has more of every resource than an erapids node, so nothing in a resource
    request can exclude it. In practice CPU tasks land on Grace, where OpenVINO does not take the
    INT8 path at all: no speedup, different detections, and nothing to say so.

    The INT8 speedup also differs sharply *within* x86 -- AMX (Emerald Rapids) is roughly twice
    VNNI (Genoa) on this kind of model -- so the accelerator is reported alongside the timings and
    a measurement is only comparable to another on the same one.
    """
    import platform  # noqa: PLC0415

    machine = platform.machine()
    if machine not in ("x86_64", "AMD64"):
        raise RuntimeError(
            f"Layout quantization must be tuned on x86, but this task landed on {machine}. "
            "OpenVINO's ARM plugin does not run the INT8 path, so any result here would be "
            "meaningless. Run on a cluster whose CPU pool is x86 (cw-us-east-02a is)."
        )

    flags = ""
    try:
        flags = pathlib.Path("/proc/cpuinfo").read_text()
    except OSError:
        pass
    accelerator = "amx" if "amx_int8" in flags else ("vnni" if "avx512_vnni" in flags else "none")
    logger.info("Running on %s with INT8 accelerator: %s", machine, accelerator)
    return accelerator


def _render_pages(source_dir: str, limit: int, skip: int = 0) -> list[np.ndarray]:
    """Render page images from the fetched corpus, as uint8 HWC arrays at docling's layout scale."""
    import pymupdf  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    filesystem, path = url_to_fs(source_dir)
    shards = sorted(filesystem.glob(f"{path}/*.parquet"))[:_SOURCE_SHARDS]
    pages: list[np.ndarray] = []
    seen = 0
    for shard in shards:
        with filesystem.open(shard, "rb") as stream:
            table = pq.read_table(stream, columns=["pdf"])
        for pdf in table.column("pdf").to_pylist():
            if len(pages) >= limit:
                return pages
            try:
                with pymupdf.open(stream=pdf, filetype="pdf") as document:
                    for page_index in range(min(2, document.page_count)):
                        if len(pages) >= limit:
                            return pages
                        seen += 1
                        if seen <= skip:
                            continue
                        pixmap = document.load_page(page_index).get_pixmap()
                        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
                        pages.append(np.asarray(image, dtype=np.uint8))
            except Exception:
                logger.debug("Skipping an unrenderable document", exc_info=True)
    return pages


def _calibration_tensors(pages: list[np.ndarray]) -> Iterator[dict]:
    from PIL import Image  # noqa: PLC0415

    target = np.array([[LAYOUT_IMAGE_SIZE, LAYOUT_IMAGE_SIZE]], dtype=np.int64)
    for page in pages:
        resized = Image.fromarray(page).resize((LAYOUT_IMAGE_SIZE, LAYOUT_IMAGE_SIZE), Image.BILINEAR)
        yield {
            "images": np.asarray(resized, dtype=np.uint8).transpose(2, 0, 1)[np.newaxis, ...],
            "orig_target_sizes": target,
        }


def _fold_in_preprocessing(model):
    """Give the graph a uint8 NHWC input of any size. Clones first: ``build`` mutates its input."""
    from openvino import Layout, Type  # noqa: PLC0415
    from openvino.preprocess import PrePostProcessor, ResizeAlgorithm  # noqa: PLC0415

    processor = PrePostProcessor(model.clone())
    pixels = processor.input("images")
    pixels.tensor().set_element_type(Type.u8).set_layout(Layout("NHWC")).set_shape([1, -1, -1, 3])
    processor.input("orig_target_sizes").tensor().set_shape([1, 2])
    pixels.preprocess().resize(
        ResizeAlgorithm.RESIZE_BILINEAR_PILLOW, LAYOUT_IMAGE_SIZE, LAYOUT_IMAGE_SIZE
    ).convert_layout(Layout("NCHW"))
    return processor.build()


def _detection_head_sigmoid(model) -> str:
    """Name the sigmoid that produces class logits, by walking back from the score top-k."""
    ops = {op.get_friendly_name(): op for op in model.get_ordered_ops()}
    frontier = [ops["scores"]]
    seen: set[str] = set()
    while frontier:
        node = frontier.pop(0)
        for port in node.inputs():
            source = port.get_source_output().get_node()
            name = source.get_friendly_name()
            if name in seen:
                continue
            seen.add(name)
            if source.get_type_name() == "Sigmoid":
                return name
            frontier.append(source)
    raise RuntimeError("Found no sigmoid feeding the score top-k; the graph's head has changed")


def _detect(compiled, page: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pixels = page[np.newaxis, ...]
    target = np.array([[page.shape[0], page.shape[1]]], dtype=np.int64)
    result = compiled((pixels, target))
    labels, boxes, scores = result[0][0], result[1][0], result[2][0]
    keep = scores >= _SCORE_THRESHOLD
    return labels[keep], boxes[keep], scores[keep]


def _iou_matrix(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    if len(first) == 0 or len(second) == 0:
        return np.zeros((len(first), len(second)))
    left = np.maximum(first[:, None, 0], second[None, :, 0])
    top = np.maximum(first[:, None, 1], second[None, :, 1])
    right = np.minimum(first[:, None, 2], second[None, :, 2])
    bottom = np.minimum(first[:, None, 3], second[None, :, 3])
    overlap = np.clip(right - left, 0, None) * np.clip(bottom - top, 0, None)
    area = lambda box: np.clip(box[:, 2] - box[:, 0], 0, None) * np.clip(box[:, 3] - box[:, 1], 0, None)  # noqa: E731
    union = area(first)[:, None] + area(second)[None, :] - overlap
    return np.where(union > 0, overlap / union, 0.0)


def _compare(reference, candidate) -> tuple[float, float, float]:
    """Fraction of FP32 detections matched, label agreement among matches, and mean box shift."""
    ref_labels, ref_boxes, _ = reference
    cand_labels, cand_boxes, _ = candidate
    if len(ref_boxes) == 0:
        return 1.0 if len(cand_boxes) == 0 else 0.0, 1.0, 0.0

    iou = _iou_matrix(ref_boxes, cand_boxes)
    best = iou.argmax(axis=1) if len(cand_boxes) else np.zeros(len(ref_boxes), dtype=int)
    matched = iou.max(axis=1) >= _MATCH_IOU if len(cand_boxes) else np.zeros(len(ref_boxes), dtype=bool)
    if not matched.any():
        return 0.0, 0.0, float("inf")
    agree = float((ref_labels[matched] == cand_labels[best[matched]]).mean())
    shift = float(np.abs(ref_boxes[matched] - cand_boxes[best[matched]]).mean())
    return float(matched.mean()), agree, shift


def tune(output_path: str, source_output_path: str) -> QuantizationReport:
    """Quantize the layout model several ways and report what each costs in detections."""
    import nncf  # noqa: PLC0415
    import openvino as ov  # noqa: PLC0415
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    nncf.set_log_level(logging.ERROR)
    accelerator = _require_x86()
    source = read_artifact(source_output_path, PdfSourceData)

    calibration = _render_pages(source.main_output_dir, CALIBRATION_PAGES)
    evaluation = _render_pages(source.main_output_dir, EVALUATION_PAGES, skip=CALIBRATION_PAGES)
    logger.info("Rendered %d calibration and %d evaluation pages", len(calibration), len(evaluation))
    dataset = nncf.Dataset(list(_calibration_tensors(calibration)))

    base = ov.convert_model(hf_hub_download(HERON_ONNX_REPO, HERON_ONNX_FILENAME, revision=HERON_ONNX_REVISION))
    head = _detection_head_sigmoid(base)
    logger.info("Detection head sigmoid: %s", head)
    ignored = nncf.IgnoredScope(subgraphs=[nncf.Subgraph(inputs=[head], outputs=["boxes", "labels", "scores"])])

    core = ov.Core()
    runtime = {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": "1"}

    def measure(model) -> tuple[float, list]:
        compiled = core.compile_model(_fold_in_preprocessing(model), "CPU", runtime)
        _detect(compiled, evaluation[0])
        start = time.perf_counter()
        detections = [_detect(compiled, page) for page in evaluation]
        return (time.perf_counter() - start) / len(evaluation) * 1000, detections

    fp32_ms, fp32_detections = measure(base)
    fp32_mean = float(np.mean([len(labels) for labels, _, _ in fp32_detections]))
    logger.info("FP32: %.0f ms/page, %.1f detections/page", fp32_ms, fp32_mean)

    recipes = {
        "mixed": {"preset": nncf.QuantizationPreset("mixed")},
        "mixed+transformer": {
            "preset": nncf.QuantizationPreset("mixed"),
            "model_type": nncf.ModelType.TRANSFORMER,
        },
        "mixed+ignore_head": {"preset": nncf.QuantizationPreset("mixed"), "ignored_scope": ignored},
        "mixed+transformer+ignore_head": {
            "preset": nncf.QuantizationPreset("mixed"),
            "model_type": nncf.ModelType.TRANSFORMER,
            "ignored_scope": ignored,
        },
        "performance+transformer+ignore_head": {
            "preset": nncf.QuantizationPreset("performance"),
            "model_type": nncf.ModelType.TRANSFORMER,
            "ignored_scope": ignored,
        },
    }

    results = []
    for name, options in recipes.items():
        logger.info("Quantizing: %s", name)
        quantized = nncf.quantize(
            base, dataset, target_device=nncf.TargetDevice.CPU, subset_size=len(calibration), **options
        )
        milliseconds, detections = measure(quantized)
        comparisons = [_compare(ref, cand) for ref, cand in zip(fp32_detections, detections, strict=True)]
        results.append(
            RecipeResult(
                recipe=name,
                milliseconds_per_page=milliseconds,
                speedup=fp32_ms / milliseconds,
                mean_detections=float(np.mean([len(labels) for labels, _, _ in detections])),
                fp32_mean_detections=fp32_mean,
                matched_fraction=float(np.mean([matched for matched, _, _ in comparisons])),
                label_agreement=float(np.mean([agree for _, agree, _ in comparisons])),
                mean_box_shift=float(np.mean([shift for _, _, shift in comparisons if np.isfinite(shift)])),
                # The failure mode this exists to catch: the head saturates and every query fires.
                collapsed_pages=sum(1 for labels, _, _ in detections if len(labels) >= 300),
            )
        )
        logger.info("  %s", results[-1].model_dump())

    report = QuantizationReport(
        int8_accelerator=accelerator,
        calibration_pages=len(calibration),
        evaluation_pages=len(evaluation),
        fp32_milliseconds_per_page=fp32_ms,
        results=results,
    )
    logger.info("=== LAYOUT QUANTIZATION REPORT (INT8 accelerator: %s) ===", accelerator)
    logger.info("  FP32 %.0f ms/page, %.1f dets/page", fp32_ms, fp32_mean)
    for result in results:
        logger.info(
            "  %-36s %6.0f ms (%.2fx)  dets %5.1f  matched %.3f  labels %.3f  shift %.1f px  collapsed %d",
            result.recipe,
            result.milliseconds_per_page,
            result.speedup,
            result.mean_detections,
            result.matched_fraction,
            result.label_agreement,
            result.mean_box_shift,
            result.collapsed_pages,
        )
    return report


def tuning_step(source_output_path: str) -> StepSpec:
    """Build the tuning step. The fetch step is named by path, not depended on, so it never runs."""
    return StepSpec(
        name="data/datakit/tune/docling_layout_quantization",
        hash_attrs={
            "source_output_path": source_output_path,
            "revision": HERON_ONNX_REVISION,
            "calibration_pages": CALIBRATION_PAGES,
            "evaluation_pages": EVALUATION_PAGES,
            "attempt": 1,
        },
        fn=remote(
            partial(tune, source_output_path=source_output_path),
            resources=_RESOURCES,
            pip_dependency_groups=["datakit"],
        ),
    )


def main() -> None:
    configure_logging(logging.INFO)
    fetch = fetch_step(plan_step())
    logger.info("Reading fetched PDFs from %s", fetch.output_path)
    StepRunner().run([tuning_step(fetch.output_path)])


if __name__ == "__main__":
    main()
