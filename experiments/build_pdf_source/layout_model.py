# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 4: build an INT8 OpenVINO version of docling's layout model, calibrated on this crawl.

Layout detection is what extraction costs. Every page of every text-extractable document runs
RT-DETR before a character is read, and the fleet is CPU-only Emerald Rapids, so this is the model
worth quantizing. FinePDFs reached the same conclusion; the difference is that docling now
publishes an ONNX export whose graph already contains RT-DETR's post-processing, so this step is a
format conversion and a calibration rather than a reimplementation of the detection head.

Three stages, all in one job:

1. **Convert.** ``docling-layout-heron-onnx`` at a pinned revision becomes an OpenVINO model.
2. **Calibrate and quantize.** NNCF runs post-training quantization over pages rendered from the
   crawl's own PDFs. Calibrating on the target distribution is the point: a layout model quantized
   against clean born-digital documents sees different activation ranges than one quantized against
   Common Crawl, which is full of scanned forms, multi-column journal articles and government PDFs
   from 2003. The calibration pages come from the text-extractable route, which is what will run.
3. **Fold in preprocessing.** The saved graph accepts a raw uint8 page image at its natural size
   and does the resize and layout transpose itself, so inference does no per-page work in Python.

The graph is staged in the marin prefix and pinned by the step's hash, so any extraction run is
attributable to one exact build. ``experiments/build_pdf_source/_compare_layout_backends.py``
measures what the quantization cost in extracted text.
"""

import logging
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from marin.execution.artifact import read_artifact
from marin.execution.remote import remote
from marin.execution.step_spec import StepSpec
from rigging.filesystem import StoragePath, prefix_join, url_to_fs

from experiments.build_pdf_source.common import LayoutModelData, PdfSourceData
from experiments.build_pdf_source.docling_extract.model_spec import (
    HERON_ONNX_FILENAME,
    HERON_ONNX_REPO,
    HERON_ONNX_REVISION,
    LAYOUT_IMAGE_SIZE,
)

logger = logging.getLogger(__name__)

STAGED_MODEL_FILENAME = "layout_heron_int8.xml"
# NNCF's default is 300. More calibration pages tighten the activation ranges at a linear cost in
# build time, and this graph is built once and used across the whole corpus.
CALIBRATION_IMAGES = 1_000
# How many source shards to draw calibration pages from. Each holds ~180 documents from one packed
# fetch task, and a task's ranges span several WARCs, so a handful of shards is already a wide
# sample of the crawl rather than one site.
CALIBRATION_SHARDS = 8
# Pages per document, taken from the front. Calibration wants many documents rather than many pages
# of the same document, and layout varies most between documents.
CALIBRATION_PAGES_PER_DOCUMENT = 2

# Chosen by measurement, not by reasoning. Five recipes were quantized on this crawl's own pages
# and compared against the FP32 graph on held-out pages from it.
#
# MIXED keeps activations asymmetric and weights symmetric per-channel; PERFORMANCE quantizes both
# symmetrically. The two scored within noise of each other, so MIXED is kept as the safer default.
#
# TRANSFORMER is the setting that actually mattered. RT-DETR is a DETR-family model, and telling
# NNCF so raises the share of FP32 detections the INT8 graph reproduces from 0.941 to 0.980 and cuts
# the mean box shift from 1.8px to 0.8px, at no measurable cost in speed.
_QUANTIZATION_PRESET = "mixed"
_QUANTIZATION_MODEL_TYPE = "transformer"

_RESOURCES = ResourceConfig(cpu=16, ram="64g", disk="32g")
_MEGABYTE = 1 << 20


def _render_calibration_pages(source_dir: str, limit: int) -> Iterator[np.ndarray]:
    """Render pages from the fetched corpus as uint8 NCHW tensors, ready for the layout graph.

    Resize matches the model's ``preprocessor_config.json`` exactly: bilinear to 640x640 with no
    rescale and no mean/std normalisation. The graph's ``images`` input is itself ``uint8``, so
    calibration must feed it uint8 -- handing it float32 makes NNCF collect its activation
    statistics on a tensor the model will never see. Docling renders pages for layout at scale 1.0,
    so these are rendered at 1.0 too: calibration should see the resolution inference will see.
    """
    import pymupdf  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    filesystem, path = url_to_fs(source_dir)
    shards = sorted(filesystem.glob(f"{path}/*.parquet"))[:CALIBRATION_SHARDS]
    if not shards:
        raise RuntimeError(f"No fetched PDFs under {source_dir}")
    logger.info("Rendering up to %d calibration pages from %d shards", limit, len(shards))

    rendered = 0
    for shard in shards:
        with filesystem.open(shard, "rb") as stream:
            table = pq.read_table(stream, columns=["pdf"])
        for row in table.column("pdf").to_pylist():
            if rendered >= limit:
                return
            try:
                with pymupdf.open(stream=row, filetype="pdf") as document:
                    for page_index in range(min(CALIBRATION_PAGES_PER_DOCUMENT, document.page_count)):
                        if rendered >= limit:
                            return
                        pixmap = document.load_page(page_index).get_pixmap()
                        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
                        resized = image.resize((LAYOUT_IMAGE_SIZE, LAYOUT_IMAGE_SIZE), Image.BILINEAR)
                        yield np.asarray(resized, dtype=np.uint8).transpose(2, 0, 1)[np.newaxis, ...]
                        rendered += 1
            except Exception:
                # A document the classifier could read may still fail to render a page. Calibration
                # only needs a representative sample, so a bad document is skipped, not fatal.
                logger.debug("Skipping an unrenderable calibration document", exc_info=True)


def _calibration_inputs(source_dir: str, limit: int) -> Iterator[dict]:
    """Pair each rendered page with the target size the detection head rescales boxes to."""
    target_size = np.array([[LAYOUT_IMAGE_SIZE, LAYOUT_IMAGE_SIZE]], dtype=np.int64)
    for pixels in _render_calibration_pages(source_dir, limit):
        yield {"images": pixels, "orig_target_sizes": target_size}


def _fold_in_preprocessing(model):
    """Make the graph accept a raw uint8 NHWC page image at any size.

    Everything the HF image processor would do in Python -- bilinear resize to the model's input
    size and channels-last to channels-first -- becomes graph nodes, so inference hands over the
    rendered page and nothing else. There is deliberately no type conversion and no scaling: this
    export's ``images`` input is already ``uint8``, and heron sets both ``do_rescale`` and
    ``do_normalize`` to false, so 0-255 values go in untouched.

    The model is cloned first because :meth:`PrePostProcessor.build` mutates the model handed to
    it, which would otherwise leave the caller's FP32 graph expecting NHWC input.
    """
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


def build_layout_model(output_path: str, source_output_path: str) -> LayoutModelData:
    """Convert, quantize and stage docling's layout model."""
    import json  # noqa: PLC0415

    import nncf  # noqa: PLC0415
    import openvino as ov  # noqa: PLC0415
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    source = read_artifact(source_output_path, PdfSourceData)

    onnx_path = hf_hub_download(HERON_ONNX_REPO, HERON_ONNX_FILENAME, revision=HERON_ONNX_REVISION)
    config_path = hf_hub_download(HERON_ONNX_REPO, "config.json", revision=HERON_ONNX_REVISION)
    label_map = {int(index): name for index, name in json.loads(Path(config_path).read_text())["id2label"].items()}
    logger.info("Converting %s (%d labels) to OpenVINO", HERON_ONNX_REPO, len(label_map))

    model = ov.convert_model(onnx_path)

    images = list(_calibration_inputs(source.main_output_dir, CALIBRATION_IMAGES))
    if not images:
        raise RuntimeError(f"Rendered no calibration pages from {source.main_output_dir}")
    logger.info("Quantizing over %d calibration pages", len(images))

    quantized = nncf.quantize(
        model,
        nncf.Dataset(images),
        target_device=nncf.TargetDevice.CPU,
        preset=nncf.QuantizationPreset(_QUANTIZATION_PRESET),
        model_type=nncf.ModelType(_QUANTIZATION_MODEL_TYPE),
        subset_size=len(images),
    )

    with TemporaryDirectory() as workspace:
        fp32_xml = Path(workspace) / "layout_heron_fp32.xml"
        int8_xml = Path(workspace) / STAGED_MODEL_FILENAME
        ov.save_model(model, fp32_xml, compress_to_fp16=False)
        ov.save_model(_fold_in_preprocessing(quantized), int8_xml, compress_to_fp16=False)

        model_path = prefix_join(output_path, STAGED_MODEL_FILENAME)
        weights_path = prefix_join(output_path, f"{int8_xml.stem}.bin")
        StoragePath(model_path).write_bytes(int8_xml.read_bytes())
        StoragePath(weights_path).write_bytes(int8_xml.with_suffix(".bin").read_bytes())

        fp32_bytes = fp32_xml.with_suffix(".bin").stat().st_size
        int8_bytes = int8_xml.with_suffix(".bin").stat().st_size

    logger.info(
        "Staged INT8 layout model at %s (%.1f MB, from %.1f MB)",
        model_path,
        int8_bytes / _MEGABYTE,
        fp32_bytes / _MEGABYTE,
    )
    return LayoutModelData(
        model_path=model_path,
        source_repo=HERON_ONNX_REPO,
        source_revision=HERON_ONNX_REVISION,
        label_map=label_map,
        calibration_images=len(images),
        fp32_megabytes=fp32_bytes / _MEGABYTE,
        int8_megabytes=int8_bytes / _MEGABYTE,
    )


def layout_model_step(source: StepSpec) -> StepSpec:
    """Build the step that converts and quantizes docling's layout model.

    Depends on the fetch step because the calibration set is drawn from the fetched corpus: the
    quantized graph is a function of this crawl, not just of the source weights.
    """
    return StepSpec(
        name="data/datakit/model/docling_layout_heron_int8",
        deps=[source],
        hash_attrs={
            "repo": HERON_ONNX_REPO,
            "revision": HERON_ONNX_REVISION,
            "calibration_images": CALIBRATION_IMAGES,
            "calibration_shards": CALIBRATION_SHARDS,
            "preset": _QUANTIZATION_PRESET,
            "model_type": _QUANTIZATION_MODEL_TYPE,
            "image_size": LAYOUT_IMAGE_SIZE,
            "schema_version": 1,
        },
        fn=remote(
            partial(build_layout_model, source_output_path=source.output_path),
            resources=_RESOURCES,
            pip_dependency_groups=["datakit"],
        ),
    )
