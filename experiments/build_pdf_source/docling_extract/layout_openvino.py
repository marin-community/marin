# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run docling's layout model as an INT8 OpenVINO graph on the CPU fleet.

Layout detection is the cost of this pipeline. Every page of every non-OCR document goes through
RT-DETR before a single character is read, and on a CPU that one model dominates the per-page time.
FinePDFs' answer was to convert the layout model to OpenVINO and quantize it to INT8; this is that
idea carried onto docling 2.117, where it fits without a fork.

Two things changed since their version, both simplifications:

* docling now publishes ``docling-project/docling-layout-heron-onnx``, an ONNX export that already
  contains RT-DETR's post-processing -- sigmoid, top-k, box decode, rescale to the original page.
  So the conversion is ONNX to OpenVINO IR, and none of the detection maths is reimplemented here.
  FinePDFs had to trace the torch model and hand-write that post-processing.
* layout inference is a pluggable stage rather than a hard-coded call into ``docling-ibm-models``,
  so this is an engine and a stage subclass instead of a patch to a vendored file.

Preprocessing is folded into the graph, as FinePDFs did: the compiled model accepts a raw
uint8 NHWC page image and does the resize, the type conversion, and the layout transpose itself,
which keeps a 640x640 PIL resize and a float32 copy per page out of Python. The recipe follows the
model's own ``preprocessor_config.json`` -- bilinear resize to 640x640 and nothing else, because
heron sets both ``do_rescale`` and ``do_normalize`` to false and consumes 0-255 values directly.

The quantized graph is an artifact of :mod:`experiments.build_pdf_source.layout_model`, built once
and staged in the marin prefix; this module only consumes it.
"""

import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import LayoutObjectDetectionOptions
from docling.models.inference_engines.object_detection.base import (
    BaseObjectDetectionEngine,
    BaseObjectDetectionEngineOptions,
    ObjectDetectionEngineInput,
    ObjectDetectionEngineOutput,
)
from docling.models.stages.layout.layout_object_detection_model import (
    LayoutObjectDetectionModel,
)
from rigging.filesystem import StoragePath

from experiments.build_pdf_source.docling_extract.model_spec import LAYOUT_IMAGE_SIZE  # noqa: F401

logger = logging.getLogger(__name__)

# OpenVINO tunables. LATENCY is right because Zephyr costs each map task at one CPU and runs many
# per worker: the throughput hint would have every task claim every core and thrash.
_PERFORMANCE_HINT = "LATENCY"
# Bounds the compiled-kernel cache. Higher values retune per shape and cost memory; layout runs at
# one fixed shape, so a small cache is all that is ever used.
_RUNTIME_CACHE_CAPACITY = "3"


def stage_model_locally(model_path: str) -> Path:
    """Copy an OpenVINO IR out of object storage and return the local ``.xml``.

    OpenVINO reads from the filesystem only -- handing ``ov.Core().read_model`` an ``s3://`` URI
    fails inside the C++ core with a bare "Exception from core.cpp". An IR is also two files, and
    the ``.xml`` names its ``.bin`` as a bare filename, so both have to land in one directory with
    matching stems or the weights are silently not found.

    The copy is per process and lives for the process's lifetime, which is what we want: a Zephyr
    task builds one converter and reuses it for every document in its shard.
    """
    source = StoragePath(model_path)
    if source.is_local:
        return Path(model_path)

    destination = Path(tempfile.mkdtemp(prefix="layout-int8-"))
    stem = source.name.rsplit(".", 1)[0]
    for suffix in (".xml", ".bin"):
        StoragePath(f"{source.parent}/{stem}{suffix}").download_to(str(destination / f"{stem}{suffix}"))
    logger.info("Staged the INT8 layout graph from %s to %s", model_path, destination)
    return destination / f"{stem}.xml"


class OpenVinoLayoutEngine(BaseObjectDetectionEngine):
    """An object-detection engine backed by a compiled OpenVINO layout graph."""

    def __init__(
        self,
        *,
        options: BaseObjectDetectionEngineOptions,
        model_path: str | Path,
        label_map: dict[int, str],
        num_threads: int,
    ):
        # The *engine* options, not the layout stage's: ``score_threshold`` lives here, and
        # ``LayoutObjectDetectionOptions`` does not carry one.
        super().__init__(options=options)
        self.model_path = str(model_path)
        self._label_map = label_map
        self._num_threads = num_threads
        self._compiled: Any | None = None

    def initialize(self) -> None:
        import openvino as ov  # noqa: PLC0415

        local_model = stage_model_locally(self.model_path)
        logger.info("Compiling OpenVINO layout graph from %s", local_model)
        core = ov.Core()
        self._compiled = core.compile_model(
            core.read_model(local_model),
            device_name="CPU",
            config={
                "PERFORMANCE_HINT": _PERFORMANCE_HINT,
                "CPU_RUNTIME_CACHE_CAPACITY": _RUNTIME_CACHE_CAPACITY,
                "INFERENCE_NUM_THREADS": str(self._num_threads),
            },
        )
        self._initialized = True

    def get_label_mapping(self) -> dict[int, str]:
        return dict(self._label_map)

    def predict_batch(self, input_batch: list[ObjectDetectionEngineInput]) -> list[ObjectDetectionEngineOutput]:
        """Detect layout on each input image.

        The compiled graph is built for one image at a time, matching how the layout stage calls it
        and how the LATENCY hint is tuned, so a batch is run as a loop rather than reshaped.
        """
        if self._compiled is None:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        outputs = []
        for item in input_batch:
            image = item.image.convert("RGB")
            # The graph takes the page at its own resolution and resizes internally; it needs the
            # original size separately to map detections back onto page coordinates.
            pixels = np.asarray(image, dtype=np.uint8)[np.newaxis, ...]
            target_size = np.array([[image.height, image.width]], dtype=np.int64)
            # Index the result rather than unpacking it: an OpenVINO result is a mapping, so
            # tuple-unpacking yields its output *ports*, not the arrays behind them.
            result = self._compiled((pixels, target_size), share_outputs=True)
            outputs.append(
                self._build_output(
                    input_item=item,
                    labels=result[0][0],
                    boxes=result[1][0],
                    scores=result[2][0],
                )
            )
        return outputs

    def _build_output(
        self,
        *,
        input_item: ObjectDetectionEngineInput,
        labels,
        scores,
        boxes,
    ) -> ObjectDetectionEngineOutput:
        keep = [index for index, score in enumerate(scores) if float(score) >= self.options.score_threshold]
        return ObjectDetectionEngineOutput(
            label_ids=[int(labels[index]) for index in keep],
            scores=[float(scores[index]) for index in keep],
            bboxes=[[float(value) for value in boxes[index]] for index in keep],
            metadata=input_item.metadata.copy(),
        )


class OpenVinoLayoutModel(LayoutObjectDetectionModel):
    """Docling's layout stage, running on a compiled OpenVINO graph.

    Only engine construction differs from the base stage; detection post-processing, the mapping
    from model labels to :class:`DocItemLabel`, and cluster building are all inherited.
    """

    def __init__(
        self,
        *,
        options: LayoutObjectDetectionOptions,
        model_path: str | Path,
        label_map: dict[int, str],
        accelerator_options: AcceleratorOptions,
    ):
        self.options = options
        # The engine gets ``options.engine_options``, which is where ``score_threshold`` lives; the
        # stage keeps the layout options, which carry the cluster-building settings it inherits.
        self.engine = OpenVinoLayoutEngine(
            options=options.engine_options,
            model_path=model_path,
            label_map=label_map,
            num_threads=accelerator_options.num_threads,
        )
        self.engine.initialize()
        self._label_map = self._build_label_map()
