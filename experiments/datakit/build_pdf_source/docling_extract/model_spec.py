# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Constants and switches that describe the extractor without importing it.

Everything here is pure data, so this module imports in the entrypoint job, which runs a plain
``uv sync`` with no extras and therefore has no docling, no PyMuPDF and no OpenVINO. Steps are
built there, so anything a :class:`StepSpec` needs -- names to hash, a pinned model revision, the
backend a run selected -- has to be reachable without the packages that do the work. The same
split is why :mod:`experiments.datakit.build_pdf_source.ocr_feature_names` exists for the classifier.

Import the rest of :mod:`experiments.datakit.build_pdf_source.docling_extract` only inside functions that
run on a worker.
"""

from enum import StrEnum

# The ONNX export the INT8 layout graph is built from, pinned so a build is attributable to exact
# weights. This export already contains RT-DETR's post-processing -- sigmoid, top-k, box decode and
# rescale to page coordinates -- so nothing downstream reimplements the detection head.
HERON_ONNX_REPO = "docling-project/docling-layout-heron-onnx"
HERON_ONNX_REVISION = "40bde044036bb181c130ddf6c51792187268748f"
HERON_ONNX_FILENAME = "model.onnx"

# The graph's input geometry, from the model's own preprocessor_config.json. Heron sets both
# do_rescale and do_normalize to false, so it consumes 0-255 values with no scaling: preprocessing
# is a bilinear resize to this size and nothing else.
LAYOUT_IMAGE_SIZE = 640


class TableBackend(StrEnum):
    """Where a detected table's cells come from.

    ``PYMUPDF`` recovers the grid from drawn ruling lines and never runs TableFormer: exact on
    ruled tables, blind to unruled ones. ``DOCLING`` runs TableFormer, which infers a grid from
    text positions: never exact, never blind. Which suits this corpus is an open question, so both
    are reachable and comparable.
    """

    PYMUPDF = "pymupdf"
    DOCLING = "docling"


class LayoutBackend(StrEnum):
    """Which build of the layout model runs."""

    TORCH_HERON = "torch_heron"
    INT8 = "openvino_int8"
