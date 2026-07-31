# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Assemble the pieces into a converter, and turn one PDF's bytes into text.

This is the module the pipeline talks to. Everything configurable about extraction is
:class:`ExtractionOptions`, and everything expensive -- the layout graph, the compiled regexes, the
converter itself -- is built once per worker process by :func:`build_converter` and reused.

The two backends worth comparing are switches here rather than forks:

``table_backend``
    :attr:`TableBackend.PYMUPDF` recovers table grids from ruling lines and never runs TableFormer.
    :attr:`TableBackend.DOCLING` runs TableFormer, which infers a grid from text positions. See
    :mod:`.tables` for why the two fail differently.
``layout_backend``
    :attr:`LayoutBackend.TORCH_HERON` is docling's own model, unmodified. :attr:`LayoutBackend.INT8`
    is the quantized OpenVINO graph from :mod:`.layout_openvino`.

Docling logs warnings per malformed page and PyMuPDF writes to stderr; both are captured per
document and returned rather than printed, so a shard's log stays readable and a document's
complaints stay attached to that document.
"""

import io
import logging
from dataclasses import dataclass, field

import numpy as np
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.object_detection_engine_options import (
    OnnxRuntimeObjectDetectionEngineOptions,
)
from docling.datamodel.pipeline_options import (
    LayoutObjectDetectionOptions,
    ThreadedPdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling_core.types.io import DocumentStream
from pydantic import Field

from experiments.build_pdf_source.docling_extract.assemble import SpanAwarePageAssembleModel
from experiments.build_pdf_source.docling_extract.backend import PyMuPdfDocumentBackend
from experiments.build_pdf_source.docling_extract.model_spec import LayoutBackend, TableBackend
from experiments.build_pdf_source.docling_extract.postprocess import postprocess_document
from experiments.build_pdf_source.docling_extract.reading_order import SpanAwareReadingOrderModel
from experiments.build_pdf_source.docling_extract.serializer import (
    DEFAULT_ALPHA_RATIO,
    PAGE_BREAK,
    serialize_document,
)
from experiments.build_pdf_source.docling_extract.tables import PyMuPdfTableReader

logger = logging.getLogger(__name__)

# Left for docling to notice the clock before the surrounding task's own timeout fires.
_TIMEOUT_MARGIN = 2.0


@dataclass(frozen=True)
class ExtractionOptions:
    """Everything that changes what text comes out of a PDF."""

    table_backend: TableBackend = TableBackend.PYMUPDF
    layout_backend: LayoutBackend = LayoutBackend.TORCH_HERON
    # Required by, and only used by, LayoutBackend.INT8.
    layout_model_path: str | None = None
    layout_label_map: dict[int, str] = field(default_factory=dict)
    picture_alpha_ratio: float = DEFAULT_ALPHA_RATIO
    document_timeout: float = 10 * 60
    fix_page_numbers: bool = True

    def __post_init__(self):
        if self.layout_backend is LayoutBackend.INT8 and not self.layout_model_path:
            raise ValueError("LayoutBackend.INT8 needs layout_model_path, the staged OpenVINO graph")


class MarinPdfPipelineOptions(ThreadedPdfPipelineOptions):
    """Docling's pipeline options, carrying this extractor's two backend switches.

    They ride on the options object because docling constructs the pipeline itself, from the
    options alone -- there is nowhere else to pass them through ``PdfFormatOption``.
    """

    table_backend: TableBackend = TableBackend.PYMUPDF
    layout_backend: LayoutBackend = LayoutBackend.TORCH_HERON
    layout_model_path: str | None = None
    layout_label_map: dict[int, str] = Field(default_factory=dict)


class MarinPdfPipeline(StandardPdfPipeline):
    """Docling's PDF pipeline with the span-aware stages, and optionally the quantized layout model.

    The models are replaced after the base constructor rather than instead of it, because the
    pipeline builds its stage graph per run from these attributes; everything else about the
    pipeline -- queueing, back-pressure, page failure handling -- is untouched.
    """

    def __init__(self, pipeline_options: MarinPdfPipelineOptions) -> None:
        super().__init__(pipeline_options)
        self.pipeline_options: MarinPdfPipelineOptions

        table_reader = PyMuPdfTableReader() if pipeline_options.table_backend is TableBackend.PYMUPDF else None
        self.assemble_model = SpanAwarePageAssembleModel(
            options=self.assemble_model.options,
            table_reader=table_reader,
        )
        self.reading_order_model = SpanAwareReadingOrderModel(options=self.reading_order_model.options)

        if pipeline_options.layout_backend is LayoutBackend.INT8:
            from experiments.build_pdf_source.docling_extract.layout_openvino import (  # noqa: PLC0415
                OpenVinoLayoutModel,
            )

            # ``engine_options`` is required and is where ``score_threshold`` lives. The ONNX
            # options class is used purely as the concrete carrier docling ships -- the OpenVINO
            # engine reads only the threshold from it and never consults the runtime settings.
            self.layout_model = OpenVinoLayoutModel(
                options=LayoutObjectDetectionOptions(
                    engine_options=OnnxRuntimeObjectDetectionEngineOptions(),
                ),
                model_path=pipeline_options.layout_model_path,
                label_map=pipeline_options.layout_label_map,
                accelerator_options=pipeline_options.accelerator_options,
            )


def build_converter(options: ExtractionOptions) -> DocumentConverter:
    """Build the docling converter for these options. Expensive: build one per worker process."""
    pipeline_options = MarinPdfPipelineOptions(
        # TableFormer runs only when docling owns table structure; the PyMuPDF reader replaces it.
        do_table_structure=options.table_backend is TableBackend.DOCLING,
        do_ocr=False,
        document_timeout=max(options.document_timeout - _TIMEOUT_MARGIN, 1.0),
        # One thread per document, because parallelism here is across documents: Zephyr costs each
        # map task at one CPU and runs several per worker.
        accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU, num_threads=1),
        table_backend=options.table_backend,
        layout_backend=options.layout_backend,
        layout_model_path=options.layout_model_path,
        layout_label_map=options.layout_label_map,
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                pipeline_cls=MarinPdfPipeline,
                backend=PyMuPdfDocumentBackend,
            )
        }
    )


@dataclass(frozen=True)
class ExtractedText:
    """One document's extracted text and what happened while extracting it."""

    text: str
    num_pages: int
    page_offsets: list[int]
    status: str
    extraction_error: str | None = None


def extract_text(
    converter: DocumentConverter,
    pdf: bytes,
    options: ExtractionOptions,
    name: str = "document.pdf",
) -> ExtractedText:
    """Convert one PDF to text.

    ``page_offsets`` are cumulative character counts, so a span of the returned text can be traced
    back to the page it came from without re-running the conversion.
    """
    converted = converter.convert(DocumentStream(name=name, stream=io.BytesIO(pdf)), raises_on_error=True)

    document = postprocess_document(converted.document, fix_page_numbers=options.fix_page_numbers)
    serialized = serialize_document(document, alpha_ratio=options.picture_alpha_ratio)
    pages = serialized.split(PAGE_BREAK)

    error = None
    if converted.status not in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS):
        error = f"conversion status {converted.status.value}: " + "; ".join(
            item.error_message for item in converted.errors
        )

    return ExtractedText(
        text="".join(pages),
        num_pages=len(pages),
        page_offsets=np.cumsum([len(page) for page in pages]).tolist(),
        status=converted.status.value,
        extraction_error=error,
    )
