# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The converter-pool handler for docling extraction, and its wire format.

One converter process calls :func:`build_handler` once, pays for the compiled layout graph there,
and then answers ``POST /v1/convert`` requests for the life of the process. The request body is one
PDF's raw bytes; the response is always HTTP 200 with a JSON body -- a document docling cannot
convert is data, not a transport failure, so it comes back as ``status: "failure"`` rather than as
an error status the sender would have to distinguish from a broker fault.

:class:`ConvertedDocument` and :func:`parse_converted` live here too, so the JSON the handler writes
and the shape the sender reads are defined in one module.

Top-level imports stay light on purpose: this module is imported wherever the pool's
``handler_factory`` partial is pickled or unpickled, including supervisor processes that must not
pay for docling.
"""

import json
import time
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass

from marin.inference.types import InferenceRequest, InferenceResponse
from marin.inference.worker import inference_error_response

CONVERT_PATH = "/v1/convert"
# Carries the document's source URL, used as the converter's filename hint and in its logs. Header
# values must be latin-1, so the sender percent-encodes and the handler unquotes.
SOURCE_URL_HEADER = "x-marin-source-url"


@dataclass(frozen=True)
class ConvertedDocument:
    """One document's conversion result, exactly as it crosses the wire."""

    text: str
    num_pages: int
    page_offsets: list[int]
    status: str
    error: str | None
    seconds: float


def parse_converted(payload: bytes) -> ConvertedDocument:
    """Parse the handler's JSON response body."""
    return ConvertedDocument(**json.loads(payload))


def build_handler(options: "ExtractionOptions") -> Callable[[InferenceRequest], InferenceResponse]:  # noqa: F821
    """Build the converter once and return the per-request handler closed over it.

    The fleet passes ``partial(build_handler, options)`` as the pool's ``handler_factory``, so this
    function must stay module-level and its docling imports must stay inside it: the partial is
    pickled with the standard pickle module and unpickled in processes that only then import the
    heavy extraction stack.
    """
    from experiments.build_pdf_source.docling_extract.converter import (  # noqa: PLC0415
        build_converter,
        extract_text,
    )

    converter = build_converter(options)

    def handle(request: InferenceRequest) -> InferenceResponse:
        if request.path != CONVERT_PATH:
            return inference_error_response(request, 404, f"no route for {request.path}")
        if request.method != "POST":
            return inference_error_response(request, 405, f"{CONVERT_PATH} only accepts POST")

        name = urllib.parse.unquote(dict(request.headers).get(SOURCE_URL_HEADER, "document.pdf"))
        start = time.perf_counter()
        try:
            extracted = extract_text(converter, request.payload, options, name=name)
        except Exception as exc:
            body = {
                "text": "",
                "num_pages": 0,
                "page_offsets": [],
                "status": "failure",
                "error": f"{type(exc).__name__}: {exc}",
                "seconds": time.perf_counter() - start,
            }
        else:
            body = {
                "text": extracted.text,
                "num_pages": extracted.num_pages,
                "page_offsets": extracted.page_offsets,
                "status": extracted.status,
                "error": extracted.extraction_error,
                "seconds": time.perf_counter() - start,
            }
        return InferenceResponse(
            request_id=request.request_id,
            status_code=200,
            payload=json.dumps(body).encode(),
            headers=(("content-type", "application/json"),),
        )

    return handle
