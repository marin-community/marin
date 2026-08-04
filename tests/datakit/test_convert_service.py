# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The converter-pool handler's wire contract.

The handler is the seam between the fleet machinery (tested in tests/inference/test_converter_pool)
and docling extraction (tested in tests/datakit/test_docling_extract). What matters here is the
contract the sender relies on: a converted document and an unconvertible document both come back as
HTTP 200 JSON that ``parse_converted`` round-trips, the source-URL header survives percent-encoding,
and unexpected routes get the standard error envelope instead of reaching the converter.

The docling converter itself is replaced at the module seam: building the real one downloads model
weights, and nothing below exercises conversion -- only the envelope around it.
"""

import json
import urllib.parse

import pytest

pytest.importorskip("docling")

from marin.inference.types import InferenceRequest

import experiments.build_pdf_source.docling_extract.converter as converter_module
from experiments.build_pdf_source.docling_extract.converter import ExtractedText, ExtractionOptions
from experiments.build_pdf_source.docling_extract.model_spec import LayoutBackend
from experiments.build_pdf_source.docling_extract.service import (
    CONVERT_PATH,
    SOURCE_URL_HEADER,
    build_arch_adaptive_handler,
    build_handler,
    parse_converted,
)

_OPTIONS = ExtractionOptions()


@pytest.fixture
def handler(monkeypatch: pytest.MonkeyPatch):
    """A handler whose converter records the names it was asked to convert."""
    seen_names: list[str] = []

    def fake_build_converter(options: ExtractionOptions) -> object:
        return object()

    def fake_extract_text(converter, pdf: bytes, options: ExtractionOptions, name: str) -> ExtractedText:
        seen_names.append(name)
        if pdf == b"broken":
            raise ValueError("not a PDF")
        return ExtractedText(
            text="page one\x0cpage two",
            num_pages=2,
            page_offsets=[9, 17],
            status="success",
            extraction_error=None,
        )

    monkeypatch.setattr(converter_module, "build_converter", fake_build_converter)
    monkeypatch.setattr(converter_module, "extract_text", fake_extract_text)
    handle = build_handler(_OPTIONS)
    handle.seen_names = seen_names  # type: ignore[attr-defined]
    return handle


def _convert_request(payload: bytes, headers: tuple[tuple[str, str], ...] = ()) -> InferenceRequest:
    return InferenceRequest(request_id="req-1", method="POST", path=CONVERT_PATH, payload=payload, headers=headers)


def test_a_converted_document_round_trips_through_the_wire_format(handler) -> None:
    response = handler(_convert_request(b"%PDF-1.4 ..."))

    assert response.status_code == 200
    document = parse_converted(response.payload)
    assert document.text == "page one\x0cpage two"
    assert document.num_pages == 2
    assert document.page_offsets == [9, 17]
    assert document.status == "success"
    assert document.error is None
    assert document.seconds >= 0.0
    assert document.backend == str(LayoutBackend.TORCH_HERON)


def test_an_unconvertible_document_is_a_failure_payload_not_an_error_status(handler) -> None:
    """The sender must be able to tell 'bad document' from 'broken fleet' by status code alone."""
    response = handler(_convert_request(b"broken"))

    assert response.status_code == 200
    document = parse_converted(response.payload)
    assert document.status == "failure"
    assert document.error == "ValueError: not a PDF"
    assert document.text == ""


def test_the_source_url_header_is_unquoted_into_the_converter_name(handler) -> None:
    url = "https://example.org/ré sumé.pdf"
    quoted = urllib.parse.quote(url, safe=":/?#[]@!$&'()*+,;=%")

    handler(_convert_request(b"%PDF", headers=((SOURCE_URL_HEADER, quoted),)))

    assert handler.seen_names == [url]


@pytest.mark.parametrize(
    ("machine", "expected_backend"),
    [("x86_64", LayoutBackend.INT8), ("aarch64", LayoutBackend.TORCH_HERON)],
)
def test_the_adaptive_handler_picks_the_backend_for_the_arch_it_lands_on(
    monkeypatch: pytest.MonkeyPatch, machine: str, expected_backend: LayoutBackend
) -> None:
    """The INT8 graph is only usable where VNNI exists, so the choice must follow placement."""
    monkeypatch.setattr(converter_module, "build_converter", lambda options: object())
    monkeypatch.setattr(
        converter_module,
        "extract_text",
        lambda converter, pdf, options, name: ExtractedText(
            text="hello", num_pages=1, page_offsets=[5], status="success", extraction_error=None
        ),
    )
    monkeypatch.setattr("platform.machine", lambda: machine)

    x86_options = ExtractionOptions(
        layout_backend=LayoutBackend.INT8, layout_model_path="unused", layout_label_map={0: "text"}
    )
    handle = build_arch_adaptive_handler(x86_options, ExtractionOptions())
    document = parse_converted(handle(_convert_request(b"%PDF")).payload)

    assert document.backend == str(expected_backend)


def test_unexpected_routes_get_the_error_envelope_without_reaching_the_converter(handler) -> None:
    wrong_path = handler(InferenceRequest(request_id="req-2", method="POST", path="/v1/chat/completions", payload=b""))
    wrong_method = handler(InferenceRequest(request_id="req-3", method="GET", path=CONVERT_PATH, payload=b""))

    assert wrong_path.status_code == 404
    assert wrong_method.status_code == 405
    assert "error" in json.loads(wrong_path.payload)
    assert handler.seen_names == []
