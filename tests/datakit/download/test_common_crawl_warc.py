# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
import hashlib
import io
import uuid
from collections.abc import Mapping
from dataclasses import replace

import pytest
import requests
from marin.datakit.download.common_crawl_warc import (
    CommonCrawlClient,
    CommonCrawlDownloadError,
    CommonCrawlRecordLocation,
    CommonCrawlRequestRejectedError,
    CommonCrawlTransientError,
    OriginResponseStatusError,
    RecordVerificationError,
    WarcPayloadTooLargeError,
    WarcRecordTooLargeError,
    WarcRevisitError,
)
from requests.adapters import BaseAdapter
from warcio.statusandheaders import StatusAndHeaders
from warcio.warcwriter import WARCWriter

RECORD_ID = "<urn:uuid:019f8700-d21d-78d8-8eb1-99eaa22579da>"
TARGET_URL = "https://example.com/document.docx"


def _sha1_digest(payload: bytes) -> str:
    digest = hashlib.sha1(payload, usedforsecurity=False).digest()
    return f"sha1:{base64.b32encode(digest).decode().rstrip('=')}"


def _warc_response(
    payload: bytes,
    *,
    target_url: str = TARGET_URL,
    record_id: str = RECORD_ID,
    http_status: str = "200 OK",
) -> bytes:
    output = io.BytesIO()
    writer = WARCWriter(output, gzip=True)
    http_headers = StatusAndHeaders(http_status, [("Content-Type", "application/octet-stream")], protocol="HTTP/1.1")
    record = writer.create_warc_record(
        target_url,
        "response",
        payload=io.BytesIO(payload),
        http_headers=http_headers,
        warc_headers_dict={
            "WARC-Record-ID": record_id,
            "WARC-Date": "2026-07-21T21:48:44Z",
            "WARC-Identified-Payload-Type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        },
    )
    writer.write_record(record)
    return output.getvalue()


def _warc_revisit() -> bytes:
    output = io.BytesIO()
    writer = WARCWriter(output, gzip=True)
    record = writer.create_warc_record(
        TARGET_URL,
        "revisit",
        warc_headers_dict={
            "WARC-Record-ID": RECORD_ID,
            "WARC-Refers-To": "<urn:uuid:119f8700-d21d-78d8-8eb1-99eaa22579da>",
        },
    )
    writer.write_record(record)
    return output.getvalue()


class _RangeAdapter(BaseAdapter):
    def __init__(self, content: bytes, content_range: str | None, response_status: int) -> None:
        super().__init__()
        self.content = content
        self.content_range = content_range
        self.response_status = response_status

    def send(
        self,
        request: requests.PreparedRequest,
        stream: bool = False,
        timeout: float | tuple[float, float] | tuple[float, None] | None = None,
        verify: bool | str = True,
        cert: bytes | str | tuple[bytes | str, bytes | str] | None = None,
        proxies: Mapping[str, str] | None = None,
    ) -> requests.Response:
        del stream, timeout, verify, cert, proxies
        requested_range = request.headers["Range"]
        start_text, end_text = requested_range.removeprefix("bytes=").split("-")
        start, end = int(start_text), int(end_text)
        body = self.content[start : end + 1]

        response = requests.Response()
        response.status_code = self.response_status
        response.headers["Content-Length"] = str(len(body))
        response.headers["Content-Range"] = self.content_range or f"bytes {start}-{end}/{len(self.content)}"
        response.raw = io.BytesIO(body)
        response.request = request
        return response

    def close(self) -> None:
        pass


def _range_session(
    content: bytes,
    *,
    content_range: str | None = None,
    response_status: int = requests.codes.partial_content,
) -> requests.Session:
    session = requests.Session()
    session.mount("https://", _RangeAdapter(content, content_range, response_status))
    return session


def _location(warc: bytes, payload: bytes) -> CommonCrawlRecordLocation:
    return CommonCrawlRecordLocation(
        crawl_id="CC-MAIN-2026-30",
        url=TARGET_URL,
        warc_filename="crawl-data/test.warc.gz",
        warc_record_offset=0,
        warc_record_length=len(warc),
        warc_record_id=RECORD_ID,
        content_digest=_sha1_digest(payload),
    )


def _client(session: requests.Session, *, maximum_warc_record_bytes: int = 1 << 20) -> CommonCrawlClient:
    return CommonCrawlClient(
        session=session,
        maximum_warc_record_bytes=maximum_warc_record_bytes,
        maximum_payload_bytes=1 << 20,
    )


def test_record_location_from_url_index_row_normalizes_uuid_blob() -> None:
    record_id = uuid.UUID("019f8700-d21d-78d8-8eb1-99eaa22579da")

    location = CommonCrawlRecordLocation.from_url_index_row(
        {
            "url": TARGET_URL,
            "warc_filename": "crawl-data/test.warc.gz",
            "warc_record_offset": 42,
            "warc_record_length": 100,
            "warc_record_id": record_id.bytes,
            "content_digest": "sha1:ABC",
        },
        crawl_id="CC-MAIN-2026-30",
    )

    assert location.warc_record_id == RECORD_ID
    assert location.byte_range == (42, 141)

    string_location = CommonCrawlRecordLocation.from_url_index_row(
        {
            "url": TARGET_URL,
            "warc_filename": "crawl-data/test.warc.gz",
            "warc_record_offset": 42,
            "warc_record_length": 100,
            "warc_record_id": "urn:uuid:019f8700-d21d-78d8-8eb1-99eaa22579da",
            "content_digest": "ABC",
        },
        crawl_id="CC-MAIN-2026-30",
    )
    assert string_location.warc_record_id == RECORD_ID
    assert string_location.content_digest == "sha1:ABC"


def test_client_default_session_constructs_and_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    sessions = []

    class TrackingSession(requests.Session):
        def __init__(self) -> None:
            super().__init__()
            self.closed = False
            sessions.append(self)

        def close(self) -> None:
            self.closed = True
            super().close()

    monkeypatch.setattr(requests, "Session", TrackingSession)

    with CommonCrawlClient(maximum_warc_record_bytes=1 << 20, maximum_payload_bytes=1 << 20):
        pass

    assert len(sessions) == 1
    assert sessions[0].closed


def test_fetch_record_returns_verified_payload() -> None:
    payload = b"PK\x03\x04example docx bytes"
    warc = _warc_response(payload)
    location = _location(warc, payload)

    with _range_session(warc) as session, _client(session) as client:
        record = client.fetch_record(location)

    assert record.payload == payload
    assert record.warc_record_id == RECORD_ID
    assert record.target_url == TARGET_URL
    assert record.http_status == 200
    assert record.http_content_type == "application/octet-stream"
    assert record.warc_date == "2026-07-21T21:48:44Z"
    assert record.identified_payload_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def test_fetch_record_rejects_incorrect_content_range() -> None:
    payload = b"document"
    warc = _warc_response(payload)
    location = _location(warc, payload)

    with _range_session(warc, content_range=f"bytes 1-{len(warc)}/{len(warc)}") as session:
        with _client(session) as client:
            with pytest.raises(CommonCrawlTransientError):
                client.fetch_record(location)


def test_fetch_record_rejects_full_response_to_range_request() -> None:
    payload = b"document"
    warc = _warc_response(payload)

    with _range_session(warc, response_status=requests.codes.ok) as session, _client(session) as client:
        with pytest.raises(CommonCrawlTransientError):
            client.fetch_record(_location(warc, payload))


@pytest.mark.parametrize("status", [400, 404])
def test_fetch_record_classifies_permanent_object_error(status: int) -> None:
    payload = b"missing"
    warc = _warc_response(payload)

    with _range_session(warc, response_status=status) as session, _client(session) as client:
        with pytest.raises(CommonCrawlRequestRejectedError) as error:
            client.fetch_record(_location(warc, payload))

    assert error.value.status == status
    assert error.value.url.endswith("/crawl-data/test.warc.gz")


def test_fetch_record_classifies_rate_limit_as_transient() -> None:
    payload = b"rate limited"
    warc = _warc_response(payload)

    with _range_session(warc, response_status=403) as session, _client(session) as client:
        with pytest.raises(CommonCrawlDownloadError):
            client.fetch_record(_location(warc, payload))


def test_fetch_record_rejects_payload_digest_mismatch() -> None:
    payload = b"document"
    warc = _warc_response(payload)
    location = replace(_location(warc, payload), content_digest="sha1:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")

    with _range_session(warc) as session, _client(session) as client:
        with pytest.raises(RecordVerificationError):
            client.fetch_record(location)


def test_fetch_record_rejects_non_success_origin_response() -> None:
    payload = b"not found"
    warc = _warc_response(payload, http_status="404 Not Found")

    with _range_session(warc) as session, _client(session) as client:
        with pytest.raises(OriginResponseStatusError) as error:
            client.fetch_record(_location(warc, payload))

    assert error.value.status == 404


def test_fetch_record_distinguishes_revisit_record() -> None:
    warc = _warc_revisit()
    location = _location(warc, b"")

    with _range_session(warc) as session, _client(session) as client:
        with pytest.raises(WarcRevisitError):
            client.fetch_record(location)


def test_fetch_record_rejects_index_range_over_limit_before_download() -> None:
    payload = b"document"
    warc = _warc_response(payload)

    with _range_session(warc) as session, _client(session, maximum_warc_record_bytes=len(warc) - 1) as client:
        with pytest.raises(WarcRecordTooLargeError):
            client.fetch_record(_location(warc, payload))


def test_fetch_record_rejects_payload_over_limit() -> None:
    payload = b"document"
    warc = _warc_response(payload)

    with _range_session(warc) as session:
        with CommonCrawlClient(
            session=session,
            maximum_warc_record_bytes=1 << 20,
            maximum_payload_bytes=len(payload) - 1,
        ) as client:
            with pytest.raises(WarcPayloadTooLargeError):
                client.fetch_record(_location(warc, payload))
