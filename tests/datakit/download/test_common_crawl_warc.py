# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
import gzip
import hashlib
import io
import uuid
from collections.abc import Mapping
from dataclasses import replace

import pytest
import requests
from marin.datakit.download.common_crawl_plan import (
    CommonCrawlIndexKind,
    CommonCrawlSource,
    PlannedCommonCrawlRange,
)
from marin.datakit.download.common_crawl_warc import (
    CommonCrawlClient,
    CommonCrawlDownloadError,
    CommonCrawlIndexManifestError,
    CommonCrawlRequestRejectedError,
    CommonCrawlTransientError,
    MainIndexedRecord,
    MissingPlannedRecordError,
    OriginResponseStatusError,
    RecordVerificationError,
    SupplementalIndexedRecord,
    WarcParsingError,
    WarcPayloadTooLargeError,
    WarcRecordRange,
    WarcRecordTooLargeError,
    WarcRevisitError,
    common_crawl_index_partitions,
    content_digest,
    main_record_from_index_row,
    supplemental_record_from_index_row,
    verify_supplemental_record,
    verify_url_index_record,
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


def _warc_record(
    record_type: str,
    *,
    include_http_headers: bool = True,
    gzip_output: bool = True,
    payload: bytes = b"payload",
) -> bytes:
    output = io.BytesIO()
    writer = WARCWriter(output, gzip=gzip_output)
    http_headers = StatusAndHeaders("200 OK", [], protocol="HTTP/1.1") if include_http_headers else None
    record = writer.create_warc_record(
        TARGET_URL,
        record_type,
        payload=io.BytesIO(payload),
        http_headers=http_headers,
        warc_headers_dict={"WARC-Record-ID": RECORD_ID},
    )
    writer.write_record(record)
    return output.getvalue()


class _RangeAdapter(BaseAdapter):
    def __init__(
        self,
        content: bytes,
        content_range: str | None,
        response_status: int,
        body_override: bytes | None,
        content_length: str | None,
    ) -> None:
        super().__init__()
        self.content = content
        self.content_range = content_range
        self.response_status = response_status
        self.body_override = body_override
        self.content_length = content_length

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
        if self.body_override is not None:
            body = self.body_override

        response = requests.Response()
        response.status_code = self.response_status
        if self.content_length is not None:
            response.headers["Content-Length"] = self.content_length
        response.headers["Content-Range"] = self.content_range or f"bytes {start}-{end}/{len(self.content)}"
        response.raw = io.BytesIO(body)
        response.request = request
        return response

    def close(self) -> None:
        pass


class _ManifestAdapter(BaseAdapter):
    def __init__(self, body: bytes, status: int = requests.codes.ok) -> None:
        super().__init__()
        self.body = body
        self.status = status

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
        response = requests.Response()
        response.status_code = self.status
        response.raw = io.BytesIO(self.body)
        response.request = request
        return response

    def close(self) -> None:
        pass


class _TruncatingRangeAdapter(BaseAdapter):
    def __init__(self, content: bytes) -> None:
        super().__init__()
        self.content = content
        self.requested_ranges: list[tuple[int, int]] = []

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
        start_text, end_text = request.headers["Range"].removeprefix("bytes=").split("-")
        start, end = int(start_text), int(end_text)
        self.requested_ranges.append((start, end))
        body = self.content[start : end + 1]
        if len(self.requested_ranges) == 1:
            body = body[: len(body) // 2]

        response = requests.Response()
        response.status_code = requests.codes.partial_content
        response.headers["Content-Length"] = str(end - start + 1)
        response.headers["Content-Range"] = f"bytes {start}-{end}/{len(self.content)}"
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
    body_override: bytes | None = None,
    content_length: str | None = None,
    include_content_length: bool = True,
) -> requests.Session:
    session = requests.Session()
    resolved_content_length = (
        content_length
        if content_length is not None
        else str(len(body_override if body_override is not None else content))
    )
    if not include_content_length:
        resolved_content_length = None
    session.mount(
        "https://",
        _RangeAdapter(content, content_range, response_status, body_override, resolved_content_length),
    )
    return session


def _indexed_record(warc: bytes, payload: bytes) -> MainIndexedRecord:
    return main_record_from_index_row(
        {
            "url": TARGET_URL,
            "warc_filename": "crawl-data/test.warc.gz",
            "warc_record_offset": 0,
            "warc_record_length": len(warc),
            "warc_record_id": RECORD_ID,
            "content_digest": _sha1_digest(payload),
        },
        crawl_id="CC-MAIN-2026-30",
    )


def _indexed_record_at(warc: bytes, payload: bytes, *, offset: int, url: str) -> MainIndexedRecord:
    return main_record_from_index_row(
        {
            "url": url,
            "warc_filename": "crawl-data/test.warc.gz",
            "warc_record_offset": offset,
            "warc_record_length": len(warc),
            "warc_record_id": RECORD_ID,
            "content_digest": _sha1_digest(payload),
        },
        crawl_id="CC-MAIN-2026-30",
    )


def _planned_range(warc: bytes, records: tuple[MainIndexedRecord, ...]) -> PlannedCommonCrawlRange:
    return PlannedCommonCrawlRange(
        source=CommonCrawlSource(
            crawl_id="CC-MAIN-2026-30",
            index_kind=CommonCrawlIndexKind.MAIN,
            paths_manifest_url="https://index.commoncrawl.org/test.paths.gz",
        ),
        warc_filename="crawl-data/test.warc.gz",
        start=0,
        stop=len(warc),
        records=records,
    )


def _record_range(warc: bytes) -> WarcRecordRange:
    return WarcRecordRange(
        crawl_id="CC-MAIN-2026-30",
        warc_filename="crawl-data/test.warc.gz",
        offset=0,
        length=len(warc),
    )


def _client(session: requests.Session, *, maximum_warc_record_bytes: int = 1 << 20) -> CommonCrawlClient:
    return CommonCrawlClient(
        session=session,
        maximum_warc_record_bytes=maximum_warc_record_bytes,
        maximum_payload_bytes=1 << 20,
    )


@pytest.mark.parametrize("compressed", [False, True])
def test_common_crawl_index_partitions_reads_manifest_and_selects_subset(compressed: bool) -> None:
    manifest = (
        b"cc-index/table/cc-main/warc/crawl=CC-MAIN-2026-30/subset=crawldiagnostics/part-00000.parquet\n"
        b"cc-index/table/cc-main/warc/crawl=CC-MAIN-2026-30/subset=warc/part-00000.parquet\n"
        b"cc-index/table/cc-main/warc/crawl=CC-MAIN-2026-30/subset=warc/part-00001.parquet\n"
    )
    body = gzip.compress(manifest) if compressed else manifest
    with requests.Session() as session:
        session.mount("https://", _ManifestAdapter(body))
        partitions = common_crawl_index_partitions(
            "https://data.commoncrawl.org/manifest.paths.gz",
            crawl_id="CC-MAIN-2026-30",
            session=session,
        )

    assert partitions == (
        "cc-index/table/cc-main/warc/crawl=CC-MAIN-2026-30/subset=warc/part-00000.parquet",
        "cc-index/table/cc-main/warc/crawl=CC-MAIN-2026-30/subset=warc/part-00001.parquet",
    )


@pytest.mark.parametrize(
    "body",
    [
        b"\x1f\x8bnot-gzip",
        gzip.compress(b""),
        gzip.compress(b"cc-index/table/crawl=CC-MAIN-2025-30/subset=warc/part-00000.parquet\n"),
        gzip.compress(b"cc-index/table/crawl=CC-MAIN-2026-30/subset=warc/not-parquet.txt\n"),
    ],
)
def test_common_crawl_index_partitions_rejects_invalid_manifest(body: bytes) -> None:
    with requests.Session() as session:
        session.mount("https://", _ManifestAdapter(body))
        with pytest.raises(CommonCrawlIndexManifestError):
            common_crawl_index_partitions(
                "https://data.commoncrawl.org/manifest.paths.gz",
                crawl_id="CC-MAIN-2026-30",
                session=session,
            )


@pytest.mark.parametrize(
    ("status", "error_type"),
    [(403, CommonCrawlDownloadError), (404, CommonCrawlRequestRejectedError)],
)
def test_common_crawl_index_partitions_classifies_http_failure(status: int, error_type: type[Exception]) -> None:
    with requests.Session() as session:
        session.mount("https://", _ManifestAdapter(b"", status=status))
        with pytest.raises(error_type):
            common_crawl_index_partitions(
                "https://data.commoncrawl.org/manifest.paths.gz",
                crawl_id="CC-MAIN-2026-30",
                session=session,
            )


def test_content_digest_uses_common_crawl_encoding() -> None:
    assert content_digest(b"abc") == "sha1:VGMT4NSHA2AWVOR6EVYXQUGCNSONBWE5"


def test_main_record_from_index_row_normalizes_uuid_blob() -> None:
    record_id = uuid.UUID("019f8700-d21d-78d8-8eb1-99eaa22579da")
    index_digest = _sha1_digest(b"index row")

    indexed_record = main_record_from_index_row(
        {
            "url": TARGET_URL,
            "warc_filename": "crawl-data/test.warc.gz",
            "warc_record_offset": 42,
            "warc_record_length": 100,
            "warc_record_id": record_id.bytes,
            "content_digest": index_digest,
        },
        crawl_id="CC-MAIN-2026-30",
    )

    assert indexed_record.expectation.warc_record_id == RECORD_ID
    assert indexed_record.record_range.http_range == (42, 141)

    string_record = main_record_from_index_row(
        {
            "url": TARGET_URL,
            "warc_filename": "crawl-data/test.warc.gz",
            "warc_record_offset": 42,
            "warc_record_length": 100,
            "warc_record_id": "urn:uuid:019f8700-d21d-78d8-8eb1-99eaa22579da",
            "content_digest": index_digest.removeprefix("sha1:"),
        },
        crawl_id="CC-MAIN-2026-30",
    )
    assert string_record.expectation.warc_record_id == RECORD_ID
    assert string_record.expectation.content_digest == index_digest


@pytest.mark.parametrize(
    "digest",
    ["sha1:ABC", "sha1:0123456789abcdef0123456789abcdef01234567", "md5:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"],
)
def test_index_row_rejects_non_base32_sha1_digest(digest: str) -> None:
    row = {
        "url": TARGET_URL,
        "warc_filename": "crawl-data/test.warc.gz",
        "warc_record_offset": 0,
        "warc_record_length": 100,
        "warc_record_id": RECORD_ID,
        "content_digest": digest,
    }

    with pytest.raises(ValueError):
        main_record_from_index_row(row, crawl_id="CC-MAIN-2026-30")


def test_supplemental_record_from_index_row_does_not_require_record_id() -> None:
    indexed_record = supplemental_record_from_index_row(
        {
            "url": TARGET_URL,
            "warc_filename": "projects/cc-open-athena-test/sample.warc.gz",
            "warc_record_offset": 908,
            "warc_record_length": 1690,
            "content_digest": "OUMEM2M6RLXNWZ5I6BRHVBBI3YURVZMP",
        },
        crawl_id="CC-SUPPLEMENTAL-2026-22",
    )

    assert isinstance(indexed_record, SupplementalIndexedRecord)
    assert indexed_record.record_range.http_range == (908, 2597)
    assert indexed_record.expectation.content_digest == "sha1:OUMEM2M6RLXNWZ5I6BRHVBBI3YURVZMP"


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


def test_fetch_record_returns_parsed_payload_then_main_verification_succeeds() -> None:
    payload = b"PK\x03\x04example docx bytes"
    warc = _warc_response(payload)
    indexed_record = _indexed_record(warc, payload)

    with _range_session(warc) as session, _client(session) as client:
        record = client.fetch_record(indexed_record.record_range)

    verify_url_index_record(record, indexed_record.expectation)

    assert record.payload == payload
    assert record.payload_digest == _sha1_digest(payload)
    assert record.warc_record_id == RECORD_ID
    assert record.target_url == TARGET_URL
    assert record.http_status == 200
    assert record.http_content_type == "application/octet-stream"
    assert record.warc_date == "2026-07-21T21:48:44Z"
    assert record.identified_payload_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def test_fetch_range_returns_planned_records_and_ignores_intervening_record() -> None:
    first_payload = b"first"
    skipped_payload = b"not selected"
    third_payload = b"third"
    first = _warc_response(first_payload, target_url="https://example.com/first")
    skipped = _warc_response(skipped_payload, target_url="https://example.com/skipped")
    third = _warc_response(third_payload, target_url="https://example.com/third")
    warc = first + skipped + third
    planned = _planned_range(
        warc,
        (
            _indexed_record_at(first, first_payload, offset=0, url="https://example.com/first"),
            _indexed_record_at(
                third,
                third_payload,
                offset=len(first) + len(skipped),
                url="https://example.com/third",
            ),
        ),
    )

    with _range_session(warc) as session, _client(session) as client:
        records = client.fetch_range(planned)

    assert [record.payload for record in records] == [first_payload, third_payload]


def test_fetch_range_resumes_from_first_unwritten_byte() -> None:
    payload = b"document"
    warc = _warc_response(payload)
    planned = _planned_range(warc, (_indexed_record_at(warc, payload, offset=0, url=TARGET_URL),))
    adapter = _TruncatingRangeAdapter(warc)

    with requests.Session() as session:
        session.mount("https://", adapter)
        with _client(session) as client:
            [record] = client.fetch_range(planned)

    assert record.payload == payload
    assert adapter.requested_ranges == [(0, len(warc) - 1), (len(warc) // 2, len(warc) - 1)]


def test_fetch_range_fails_when_a_planned_offset_is_missing() -> None:
    first_payload = b"first"
    second_payload = b"second"
    first = _warc_response(first_payload, target_url="https://example.com/first")
    second = _warc_response(second_payload, target_url="https://example.com/second")
    warc = first + second
    missing = _indexed_record_at(
        second[:-1],
        second_payload,
        offset=len(first) + 1,
        url="https://example.com/second",
    )
    planned = _planned_range(
        warc,
        (
            _indexed_record_at(first, first_payload, offset=0, url="https://example.com/first"),
            missing,
        ),
    )

    with _range_session(warc) as session, _client(session) as client:
        with pytest.raises(MissingPlannedRecordError):
            client.fetch_range(planned)


@pytest.mark.parametrize(
    "observed_record_id",
    [
        "urn:uuid:019f8700-d21d-78d8-8eb1-99eaa22579da",
        "<urn:uuid:019F8700-D21D-78D8-8EB1-99EAA22579DA>",
    ],
)
def test_url_index_verification_accepts_equivalent_uuid_spelling(observed_record_id: str) -> None:
    payload = b"document"
    warc = _warc_response(payload, record_id=observed_record_id)
    indexed_record = _indexed_record(warc, payload)

    with _range_session(warc) as session, _client(session) as client:
        record = client.fetch_record(indexed_record.record_range)

    verify_url_index_record(record, indexed_record.expectation)


def test_fetch_record_rejects_incorrect_content_range() -> None:
    payload = b"document"
    warc = _warc_response(payload)
    record_range = _record_range(warc)

    with _range_session(warc, content_range=f"bytes 1-{len(warc)}/{len(warc)}") as session:
        with _client(session) as client:
            with pytest.raises(CommonCrawlTransientError):
                client.fetch_record(record_range)


def test_fetch_record_rejects_full_response_to_range_request() -> None:
    payload = b"document"
    warc = _warc_response(payload)

    with _range_session(warc, response_status=requests.codes.ok) as session, _client(session) as client:
        with pytest.raises(CommonCrawlTransientError):
            client.fetch_record(_record_range(warc))


@pytest.mark.parametrize("status", [400, 404])
def test_fetch_record_classifies_permanent_object_error(status: int) -> None:
    payload = b"missing"
    warc = _warc_response(payload)

    with _range_session(warc, response_status=status) as session, _client(session) as client:
        with pytest.raises(CommonCrawlRequestRejectedError) as error:
            client.fetch_record(_record_range(warc))

    assert error.value.status == status
    assert error.value.url.endswith("/crawl-data/test.warc.gz")


def test_fetch_record_classifies_rate_limit_as_transient() -> None:
    payload = b"rate limited"
    warc = _warc_response(payload)

    with _range_session(warc, response_status=403) as session, _client(session) as client:
        with pytest.raises(CommonCrawlDownloadError):
            client.fetch_record(_record_range(warc))


def test_url_index_verification_rejects_payload_digest_mismatch() -> None:
    payload = b"document"
    warc = _warc_response(payload)
    indexed_record = _indexed_record(warc, payload)
    expectation = replace(
        indexed_record.expectation,
        content_digest="sha1:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    )

    with _range_session(warc) as session, _client(session) as client:
        record = client.fetch_record(indexed_record.record_range)

    with pytest.raises(RecordVerificationError):
        verify_url_index_record(record, expectation)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("url", "https://example.com/different.docx"),
        ("warc_record_id", "<urn:uuid:119f8700-d21d-78d8-8eb1-99eaa22579da>"),
    ],
)
def test_url_index_verification_rejects_identity_mismatch(field: str, value: str) -> None:
    payload = b"document"
    warc = _warc_response(payload)
    indexed_record = _indexed_record(warc, payload)
    expectation = replace(indexed_record.expectation, **{field: value})

    with _range_session(warc) as session, _client(session) as client:
        record = client.fetch_record(indexed_record.record_range)

    with pytest.raises(RecordVerificationError):
        verify_url_index_record(record, expectation)


def test_url_index_verification_rejects_non_success_origin_response() -> None:
    payload = b"not found"
    warc = _warc_response(payload, http_status="404 Not Found")

    with _range_session(warc) as session, _client(session) as client:
        record = client.fetch_record(_record_range(warc))

    with pytest.raises(OriginResponseStatusError) as error:
        verify_url_index_record(record, _indexed_record(warc, payload).expectation)

    assert error.value.status == 404


def test_url_index_verification_reports_identity_mismatch_before_origin_status() -> None:
    payload = b"wrong response"
    warc = _warc_response(payload, target_url="https://example.com/wrong", http_status="404 Not Found")
    indexed_record = _indexed_record(warc, payload)

    with _range_session(warc) as session, _client(session) as client:
        record = client.fetch_record(indexed_record.record_range)

    with pytest.raises(RecordVerificationError):
        verify_url_index_record(record, indexed_record.expectation)


def test_fetch_record_distinguishes_revisit_record() -> None:
    warc = _warc_revisit()

    with _range_session(warc) as session, _client(session) as client:
        with pytest.raises(WarcRevisitError):
            client.fetch_record(_record_range(warc))


def test_fetch_record_rejects_non_response_record() -> None:
    warc = _warc_record("metadata", include_http_headers=False)

    with _range_session(warc) as session, _client(session) as client:
        with pytest.raises(WarcParsingError):
            client.fetch_record(_record_range(warc))


def test_fetch_record_rejects_response_without_http_headers() -> None:
    metadata_warc = _warc_record("metadata", include_http_headers=False, gzip_output=False, payload=b"")
    warc = metadata_warc.replace(b"WARC-Type: metadata", b"WARC-Type: response", 1)

    with _range_session(warc) as session, _client(session) as client:
        with pytest.raises(WarcParsingError):
            client.fetch_record(_record_range(warc))


def test_fetch_record_rejects_multiple_records_in_range() -> None:
    warc = _warc_response(b"first") + _warc_response(b"second")

    with _range_session(warc) as session, _client(session) as client:
        with pytest.raises(WarcParsingError):
            client.fetch_record(_record_range(warc))


@pytest.mark.parametrize("body_delta", [-1, 1])
def test_fetch_record_rejects_body_length_different_from_range(body_delta: int) -> None:
    warc = _warc_response(b"document")
    body = warc[:-1] if body_delta < 0 else warc + b"x"

    with _range_session(warc, body_override=body, include_content_length=False) as session, _client(session) as client:
        with pytest.raises(CommonCrawlTransientError):
            client.fetch_record(_record_range(warc))


def test_fetch_record_rejects_invalid_content_length() -> None:
    warc = _warc_response(b"document")

    with _range_session(warc, content_length="not-an-integer") as session, _client(session) as client:
        with pytest.raises(CommonCrawlTransientError):
            client.fetch_record(_record_range(warc))


def test_fetch_record_rejects_index_range_over_limit_before_download() -> None:
    payload = b"document"
    warc = _warc_response(payload)

    with _range_session(warc) as session, _client(session, maximum_warc_record_bytes=len(warc) - 1) as client:
        with pytest.raises(WarcRecordTooLargeError):
            client.fetch_record(_record_range(warc))


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
                client.fetch_record(_record_range(warc))


def test_supplemental_verification_uses_url_and_digest_without_expected_record_id() -> None:
    payload = b"supplemental document"
    warc = _warc_response(payload)
    indexed_record = supplemental_record_from_index_row(
        {
            "url": TARGET_URL,
            "warc_filename": "crawl-data/test.warc.gz",
            "warc_record_offset": 0,
            "warc_record_length": len(warc),
            "content_digest": _sha1_digest(payload),
        },
        crawl_id="CC-SUPPLEMENTAL-2026-22",
    )

    with _range_session(warc) as session, _client(session) as client:
        record = client.fetch_record(indexed_record.record_range)

    verify_supplemental_record(record, indexed_record.expectation)
    assert record.warc_record_id == RECORD_ID


def test_supplemental_verification_preserves_non_uuid_record_id() -> None:
    payload = b"supplemental document"
    warc = _warc_response(payload, record_id="<https://example.com/rec/1>")
    indexed_record = supplemental_record_from_index_row(
        {
            "url": TARGET_URL,
            "warc_filename": "crawl-data/test.warc.gz",
            "warc_record_offset": 0,
            "warc_record_length": len(warc),
            "content_digest": _sha1_digest(payload),
        },
        crawl_id="CC-SUPPLEMENTAL-2026-22",
    )

    with _range_session(warc) as session, _client(session) as client:
        record = client.fetch_record(indexed_record.record_range)

    verify_supplemental_record(record, indexed_record.expectation)
    assert record.warc_record_id == "<https://example.com/rec/1>"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("url", "https://example.com/different.docx"),
        ("content_digest", "sha1:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"),
    ],
)
def test_supplemental_verification_rejects_index_mismatch(field: str, value: str) -> None:
    payload = b"supplemental document"
    warc = _warc_response(payload)
    indexed_record = supplemental_record_from_index_row(
        {
            "url": TARGET_URL,
            "warc_filename": "crawl-data/test.warc.gz",
            "warc_record_offset": 0,
            "warc_record_length": len(warc),
            "content_digest": _sha1_digest(payload),
        },
        crawl_id="CC-SUPPLEMENTAL-2026-22",
    )
    expectation = replace(indexed_record.expectation, **{field: value})

    with _range_session(warc) as session, _client(session) as client:
        record = client.fetch_record(indexed_record.record_range)

    with pytest.raises(RecordVerificationError):
        verify_supplemental_record(record, expectation)
