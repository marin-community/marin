# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fetch and verify individual records from Common Crawl WARC files."""

import base64
import hashlib
import io
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass

import requests
from warcio.archiveiterator import ArchiveIterator
from warcio.exceptions import ArchiveLoadFailed

from marin.datakit.download.http_session import build_retrying_session

COMMON_CRAWL_DATA_URL = "https://data.commoncrawl.org"
_DOWNLOAD_CHUNK_BYTES = 1 << 20
_RETRY_STATUS = (403, 429, 500, 502, 503, 504)
_RETRY_TOTAL = 10
_RETRY_BACKOFF_FACTOR = 2.0
_RETRY_BACKOFF_JITTER = 10.0
_REQUEST_TIMEOUT = (30, 300)
_CONTENT_RANGE_PATTERN = re.compile(r"bytes (\d+)-(\d+)/(?:\d+|\*)")


class CommonCrawlWarcError(RuntimeError):
    """Base class for Common Crawl record retrieval and validation failures."""


class CommonCrawlTransientError(CommonCrawlWarcError):
    """Base class for failures that a task-level retry may resolve."""


class CommonCrawlDownloadError(CommonCrawlTransientError):
    """Raised after retryable HTTP download attempts are exhausted."""


class CommonCrawlRequestRejectedError(CommonCrawlWarcError):
    """Raised when Common Crawl permanently rejects an object request."""

    def __init__(self, url: str, status: int) -> None:
        self.url = url
        self.status = status
        super().__init__(f"Common Crawl object request returned HTTP {status}: {url}")


class RangeResponseError(CommonCrawlTransientError):
    """Raised when a server does not honor an exact WARC byte-range request."""


class WarcRecordTooLargeError(CommonCrawlWarcError):
    """Raised when the URL Index range exceeds the configured WARC record limit."""


class WarcPayloadTooLargeError(CommonCrawlWarcError):
    """Raised when a parsed response payload exceeds the configured payload limit."""


class WarcParsingError(CommonCrawlWarcError):
    """Raised when a byte range does not contain exactly one WARC response record."""


class WarcRevisitError(WarcParsingError):
    """Raised when a selected range contains a revisit rather than a response payload."""


class RecordVerificationError(CommonCrawlWarcError):
    """Raised when a fetched WARC record does not match its URL Index row."""


class OriginResponseStatusError(CommonCrawlWarcError):
    """Raised when a WARC response contains a non-successful origin status."""

    def __init__(self, status: int) -> None:
        self.status = status
        super().__init__(f"Origin HTTP status {status} is not successful")


@dataclass(frozen=True)
class CommonCrawlRecordLocation:
    """Record-level location and integrity fields from the Common Crawl URL Index."""

    crawl_id: str
    url: str
    warc_filename: str
    warc_record_offset: int
    warc_record_length: int
    warc_record_id: str
    content_digest: str

    def __post_init__(self) -> None:
        if self.warc_record_offset < 0:
            raise ValueError("warc_record_offset must be non-negative")
        if self.warc_record_length <= 0:
            raise ValueError("warc_record_length must be positive")

    @property
    def byte_range(self) -> tuple[int, int]:
        """Return the inclusive HTTP byte range for this record."""
        return self.warc_record_offset, self.warc_record_offset + self.warc_record_length - 1

    @classmethod
    def from_url_index_row(cls, row: Mapping[str, object], *, crawl_id: str) -> "CommonCrawlRecordLocation":
        """Build a location from one URL Index Parquet row."""
        return cls(
            crawl_id=crawl_id,
            url=_required_string(row, "url"),
            warc_filename=_required_string(row, "warc_filename"),
            warc_record_offset=_required_int(row, "warc_record_offset"),
            warc_record_length=_required_int(row, "warc_record_length"),
            warc_record_id=_canonical_record_id(row.get("warc_record_id")),
            content_digest=_canonical_content_digest(_required_string(row, "content_digest")),
        )


@dataclass(frozen=True)
class CommonCrawlWarcRecord:
    """Verified HTTP response extracted from a Common Crawl WARC record."""

    payload: bytes
    warc_record_id: str
    target_url: str
    http_status: int
    http_content_type: str | None
    warc_date: str | None
    identified_payload_type: str | None


class CommonCrawlClient:
    """Retrying client for exact Common Crawl WARC record range requests."""

    def __init__(
        self,
        *,
        maximum_warc_record_bytes: int,
        maximum_payload_bytes: int,
        base_url: str = COMMON_CRAWL_DATA_URL,
        request_timeout: tuple[int, int] = _REQUEST_TIMEOUT,
        session: requests.Session | None = None,
    ) -> None:
        if maximum_warc_record_bytes <= 0:
            raise ValueError("maximum_warc_record_bytes must be positive")
        if maximum_payload_bytes <= 0:
            raise ValueError("maximum_payload_bytes must be positive")
        self._base_url = base_url.rstrip("/")
        self._request_timeout = request_timeout
        self._maximum_warc_record_bytes = maximum_warc_record_bytes
        self._maximum_payload_bytes = maximum_payload_bytes
        self._session = session or build_retrying_session(
            total=_RETRY_TOTAL,
            backoff_factor=_RETRY_BACKOFF_FACTOR,
            backoff_jitter=_RETRY_BACKOFF_JITTER,
            status_forcelist=_RETRY_STATUS,
        )
        self._owns_session = session is None

    def __enter__(self) -> "CommonCrawlClient":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if self._owns_session:
            self._session.close()

    def fetch_record(self, location: CommonCrawlRecordLocation) -> CommonCrawlWarcRecord:
        """Range-fetch, parse, and verify one URL Index record."""
        if location.warc_record_length > self._maximum_warc_record_bytes:
            raise WarcRecordTooLargeError(
                f"WARC record length {location.warc_record_length} exceeds limit {self._maximum_warc_record_bytes}"
            )

        start, end = location.byte_range
        url = f"{self._base_url}/{location.warc_filename.lstrip('/')}"
        try:
            with self._session.get(
                url,
                headers={"Range": f"bytes={start}-{end}"},
                stream=True,
                timeout=self._request_timeout,
            ) as response:
                response.raise_for_status()
                _validate_range_response_headers(
                    response,
                    start=start,
                    end=end,
                    expected_length=location.warc_record_length,
                )
                content = _read_exact_response(response, expected_length=location.warc_record_length)
        except requests.HTTPError as error:
            response = error.response
            status = response.status_code if response is not None else None
            if status is not None and 400 <= status < 500 and status not in _RETRY_STATUS:
                raise CommonCrawlRequestRejectedError(url, status) from error
            raise CommonCrawlDownloadError(f"Failed to download WARC record from {url}") from error
        except requests.RequestException as error:
            raise CommonCrawlDownloadError(f"Failed to download WARC record from {url}") from error

        record = _parse_warc_response(content, maximum_payload_bytes=self._maximum_payload_bytes)
        _verify_record(record, location)
        return record


def _required_string(row: Mapping[str, object], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _required_int(row: Mapping[str, object], field: str) -> int:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _canonical_record_id(value: object) -> str:
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, bytes):
        if len(value) != 16:
            raise ValueError("warc_record_id BLOB must contain a 16-byte UUID")
        record_uuid = uuid.UUID(bytes=value)
    elif isinstance(value, str):
        text = value.strip()
        if text.startswith("<urn:uuid:") and text.endswith(">"):
            text = text[len("<urn:uuid:") : -1]
        elif text.startswith("urn:uuid:"):
            text = text[len("urn:uuid:") :]
        try:
            record_uuid = uuid.UUID(text)
        except ValueError as error:
            raise ValueError(f"Invalid WARC record ID: {value!r}") from error
    else:
        raise ValueError("warc_record_id must be a UUID string or 16-byte BLOB")
    return f"<urn:uuid:{record_uuid}>"


def _canonical_content_digest(value: str) -> str:
    algorithm, separator, encoded_digest = value.partition(":")
    if not separator:
        algorithm, encoded_digest = "sha1", algorithm
    if algorithm.lower() != "sha1" or not encoded_digest:
        raise ValueError(f"Unsupported Common Crawl content digest: {value!r}")
    return f"sha1:{encoded_digest.upper().rstrip('=')}"


def _validate_range_response_headers(
    response: requests.Response,
    *,
    start: int,
    end: int,
    expected_length: int,
) -> None:
    if response.status_code != requests.codes.partial_content:
        raise RangeResponseError(f"Expected HTTP 206 for WARC range request, received {response.status_code}")

    content_range = response.headers.get("Content-Range", "")
    match = _CONTENT_RANGE_PATTERN.fullmatch(content_range)
    if match is None or (int(match.group(1)), int(match.group(2))) != (start, end):
        raise RangeResponseError(f"Expected Content-Range bytes {start}-{end}, received {content_range!r}")
    content_length = response.headers.get("Content-Length")
    if content_length is not None:
        try:
            observed_length = int(content_length)
        except ValueError as error:
            raise RangeResponseError(f"Invalid Content-Length {content_length!r}") from error
        if observed_length != expected_length:
            raise RangeResponseError(f"Expected Content-Length {expected_length}, received {content_length!r}")


def _read_exact_response(response: requests.Response, *, expected_length: int) -> bytes:
    content = bytearray()
    for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_BYTES):
        content.extend(chunk)
        if len(content) > expected_length:
            raise RangeResponseError(f"WARC range exceeded expected length {expected_length}")
    if len(content) != expected_length:
        raise RangeResponseError(f"Expected {expected_length} WARC bytes, received {len(content)}")
    return bytes(content)


def _parse_warc_response(content: bytes, *, maximum_payload_bytes: int) -> CommonCrawlWarcRecord:
    iterator = iter(ArchiveIterator(io.BytesIO(content)))
    try:
        record = next(iterator)
    except StopIteration as error:
        raise WarcParsingError("WARC byte range contained no records") from error
    except ArchiveLoadFailed as error:
        raise WarcParsingError("WARC byte range could not be parsed") from error

    if record.rec_type == "revisit":
        raise WarcRevisitError("WARC range contains a revisit record without an embedded payload")
    if record.rec_type != "response":
        raise WarcParsingError(f"Expected a WARC response record, received {record.rec_type!r}")
    if record.http_headers is None:
        raise WarcParsingError("WARC response record did not contain HTTP headers")

    # Common Crawl's content digest covers the decoded HTTP entity returned by
    # warcio's content_stream(), not the transfer/content-encoded wire bytes.
    payload = record.content_stream().read(maximum_payload_bytes + 1)
    if len(payload) > maximum_payload_bytes:
        raise WarcPayloadTooLargeError(f"WARC payload exceeds limit {maximum_payload_bytes}")
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise WarcParsingError("WARC byte range contained more than one record")

    record_id = record.rec_headers.get_header("WARC-Record-ID")
    target_url = record.rec_headers.get_header("WARC-Target-URI")
    if record_id is None or target_url is None:
        raise WarcParsingError("WARC response record is missing its record ID or target URI")
    status_code = record.http_headers.get_statuscode()
    if status_code is None or not status_code.isdigit():
        raise WarcParsingError(f"WARC response record has invalid HTTP status {status_code!r}")

    return CommonCrawlWarcRecord(
        payload=payload,
        warc_record_id=_canonical_record_id(record_id),
        target_url=target_url,
        http_status=int(status_code),
        http_content_type=record.http_headers.get_header("Content-Type"),
        warc_date=record.rec_headers.get_header("WARC-Date"),
        identified_payload_type=record.rec_headers.get_header("WARC-Identified-Payload-Type"),
    )


def _verify_record(record: CommonCrawlWarcRecord, location: CommonCrawlRecordLocation) -> None:
    if not 200 <= record.http_status < 300:
        raise OriginResponseStatusError(record.http_status)

    mismatches = []
    if record.warc_record_id != location.warc_record_id:
        mismatches.append(f"record ID {record.warc_record_id!r} != {location.warc_record_id!r}")
    if record.target_url != location.url:
        mismatches.append(f"target URL {record.target_url!r} != {location.url!r}")

    digest = hashlib.sha1(record.payload, usedforsecurity=False).digest()
    observed_digest = f"sha1:{base64.b32encode(digest).decode().rstrip('=')}"
    if observed_digest != location.content_digest:
        mismatches.append(f"content digest {observed_digest!r} != {location.content_digest!r}")
    if mismatches:
        raise RecordVerificationError("; ".join(mismatches))
