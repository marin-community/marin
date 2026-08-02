# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Discover Common Crawl index partitions and fetch and verify WARC records."""

import base64
import gzip
import hashlib
import http.client
import io
import re
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import BinaryIO, Protocol

import requests
from warcio.archiveiterator import ArchiveIterator
from warcio.exceptions import ArchiveLoadFailed

from marin.datakit.download.http_session import build_retrying_session

COMMON_CRAWL_DATA_URL = "https://data.commoncrawl.org"
COMMON_CRAWL_USER_AGENT = "marin-common-crawl-ingress/1.0"
_DOWNLOAD_CHUNK_BYTES = 1 << 20
_RETRY_STATUS = (403, 429, 500, 502, 503, 504)
_RETRY_TOTAL = 10
_RETRY_BACKOFF_FACTOR = 2.0
_RETRY_BACKOFF_JITTER = 10.0
_REQUEST_TIMEOUT = (30, 300)
_CONTENT_RANGE_PATTERN = re.compile(r"bytes (\d+)-(\d+)/(?:\d+|\*)")
_SHA1_BASE32_PATTERN = re.compile(r"[A-Z2-7]{32}")
_MAXIMUM_MANIFEST_BYTES = 16 << 20
_MAXIMUM_DECOMPRESSED_MANIFEST_BYTES = 128 << 20
_MAXIMUM_DOWNLOAD_STALLS = 8


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
    """Raised when a fetched WARC record does not match its index expectations."""


class MissingPlannedRecordError(CommonCrawlWarcError):
    """Raised when a coalesced range omits an expected record offset."""


class OriginResponseStatusError(CommonCrawlWarcError):
    """Raised when a WARC response contains a non-successful origin status."""

    def __init__(self, status: int) -> None:
        self.status = status
        super().__init__(f"Origin HTTP status {status} is not successful")


class CommonCrawlIndexManifestError(ValueError):
    """Raised when an index partition manifest is invalid for the requested crawl."""


@dataclass(frozen=True)
class WarcRecordRange:
    """Coordinates for one independently compressed WARC record."""

    crawl_id: str
    warc_filename: str
    offset: int
    length: int

    def __post_init__(self) -> None:
        if self.offset < 0:
            raise ValueError("offset must be non-negative")
        if self.length <= 0:
            raise ValueError("length must be positive")

    @property
    def stop(self) -> int:
        """Return the exclusive end offset."""
        return self.offset + self.length

    @property
    def http_range(self) -> tuple[int, int]:
        """Return the inclusive HTTP byte range."""
        return self.offset, self.stop - 1


@dataclass(frozen=True)
class UrlIndexRecordExpectation:
    """Identity fields available from the current CC-MAIN URL Index."""

    url: str
    warc_record_id: str
    content_digest: str


@dataclass(frozen=True)
class SupplementalRecordExpectation:
    """Identity fields available from CC-SUPPLEMENTAL URL indexes."""

    url: str
    content_digest: str


@dataclass(frozen=True)
class MainIndexedRecord:
    """Transport coordinates and identity fields from a CC-MAIN index row."""

    record_range: WarcRecordRange
    expectation: UrlIndexRecordExpectation


@dataclass(frozen=True)
class SupplementalIndexedRecord:
    """Transport coordinates and identity fields from a supplemental index row."""

    record_range: WarcRecordRange
    expectation: SupplementalRecordExpectation


@dataclass(frozen=True)
class CommonCrawlWarcRecord:
    """Observed HTTP response extracted from a Common Crawl WARC record."""

    payload: bytes
    payload_digest: str
    warc_record_id: str
    target_url: str
    http_status: int
    http_content_type: str | None
    warc_date: str | None
    identified_payload_type: str | None


class CommonCrawlRangeSource(Protocol):
    """Source fields required by coalesced range transport."""

    base_url: str


class PlannedCommonCrawlRange(Protocol):
    """Structural boundary implemented by the shared planner's range type."""

    source: CommonCrawlRangeSource
    warc_filename: str
    start: int
    stop: int
    records: Sequence[MainIndexedRecord | SupplementalIndexedRecord]


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
        self._session = session or _build_common_crawl_session()
        self._owns_session = session is None

    def __enter__(self) -> "CommonCrawlClient":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if self._owns_session:
            self._session.close()

    def fetch_record(self, record_range: WarcRecordRange) -> CommonCrawlWarcRecord:
        """Range-fetch and parse one WARC record without source-specific verification."""
        if record_range.length > self._maximum_warc_record_bytes:
            raise WarcRecordTooLargeError(
                f"WARC record length {record_range.length} exceeds limit {self._maximum_warc_record_bytes}"
            )

        start, end = record_range.http_range
        url = f"{self._base_url}/{record_range.warc_filename.lstrip('/')}"
        try:
            with self._session.get(
                url,
                headers={"Range": f"bytes={start}-{end}", "user-agent": COMMON_CRAWL_USER_AGENT},
                stream=True,
                timeout=self._request_timeout,
            ) as response:
                response.raise_for_status()
                _validate_range_response_headers(
                    response,
                    start=start,
                    end=end,
                    expected_length=record_range.length,
                )
                content = _read_exact_response(response, expected_length=record_range.length)
        except requests.HTTPError as error:
            error_response = error.response
            status = error_response.status_code if error_response is not None else None
            if status is not None and 400 <= status < 500 and status not in _RETRY_STATUS:
                raise CommonCrawlRequestRejectedError(url, status) from error
            raise CommonCrawlDownloadError(f"Failed to download WARC record from {url}") from error
        except requests.RequestException as error:
            raise CommonCrawlDownloadError(f"Failed to download WARC record from {url}") from error

        return _parse_warc_response(content, maximum_payload_bytes=self._maximum_payload_bytes)

    def fetch_range(self, planned_range: PlannedCommonCrawlRange) -> tuple[CommonCrawlWarcRecord, ...]:
        """Fetch a coalesced range and return its verified planned records in offset order."""
        if planned_range.stop <= planned_range.start:
            raise ValueError("planned range stop must be greater than start")
        expected: dict[int, MainIndexedRecord | SupplementalIndexedRecord] = {}
        previous_stop = planned_range.start
        for indexed in planned_range.records:
            location = indexed.record_range
            if location.warc_filename != planned_range.warc_filename:
                raise ValueError("planned record belongs to another WARC")
            if location.offset < planned_range.start or location.stop > planned_range.stop:
                raise ValueError("planned record lies outside the coalesced range")
            if location.offset < previous_stop:
                raise ValueError("planned records overlap or are out of order")
            if location.length > self._maximum_warc_record_bytes:
                raise WarcRecordTooLargeError(
                    f"WARC record length {location.length} exceeds limit {self._maximum_warc_record_bytes}"
                )
            expected[location.offset] = indexed
            previous_stop = location.stop

        base_url = planned_range.source.base_url.rstrip("/") or self._base_url
        url = f"{base_url}/{planned_range.warc_filename.lstrip('/')}"
        with tempfile.TemporaryFile(prefix="common-crawl-range-", suffix=".warc.gz") as stream:
            self._download_range(
                url,
                start=planned_range.start,
                stop=planned_range.stop,
                destination=stream,
            )
            return _parse_planned_records(
                stream,
                range_start=planned_range.start,
                expected=expected,
                maximum_payload_bytes=self._maximum_payload_bytes,
            )

    def _download_range(self, url: str, *, start: int, stop: int, destination: BinaryIO) -> None:
        """Stream an exact byte range, resuming from the first unwritten byte after interruption."""
        expected_bytes = stop - start
        stalls = 0
        while destination.tell() < expected_bytes:
            written = destination.tell()
            request_start = start + written
            error: Exception | None = None
            try:
                with self._session.get(
                    url,
                    headers={"Range": f"bytes={request_start}-{stop - 1}", "user-agent": _USER_AGENT},
                    stream=True,
                    timeout=self._request_timeout,
                ) as response:
                    response.raise_for_status()
                    _validate_range_response_headers(
                        response,
                        start=request_start,
                        end=stop - 1,
                        expected_length=stop - request_start,
                    )
                    for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_BYTES):
                        if chunk:
                            destination.write(chunk)
            except requests.HTTPError as caught:
                status = caught.response.status_code if caught.response is not None else None
                if status is not None and 400 <= status < 500 and status not in _RETRY_STATUS:
                    raise CommonCrawlRequestRejectedError(url, status) from caught
                error = caught
            except (requests.RequestException, http.client.IncompleteRead) as caught:
                error = caught

            if destination.tell() > written:
                stalls = 0
            else:
                stalls += 1
            if destination.tell() > expected_bytes:
                raise RangeResponseError(f"WARC range exceeded expected length {expected_bytes}")
            if stalls > _MAXIMUM_DOWNLOAD_STALLS:
                raise CommonCrawlDownloadError(
                    f"WARC range download stalled at {destination.tell()}/{expected_bytes} bytes"
                ) from error
        destination.seek(0)


def main_record_from_index_row(row: Mapping[str, object], *, crawl_id: str) -> MainIndexedRecord:
    """Build transport coordinates and CC-MAIN expectations from one index row."""
    return MainIndexedRecord(
        record_range=_record_range_from_index_row(row, crawl_id=crawl_id),
        expectation=UrlIndexRecordExpectation(
            url=_required_string(row, "url"),
            warc_record_id=_canonical_record_id(row.get("warc_record_id")),
            content_digest=_canonical_content_digest(_required_string(row, "content_digest")),
        ),
    )


def supplemental_record_from_index_row(row: Mapping[str, object], *, crawl_id: str) -> SupplementalIndexedRecord:
    """Build transport coordinates and supplemental expectations from one index row."""
    return SupplementalIndexedRecord(
        record_range=_record_range_from_index_row(row, crawl_id=crawl_id),
        expectation=SupplementalRecordExpectation(
            url=_required_string(row, "url"),
            content_digest=_canonical_content_digest(_required_string(row, "content_digest")),
        ),
    )


def common_crawl_index_partitions(
    paths_manifest_url: str,
    *,
    crawl_id: str,
    subset: str = "warc",
    session: requests.Session | None = None,
    request_timeout: tuple[int, int] = _REQUEST_TIMEOUT,
) -> tuple[str, ...]:
    """Read and validate URL Index Parquet paths for one crawl and subset."""
    if not crawl_id:
        raise ValueError("crawl_id must be non-empty")
    if not subset:
        raise ValueError("subset must be non-empty")
    owns_session = session is None
    active_session = session or _build_common_crawl_session()
    try:
        with active_session.get(
            paths_manifest_url,
            headers={"user-agent": COMMON_CRAWL_USER_AGENT},
            stream=True,
            timeout=request_timeout,
        ) as response:
            response.raise_for_status()
            encoded_manifest = _read_bounded_response(response, maximum_bytes=_MAXIMUM_MANIFEST_BYTES)
    except requests.HTTPError as error:
        error_response = error.response
        status = error_response.status_code if error_response is not None else None
        if status is not None and 400 <= status < 500 and status not in _RETRY_STATUS:
            raise CommonCrawlRequestRejectedError(paths_manifest_url, status) from error
        raise CommonCrawlDownloadError(
            f"Failed to download Common Crawl index manifest: {paths_manifest_url}"
        ) from error
    except requests.RequestException as error:
        raise CommonCrawlDownloadError(
            f"Failed to download Common Crawl index manifest: {paths_manifest_url}"
        ) from error
    finally:
        if owns_session:
            active_session.close()

    manifest = _decode_paths_manifest(encoded_manifest, paths_manifest_url=paths_manifest_url)
    paths = tuple(line.strip() for line in manifest.splitlines() if line.strip())
    invalid_paths = tuple(path for path in paths if not path.endswith(".parquet") or crawl_id not in path)
    if invalid_paths:
        raise CommonCrawlIndexManifestError(
            f"Common Crawl index manifest contains paths outside crawl {crawl_id!r} or non-Parquet paths"
        )
    expected_subset_component = f"subset={subset}"
    partitions = tuple(path for path in paths if expected_subset_component in path.split("/"))
    if not partitions:
        raise CommonCrawlIndexManifestError(
            f"Common Crawl index manifest has no partitions for crawl {crawl_id!r}, subset {subset!r}"
        )
    return partitions


def _build_common_crawl_session() -> requests.Session:
    return build_retrying_session(
        total=_RETRY_TOTAL,
        backoff_factor=_RETRY_BACKOFF_FACTOR,
        backoff_jitter=_RETRY_BACKOFF_JITTER,
        status_forcelist=_RETRY_STATUS,
    )


def _read_bounded_response(response: requests.Response, *, maximum_bytes: int) -> bytes:
    content = bytearray()
    for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_BYTES):
        content.extend(chunk)
        if len(content) > maximum_bytes:
            raise CommonCrawlIndexManifestError(f"Common Crawl index manifest exceeds {maximum_bytes} bytes")
    return bytes(content)


def _decode_paths_manifest(encoded_manifest: bytes, *, paths_manifest_url: str) -> str:
    try:
        if encoded_manifest.startswith(b"\x1f\x8b"):
            with gzip.GzipFile(fileobj=io.BytesIO(encoded_manifest)) as stream:
                decoded_manifest = stream.read(_MAXIMUM_DECOMPRESSED_MANIFEST_BYTES + 1)
        else:
            decoded_manifest = encoded_manifest
        if len(decoded_manifest) > _MAXIMUM_DECOMPRESSED_MANIFEST_BYTES:
            raise CommonCrawlIndexManifestError(
                f"Common Crawl index manifest expands beyond {_MAXIMUM_DECOMPRESSED_MANIFEST_BYTES} bytes"
            )
        return decoded_manifest.decode()
    except (gzip.BadGzipFile, EOFError, UnicodeDecodeError) as error:
        raise CommonCrawlIndexManifestError(f"Invalid Common Crawl index manifest: {paths_manifest_url}") from error


def _record_range_from_index_row(row: Mapping[str, object], *, crawl_id: str) -> WarcRecordRange:
    return WarcRecordRange(
        crawl_id=crawl_id,
        warc_filename=_required_string(row, "warc_filename"),
        offset=_required_int(row, "warc_record_offset"),
        length=_required_int(row, "warc_record_length"),
    )


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
    encoded_digest = encoded_digest.upper().rstrip("=")
    if algorithm.lower() != "sha1" or _SHA1_BASE32_PATTERN.fullmatch(encoded_digest) is None:
        raise ValueError(f"Unsupported Common Crawl content digest: {value!r}")
    return f"sha1:{encoded_digest}"


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
        payload_digest=content_digest(payload),
        # WARC-Record-ID permits any absolute URI. CC-MAIN UUID expectations
        # are normalized separately and compared during source verification.
        warc_record_id=record_id,
        target_url=target_url,
        http_status=int(status_code),
        http_content_type=record.http_headers.get_header("Content-Type"),
        warc_date=record.rec_headers.get_header("WARC-Date"),
        identified_payload_type=record.rec_headers.get_header("WARC-Identified-Payload-Type"),
    )


def _parse_planned_records(
    stream: BinaryIO,
    *,
    range_start: int,
    expected: Mapping[int, MainIndexedRecord | SupplementalIndexedRecord],
    maximum_payload_bytes: int,
) -> tuple[CommonCrawlWarcRecord, ...]:
    """Walk a downloaded interval and materialize only records named by absolute offset."""
    found: dict[int, CommonCrawlWarcRecord] = {}
    records = ArchiveIterator(stream)
    try:
        for record in records:
            payload = record.content_stream().read(maximum_payload_bytes + 1)
            absolute_offset = range_start + records.get_record_offset()
            indexed = expected.get(absolute_offset)
            if indexed is None:
                continue
            if absolute_offset in found:
                raise WarcParsingError(f"WARC range contained duplicate record offset {absolute_offset}")
            if record.rec_type == "revisit":
                raise WarcRevisitError(f"Planned offset {absolute_offset} contains a revisit record")
            if record.rec_type != "response" or record.http_headers is None:
                raise WarcParsingError(f"Planned offset {absolute_offset} is not a WARC response record")
            if len(payload) > maximum_payload_bytes:
                raise WarcPayloadTooLargeError(f"WARC payload exceeds limit {maximum_payload_bytes}")
            parsed = _warc_record_from_archive(record, payload)
            if isinstance(indexed, MainIndexedRecord):
                verify_url_index_record(parsed, indexed.expectation)
            else:
                verify_supplemental_record(parsed, indexed.expectation)
            found[absolute_offset] = parsed
    except ArchiveLoadFailed as error:
        raise WarcParsingError("WARC byte range could not be parsed") from error

    missing = sorted(set(expected) - set(found))
    if missing:
        raise MissingPlannedRecordError(f"WARC range omitted {len(missing)} planned records: {missing[:8]}")
    return tuple(found[offset] for offset in sorted(found))


def _warc_record_from_archive(record: object, payload: bytes) -> CommonCrawlWarcRecord:
    """Build the shared observed-record shape from one consumed warcio record."""
    rec_headers = record.rec_headers  # pyrefly: ignore[missing-attribute]
    http_headers = record.http_headers  # pyrefly: ignore[missing-attribute]
    record_id = rec_headers.get_header("WARC-Record-ID")
    target_url = rec_headers.get_header("WARC-Target-URI")
    if record_id is None or target_url is None:
        raise WarcParsingError("WARC response record is missing its record ID or target URI")
    status_code = http_headers.get_statuscode()
    if status_code is None or not status_code.isdigit():
        raise WarcParsingError(f"WARC response record has invalid HTTP status {status_code!r}")
    return CommonCrawlWarcRecord(
        payload=payload,
        payload_digest=content_digest(payload),
        warc_record_id=record_id,
        target_url=target_url,
        http_status=int(status_code),
        http_content_type=http_headers.get_header("Content-Type"),
        warc_date=rec_headers.get_header("WARC-Date"),
        identified_payload_type=rec_headers.get_header("WARC-Identified-Payload-Type"),
    )


def verify_url_index_record(record: CommonCrawlWarcRecord, expected: UrlIndexRecordExpectation) -> None:
    """Verify an observed record against the fields available from CC-MAIN."""
    mismatches = []
    if _comparable_record_id(record.warc_record_id) != _comparable_record_id(expected.warc_record_id):
        mismatches.append(f"record ID {record.warc_record_id!r} != {expected.warc_record_id!r}")
    _verify_url_and_digest(record, expected.url, expected.content_digest, mismatches)
    if mismatches:
        raise RecordVerificationError("; ".join(mismatches))
    _verify_successful_origin_response(record)


def verify_supplemental_record(record: CommonCrawlWarcRecord, expected: SupplementalRecordExpectation) -> None:
    """Verify an observed record against the fields available from a supplemental index."""
    mismatches = []
    _verify_url_and_digest(record, expected.url, expected.content_digest, mismatches)
    if mismatches:
        raise RecordVerificationError("; ".join(mismatches))
    _verify_successful_origin_response(record)


def _verify_successful_origin_response(record: CommonCrawlWarcRecord) -> None:
    if not 200 <= record.http_status < 300:
        raise OriginResponseStatusError(record.http_status)


def _comparable_record_id(record_id: str) -> str:
    try:
        return _canonical_record_id(record_id)
    except ValueError:
        return record_id


def _verify_url_and_digest(
    record: CommonCrawlWarcRecord,
    expected_url: str,
    expected_digest: str,
    mismatches: list[str],
) -> None:
    if record.target_url != expected_url:
        mismatches.append(f"target URL {record.target_url!r} != {expected_url!r}")
    if record.payload_digest != expected_digest:
        mismatches.append(f"content digest {record.payload_digest!r} != {expected_digest!r}")


def content_digest(payload: bytes) -> str:
    """Return Common Crawl's unpadded Base32 SHA-1 payload digest."""
    digest = hashlib.sha1(payload, usedforsecurity=False).digest()
    return f"sha1:{base64.b32encode(digest).decode().rstrip('=')}"
