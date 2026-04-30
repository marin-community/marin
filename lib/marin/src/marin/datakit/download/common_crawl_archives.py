# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sample-capped Common Crawl WARC/WAT materializers.

This module streams a tiny, deterministic text slice from one pinned Common Crawl
archive without mirroring whole WARC/WAT shards. It preserves archive structure by
emitting the original WARC headers plus the decoded payload body for selected
records.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import posixpath
import re
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from rigging.filesystem import open_url
from urllib3.util import Retry
from zephyr.writers import atomic_rename

from marin.datakit.ingestion_manifest import (
    IdentityTreatment,
    IngestionPolicy,
    IngestionSourceManifest,
    JsonValue,
    MaterializedOutputMetadata,
    SampleCapConfig,
    SecretRedaction,
    StagingMetadata,
    UsagePolicy,
    write_ingestion_metadata_json,
)
from marin.execution.executor import THIS_OUTPUT_PATH
from marin.execution.step_spec import StepSpec
from marin.utils import fsspec_mkdirs

logger = logging.getLogger(__name__)

COMMON_CRAWL_ISSUE = 5056
COMMON_CRAWL_EPIC = 5005
COMMON_CRAWL_BASE_URL = "https://data.commoncrawl.org/"
COMMON_CRAWL_GET_STARTED_URL = "https://commoncrawl.org/get-started"
COMMON_CRAWL_TERMS_URL = "https://commoncrawl.org/terms-of-use"
COMMON_CRAWL_CRAWL_ID = "CC-MAIN-2026-12"
COMMON_CRAWL_HTTP_TIMEOUT = 120
COMMON_CRAWL_MAX_FILES = 4
COMMON_CRAWL_MAX_OUTPUT_BYTES = 30 * 1024 * 1024
COMMON_CRAWL_MAX_RECORD_BYTES = 512 * 1024
COMMON_CRAWL_OUTPUT_FILENAME = "data.jsonl.gz"
COMMON_CRAWL_WARC_SLICE_KEY = "raw_web_markup/common_crawl/warc"
COMMON_CRAWL_WAT_SLICE_KEY = "raw_web_markup/common_crawl/wat"
_SEGMENT_RE = re.compile(r"/segments/([^/]+)/")
_TEXTUAL_HTTP_TYPES = (
    "text/",
    "application/json",
    "application/ld+json",
    "application/javascript",
    "application/xml",
    "application/xhtml+xml",
    "application/rss+xml",
    "application/atom+xml",
    "image/svg+xml",
)


class CommonCrawlArchiveKind(StrEnum):
    """Archive kinds exposed by Common Crawl."""

    WARC = "warc"
    WAT = "wat"


@dataclass(frozen=True)
class CommonCrawlArchiveSource:
    """Config for one capped Common Crawl archive slice."""

    manifest: IngestionSourceManifest
    archive_kind: CommonCrawlArchiveKind
    crawl_id: str = COMMON_CRAWL_CRAWL_ID
    base_url: str = COMMON_CRAWL_BASE_URL
    max_output_bytes: int = COMMON_CRAWL_MAX_OUTPUT_BYTES
    max_record_bytes: int = COMMON_CRAWL_MAX_RECORD_BYTES
    max_files: int = COMMON_CRAWL_MAX_FILES
    source_label: str = ""

    def resolved_source_label(self) -> str:
        return self.source_label or self.manifest.slice_key

    @property
    def path_list_url(self) -> str:
        return urljoin(self.base_url, f"crawl-data/{self.crawl_id}/{self.archive_kind.value}.paths.gz")

    def validate(self) -> None:
        if not self.base_url.endswith("/"):
            raise ValueError("base_url must end with '/'")
        if not self.crawl_id.startswith("CC-MAIN-"):
            raise ValueError(f"unexpected Common Crawl id {self.crawl_id!r}")
        if self.max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be positive")
        if self.max_record_bytes <= 0:
            raise ValueError("max_record_bytes must be positive")
        if self.max_files <= 0:
            raise ValueError("max_files must be positive")


@dataclass
class DownloadCommonCrawlSampleConfig:
    """Runtime config for :func:`download_common_crawl_sample`."""

    source: CommonCrawlArchiveSource
    output_path: str = THIS_OUTPUT_PATH
    output_filename: str = COMMON_CRAWL_OUTPUT_FILENAME
    http_timeout: int = COMMON_CRAWL_HTTP_TIMEOUT


@dataclass(frozen=True)
class _WarcRecord:
    version_line: str
    header_lines: tuple[str, ...]
    headers: tuple[tuple[str, str], ...]
    body: bytes

    def header(self, name: str) -> str | None:
        folded = name.casefold()
        for key, value in self.headers:
            if key.casefold() == folded:
                return value
        return None


def _common_crawl_policy() -> IngestionPolicy:
    return IngestionPolicy(
        usage_policy=UsagePolicy.EVAL_ONLY,
        use_policy="Eval-only raw-web probe slice. Do not mix this held-out slice into training.",
        requires_sanitization=False,
        identity_treatment=IdentityTreatment.PRESERVE,
        secret_redaction=SecretRedaction.NONE,
        contamination_risk="high: direct eval contamination if this fixed slice is copied into training",
        provenance_notes=(
            "Pinned Common Crawl archive sample. Access is subject to Common Crawl Terms of Use and "
            "underlying crawled-content terms."
        ),
    )


def _common_crawl_manifest(*, slice_key: str, archive_kind: CommonCrawlArchiveKind) -> IngestionSourceManifest:
    surface_form = "warc_headers_http_metadata_raw_response_body"
    if archive_kind == CommonCrawlArchiveKind.WAT:
        surface_form = "wat_warc_headers_json_metadata_body"
    return IngestionSourceManifest(
        dataset_key=f"commoncrawl/{archive_kind.value}",
        slice_key=slice_key,
        source_label=slice_key.rsplit("/", maxsplit=1)[-1],
        source_urls=(COMMON_CRAWL_GET_STARTED_URL, COMMON_CRAWL_TERMS_URL),
        source_license="Common Crawl Terms of Use; underlying crawled content may carry separate terms",
        source_format=f"common_crawl_{archive_kind.value}_gzip",
        surface_form=surface_form,
        policy=_common_crawl_policy(),
        staging=StagingMetadata(
            transform_name="download_common_crawl_sample",
            metadata={
                "archive_kind": archive_kind.value,
                "crawl_id": COMMON_CRAWL_CRAWL_ID,
                "output_filename": COMMON_CRAWL_OUTPUT_FILENAME,
                "provenance_fields": ["id", "source_file", "record_index", "warc_type", "target_uri"],
            },
        ),
        epic_issue=COMMON_CRAWL_EPIC,
        issue_numbers=(COMMON_CRAWL_ISSUE,),
        sample_caps=SampleCapConfig(
            max_bytes_per_source=COMMON_CRAWL_MAX_OUTPUT_BYTES,
            max_bytes_per_document=COMMON_CRAWL_MAX_RECORD_BYTES,
            max_files=COMMON_CRAWL_MAX_FILES,
        ),
        source_metadata={"crawl_id": COMMON_CRAWL_CRAWL_ID, "archive_kind": archive_kind.value},
    )


COMMON_CRAWL_WARC_SOURCE = CommonCrawlArchiveSource(
    manifest=_common_crawl_manifest(
        slice_key=COMMON_CRAWL_WARC_SLICE_KEY,
        archive_kind=CommonCrawlArchiveKind.WARC,
    ),
    archive_kind=CommonCrawlArchiveKind.WARC,
)
COMMON_CRAWL_WAT_SOURCE = CommonCrawlArchiveSource(
    manifest=_common_crawl_manifest(
        slice_key=COMMON_CRAWL_WAT_SLICE_KEY,
        archive_kind=CommonCrawlArchiveKind.WAT,
    ),
    archive_kind=CommonCrawlArchiveKind.WAT,
)


def _build_session() -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session = requests.Session()
    session.mount("http://", HTTPAdapter(max_retries=retry))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _segment_key(relative_path: str) -> str:
    match = _SEGMENT_RE.search(relative_path)
    if match is None:
        raise ValueError(f"could not parse segment from {relative_path!r}")
    return match.group(1)


def _selected_archive_paths(
    session: requests.Session,
    source: CommonCrawlArchiveSource,
    *,
    http_timeout: int,
) -> list[str]:
    selected: list[str] = []
    seen_segments: set[str] = set()
    with session.get(source.path_list_url, stream=True, timeout=http_timeout) as response:
        response.raise_for_status()
        with gzip.GzipFile(fileobj=response.raw) as gz:
            for raw_line in gz:
                relative_path = raw_line.decode("utf-8", errors="replace").strip()
                if not relative_path:
                    continue
                segment = _segment_key(relative_path)
                if segment in seen_segments:
                    continue
                seen_segments.add(segment)
                selected.append(relative_path)
                if len(selected) >= source.max_files:
                    break
    if not selected:
        raise ValueError(f"no {source.archive_kind.value} archive paths found at {source.path_list_url}")
    return selected


def _iter_warc_records(stream: gzip.GzipFile) -> Iterator[_WarcRecord]:
    pending_line: bytes | None = None
    while True:
        line = pending_line if pending_line is not None else stream.readline()
        pending_line = None
        while line in (b"\n", b"\r\n"):
            line = stream.readline()
        if not line:
            return

        version_line = line.decode("utf-8", errors="replace").rstrip("\r\n")
        if not version_line.startswith("WARC/"):
            raise ValueError(f"expected WARC record header, got {version_line!r}")

        header_lines: list[str] = []
        headers: list[tuple[str, str]] = []
        while True:
            header_raw = stream.readline()
            if not header_raw:
                raise EOFError("unexpected EOF while reading WARC headers")
            if header_raw in (b"\n", b"\r\n"):
                break
            header_line = header_raw.decode("utf-8", errors="replace").rstrip("\r\n")
            header_lines.append(header_line)
            name, sep, value = header_line.partition(":")
            if not sep:
                raise ValueError(f"malformed WARC header line: {header_line!r}")
            headers.append((name.strip(), value.lstrip()))

        content_length = _header_int(headers, "Content-Length")
        body = stream.read(content_length)
        if len(body) != content_length:
            raise EOFError(f"expected {content_length} body bytes, found {len(body)}")

        pending_line = stream.readline()
        while pending_line in (b"\n", b"\r\n"):
            pending_line = stream.readline()

        yield _WarcRecord(
            version_line=version_line,
            header_lines=tuple(header_lines),
            headers=tuple(headers),
            body=body,
        )
        if not pending_line:
            pending_line = None


def _header_int(headers: tuple[tuple[str, str], ...] | list[tuple[str, str]], name: str) -> int:
    folded = name.casefold()
    for key, value in headers:
        if key.casefold() == folded:
            return int(value)
    raise ValueError(f"missing required header {name!r}")


def _http_content_type(body: bytes) -> str | None:
    preview = body[:4096].decode("iso-8859-1", errors="replace")
    header_block, _, _ = preview.partition("\r\n\r\n")
    if not header_block:
        header_block, _, _ = preview.partition("\n\n")
    for line in header_block.splitlines():
        if line.lower().startswith("content-type:"):
            return line.split(":", maxsplit=1)[1].strip()
    return None


def _looks_utf8_text(body: bytes) -> bool:
    preview_bytes = body[:8192]
    if not preview_bytes:
        return False
    if b"\x00" in preview_bytes:
        return False
    preview = preview_bytes.decode("utf-8", errors="replace")
    replacement_ratio = preview.count("\ufffd") / max(1, len(preview))
    control_chars = sum(1 for char in preview if ord(char) < 32 and char not in "\n\r\t\f")
    return replacement_ratio <= 0.03 and control_chars <= max(2, len(preview) // 100)


def _is_textual_content_type(content_type: str | None) -> bool:
    if content_type is None:
        return False
    folded = content_type.casefold()
    return any(marker in folded for marker in _TEXTUAL_HTTP_TYPES)


def _keep_record(record: _WarcRecord, source: CommonCrawlArchiveSource) -> bool:
    if len(record.body) > source.max_record_bytes:
        return False
    if source.archive_kind == CommonCrawlArchiveKind.WAT:
        stripped = record.body.lstrip()
        return stripped.startswith((b"{", b"[")) and _looks_utf8_text(record.body)

    warc_type = (record.header("WARC-Type") or "").casefold()
    if warc_type not in {"response", "request", "metadata", "warcinfo"}:
        return False
    if warc_type == "response" and _is_textual_content_type(_http_content_type(record.body)):
        return True
    return _looks_utf8_text(record.body)


def _render_record_text(record: _WarcRecord) -> str:
    body_text = record.body.decode("utf-8", errors="replace")
    return "\n".join((record.version_line, *record.header_lines, "", body_text)).rstrip("\n")


def _record_payload(
    *,
    record: _WarcRecord,
    source: CommonCrawlArchiveSource,
    source_file: str,
    record_index: int,
) -> dict[str, str | int]:
    row_key = f"{source.manifest.slice_key}:{source_file}:{record_index}"
    return {
        "id": hashlib.sha256(row_key.encode("utf-8")).hexdigest(),
        "text": _render_record_text(record),
        "source": source.resolved_source_label(),
        "crawl_id": source.crawl_id,
        "source_file": source_file,
        "record_index": record_index,
        "warc_type": record.header("WARC-Type") or "",
        "target_uri": record.header("WARC-Target-URI") or "",
    }


def _write_source_metadata(
    *,
    source: CommonCrawlArchiveSource,
    output_path: str,
    output_file: str,
    record_count: int,
    bytes_written: int,
    metadata: dict[str, JsonValue],
) -> str:
    return write_ingestion_metadata_json(
        manifest=source.manifest,
        materialized_output=MaterializedOutputMetadata(
            input_path=source.path_list_url,
            output_path=output_path,
            output_file=output_file,
            record_count=record_count,
            bytes_written=bytes_written,
            metadata=metadata,
        ),
    )


def download_common_crawl_sample(config: DownloadCommonCrawlSampleConfig) -> dict[str, Any]:
    """Download a tiny deterministic Common Crawl WARC/WAT sample."""

    source = config.source
    source.validate()
    output_path = str(config.output_path)
    output_file = posixpath.join(output_path, config.output_filename)
    fsspec_mkdirs(output_path, exist_ok=True)

    counters = {
        "records_seen": 0,
        "records_kept": 0,
        "records_skipped_nontxt": 0,
        "records_skipped_too_large": 0,
        "files_opened": 0,
    }
    bytes_written = 0
    record_types: dict[str, int] = {}

    session = _build_session()
    try:
        selected_paths = _selected_archive_paths(session, source, http_timeout=config.http_timeout)
        with atomic_rename(output_file) as temp_path:
            with open_url(temp_path, "wt", encoding="utf-8", compression="gzip") as writer:
                for relative_path in selected_paths:
                    archive_url = urljoin(source.base_url, relative_path)
                    counters["files_opened"] += 1
                    logger.info("Streaming %s archive %s", source.archive_kind.value, archive_url)
                    with session.get(archive_url, stream=True, timeout=config.http_timeout) as response:
                        response.raise_for_status()
                        with gzip.GzipFile(fileobj=response.raw) as gz:
                            for record_index, record in enumerate(_iter_warc_records(gz)):
                                counters["records_seen"] += 1
                                if len(record.body) > source.max_record_bytes:
                                    counters["records_skipped_too_large"] += 1
                                    continue
                                if not _keep_record(record, source):
                                    counters["records_skipped_nontxt"] += 1
                                    continue

                                payload = _record_payload(
                                    record=record,
                                    source=source,
                                    source_file=relative_path,
                                    record_index=record_index,
                                )
                                serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
                                serialized_bytes = len(serialized.encode("utf-8")) + 1
                                if bytes_written + serialized_bytes > source.max_output_bytes:
                                    metadata_path = _write_source_metadata(
                                        source=source,
                                        output_path=output_path,
                                        output_file=output_file,
                                        record_count=counters["records_kept"],
                                        bytes_written=bytes_written,
                                        metadata={
                                            "archive_kind": source.archive_kind.value,
                                            "crawl_id": source.crawl_id,
                                            "selected_paths": selected_paths,
                                            "sample_limits": {
                                                "max_output_bytes": source.max_output_bytes,
                                                "max_record_bytes": source.max_record_bytes,
                                                "max_files": source.max_files,
                                            },
                                            "counters": counters,
                                            "record_type_counts": record_types,
                                        },
                                    )
                                    return {
                                        "output_file": output_file,
                                        "metadata_file": metadata_path,
                                        "selected_paths": selected_paths,
                                        "bytes_written": bytes_written,
                                        "record_count": counters["records_kept"],
                                        "counters": counters,
                                        "record_type_counts": record_types,
                                    }

                                writer.write(serialized)
                                writer.write("\n")
                                counters["records_kept"] += 1
                                bytes_written += serialized_bytes
                                warc_type = payload["warc_type"] or "<missing>"
                                record_types[str(warc_type)] = record_types.get(str(warc_type), 0) + 1
    finally:
        session.close()

    metadata_path = _write_source_metadata(
        source=source,
        output_path=output_path,
        output_file=output_file,
        record_count=counters["records_kept"],
        bytes_written=bytes_written,
        metadata={
            "archive_kind": source.archive_kind.value,
            "crawl_id": source.crawl_id,
            "selected_paths": selected_paths,
            "sample_limits": {
                "max_output_bytes": source.max_output_bytes,
                "max_record_bytes": source.max_record_bytes,
                "max_files": source.max_files,
            },
            "counters": counters,
            "record_type_counts": record_types,
        },
    )
    return {
        "output_file": output_file,
        "metadata_file": metadata_path,
        "selected_paths": selected_paths,
        "bytes_written": bytes_written,
        "record_count": counters["records_kept"],
        "counters": counters,
        "record_type_counts": record_types,
    }


def common_crawl_sample_step(
    source: CommonCrawlArchiveSource,
    *,
    name: str | None = None,
    http_timeout: int = COMMON_CRAWL_HTTP_TIMEOUT,
) -> StepSpec:
    """Create the StepSpec for one Common Crawl WARC/WAT sample."""

    source.validate()
    step_name = name or f"raw/{source.manifest.slice_key}"
    return StepSpec(
        name=step_name,
        fn=lambda output_path: download_common_crawl_sample(
            DownloadCommonCrawlSampleConfig(
                source=source,
                output_path=output_path,
                http_timeout=http_timeout,
            )
        ),
        hash_attrs={
            "slice_key": source.manifest.slice_key,
            "manifest_fingerprint": source.manifest.fingerprint(),
            "archive_kind": source.archive_kind.value,
            "crawl_id": source.crawl_id,
            "base_url": source.base_url,
            "max_output_bytes": source.max_output_bytes,
            "max_record_bytes": source.max_record_bytes,
            "max_files": source.max_files,
            "http_timeout": http_timeout,
        },
    )
