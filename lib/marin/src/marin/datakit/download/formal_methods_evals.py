# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""HTTP-archive downloader for formal-methods and hardware-RTL PPL eval slices.

Issue: https://github.com/marin-community/marin/issues/5060 (parent #5005).

Each configured source fetches a single upstream archive (tar/zip/tar.zst), filters members
by glob, and writes one JSONL.gz file with one record per source file::

    {"id": str, "text": str, "source": str, "filename": str}

The ``text`` field preserves raw file bytes verbatim (decoded as UTF-8 with ``errors="replace"``)
so PPL eval keeps file-level syntax, comments, long symbols, generated identifiers, module
boundaries, and solver status markers intact (issue #5060 DoD).

The compressed JSONL output is capped per-source (default 40 MB) so gap-report runs do not
ingest the full upstream benchmark — downloads truncate once the gzip buffer exceeds the budget.

Supported archive formats: ``tar``, ``tar.gz`` (``.tgz``), ``tar.bz2``, ``tar.xz``, ``tar.zst``,
``zip``. Binary archives without a textual form (e.g. AIGER ``.aig``) are out of scope per
the issue discussion and not supported by this module.
"""

from __future__ import annotations

import fnmatch
import gzip
import io
import json
import logging
import posixpath
import tarfile
import zipfile
import zstandard
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from rigging.filesystem import open_url
from urllib3.util import Retry
from zephyr.writers import atomic_rename

from marin.datakit.ingestion_manifest import (
    IngestionSourceManifest,
    JsonValue,
    MaterializedOutputMetadata,
    write_ingestion_metadata_json,
)
from marin.execution.executor import THIS_OUTPUT_PATH
from marin.execution.step_spec import StepSpec
from marin.utils import fsspec_mkdirs

logger = logging.getLogger(__name__)

DEFAULT_MAX_COMPRESSED_BYTES: int = 40 * 1024 * 1024
"""Per-source compressed-JSONL budget (issue #5060: 20-40 MB compressed per source)."""

DEFAULT_HTTP_TIMEOUT_SECONDS: int = 600
DEFAULT_OUTPUT_FILENAME: str = "data.jsonl.gz"

RAW_FILE_CONTENT_MODE: str = "raw_file"
"""Wrap each archive member verbatim as {text: <file bytes>}."""

JSONL_TEXT_COLUMN_CONTENT_MODE: str = "jsonl_text_column"
"""Treat line-oriented JSON archive members as records and emit text from a configured column."""

_VALID_CONTENT_MODES: frozenset[str] = frozenset({RAW_FILE_CONTENT_MODE, JSONL_TEXT_COLUMN_CONTENT_MODE})


@dataclass(frozen=True)
class ArchiveSourceConfig:
    """Manifest-backed description of one upstream archive slice."""

    manifest: IngestionSourceManifest

    @property
    def slice_key(self) -> str:
        return self.manifest.slice_key

    @property
    def archive_url(self) -> str:
        if not self.manifest.source_urls:
            raise ValueError("source_urls must contain the primary archive URL")
        return self.manifest.source_urls[0]

    @property
    def archive_format(self) -> str:
        return _required_metadata_str(self.manifest, "archive_format")

    @property
    def include_globs(self) -> tuple[str, ...]:
        return _required_metadata_str_tuple(self.manifest, "include_globs")

    @property
    def exclude_globs(self) -> tuple[str, ...]:
        return _optional_metadata_str_tuple(self.manifest, "exclude_globs")

    @property
    def content_mode(self) -> str:
        value = _optional_metadata_str(self.manifest, "content_mode", default=RAW_FILE_CONTENT_MODE)
        assert value is not None
        return value

    @property
    def jsonl_text_column(self) -> str | None:
        return _optional_metadata_str(self.manifest, "jsonl_text_column")

    @property
    def max_compressed_bytes(self) -> int:
        return self.manifest.sample_caps.max_bytes_per_source or DEFAULT_MAX_COMPRESSED_BYTES

    @property
    def max_files(self) -> int | None:
        return self.manifest.sample_caps.max_files

    def resolved_source_label(self) -> str:
        return self.manifest.source_label or self.slice_key

    def validate(self) -> None:
        if self.content_mode not in _VALID_CONTENT_MODES:
            raise ValueError(f"unknown content_mode: {self.content_mode!r}")
        if self.content_mode == JSONL_TEXT_COLUMN_CONTENT_MODE and not self.jsonl_text_column:
            raise ValueError("jsonl_text_column must be set when content_mode is jsonl_text_column")
        if not self.include_globs:
            raise ValueError("include_globs must not be empty")
        if self.archive_format not in _ARCHIVE_FORMATS:
            raise ValueError(f"unsupported archive_format: {self.archive_format!r}")
        if self.max_compressed_bytes <= 0:
            raise ValueError("max_compressed_bytes must be positive")


@dataclass
class DownloadArchiveSliceConfig:
    """Runtime config for :func:`download_archive_slice`."""

    source: ArchiveSourceConfig
    output_path: str = THIS_OUTPUT_PATH
    output_filename: str = DEFAULT_OUTPUT_FILENAME
    http_timeout_seconds: int = DEFAULT_HTTP_TIMEOUT_SECONDS


@dataclass(frozen=True)
class _SourceFile:
    filename: str
    content: bytes


_ARCHIVE_FORMATS: frozenset[str] = frozenset({"tar", "tar.gz", "tgz", "tar.bz2", "tar.xz", "tar.zst", "zip"})


def _required_metadata(manifest: IngestionSourceManifest, key: str) -> JsonValue:
    if key not in manifest.staging.metadata:
        raise ValueError(f"staging.metadata[{key!r}] must be set")
    return manifest.staging.metadata[key]


def _required_metadata_str(manifest: IngestionSourceManifest, key: str) -> str:
    value = _required_metadata(manifest, key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"staging.metadata[{key!r}] must be a non-empty string")
    return value


def _required_metadata_str_tuple(manifest: IngestionSourceManifest, key: str) -> tuple[str, ...]:
    value = _required_metadata(manifest, key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"staging.metadata[{key!r}] must be a non-empty string list")
    if not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"staging.metadata[{key!r}] must contain only non-empty strings")
    return tuple(value)


def _optional_metadata_str_tuple(manifest: IngestionSourceManifest, key: str) -> tuple[str, ...]:
    value = manifest.staging.metadata.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"staging.metadata[{key!r}] must be a string list")
    if not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"staging.metadata[{key!r}] must contain only non-empty strings")
    return tuple(value)


def _optional_metadata_str(manifest: IngestionSourceManifest, key: str, *, default: str | None = None) -> str | None:
    value = manifest.staging.metadata.get(key, default)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"staging.metadata[{key!r}] must be a non-empty string")
    return value


def _tar_mode(archive_format: str) -> str:
    return {
        "tar": "r:",
        "tar.gz": "r:gz",
        "tgz": "r:gz",
        "tar.bz2": "r:bz2",
        "tar.xz": "r:xz",
    }[archive_format]


def _iter_tar_members(archive_bytes: bytes, archive_format: str) -> Iterator[_SourceFile]:
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode=_tar_mode(archive_format)) as tf:
        for member in tf:
            if not member.isreg():
                continue
            handle = tf.extractfile(member)
            if handle is None:
                continue
            yield _SourceFile(filename=member.name, content=handle.read())


def _iter_tar_zst_members(archive_bytes: bytes) -> Iterator[_SourceFile]:
    dctx = zstandard.ZstdDecompressor()
    decompressed = dctx.decompress(archive_bytes, max_output_size=2**34)  # 16 GB ceiling
    yield from _iter_tar_members(decompressed, "tar")


def _iter_zip_members(archive_bytes: bytes) -> Iterator[_SourceFile]:
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            with zf.open(info) as handle:
                yield _SourceFile(filename=info.filename, content=handle.read())


def _iter_archive_members(archive_bytes: bytes, archive_format: str) -> Iterator[_SourceFile]:
    if archive_format == "zip":
        yield from _iter_zip_members(archive_bytes)
    elif archive_format == "tar.zst":
        yield from _iter_tar_zst_members(archive_bytes)
    else:
        yield from _iter_tar_members(archive_bytes, archive_format)


def _filter_members(
    files: Iterable[_SourceFile], include_globs: tuple[str, ...], exclude_globs: tuple[str, ...]
) -> Iterator[_SourceFile]:
    for sf in files:
        if not any(fnmatch.fnmatch(sf.filename, pat) for pat in include_globs):
            continue
        if any(fnmatch.fnmatch(sf.filename, pat) for pat in exclude_globs):
            continue
        yield sf


def _iter_json_records(raw: bytes) -> Iterator[dict[str, Any]]:
    if raw[:2] == b"\x1f\x8b":  # gzip magic
        raw = gzip.decompress(raw)
    stripped = raw.lstrip()
    if stripped.startswith(b"["):
        records = json.loads(raw)
        if not isinstance(records, list):
            raise ValueError("JSON archive member must contain a list of records")
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("JSON archive member list entries must be objects")
            yield record
        return
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed JSONL line {line_number}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"JSONL line {line_number} must be an object")
        yield record


def _text_values(record: dict[str, Any], column: str) -> Iterator[str]:
    value = record.get(column)
    if isinstance(value, str):
        if value:
            yield value
        return
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item:
                yield item


def _emit_records(source: ArchiveSourceConfig, filtered: Iterable[_SourceFile]) -> Iterator[tuple[str, str]]:
    """Yield (filename, text) for each downstream record."""
    if source.content_mode == RAW_FILE_CONTENT_MODE:
        for sf in filtered:
            yield sf.filename, sf.content.decode("utf-8", errors="replace")
        return
    # jsonl_text_column
    assert source.jsonl_text_column is not None
    column = source.jsonl_text_column
    for sf in filtered:
        for idx, record in enumerate(_iter_json_records(sf.content)):
            for value_idx, text in enumerate(_text_values(record, column)):
                yield f"{sf.filename}#{idx}:{value_idx}", text


def _http_session(timeout_seconds: int) -> requests.Session:
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1.5, status_forcelist=[500, 502, 503, 504], allowed_methods=["GET"])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _fetch_archive_bytes(url: str, timeout_seconds: int) -> bytes:
    """Fetch the archive into memory. For `file://` URLs and any fsspec path we fall back
    to ``open_url`` so tests and local fixtures work without a network round-trip."""
    if url.startswith("http://") or url.startswith("https://"):
        logger.info("fetching archive from %s", url)
        with _http_session(timeout_seconds) as session:
            response = session.get(url, timeout=timeout_seconds, stream=True)
            response.raise_for_status()
            return response.content
    logger.info("reading archive via fsspec from %s", url)
    with open_url(url, "rb") as fh:
        return fh.read()


def _write_source_metadata(
    *,
    source: ArchiveSourceConfig,
    output_path: str,
    output_file: str,
    record_count: int,
    bytes_written: int,
) -> str:
    return write_ingestion_metadata_json(
        manifest=source.manifest,
        materialized_output=MaterializedOutputMetadata(
            input_path=source.archive_url,
            output_path=output_path,
            output_file=output_file,
            record_count=record_count,
            bytes_written=bytes_written,
            metadata={
                "archive_format": source.archive_format,
                "include_globs": list(source.include_globs),
                "exclude_globs": list(source.exclude_globs),
                "content_mode": source.content_mode,
                "jsonl_text_column": source.jsonl_text_column,
            },
        ),
    )


def _write_budgeted_jsonl_gz(
    records: Iterable[tuple[str, str]],
    *,
    output_path: str,
    source_label: str,
    max_compressed_bytes: int,
    max_files: int | None,
) -> dict[str, Any]:
    """Write records to gzipped JSONL, truncating before the next record would exceed the budget."""
    buf = io.BytesIO()
    written = 0
    for idx, (filename, text) in enumerate(records):
        if max_files is not None and written >= max_files:
            break
        record = {
            "id": f"{source_label}#{idx:06d}",
            "text": text,
            "source": source_label,
            "filename": filename,
        }
        member = gzip.compress(
            (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"),
            compresslevel=9,
            mtime=0,
        )
        next_size = buf.tell() + len(member)
        if next_size > max_compressed_bytes:
            logger.info("byte budget reached after %d records", written)
            break
        buf.write(member)
        written += 1
    payload = buf.getvalue()
    with atomic_rename(output_path) as temp_path, open_url(temp_path, "wb") as out:
        out.write(payload)
    logger.info("wrote %d records (%d compressed bytes) to %s", written, len(payload), output_path)
    return {"records": written, "compressed_bytes": len(payload), "output_file": output_path}


def download_archive_slice(cfg: DownloadArchiveSliceConfig) -> dict[str, Any]:
    """Download, filter, and write one archive source into ``<output_path>/<output_filename>``.

    Returns a metadata dict suitable for step hash attributes and audit output.
    """
    cfg.source.validate()
    output_path = str(cfg.output_path)
    fsspec_mkdirs(output_path, exist_ok=True)
    archive_bytes = _fetch_archive_bytes(cfg.source.archive_url, cfg.http_timeout_seconds)
    members = _iter_archive_members(archive_bytes, cfg.source.archive_format)
    filtered = _filter_members(members, cfg.source.include_globs, cfg.source.exclude_globs)
    records = _emit_records(cfg.source, filtered)
    output_file = posixpath.join(output_path, cfg.output_filename)
    result = _write_budgeted_jsonl_gz(
        records,
        output_path=output_file,
        source_label=cfg.source.resolved_source_label(),
        max_compressed_bytes=cfg.source.max_compressed_bytes,
        max_files=cfg.source.max_files,
    )
    result["metadata_path"] = _write_source_metadata(
        source=cfg.source,
        output_path=output_path,
        output_file=output_file,
        record_count=result["records"],
        bytes_written=result["compressed_bytes"],
    )
    return result


def archive_slice_step(
    source: ArchiveSourceConfig,
    *,
    name: str | None = None,
    output_filename: str = DEFAULT_OUTPUT_FILENAME,
) -> StepSpec:
    """Build a StepSpec that downloads ``source`` and writes ``output_filename``."""
    source.validate()
    step_name = name or f"raw/{source.slice_key}"
    return StepSpec(
        name=step_name,
        fn=lambda output_path: download_archive_slice(
            DownloadArchiveSliceConfig(
                source=source,
                output_path=output_path,
                output_filename=output_filename,
            )
        ),
        hash_attrs={
            "manifest_content_fingerprint": source.manifest.fingerprint(),
            "output_filename": output_filename,
        },
    )
