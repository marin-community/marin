# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""WebDataCommons WebTables sample staging.

The public WebTables samples use two small archive layouts:

* ``sample10.zip`` contains CSV table files next to JSON page metadata.
* ``sample1K.zip`` contains tar members, each with gzipped CSV tables.

Both layouts already store table cells as source-formatted CSV text. The
staging path therefore preserves the CSV bytes after UTF-8 decoding and only
splits documents at source line boundaries.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import logging
import posixpath
import tarfile
import zipfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from urllib.parse import urlparse

import requests
from rigging.filesystem import open_url, url_to_fs
from zephyr.writers import atomic_rename

from marin.datakit.ingestion_manifest import (
    IngestionSourceManifest,
    MaterializedOutputMetadata,
    write_ingestion_metadata_json,
)
from marin.utils import fsspec_mkdirs

logger = logging.getLogger(__name__)

DEFAULT_MAX_BYTES_PER_SOURCE = 30 * 1024 * 1024
DEFAULT_MAX_BYTES_PER_DOCUMENT = 32 * 1024
DEFAULT_MAX_ZIP_BYTES = 64 * 1024 * 1024
DEFAULT_REQUEST_TIMEOUT = 120
DOWNLOAD_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class WebDataCommonsStagingConfig:
    """Configuration for staging WebDataCommons WebTables samples into JSONL."""

    sample_url: str
    output_path: str
    source_label: str
    sample_name: str
    max_bytes_per_source: int = DEFAULT_MAX_BYTES_PER_SOURCE
    max_bytes_per_document: int = DEFAULT_MAX_BYTES_PER_DOCUMENT
    preserve_header: bool = True
    max_zip_bytes: int = DEFAULT_MAX_ZIP_BYTES
    request_timeout: int = DEFAULT_REQUEST_TIMEOUT
    output_filename: str = "staged.jsonl.gz"
    extra_metadata: dict[str, str] = field(default_factory=dict)
    source_manifest: IngestionSourceManifest | None = None
    content_fingerprint: str = ""


@dataclass(frozen=True)
class _CsvEntry:
    zip_member: str
    table_member: str
    csv_bytes: bytes
    metadata: Mapping[str, object]


def serialize_csv_document(header_line: str | None, body_lines: Iterator[str] | list[str]) -> str:
    """Concatenate a header line and body lines verbatim."""
    parts: list[str] = []
    if header_line is not None:
        parts.append(header_line)
    parts.extend(body_lines)
    return "".join(parts)


def chunk_lines_by_bytes(
    lines: Iterator[str] | list[str],
    *,
    max_bytes_per_chunk: int,
    header_line: str | None = None,
) -> Iterator[list[str]]:
    """Split lines into chunks whose UTF-8 sizes stay under the target cap."""
    if max_bytes_per_chunk <= 0:
        raise ValueError(f"max_bytes_per_chunk must be positive, got {max_bytes_per_chunk}")

    header_bytes = len(header_line.encode("utf-8")) if header_line is not None else 0
    body_budget = max_bytes_per_chunk - header_bytes
    if body_budget <= 0:
        raise ValueError(
            f"max_bytes_per_chunk={max_bytes_per_chunk} is smaller than header size {header_bytes}; "
            "increase the cap or disable header preservation."
        )

    current: list[str] = []
    current_bytes = 0
    for line in lines:
        line_bytes = len(line.encode("utf-8"))
        if current and current_bytes + line_bytes > body_budget:
            yield current
            current = []
            current_bytes = 0
        current.append(line)
        current_bytes += line_bytes
    if current:
        yield current


def _append_limited(buffer: io.BytesIO, chunk: bytes, *, max_zip_bytes: int, source: str) -> None:
    if buffer.tell() + len(chunk) > max_zip_bytes:
        raise ValueError(f"{source} exceeds max_zip_bytes={max_zip_bytes}")
    buffer.write(chunk)


def _read_zip_bytes(cfg: WebDataCommonsStagingConfig) -> bytes:
    if cfg.max_zip_bytes <= 0:
        raise ValueError(f"max_zip_bytes must be positive, got {cfg.max_zip_bytes}")

    buffer = io.BytesIO()
    parsed = urlparse(cfg.sample_url)
    if parsed.scheme in {"http", "https"}:
        with requests.get(cfg.sample_url, stream=True, timeout=cfg.request_timeout) as response:
            response.raise_for_status()
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) > cfg.max_zip_bytes:
                raise ValueError(f"{cfg.sample_url} exceeds max_zip_bytes={cfg.max_zip_bytes}")
            for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_BYTES):
                if chunk:
                    _append_limited(buffer, chunk, max_zip_bytes=cfg.max_zip_bytes, source=cfg.sample_url)
    else:
        with open_url(cfg.sample_url, "rb") as infile:
            while True:
                chunk = infile.read(DOWNLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                _append_limited(buffer, chunk, max_zip_bytes=cfg.max_zip_bytes, source=cfg.sample_url)

    return buffer.getvalue()


def _arc_position_from_member(member_name: str) -> str | None:
    basename = posixpath.basename(member_name)
    if "_" not in basename:
        return None
    return basename.split("_", 1)[0]


def _metadata_by_arc_position(archive: zipfile.ZipFile) -> dict[str, Mapping[str, object]]:
    metadata_by_position: dict[str, Mapping[str, object]] = {}
    for info in archive.infolist():
        if info.is_dir() or not info.filename.lower().endswith(".json"):
            continue
        metadata = json.loads(archive.read(info).decode("utf-8"))
        arc_position = metadata.get("arcfilePosition")
        key = str(arc_position) if arc_position is not None else _arc_position_from_member(info.filename)
        if key is not None:
            metadata_by_position[key] = metadata
    return metadata_by_position


def _table_metadata(metadata_by_position: Mapping[str, Mapping[str, object]], member_name: str) -> Mapping[str, object]:
    arc_position = _arc_position_from_member(member_name)
    if arc_position is None:
        return {}
    return metadata_by_position.get(arc_position, {})


def _iter_tar_csv_entries(
    *,
    zip_member: str,
    tar_bytes: bytes,
    metadata_by_position: Mapping[str, Mapping[str, object]],
) -> Iterator[_CsvEntry]:
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:*") as tar:
        members = [member for member in tar.getmembers() if member.isfile()]
        members.sort(key=lambda member: member.name)
        for member in members:
            if not member.name.lower().endswith((".csv", ".csv.gz")):
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            csv_bytes = extracted.read()
            if member.name.lower().endswith(".gz"):
                csv_bytes = gzip.decompress(csv_bytes)
            yield _CsvEntry(
                zip_member=zip_member,
                table_member=member.name,
                csv_bytes=csv_bytes,
                metadata=_table_metadata(metadata_by_position, member.name),
            )


def _iter_csv_entries(zip_bytes: bytes) -> Iterator[_CsvEntry]:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        metadata_by_position = _metadata_by_arc_position(archive)
        infos = [info for info in archive.infolist() if not info.is_dir()]
        infos.sort(key=lambda info: info.filename)
        for info in infos:
            lower_name = info.filename.lower()
            if lower_name.endswith(".csv"):
                yield _CsvEntry(
                    zip_member=info.filename,
                    table_member=info.filename,
                    csv_bytes=archive.read(info),
                    metadata=_table_metadata(metadata_by_position, info.filename),
                )
            elif lower_name.endswith(".tar"):
                yield from _iter_tar_csv_entries(
                    zip_member=info.filename,
                    tar_bytes=archive.read(info),
                    metadata_by_position=metadata_by_position,
                )


def _csv_lines(csv_bytes: bytes, table_member: str) -> list[str]:
    try:
        text = csv_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = csv_bytes.decode("utf-8", errors="replace")
        logger.warning("Non-UTF-8 bytes in %s; replaced with U+FFFD", table_member)
    return text.splitlines(keepends=True)


def _header_and_body(lines: list[str], preserve_header: bool) -> tuple[str | None, list[str]]:
    if not preserve_header or not lines or not lines[0].strip():
        return None, lines
    return lines[0], lines[1:]


def _record_id(source_label: str, table_member: str, chunk_index: int) -> str:
    basename = posixpath.basename(table_member)
    for suffix in (".gz", ".csv"):
        basename = basename.removesuffix(suffix)
    digest = hashlib.sha1(table_member.encode("utf-8")).hexdigest()[:8]
    return f"{source_label}:{basename}:{digest}:{chunk_index:04d}"


def _provenance(
    *,
    cfg: WebDataCommonsStagingConfig,
    entry: _CsvEntry,
    chunk_index: int,
    header_preserved: bool,
) -> dict[str, object]:
    metadata = entry.metadata
    provenance: dict[str, object] = {
        "sample_name": cfg.sample_name,
        "source_url": cfg.sample_url,
        "zip_member": entry.zip_member,
        "table_member": entry.table_member,
        "chunk_index": chunk_index,
        "header_preserved": header_preserved,
    }
    selected_metadata = {
        "uri": metadata.get("uri"),
        "arcfile": metadata.get("arcfile"),
        "arcfile_position": metadata.get("arcfilePosition"),
        "metadata_filename": metadata.get("filename"),
        "has_content_tables": metadata.get("hasContentTables"),
        "has_relevant_tables": metadata.get("hasRelevantTables"),
    }
    provenance.update({key: value for key, value in selected_metadata.items() if value is not None})
    provenance.update(cfg.extra_metadata)
    return provenance


def stage_web_data_commons_source(cfg: WebDataCommonsStagingConfig) -> dict[str, int | str]:
    """Stage a WebDataCommons WebTables sample archive into JSONL."""
    if cfg.max_bytes_per_source <= 0:
        raise ValueError(f"max_bytes_per_source must be positive, got {cfg.max_bytes_per_source}")
    if cfg.source_manifest is not None and cfg.content_fingerprint:
        expected = cfg.source_manifest.fingerprint()
        if cfg.content_fingerprint != expected:
            raise ValueError(
                f"content_fingerprint mismatch: config has {cfg.content_fingerprint}, source manifest has {expected}"
            )

    fsspec_mkdirs(cfg.output_path, exist_ok=True)
    out_file = posixpath.join(cfg.output_path, cfg.output_filename)
    compression = "gzip" if out_file.endswith(".gz") else None

    total_text_bytes = 0
    record_count = 0
    table_count = 0
    zip_bytes = _read_zip_bytes(cfg)

    with atomic_rename(out_file) as temp_path:
        with open_url(temp_path, "wt", encoding="utf-8", compression=compression) as outfile:
            stop = False
            for entry in _iter_csv_entries(zip_bytes):
                table_count += 1
                if total_text_bytes >= cfg.max_bytes_per_source:
                    break

                lines = _csv_lines(entry.csv_bytes, entry.table_member)
                if not lines:
                    continue
                header, body_lines = _header_and_body(lines, cfg.preserve_header)

                for chunk_index, chunk in enumerate(
                    chunk_lines_by_bytes(
                        body_lines,
                        max_bytes_per_chunk=cfg.max_bytes_per_document,
                        header_line=header if cfg.preserve_header else None,
                    )
                ):
                    text = serialize_csv_document(header if cfg.preserve_header else None, chunk)
                    text_bytes = len(text.encode("utf-8"))
                    if total_text_bytes + text_bytes > cfg.max_bytes_per_source and record_count > 0:
                        logger.info(
                            "Would exceed per-source cap with next WDC chunk (%d + %d > %d); stopping.",
                            total_text_bytes,
                            text_bytes,
                            cfg.max_bytes_per_source,
                        )
                        stop = True
                        break

                    record = {
                        "id": _record_id(cfg.source_label, entry.table_member, chunk_index),
                        "text": text,
                        "source": cfg.source_label,
                        "provenance": _provenance(
                            cfg=cfg,
                            entry=entry,
                            chunk_index=chunk_index,
                            header_preserved=cfg.preserve_header and header is not None,
                        ),
                    }
                    json.dump(record, outfile, ensure_ascii=False)
                    outfile.write("\n")
                    total_text_bytes += text_bytes
                    record_count += 1

                if stop:
                    break

    fs, _ = url_to_fs(out_file)
    output_size = int(fs.info(out_file)["size"])
    result: dict[str, int | str] = {
        "record_count": record_count,
        "bytes_written": output_size,
        "text_bytes_written": total_text_bytes,
        "output_file": out_file,
        "table_count": table_count,
    }

    if cfg.source_manifest is not None:
        metadata_path = write_ingestion_metadata_json(
            manifest=cfg.source_manifest,
            materialized_output=MaterializedOutputMetadata(
                input_path=cfg.sample_url,
                output_path=cfg.output_path,
                output_file=out_file,
                record_count=record_count,
                bytes_written=output_size,
                metadata={
                    "sample_name": cfg.sample_name,
                    "table_count": table_count,
                },
            ),
        )
        result["metadata_file"] = metadata_path

    return result
