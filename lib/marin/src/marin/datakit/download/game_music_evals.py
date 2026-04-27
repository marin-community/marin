# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Manifest-backed first-pass game/music ingestion helpers for long-tail PPL evals."""

from __future__ import annotations

import io
import json
import logging
import posixpath
import re
from dataclasses import dataclass
from typing import Any

import requests
import zstandard
from requests.adapters import HTTPAdapter
from rigging.filesystem import open_url
from urllib3.util import Retry
from zephyr.writers import atomic_rename

from marin.datakit.ingestion_manifest import (
    IngestionSourceManifest,
    MaterializedOutputMetadata,
    write_ingestion_metadata_json,
)
from marin.utils import fsspec_mkdirs

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_FILENAME = "data.jsonl.gz"
DEFAULT_HTTP_TIMEOUT_SECONDS = 120
LICHESS_SITE_RE = re.compile(r'^\[Site "(?:https?://)?lichess\.org/([^"]+)"\]$', re.MULTILINE)
LICHESS_EVENT_PREFIX = "[Event "


@dataclass(frozen=True)
class LichessPgnStagingConfig:
    """Configuration for materializing a bounded Lichess PGN sample."""

    source_url: str
    output_path: str
    source_label: str
    max_records: int
    output_filename: str = DEFAULT_OUTPUT_FILENAME
    http_timeout_seconds: int = DEFAULT_HTTP_TIMEOUT_SECONDS
    source_manifest: IngestionSourceManifest | None = None
    content_fingerprint: str = ""


@dataclass(frozen=True)
class HfJsonTextStagingConfig:
    """Configuration for staging a bounded text column from a raw HF JSON file."""

    dataset_id: str
    revision: str
    split_filename: str
    text_key: str
    output_path: str
    source_label: str
    max_examples: int
    output_filename: str = DEFAULT_OUTPUT_FILENAME
    http_timeout_seconds: int = DEFAULT_HTTP_TIMEOUT_SECONDS
    source_manifest: IngestionSourceManifest | None = None
    content_fingerprint: str = ""
    source_file_url_override: str | None = None


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


def _validate_manifest(source_manifest: IngestionSourceManifest | None, content_fingerprint: str) -> None:
    if source_manifest is None:
        return
    if not content_fingerprint:
        raise ValueError("content_fingerprint is required when source_manifest is set")
    expected = source_manifest.fingerprint()
    if content_fingerprint != expected:
        raise ValueError(
            f"content_fingerprint mismatch: config has {content_fingerprint}, source manifest has {expected}"
        )


def _write_record(record: dict[str, Any], handle: io.TextIOBase) -> int:
    text = record["text"]
    assert isinstance(text, str)
    json.dump(record, handle, ensure_ascii=False, sort_keys=True)
    handle.write("\n")
    return len(text.encode("utf-8"))


def _write_metadata(
    *,
    source_manifest: IngestionSourceManifest | None,
    input_path: str,
    output_path: str,
    output_file: str,
    record_count: int,
    bytes_written: int,
    metadata: dict[str, Any],
) -> str | None:
    if source_manifest is None:
        return None
    return write_ingestion_metadata_json(
        manifest=source_manifest,
        materialized_output=MaterializedOutputMetadata(
            input_path=input_path,
            output_path=output_path,
            output_file=output_file,
            record_count=record_count,
            bytes_written=bytes_written,
            metadata=metadata,
        ),
    )


def _iter_pgn_games(stream: io.TextIOBase):
    current_lines: list[str] = []
    saw_movetext = False

    for line in stream:
        if line.startswith(LICHESS_EVENT_PREFIX) and saw_movetext and current_lines:
            yield "".join(current_lines).rstrip("\n")
            current_lines = [line]
            saw_movetext = False
            continue

        if not current_lines and not line.startswith(LICHESS_EVENT_PREFIX):
            continue

        current_lines.append(line)
        if line.strip() and not line.startswith("["):
            saw_movetext = True

    if current_lines:
        yield "".join(current_lines).rstrip("\n")


def _lichess_record_id(game_text: str, index: int, source_label: str) -> str:
    match = LICHESS_SITE_RE.search(game_text)
    if match:
        return match.group(1)
    return f"{source_label}:{index:08d}"


def stage_lichess_pgn_sample(config: LichessPgnStagingConfig) -> dict[str, int | str]:
    """Stream a bounded official Lichess PGN sample into JSONL.gz."""

    _validate_manifest(config.source_manifest, config.content_fingerprint)
    fsspec_mkdirs(config.output_path, exist_ok=True)

    session = _build_session()
    output_file = posixpath.join(config.output_path, config.output_filename)
    record_count = 0
    bytes_written = 0
    try:
        with session.get(config.source_url, timeout=config.http_timeout_seconds, stream=True) as response:
            response.raise_for_status()
            response.raw.decode_content = False
            decompressor = zstandard.ZstdDecompressor()
            with (
                decompressor.stream_reader(response.raw) as compressed_stream,
                io.TextIOWrapper(compressed_stream, encoding="utf-8") as text_stream,
                atomic_rename(output_file) as temp_path,
                open_url(temp_path, "wt", encoding="utf-8", compression="gzip") as handle,
            ):
                for index, game_text in enumerate(_iter_pgn_games(text_stream)):
                    if not game_text.strip():
                        continue
                    record = {
                        "id": _lichess_record_id(game_text, index, config.source_label),
                        "text": game_text,
                        "source": config.source_label,
                        "provenance": {
                            "index": index,
                            "source_url": config.source_url,
                        },
                    }
                    bytes_written += _write_record(record, handle)
                    record_count += 1
                    if record_count >= config.max_records:
                        break
    finally:
        session.close()

    logger.info("Materialized %d Lichess PGN records from %s", record_count, config.source_url)
    metadata_file = _write_metadata(
        source_manifest=config.source_manifest,
        input_path=config.source_url,
        output_path=config.output_path,
        output_file=output_file,
        record_count=record_count,
        bytes_written=bytes_written,
        metadata={"source_url": config.source_url, "max_records": config.max_records},
    )
    result: dict[str, int | str] = {
        "record_count": record_count,
        "bytes_written": bytes_written,
        "output_file": output_file,
    }
    if metadata_file is not None:
        result["metadata_file"] = metadata_file
    return result


def _hf_resolve_url(dataset_id: str, revision: str, split_filename: str) -> str:
    return f"https://huggingface.co/datasets/{dataset_id}/resolve/{revision}/{split_filename}"


def _iter_json_examples(payload: Any):
    if isinstance(payload, list):
        yield from enumerate(payload)
        return
    raise ValueError(f"Expected top-level JSON list, got {type(payload).__name__}")


def stage_hf_json_text_source(config: HfJsonTextStagingConfig) -> dict[str, int | str]:
    """Stage a bounded raw text field from an HF JSON or JSONL file into JSONL.gz."""

    _validate_manifest(config.source_manifest, config.content_fingerprint)
    source_url = config.source_file_url_override or _hf_resolve_url(
        config.dataset_id, config.revision, config.split_filename
    )
    fsspec_mkdirs(config.output_path, exist_ok=True)
    output_file = posixpath.join(config.output_path, config.output_filename)

    session = _build_session()
    record_count = 0
    bytes_written = 0
    try:
        with session.get(source_url, timeout=config.http_timeout_seconds, stream=True) as response:
            response.raise_for_status()
            with (
                atomic_rename(output_file) as temp_path,
                open_url(temp_path, "wt", encoding="utf-8", compression="gzip") as handle,
            ):
                if config.split_filename.endswith(".jsonl"):
                    iterator = (
                        (index, json.loads(line))
                        for index, line in enumerate(response.iter_lines(decode_unicode=True))
                        if line
                    )
                else:
                    iterator = _iter_json_examples(response.json())

                for index, example in iterator:
                    if not isinstance(example, dict):
                        raise ValueError(f"Expected JSON object rows, got {type(example).__name__}")
                    text = example.get(config.text_key)
                    if text is None:
                        continue
                    if not isinstance(text, str):
                        raise ValueError(f"Expected {config.text_key!r} to be a string, got {type(text).__name__}")
                    if not text.strip():
                        continue
                    record = {
                        "id": f"{config.source_label}:{index:08d}",
                        "text": text,
                        "source": config.source_label,
                        "provenance": {
                            "dataset_id": config.dataset_id,
                            "revision": config.revision,
                            "split_filename": config.split_filename,
                            "index": index,
                        },
                    }
                    bytes_written += _write_record(record, handle)
                    record_count += 1
                    if record_count >= config.max_examples:
                        break
    finally:
        session.close()

    logger.info("Materialized %d text records from %s", record_count, source_url)
    metadata_file = _write_metadata(
        source_manifest=config.source_manifest,
        input_path=source_url,
        output_path=config.output_path,
        output_file=output_file,
        record_count=record_count,
        bytes_written=bytes_written,
        metadata={
            "dataset_id": config.dataset_id,
            "revision": config.revision,
            "split_filename": config.split_filename,
            "text_key": config.text_key,
        },
    )
    result: dict[str, int | str] = {
        "record_count": record_count,
        "bytes_written": bytes_written,
        "output_file": output_file,
    }
    if metadata_file is not None:
        result["metadata_file"] = metadata_file
    return result
