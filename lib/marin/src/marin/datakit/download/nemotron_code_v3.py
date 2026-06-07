# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Nemotron-Pretraining-Code-v3 metadata download and bounded materialization.

``nvidia/Nemotron-Pretraining-Code-v3`` is a GitHub source-code refresh, but
the Hugging Face dataset stores metadata only. Each row points at a raw GitHub
blob via ``repo``, ``commit_id``, and ``rel_path``. This module keeps that
metadata download separate from materialization so Marin never normalizes the
metadata as training text by accident.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import posixpath
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import quote

import pyarrow as pa
import pyarrow.parquet as pq
import requests
from fray import ResourceConfig
from rigging.filesystem import atomic_rename, open_url, url_to_fs
from zephyr import counters

from marin.datakit.download.huggingface import download_hf_step
from marin.execution.step_spec import StepSpec
from marin.utils import fsspec_mkdirs

logger = logging.getLogger(__name__)

HF_DATASET_ID = "nvidia/Nemotron-Pretraining-Code-v3"
HF_REVISION = "9b42feaec991c69006452e6654d91a58a04d935a"
METADATA_PARQUET_GLOB = "Nemotron-Code-Metadata/**/*.parquet"
GITHUB_RAW_BASE_URL = "https://raw.githubusercontent.com"

SUCCESS_DIR = "success"
FAILURES_DIR = "failures"
DEFAULT_BATCH_ROWS = 1024
_READ_CHUNK_BYTES = 64 * 1024
_PERMANENT_MISSING_STATUSES = frozenset({404, 410})
_TRANSIENT_HTTP_STATUSES = frozenset({401, 403, 408, 409, 425, 429, 500, 502, 503, 504})
_REQUEST_HEADERS = {"User-Agent": "marin-nemotron-code-v3-materializer"}


@dataclass(frozen=True)
class GitHubBlobRef:
    """Pointer to one source file in a GitHub repository."""

    repo: str
    rel_path: str
    language: str
    commit_id: str


@dataclass(frozen=True)
class NemotronCodeV3MaterializeConfig:
    """Configuration for a bounded Nemotron Code v3 materialization view."""

    view_name: str
    metadata_relative_glob: str
    allowed_languages: tuple[str, ...]
    max_rows: int | None
    max_file_bytes: int
    request_timeout_seconds: float
    retry_attempts: int
    raw_base_url: str = GITHUB_RAW_BASE_URL
    batch_rows: int = DEFAULT_BATCH_ROWS
    record_transient_failures: bool = False

    def __post_init__(self) -> None:
        if not self.view_name:
            raise ValueError("view_name must be non-empty")
        if not self.metadata_relative_glob:
            raise ValueError("metadata_relative_glob must be non-empty")
        if not self.allowed_languages:
            raise ValueError("allowed_languages must be non-empty")
        if self.max_rows is not None and self.max_rows <= 0:
            raise ValueError("max_rows must be positive when set")
        if self.max_file_bytes <= 0:
            raise ValueError("max_file_bytes must be positive")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        if self.retry_attempts <= 0:
            raise ValueError("retry_attempts must be positive")
        if self.batch_rows <= 0:
            raise ValueError("batch_rows must be positive")


PILOT_CONFIG = NemotronCodeV3MaterializeConfig(
    view_name="pilot",
    metadata_relative_glob="Nemotron-Code-Metadata/part_00000.parquet",
    allowed_languages=("Python", "JavaScript", "TypeScript", "Go", "Rust", "Java", "C++", "C"),
    max_rows=1_000_000,
    max_file_bytes=1_000_000,
    request_timeout_seconds=20.0,
    retry_attempts=3,
)


@dataclass(frozen=True)
class MaterializedBlob:
    """Successful raw GitHub blob fetch."""

    ref: GitHubBlobRef
    source_url: str
    text: str


@dataclass(frozen=True)
class MaterializationFailure:
    """Permanent skip or diagnostics-only fetch failure."""

    repo: str | None
    rel_path: str | None
    language: str | None
    commit_id: str | None
    source_url: str | None
    status: str
    http_status: int | None = None
    error: str | None = None


class TransientFetchError(RuntimeError):
    """A fetch failed for reasons that should not silently change the dataset."""


def download_nemotron_code_v3_metadata_step() -> StepSpec:
    """Download the Nemotron Code v3 GitHub blob metadata from Hugging Face."""
    return download_hf_step(
        "raw/nemotron_pretraining_code_v3_metadata",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        hf_urls_glob=[METADATA_PARQUET_GLOB],
    )


def blob_ref_from_row(row: Mapping[str, Any]) -> GitHubBlobRef:
    """Parse and validate one Nemotron Code v3 metadata row."""
    repo = _required_str(row, "repo")
    rel_path = _required_str(row, "rel_path")
    language = _required_str(row, "language")
    commit_id = _required_str(row, "commit_id")

    repo_parts = repo.split("/")
    if len(repo_parts) != 2 or not all(repo_parts):
        raise ValueError(f"repo must be owner/repo, got {repo!r}")

    path = PurePosixPath(rel_path)
    if path.is_absolute():
        raise ValueError(f"rel_path must be relative, got {rel_path!r}")
    if ".." in path.parts:
        raise ValueError(f"rel_path must not contain '..', got {rel_path!r}")
    if not path.parts:
        raise ValueError("rel_path must contain at least one path segment")

    return GitHubBlobRef(repo=repo, rel_path=rel_path, language=language, commit_id=commit_id)


def github_raw_url(ref: GitHubBlobRef, *, raw_base_url: str = GITHUB_RAW_BASE_URL) -> str:
    """Return the raw GitHub URL for a validated blob reference."""
    repo = quote(ref.repo.strip("/"), safe="/")
    commit_id = quote(ref.commit_id, safe="")
    rel_path = quote(ref.rel_path.lstrip("/"), safe="/")
    return f"{raw_base_url.rstrip('/')}/{repo}/{commit_id}/{rel_path}"


def materialize_blob(
    ref: GitHubBlobRef,
    *,
    config: NemotronCodeV3MaterializeConfig,
    session: requests.Session,
) -> MaterializedBlob | MaterializationFailure:
    """Fetch one GitHub blob or return a permanent skip diagnostic."""
    url = github_raw_url(ref, raw_base_url=config.raw_base_url)
    last_error: str | None = None
    last_failure_status = "timeout"
    last_http_status: int | None = None
    for _attempt in range(config.retry_attempts):
        try:
            with session.get(
                url,
                headers=_REQUEST_HEADERS,
                stream=True,
                timeout=config.request_timeout_seconds,
            ) as response:
                status_code = response.status_code
                if status_code in _PERMANENT_MISSING_STATUSES:
                    counters.increment("nemotron_code_v3/fetch_not_found")
                    return _failure(ref, url, "not_found", http_status=status_code)
                if status_code in _TRANSIENT_HTTP_STATUSES or status_code >= 500:
                    last_error = f"HTTP {status_code}"
                    last_failure_status = "http_error"
                    last_http_status = status_code
                    continue
                if status_code >= 400:
                    counters.increment("nemotron_code_v3/fetch_http_error")
                    return _failure(ref, url, "http_error", http_status=status_code)

                length_header = response.headers.get("Content-Length")
                if _content_length_exceeds_cap(length_header, max_file_bytes=config.max_file_bytes):
                    counters.increment("nemotron_code_v3/dropped_too_large")
                    return _failure(ref, url, "too_large", http_status=status_code)

                raw = _read_bounded_response(response, max_file_bytes=config.max_file_bytes)
                if raw is None:
                    counters.increment("nemotron_code_v3/dropped_too_large")
                    return _failure(ref, url, "too_large", http_status=status_code)

                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError as exc:
                    counters.increment("nemotron_code_v3/decode_error")
                    return _failure(ref, url, "decode_error", http_status=status_code, error=str(exc))

                if not text.strip():
                    counters.increment("nemotron_code_v3/dropped_empty")
                    return _failure(ref, url, "empty", http_status=status_code)

                counters.increment("nemotron_code_v3/kept")
                return MaterializedBlob(ref=ref, source_url=url, text=text)
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            last_failure_status = "timeout"
            last_http_status = None
            counters.increment("nemotron_code_v3/fetch_timeout")
        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            last_failure_status = "http_error"
            last_http_status = exc.response.status_code if exc.response is not None else None
            counters.increment("nemotron_code_v3/fetch_http_error")

    if config.record_transient_failures:
        return _failure(ref, url, last_failure_status, http_status=last_http_status, error=last_error)

    counters.increment("nemotron_code_v3/job_failed_transient_fetch")
    raise TransientFetchError(f"Failed to fetch {url} after {config.retry_attempts} attempts: {last_error}")


def materialize_nemotron_code_v3(
    input_path: str,
    output_path: str,
    *,
    config: NemotronCodeV3MaterializeConfig = PILOT_CONFIG,
    session_factory: Callable[[], requests.Session] = requests.Session,
) -> dict[str, int]:
    """Materialize a bounded Nemotron Code v3 metadata slice into code documents.

    Success rows are written under ``<output_path>/success`` so the downstream
    normalize step can target that directory explicitly. Permanent skip
    diagnostics are written under ``<output_path>/failures``.
    """
    success_writer = _ParquetBatchWriter(posixpath.join(output_path, SUCCESS_DIR), batch_rows=config.batch_rows)
    failure_writer = _ParquetBatchWriter(posixpath.join(output_path, FAILURES_DIR), batch_rows=config.batch_rows)
    counts = {"metadata_rows": 0, "fetch_attempts": 0, "successes": 0, "failures": 0}

    try:
        with session_factory() as session:
            for row in _iter_metadata_rows(input_path, metadata_relative_glob=config.metadata_relative_glob):
                counters.increment("nemotron_code_v3/metadata_rows")
                counts["metadata_rows"] += 1
                try:
                    ref = blob_ref_from_row(row)
                except ValueError as exc:
                    counters.increment("nemotron_code_v3/dropped_missing_field")
                    failure_writer.write(_row_failure(row, status="invalid_metadata", error=str(exc)))
                    counts["failures"] += 1
                    continue

                if ref.language not in config.allowed_languages:
                    counters.increment("nemotron_code_v3/dropped_language")
                    failure_writer.write(
                        dataclasses.asdict(
                            _failure(ref, github_raw_url(ref, raw_base_url=config.raw_base_url), "language")
                        )
                    )
                    counts["failures"] += 1
                    continue

                if config.max_rows is not None and counts["fetch_attempts"] >= config.max_rows:
                    break

                counts["fetch_attempts"] += 1
                result = materialize_blob(ref, config=config, session=session)
                if isinstance(result, MaterializedBlob):
                    success_writer.write(_success_row(result))
                    counts["successes"] += 1
                else:
                    failure_writer.write(dataclasses.asdict(result))
                    counts["failures"] += 1
    finally:
        success_writer.close()
        failure_writer.close()

    _write_metadata(output_path, config=config, counts=counts)
    logger.info("Materialized Nemotron Code v3 view %s: %s", config.view_name, counts)
    return counts


def materialize_nemotron_code_v3_step(
    metadata: StepSpec,
    *,
    config: NemotronCodeV3MaterializeConfig = PILOT_CONFIG,
) -> StepSpec:
    """Create a StepSpec that materializes one bounded Nemotron Code v3 view."""
    return StepSpec(
        name=f"processed/nemotron_code_v3/{config.view_name}",
        deps=[metadata],
        fn=lambda output_path: materialize_nemotron_code_v3(
            input_path=metadata.output_path,
            output_path=output_path,
            config=config,
        ),
        hash_attrs={
            "version": "v1",
            "view_name": config.view_name,
            "metadata_relative_glob": config.metadata_relative_glob,
            "allowed_languages": config.allowed_languages,
            "max_rows": config.max_rows,
            "max_file_bytes": config.max_file_bytes,
            "raw_base_url": config.raw_base_url,
            "batch_rows": config.batch_rows,
            "record_transient_failures": config.record_transient_failures,
        },
    )


def nemotron_code_v3_normalize_steps(
    config: NemotronCodeV3MaterializeConfig = PILOT_CONFIG,
) -> tuple[StepSpec, ...]:
    """Return the metadata -> materialize -> normalize chain for one v3 view."""
    from marin.datakit.normalize import normalize_step

    metadata = download_nemotron_code_v3_metadata_step()
    materialized = materialize_nemotron_code_v3_step(metadata, config=config)
    return (
        metadata,
        materialized,
        normalize_step(
            name=f"normalized/nemotron_code_v3/{config.view_name}",
            download=materialized,
            relative_input_path=SUCCESS_DIR,
            file_extensions=(".parquet",),
            worker_resources=ResourceConfig(cpu=1, ram="8g"),
        ),
    )


def _required_str(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _content_length_exceeds_cap(length_header: str | None, *, max_file_bytes: int) -> bool:
    if length_header is None:
        return False
    try:
        return int(length_header) > max_file_bytes
    except ValueError:
        return False


def _read_bounded_response(response: requests.Response, *, max_file_bytes: int) -> bytes | None:
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_content(chunk_size=_READ_CHUNK_BYTES):
        if not chunk:
            continue
        total += len(chunk)
        if total > max_file_bytes:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


def _failure(
    ref: GitHubBlobRef,
    source_url: str,
    status: str,
    *,
    http_status: int | None = None,
    error: str | None = None,
) -> MaterializationFailure:
    return MaterializationFailure(
        repo=ref.repo,
        rel_path=ref.rel_path,
        language=ref.language,
        commit_id=ref.commit_id,
        source_url=source_url,
        status=status,
        http_status=http_status,
        error=error,
    )


def _row_failure(row: Mapping[str, Any], *, status: str, error: str) -> dict[str, Any]:
    return {
        "repo": row.get("repo"),
        "rel_path": row.get("rel_path"),
        "language": row.get("language"),
        "commit_id": row.get("commit_id"),
        "source_url": None,
        "status": status,
        "http_status": None,
        "error": error,
    }


def _success_row(blob: MaterializedBlob) -> dict[str, Any]:
    return {
        "id": hashlib.sha256(blob.text.encode("utf-8")).hexdigest(),
        "text": blob.text,
        "source": HF_DATASET_ID,
        "repo": blob.ref.repo,
        "rel_path": blob.ref.rel_path,
        "commit_id": blob.ref.commit_id,
        "language": blob.ref.language,
        "source_url": blob.source_url,
    }


def _iter_metadata_rows(input_path: str, *, metadata_relative_glob: str) -> Iterator[dict[str, Any]]:
    for path in _metadata_file_paths(input_path, metadata_relative_glob=metadata_relative_glob):
        with open_url(path, "rb") as handle:
            parquet_file = pq.ParquetFile(handle)
            for batch in parquet_file.iter_batches(batch_size=DEFAULT_BATCH_ROWS):
                columns = batch.to_pydict()
                row_count = len(next(iter(columns.values()), []))
                for index in range(row_count):
                    yield {name: values[index] for name, values in columns.items()}


def _metadata_file_paths(input_path: str, *, metadata_relative_glob: str) -> list[str]:
    fs, resolved = url_to_fs(input_path)
    pattern = posixpath.join(resolved, metadata_relative_glob)
    paths = sorted(fs.glob(pattern))
    if not paths:
        raise ValueError(f"No metadata parquet files matched {metadata_relative_glob!r} under {input_path!r}")

    protocol = input_path.split("://", 1)[0] if "://" in input_path else ""
    return [f"{protocol}://{path}" if protocol and "://" not in path else path for path in paths]


class _ParquetBatchWriter:
    def __init__(self, output_dir: str, *, batch_rows: int) -> None:
        self.output_dir = output_dir
        self.batch_rows = batch_rows
        self.rows: list[dict[str, Any]] = []
        self.shard = 0

    def write(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.batch_rows:
            self._flush()

    def close(self) -> None:
        self._flush()

    def _flush(self) -> None:
        if not self.rows:
            return
        fsspec_mkdirs(self.output_dir, exist_ok=True)
        output_file = posixpath.join(self.output_dir, f"part-{self.shard:05d}.parquet")
        with atomic_rename(output_file) as temp_path:
            with open_url(temp_path, "wb") as handle:
                pq.write_table(pa.Table.from_pylist(self.rows), handle)
        self.rows = []
        self.shard += 1


def _write_metadata(output_path: str, *, config: NemotronCodeV3MaterializeConfig, counts: dict[str, int]) -> None:
    metadata = {
        "source": HF_DATASET_ID,
        "hf_revision": HF_REVISION,
        "config": dataclasses.asdict(config),
        "counts": counts,
    }
    metadata_path = posixpath.join(output_path, "metadata.json")
    fsspec_mkdirs(output_path, exist_ok=True)
    with atomic_rename(metadata_path) as temp_path:
        with open_url(temp_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
