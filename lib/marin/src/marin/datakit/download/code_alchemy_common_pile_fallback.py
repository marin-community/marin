# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover the final Code Alchemy sources from the pinned Stack-Edu mirror.

The driver pins the Hugging Face repository before enumerating its gzip JSONL
objects and resolves every object exactly once to a signed URL. Zephyr assigns
one object to each task. Tasks stream through gzip without materializing the
mirror, persist a durable per-file result, and stage only matching records.
After every file has been scanned, the driver validates uniqueness and writes
canonical ``blob_id, source`` Parquet partitions plus a provenance side table.
"""

import dataclasses
import gzip
import hashlib
import io
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import BinaryIO, Iterable
from urllib.parse import quote

import draccus
import polars as pl
import requests
from fray.types import ResourceConfig
from huggingface_hub import HfApi
from rigging.filesystem.atomic import atomic_rename
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.log_setup import configure_logging
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.parquet_scan import scan_parquet, storage_options_for_path

from marin.utilities.validation_utils import write_provenance_json

logger = logging.getLogger(__name__)

MIRROR_REPO_ID = "craffel/common-pile-stack-edu"
# Immutable mirror revision resolved from the public repository on 2026-08-25.
MIRROR_REVISION = "2e1049392016c35a97de8ffa3602c19d989401e2"
TARGET_BLOB_IDS = (
    "3279b5122ce66ccb594902d6b59678092d9b35bd",
    "a57b990b1b4af0d164290f44c123ece12a08ada1",
)
DEFAULT_OUTPUT_PATH = "s3://marin-us-east-02a/tmp/ttl=30d/code-alchemy-hydration/fallback-sources"
MIRROR_LANGUAGES = ("python", "shell")
SHARDS_PER_LANGUAGE = 128
DEFAULT_MAX_WORKERS = 256
DEFAULT_RESOLVE_WORKERS = 32
DEFAULT_MAX_SHARD_FAILURES = 5
OBJECT_STORE_MAX_RETRIES = 10
_SHA1 = re.compile(r"^[0-9a-f]{40}$")

_CANDIDATE_SCHEMA = {
    "blob_id": pl.String,
    "source": pl.String,
    "mirror_record_id": pl.String,
    "src_encoding": pl.String,
    "content_id": pl.String,
    "raw_sha1": pl.String,
    "git_blob_sha1": pl.String,
    "mirror_repo": pl.String,
    "mirror_path": pl.String,
    "mirror_file": pl.String,
    "mirror_revision": pl.String,
    "mirror_line_number": pl.UInt64,
    "metadata_json": pl.String,
}
_PROVENANCE_COLUMNS = tuple(name for name in _CANDIDATE_SCHEMA if name != "source")


@dataclass(frozen=True)
class CodeAlchemyCommonPileFallbackConfig:
    """Pinned source, destination, and explicitly scaled Zephyr resources."""

    repo_id: str = MIRROR_REPO_ID
    revision: str = MIRROR_REVISION
    target_blob_ids: tuple[str, ...] = TARGET_BLOB_IDS
    mirror_languages: tuple[str, ...] = MIRROR_LANGUAGES
    output_path: str = DEFAULT_OUTPUT_PATH
    resolve_workers: int = DEFAULT_RESOLVE_WORKERS
    max_workers: int = DEFAULT_MAX_WORKERS
    request_connect_timeout_seconds: float = 30.0
    request_read_timeout_seconds: float = 300.0
    worker_resources: ResourceConfig = field(default_factory=lambda: ResourceConfig(cpu=2, ram="8g", disk="4g"))
    task_resources: ResourceConfig = field(default_factory=lambda: ResourceConfig(cpu=1, ram="4g", disk="2g"))
    coordinator_resources: ResourceConfig = field(
        default_factory=lambda: ResourceConfig(cpu=2, ram="8g", disk="4g", preemptible=False)
    )


@dataclass(frozen=True)
class MirrorFile:
    path: str
    size: int
    git_blob_id: str
    lfs_sha256: str


@dataclass(frozen=True)
class MirrorScanTask:
    file_index: int
    repo_id: str
    revision: str
    manifest_fingerprint: str
    mirror_file: MirrorFile
    signed_url: str
    target_blob_ids: tuple[str, ...]
    candidate_path: str
    metrics_path: str
    request_connect_timeout_seconds: float
    request_read_timeout_seconds: float


@dataclass(frozen=True)
class MirrorFileResult:
    file_index: int
    mirror_file: str
    expected_compressed_bytes: int
    input_rows: int
    matched_rows: int
    matched_blob_ids: tuple[str, ...]
    candidate_path: str | None
    elapsed_seconds: float
    reused: bool = False


@dataclass(frozen=True)
class VerifiedRecord:
    blob_id: str
    source: str
    mirror_record_id: str
    src_encoding: str
    content_id: str
    raw_sha1: str
    git_blob_sha1: str
    mirror_repo: str
    mirror_path: str
    metadata_json: str


def _validate_config(cfg: CodeAlchemyCommonPileFallbackConfig) -> None:
    if not _SHA1.fullmatch(cfg.revision):
        raise ValueError(f"Mirror revision must be a full immutable SHA-1, got {cfg.revision!r}")
    if not cfg.target_blob_ids or len(set(cfg.target_blob_ids)) != len(cfg.target_blob_ids):
        raise ValueError("target_blob_ids must be non-empty and unique")
    invalid = [blob_id for blob_id in cfg.target_blob_ids if not _SHA1.fullmatch(blob_id)]
    if invalid:
        raise ValueError(f"Target blob IDs must be lowercase SHA-1 values: {invalid}")
    if not cfg.mirror_languages or len(set(cfg.mirror_languages)) != len(cfg.mirror_languages):
        raise ValueError("mirror_languages must be non-empty and unique")
    invalid_languages = [language for language in cfg.mirror_languages if not re.fullmatch(r"[a-z0-9_]+", language)]
    if invalid_languages:
        raise ValueError(f"Invalid mirror language names: {invalid_languages}")
    for name, value in (("resolve_workers", cfg.resolve_workers), ("max_workers", cfg.max_workers)):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    for name, value in (
        ("request_connect_timeout_seconds", cfg.request_connect_timeout_seconds),
        ("request_read_timeout_seconds", cfg.request_read_timeout_seconds),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")


def list_pinned_mirror_files(cfg: CodeAlchemyCommonPileFallbackConfig) -> list[MirrorFile]:
    """List every shard for the target languages at exactly the configured commit."""

    _validate_config(cfg)
    info = HfApi().dataset_info(cfg.repo_id, revision=cfg.revision, files_metadata=True)
    if info.sha != cfg.revision:
        raise RuntimeError(f"Hugging Face resolved {cfg.revision} to unexpected revision {info.sha}")

    patterns = {
        language: re.compile(rf"^00000_{re.escape(language)}(\d+)\.json\.gz$")
        for language in cfg.mirror_languages
    }
    selected: dict[str, dict[int, MirrorFile]] = {language: {} for language in cfg.mirror_languages}
    for sibling in info.siblings or []:
        matched = next(
            (
                (language, match)
                for language, pattern in patterns.items()
                if (match := pattern.fullmatch(sibling.rfilename)) is not None
            ),
            None,
        )
        if matched is None:
            continue
        language, match = matched
        shard_index = int(match.group(1))
        lfs = sibling.lfs
        if sibling.size is None or sibling.blob_id is None or lfs is None or not lfs.sha256:
            raise RuntimeError(f"Mirror object lacks immutable size/blob/LFS metadata: {sibling.rfilename}")
        if shard_index in selected[language]:
            raise RuntimeError(f"Duplicate {language} mirror shard index {shard_index}")
        selected[language][shard_index] = MirrorFile(
            path=sibling.rfilename,
            size=int(sibling.size),
            git_blob_id=sibling.blob_id,
            lfs_sha256=lfs.sha256,
        )

    expected_indices = set(range(SHARDS_PER_LANGUAGE))
    for language, shards in selected.items():
        actual_indices = set(shards)
        if actual_indices != expected_indices:
            raise RuntimeError(
                f"Pinned mirror has incomplete {language} shard set: "
                f"missing={sorted(expected_indices - actual_indices)}, "
                f"unexpected={sorted(actual_indices - expected_indices)}"
            )
    return sorted(
        (item for language_files in selected.values() for item in language_files.values()),
        key=lambda item: item.path,
    )


def mirror_manifest_fingerprint(repo_id: str, revision: str, files: Iterable[MirrorFile]) -> str:
    digest = hashlib.sha256()
    digest.update(f"{repo_id}\t{revision}\n".encode())
    for item in files:
        digest.update(f"{item.path}\t{item.size}\t{item.git_blob_id}\t{item.lfs_sha256}\n".encode())
    return digest.hexdigest()


def _resolve_signed_url(repo_id: str, revision: str, item: MirrorFile) -> str:
    """Resolve one pinned object once; HEAD follows the redirect without downloading it."""

    source_url = (
        f"https://huggingface.co/datasets/{quote(repo_id, safe='/')}/resolve/"
        f"{quote(revision, safe='')}/{quote(item.path, safe='/')}"
    )
    response = requests.head(source_url, allow_redirects=True, timeout=(30.0, 120.0))
    try:
        response.raise_for_status()
        final_url = response.url
        content_length = response.headers.get("Content-Length")
        if content_length is not None and int(content_length) != item.size:
            raise RuntimeError(
                f"Resolved object size changed for {item.path}: manifest={item.size}, HEAD={content_length}"
            )
        if final_url == source_url:
            raise RuntimeError(f"Hugging Face did not resolve {item.path} to an immutable signed object URL")
        return final_url
    finally:
        response.close()


def resolve_signed_urls(
    repo_id: str,
    revision: str,
    files: list[MirrorFile],
    *,
    workers: int,
) -> list[str]:
    """Resolve every object exactly once while retaining manifest order."""

    with ThreadPoolExecutor(max_workers=min(workers, len(files)), thread_name_prefix="hf-resolve") as pool:
        return list(pool.map(lambda item: _resolve_signed_url(repo_id, revision, item), files))


def _task_identity(task: MirrorScanTask) -> dict[str, object]:
    return {
        "file_index": task.file_index,
        "repo_id": task.repo_id,
        "revision": task.revision,
        "manifest_fingerprint": task.manifest_fingerprint,
        "mirror_file": dataclasses.asdict(task.mirror_file),
        "target_blob_ids": list(task.target_blob_ids),
    }


def _write_json_atomic(path: str, payload: dict[str, object]) -> None:
    StoragePath(path.rsplit("/", 1)[0]).mkdirs()
    with atomic_rename(path) as temporary_path:
        StoragePath(temporary_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _read_completed_result(task: MirrorScanTask) -> MirrorFileResult | None:
    metrics = StoragePath(task.metrics_path)
    if not metrics.exists():
        return None
    payload = json.loads(metrics.read_text())
    if payload.get("status") != "complete" or payload.get("task") != _task_identity(task):
        return None
    result = MirrorFileResult(**payload["result"])
    if result.candidate_path is not None and not StoragePath(result.candidate_path).exists():
        return None
    return dataclasses.replace(result, reused=True)


def _strict_metadata(record: dict[str, object]) -> dict[str, object]:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("matching Stack-Edu record has no metadata object")
    return metadata


def verify_record_identity(record: dict[str, object], expected_blob_id: str) -> VerifiedRecord:
    """Verify Stack-Edu raw SHA-1 and Git content identity without lossy decoding."""

    metadata = _strict_metadata(record)
    record_id = record.get("id")
    metadata_blob_id = metadata.get("blob_id")
    source = record.get("text")
    src_encoding = metadata.get("src_encoding")
    content_id = metadata.get("content_id")
    if record_id != expected_blob_id or metadata_blob_id != expected_blob_id:
        raise ValueError(
            f"matching record identity disagreement: expected={expected_blob_id}, "
            f"id={record_id!r}, metadata.blob_id={metadata_blob_id!r}"
        )
    if not isinstance(source, str) or not isinstance(src_encoding, str) or not src_encoding:
        raise ValueError(f"{expected_blob_id}: text and metadata.src_encoding must be strings")
    if not isinstance(content_id, str) or _SHA1.fullmatch(content_id) is None:
        raise ValueError(f"{expected_blob_id}: metadata.content_id is not a lowercase SHA-1")
    try:
        raw = source.encode(src_encoding, errors="strict")
    except (LookupError, UnicodeEncodeError) as error:
        raise ValueError(
            f"{expected_blob_id}: cannot reconstruct raw bytes with src_encoding={src_encoding!r}"
        ) from error

    raw_sha1 = hashlib.sha1(raw).hexdigest()
    git_blob_sha1 = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    if raw_sha1 != expected_blob_id:
        raise ValueError(
            f"{expected_blob_id}: raw Stack-Edu SHA-1 mismatch after {src_encoding!r} re-encoding: {raw_sha1}; "
            f"metadata.content_id={content_id}, computed_git_blob={git_blob_sha1}"
        )
    if git_blob_sha1 != content_id:
        raise ValueError(
            f"{expected_blob_id}: metadata.content_id mismatch: metadata={content_id}, computed={git_blob_sha1}"
        )
    length_bytes = metadata.get("length_bytes")
    if length_bytes is not None and (not isinstance(length_bytes, int) or length_bytes != len(raw)):
        raise ValueError(
            f"{expected_blob_id}: metadata.length_bytes={length_bytes!r}, reconstructed={len(raw)}"
        )

    return VerifiedRecord(
        blob_id=expected_blob_id,
        source=source,
        mirror_record_id=record_id,
        src_encoding=src_encoding,
        content_id=content_id,
        raw_sha1=raw_sha1,
        git_blob_sha1=git_blob_sha1,
        mirror_repo=str(metadata.get("repo_name") or metadata.get("repo") or ""),
        mirror_path=str(metadata.get("path") or ""),
        metadata_json=json.dumps(metadata, sort_keys=True, separators=(",", ":"), default=str),
    )


def _sink_parquet(frame: pl.DataFrame, path: str) -> None:
    options: dict[str, object] = dict(storage_options_for_path(path) or {})
    options["max_retries"] = OBJECT_STORE_MAX_RETRIES
    frame.lazy().sink_parquet(
        path,
        compression="zstd",
        compression_level=3,
        statistics=True,
        mkdir=True,
        engine="streaming",
        storage_options=options,
    )


def _candidate_frame(records: list[VerifiedRecord], task: MirrorScanTask, line_numbers: list[int]) -> pl.DataFrame:
    rows = []
    for record, line_number in zip(records, line_numbers, strict=True):
        row = dataclasses.asdict(record)
        row.update(
            {
                "mirror_file": task.mirror_file.path,
                "mirror_revision": task.revision,
                "mirror_line_number": line_number,
            }
        )
        rows.append(row)
    return pl.DataFrame(rows, schema=_CANDIDATE_SCHEMA)


def _scan_stream(stream: BinaryIO, task: MirrorScanTask) -> tuple[int, list[VerifiedRecord], list[int]]:
    target_ids = frozenset(task.target_blob_ids)
    input_rows = 0
    matches: list[VerifiedRecord] = []
    line_numbers: list[int] = []
    with gzip.GzipFile(fileobj=stream, mode="rb") as compressed:
        with io.TextIOWrapper(compressed, encoding="utf-8", errors="strict", newline="") as text_stream:
            for line_number, line in enumerate(text_stream, start=1):
                input_rows += 1
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"{task.mirror_file.path}:{line_number}: JSONL row is not an object")
                metadata = record.get("metadata")
                metadata_blob_id = metadata.get("blob_id") if isinstance(metadata, dict) else None
                candidate_ids = target_ids.intersection((record.get("id"), metadata_blob_id))
                if not candidate_ids:
                    continue
                if len(candidate_ids) != 1:
                    raise ValueError(f"{task.mirror_file.path}:{line_number}: ambiguous target identity")
                expected_blob_id = next(iter(candidate_ids))
                matches.append(verify_record_identity(record, expected_blob_id))
                line_numbers.append(line_number)
    return input_rows, matches, line_numbers


def scan_mirror_file(task: MirrorScanTask) -> MirrorFileResult:
    """Stream one signed gzip JSONL object and durably record its exhaustive outcome."""

    completed = _read_completed_result(task)
    if completed is not None:
        return completed

    started = time.monotonic()
    response = requests.get(
        task.signed_url,
        stream=True,
        timeout=(task.request_connect_timeout_seconds, task.request_read_timeout_seconds),
    )
    try:
        response.raise_for_status()
        response.raw.decode_content = False
        input_rows, matches, line_numbers = _scan_stream(response.raw, task)
    finally:
        response.close()

    candidate_path: str | None = None
    if matches:
        _sink_parquet(_candidate_frame(matches, task, line_numbers), task.candidate_path)
        candidate_path = task.candidate_path
    result = MirrorFileResult(
        file_index=task.file_index,
        mirror_file=task.mirror_file.path,
        expected_compressed_bytes=task.mirror_file.size,
        input_rows=input_rows,
        matched_rows=len(matches),
        matched_blob_ids=tuple(record.blob_id for record in matches),
        candidate_path=candidate_path,
        elapsed_seconds=time.monotonic() - started,
    )
    _write_json_atomic(
        task.metrics_path,
        {"status": "complete", "task": _task_identity(task), "result": dataclasses.asdict(result)},
    )
    return result


def _write_manifest(
    cfg: CodeAlchemyCommonPileFallbackConfig,
    files: list[MirrorFile],
    manifest_fingerprint: str,
) -> str:
    path = prefix_join(cfg.output_path, ".provenance/mirror-files.jsonl")
    text = "".join(
        json.dumps({"repo_id": cfg.repo_id, "revision": cfg.revision, **dataclasses.asdict(item)}, sort_keys=True)
        + "\n"
        for item in files
    )
    StoragePath(prefix_join(cfg.output_path, ".provenance")).mkdirs()
    with atomic_rename(path) as temporary_path:
        StoragePath(temporary_path).write_text(text)
    _write_json_atomic(
        prefix_join(cfg.output_path, ".provenance/mirror-manifest-summary.json"),
        {
            "repo_id": cfg.repo_id,
            "revision": cfg.revision,
            "languages": list(cfg.mirror_languages),
            "shards_per_language": SHARDS_PER_LANGUAGE,
            "file_count": len(files),
            "compressed_bytes": sum(item.size for item in files),
            "manifest_fingerprint": manifest_fingerprint,
            "manifest_path": path,
        },
    )
    return path


def build_scan_tasks(
    cfg: CodeAlchemyCommonPileFallbackConfig,
    files: list[MirrorFile],
    signed_urls: list[str],
    manifest_fingerprint: str,
) -> list[MirrorScanTask]:
    if len(files) != len(signed_urls):
        raise ValueError(f"Expected one signed URL per mirror file: files={len(files)}, urls={len(signed_urls)}")
    tasks = []
    for index, (item, signed_url) in enumerate(zip(files, signed_urls, strict=True)):
        stem = f"part-{index:05d}-of-{len(files):05d}"
        tasks.append(
            MirrorScanTask(
                file_index=index,
                repo_id=cfg.repo_id,
                revision=cfg.revision,
                manifest_fingerprint=manifest_fingerprint,
                mirror_file=item,
                signed_url=signed_url,
                target_blob_ids=cfg.target_blob_ids,
                candidate_path=prefix_join(cfg.output_path, f".staging/candidates/{stem}.parquet"),
                metrics_path=prefix_join(cfg.output_path, f".metrics/files/{stem}.json"),
                request_connect_timeout_seconds=cfg.request_connect_timeout_seconds,
                request_read_timeout_seconds=cfg.request_read_timeout_seconds,
            )
        )
    return tasks


def _load_complete_results(tasks: list[MirrorScanTask]) -> list[MirrorFileResult]:
    results = []
    incomplete = []
    for task in tasks:
        result = _read_completed_result(task)
        if result is None:
            incomplete.append(task.mirror_file.path)
        else:
            results.append(result)
    if incomplete:
        raise RuntimeError(f"Missing durable completion metrics for {len(incomplete)} mirror files: {incomplete[:8]}")
    return results


def compact_exact_results(
    cfg: CodeAlchemyCommonPileFallbackConfig,
    tasks: list[MirrorScanTask],
    results: list[MirrorFileResult],
) -> dict[str, int]:
    """Require one distinct source per target and write exact canonical partitions."""

    candidate_paths = tuple(result.candidate_path for result in results if result.candidate_path is not None)
    if not candidate_paths:
        raise RuntimeError("Full pinned mirror scan found none of the requested blob IDs")
    candidates = pl.concat([scan_parquet(path) for path in candidate_paths]).collect(engine="streaming")
    missing_columns = set(_CANDIDATE_SCHEMA) - set(candidates.columns)
    if missing_columns:
        raise RuntimeError(f"Candidate side table is missing columns: {sorted(missing_columns)}")

    target_set = set(cfg.target_blob_ids)
    observed_set = set(candidates.get_column("blob_id").to_list())
    unexpected = observed_set - target_set
    missing = target_set - observed_set
    if unexpected or missing:
        raise RuntimeError(f"Candidate identity coverage mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")

    distinct = candidates.select("blob_id", "source").unique()
    source_counts = dict(distinct.group_by("blob_id").agg(pl.len().alias("count")).iter_rows())
    invalid_counts = {
        blob_id: source_counts.get(blob_id, 0)
        for blob_id in cfg.target_blob_ids
        if source_counts.get(blob_id) != 1
    }
    if invalid_counts:
        raise RuntimeError(f"Expected exactly one distinct source per target blob ID, got {invalid_counts}")

    canonical = distinct.sort("blob_id")
    if canonical.height != len(cfg.target_blob_ids):
        raise RuntimeError(f"Expected {len(cfg.target_blob_ids)} canonical rows, got {canonical.height}")
    provenance = candidates.select(*_PROVENANCE_COLUMNS).unique().sort(
        ["blob_id", "mirror_file", "mirror_line_number"]
    )
    for blob_id in cfg.target_blob_ids:
        prefix = blob_id[:2]
        canonical_partition = canonical.filter(pl.col("blob_id") == blob_id).select(
            pl.col("blob_id").cast(pl.String),
            pl.col("source").cast(pl.String),
        )
        if canonical_partition.height != 1 or canonical_partition.schema != {
            "blob_id": pl.String,
            "source": pl.String,
        }:
            raise RuntimeError(f"Canonical partition invariant failed for {blob_id}")
        _sink_parquet(
            canonical_partition,
            prefix_join(cfg.output_path, f"data/blob_prefix={prefix}/part-00000.parquet"),
        )
        _sink_parquet(
            provenance.filter(pl.col("blob_id") == blob_id),
            prefix_join(cfg.output_path, f"provenance/data/blob_prefix={prefix}/part-00000.parquet"),
        )

    totals = {
        "mirror_file_count": len(tasks),
        "completed_file_count": len(results),
        "input_rows": sum(result.input_rows for result in results),
        "matched_rows": sum(result.matched_rows for result in results),
        "canonical_rows": canonical.height,
        "distinct_target_blob_ids": canonical.get_column("blob_id").n_unique(),
    }
    _write_json_atomic(prefix_join(cfg.output_path, ".metrics/summary.json"), totals)
    return totals


def recover_code_alchemy_common_pile(cfg: CodeAlchemyCommonPileFallbackConfig) -> None:
    """Plan, execute, validate, compact, and record the complete recovery."""

    files = list_pinned_mirror_files(cfg)
    manifest_fingerprint = mirror_manifest_fingerprint(cfg.repo_id, cfg.revision, files)
    manifest_path = _write_manifest(cfg, files, manifest_fingerprint)
    logger.info(
        "Pinned %d mirror gzip files (%0.2f GiB) at %s@%s; resolving each object once",
        len(files),
        sum(item.size for item in files) / 1024**3,
        cfg.repo_id,
        cfg.revision,
    )
    signed_urls = resolve_signed_urls(cfg.repo_id, cfg.revision, files, workers=cfg.resolve_workers)
    tasks = build_scan_tasks(cfg, files, signed_urls, manifest_fingerprint)
    execution_path = prefix_join(
        cfg.output_path,
        f".metrics/execution/{manifest_fingerprint}/part-{{shard:05d}}-of-{{total:05d}}.jsonl",
    )
    pipeline = Dataset.from_list(tasks).map(scan_mirror_file).write_jsonl(execution_path, skip_existing=True)
    context = ZephyrContext(
        name="code-alchemy-common-pile-fallback",
        resources=cfg.worker_resources,
        coordinator_resources=cfg.coordinator_resources,
        max_workers=min(cfg.max_workers, len(tasks)),
        max_shard_failures=DEFAULT_MAX_SHARD_FAILURES,
        max_execution_retries=10,
    )
    with context:
        context.execute(pipeline, map_task_resources=cfg.task_resources)

    results = _load_complete_results(tasks)
    totals = compact_exact_results(cfg, tasks, results)
    write_provenance_json(
        cfg.output_path,
        metadata={
            "source_dataset": cfg.repo_id,
            "source_revision": cfg.revision,
            "source_languages": list(cfg.mirror_languages),
            "shards_per_language": SHARDS_PER_LANGUAGE,
            "mirror_manifest_path": manifest_path,
            "mirror_manifest_fingerprint": manifest_fingerprint,
            "target_blob_ids": list(cfg.target_blob_ids),
            "canonical_columns": ["blob_id", "source"],
            "canonical_partition_key": "lower(blob_id[:2])",
            "provenance_side_table": prefix_join(cfg.output_path, "provenance/data"),
            "per_file_metrics": prefix_join(cfg.output_path, ".metrics/files"),
            "hash_identity": (
                "sha1(text encoded strictly with metadata.src_encoding) == id == metadata.blob_id; "
                "sha1(git blob header || reconstructed bytes) == metadata.content_id"
            ),
            "streaming_policy": "one Zephyr task per pinned gzip; gzip JSONL streamed without mirror materialization",
            "resources": {
                "max_workers": cfg.max_workers,
                "worker": dataclasses.asdict(cfg.worker_resources),
                "map_task": dataclasses.asdict(cfg.task_resources),
                "coordinator": dataclasses.asdict(cfg.coordinator_resources),
            },
            "metrics": totals,
        },
    )


@draccus.wrap()
def main(cfg: CodeAlchemyCommonPileFallbackConfig) -> None:
    configure_logging(level=logging.INFO)
    recover_code_alchemy_common_pile(cfg)


if __name__ == "__main__":
    main()
