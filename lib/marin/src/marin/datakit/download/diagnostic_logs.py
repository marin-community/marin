# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Public diagnostic-log source inventory and extraction helpers for training."""

from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
import logging
import os.path
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

import fsspec
from marin.utils import fsspec_mkdirs

from marin.datakit.download.rollout_transforms import load_parquet_batched
from marin.execution.step_spec import StepSpec

logger = logging.getLogger(__name__)

GHALOGS_RECORD_URL = "https://zenodo.org/records/14796970"
LOGCHUNKS_RECORD_URL = "https://zenodo.org/records/3632351"
LOGHUB_REPO_URL = "https://github.com/logpai/loghub"
STARCODERDATA_URL = "https://huggingface.co/datasets/bigcode/starcoderdata"

GHALOGS_TOTAL_BYTES = 143_425_404_506
LOGCHUNKS_TOTAL_BYTES = 24_108_826
LOGHUB_REPO_SIZE_BYTES = 7_513_088
STARCODERDATA_TOTAL_BYTES = 310_802_033_041

STARCODERDATA_REVISION = "9fc30b5"
DEFAULT_SAMPLE_MAX_PARQUET_FILES = 8
DEFAULT_SAMPLE_MAX_ROWS = 200_000

_PARTITION_BUCKETS = 10_000
_ISSUE_5093_HOLDOUT_BUCKETS = 100
_DEV_BUCKETS = 100
_TEST_BUCKETS = 100
_PARTITION_HASH_PERSON = b"diag-log-v1"

_PATH_SIGNAL_PATTERNS = (
    "/log/",
    "/logs/",
    "stacktrace",
    "stack-trace",
    "traceback",
    "stderr",
    "stdout",
    "golden",
    "snapshot",
    "fixture",
    "failure",
    "error",
)
_PATH_SIGNAL_RE = re.compile("|".join(re.escape(signal) for signal in _PATH_SIGNAL_PATTERNS))

_CONTENT_SIGNAL_RE = re.compile(
    r"(?im)"
    r"(traceback \(most recent call last\)|"
    r"\bexception\b|"
    r"\berror\b|"
    r"\bfailed\b|"
    r"\bpanic:|"
    r"\bstack trace\b|"
    r"\bassertionerror\b|"
    r"\bsegmentation fault\b)"
)

_REDACTION_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"), "<REDACTED_GITHUB_TOKEN>"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"), "<REDACTED_GITHUB_TOKEN>"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "<REDACTED_AWS_ACCESS_KEY>"),
    (
        re.compile(
            r"(?im)(?P<key>\b(?:api[_-]?key|token|secret|password|passwd)\b)\s*[:=]\s*['\"]?[A-Za-z0-9_\-./+=]{8,}"
        ),
        r"\g<key>=<REDACTED_SECRET>",
    ),
    (re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}\b"), "<REDACTED_EMAIL>"),
    (re.compile(r"(?:(?:/Users|/home)/[^/\s]+)"), "<REDACTED_USER_HOME>"),
    (re.compile(r"\b[A-Za-z]:\\Users\\[^\\\s]+"), "<REDACTED_WINDOWS_USER_HOME>"),
    (re.compile(r"gs://marin-[^)\s]+"), "gs://<REDACTED_INTERNAL_BUCKET>"),
)


class DiagnosticSourceStatus(StrEnum):
    """Training eligibility for a candidate source."""

    TRAINING_READY = "training_ready"
    BLOCKED_LICENSE = "blocked_license"
    EVAL_ONLY = "eval_only"


class DiagnosticPartition(StrEnum):
    """Stable split assignment for diagnostic logs."""

    TRAIN = "train"
    DEV = "dev"
    TEST = "test"
    ISSUE_5093_HOLDOUT = "issue_5093_holdout"


@dataclass(frozen=True)
class DiagnosticLogSource:
    """Metadata and policy for one diagnostic-log candidate source."""

    name: str
    source_url: str
    source_license: str
    source_format: str
    compressed_size_bytes: int | None
    contamination_risk: str
    status: DiagnosticSourceStatus
    provenance_notes: str
    rough_tokens_b: float | None


SOURCE_INVENTORY: tuple[DiagnosticLogSource, ...] = (
    DiagnosticLogSource(
        name="ghalogs",
        source_url=GHALOGS_RECORD_URL,
        source_license="unspecified (Zenodo access_right=open; no explicit rights metadata)",
        source_format="runs.json.gz, repositories.json.gz, github_run_logs.zip",
        compressed_size_bytes=GHALOGS_TOTAL_BYTES,
        contamination_risk="high: public CI logs can contain secrets and internal paths",
        status=DiagnosticSourceStatus.BLOCKED_LICENSE,
        provenance_notes="DOI 10.5281/zenodo.14796970, published 2025-02-03.",
        rough_tokens_b=None,
    ),
    DiagnosticLogSource(
        name="logchunks",
        source_url=LOGCHUNKS_RECORD_URL,
        source_license="unspecified (Zenodo access_right=open; no explicit rights metadata)",
        source_format="LogChunks.zip (XML chunk annotations)",
        compressed_size_bytes=LOGCHUNKS_TOTAL_BYTES,
        contamination_risk="medium: labeled failure snippets may include local paths and user names",
        status=DiagnosticSourceStatus.BLOCKED_LICENSE,
        provenance_notes="DOI 10.5281/zenodo.3632351, published 2020-01-31.",
        rough_tokens_b=None,
    ),
    DiagnosticLogSource(
        name="loghub",
        source_url=LOGHUB_REPO_URL,
        source_license="custom research/academic-only license",
        source_format="mixed plain-text log files grouped by dataset",
        compressed_size_bytes=LOGHUB_REPO_SIZE_BYTES,
        contamination_risk="medium: includes system identifiers and infrastructure paths",
        status=DiagnosticSourceStatus.BLOCKED_LICENSE,
        provenance_notes="LICENSE file restricts usage to research/academic work.",
        rough_tokens_b=None,
    ),
    DiagnosticLogSource(
        name="github_fixture_logs_from_source_corpora",
        source_url=STARCODERDATA_URL,
        source_license="inherits accepted source-corpus licensing and per-repo provenance",
        source_format="parquet rows with repo path + content",
        compressed_size_bytes=STARCODERDATA_TOTAL_BYTES,
        contamination_risk="medium: fixtures often include tokens, email addresses, and host paths",
        status=DiagnosticSourceStatus.TRAINING_READY,
        provenance_notes=(
            "Extract only files with diagnostic path/content signals from existing source corpora; "
            "apply sanitization before any split is materialized."
        ),
        rough_tokens_b=0.15,
    ),
    DiagnosticLogSource(
        name="marin_owned_ci_iris_zephyr_logs",
        source_url="internal",
        source_license="not public",
        source_format="internal run logs",
        compressed_size_bytes=None,
        contamination_risk="high: internal infra identifiers and sensitive traces",
        status=DiagnosticSourceStatus.EVAL_ONLY,
        provenance_notes="Eval-only until governance and sanitization policy is explicitly approved.",
        rough_tokens_b=None,
    ),
    DiagnosticLogSource(
        name="issue_5093_eval_slices",
        source_url="https://github.com/marin-community/marin/issues/5093",
        source_license="eval holdout policy",
        source_format="held-out eval slices",
        compressed_size_bytes=None,
        contamination_risk="high: direct eval contamination",
        status=DiagnosticSourceStatus.EVAL_ONLY,
        provenance_notes="Never include in training.",
        rough_tokens_b=None,
    ),
)


def source_inventory() -> tuple[DiagnosticLogSource, ...]:
    """Return the immutable source inventory for public diagnostic logs."""
    return SOURCE_INVENTORY


def training_ready_sources() -> tuple[DiagnosticLogSource, ...]:
    """Return only source entries that are approved for training ingestion."""
    return tuple(source for source in SOURCE_INVENTORY if source.status == DiagnosticSourceStatus.TRAINING_READY)


def sanitize_diagnostic_log_text(text: str) -> str:
    """Redact sensitive tokens, identities, and internal paths from log text."""
    sanitized = text
    for pattern, replacement in _REDACTION_RULES:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized


def looks_like_diagnostic_log_row(path: str, content: str) -> bool:
    """Return True if path/content indicate a diagnostic log fixture or stack trace."""
    path_value = path.lower()
    if not _PATH_SIGNAL_RE.search(path_value):
        return False
    return _CONTENT_SIGNAL_RE.search(content) is not None


def assign_partition(split_key: str) -> DiagnosticPartition:
    """Assign a stable partition with a dedicated #5093 holdout slice."""
    digest = hashlib.blake2b(split_key.encode("utf-8"), digest_size=8, person=_PARTITION_HASH_PERSON).digest()
    bucket = int.from_bytes(digest, byteorder="big") % _PARTITION_BUCKETS

    if bucket < _ISSUE_5093_HOLDOUT_BUCKETS:
        return DiagnosticPartition.ISSUE_5093_HOLDOUT
    if bucket < _ISSUE_5093_HOLDOUT_BUCKETS + _DEV_BUCKETS:
        return DiagnosticPartition.DEV
    if bucket < _ISSUE_5093_HOLDOUT_BUCKETS + _DEV_BUCKETS + _TEST_BUCKETS:
        return DiagnosticPartition.TEST
    return DiagnosticPartition.TRAIN


def starcoder_fixture_row_to_record(row: Mapping[str, object]) -> dict[str, str] | None:
    """Convert one StarCoder row into a sanitized diagnostic-log record."""
    content = row.get("content")
    path = row.get("max_stars_repo_path")
    repo = row.get("max_stars_repo_name")

    if not isinstance(content, str) or not content.strip():
        return None
    if not isinstance(path, str) or not path:
        return None
    if not isinstance(repo, str) or not repo:
        return None
    if not looks_like_diagnostic_log_row(path, content):
        return None

    split_key = f"{repo}:{path}"
    partition = assign_partition(split_key)
    sanitized = sanitize_diagnostic_log_text(content)
    row_id = hashlib.sha256(split_key.encode("utf-8")).hexdigest()

    return {
        "id": row_id,
        "text": sanitized,
        "source": "github_fixture_logs",
        "repo_name": repo,
        "repo_path": path,
        "partition": partition.value,
    }


def _source_path(fs: fsspec.AbstractFileSystem, relative_path: str) -> str:
    protocol = fs.protocol
    if isinstance(protocol, tuple):
        protocol = protocol[0]
    if protocol in (None, "", "file"):
        return relative_path
    return f"{protocol}://{relative_path}"


def _list_parquet_files(input_path: str) -> list[str]:
    fs, relative_root = fsspec.core.url_to_fs(input_path)
    pattern = os.path.join(relative_root.rstrip("/"), "**", "*.parquet")
    paths = sorted(fs.glob(pattern, recursive=True))
    return [_source_path(fs, path) for path in paths]


def extract_starcoder_fixture_logs(
    input_path: str,
    output_path: str,
    *,
    max_parquet_files: int = DEFAULT_SAMPLE_MAX_PARQUET_FILES,
    max_rows: int = DEFAULT_SAMPLE_MAX_ROWS,
) -> None:
    """Extract a capped sample of partitioned diagnostic fixture logs from StarCoderData parquet shards."""
    if max_parquet_files <= 0:
        raise ValueError(f"max_parquet_files must be positive, got {max_parquet_files}")
    if max_rows <= 0:
        raise ValueError(f"max_rows must be positive, got {max_rows}")

    counters = {"seen_rows": 0, "kept_rows": 0}
    partition_counts = {partition.value: 0 for partition in DiagnosticPartition}
    parquet_files = _list_parquet_files(input_path)

    if not parquet_files:
        raise ValueError(f"No parquet files found at {input_path}")

    sampled_files = parquet_files[:max_parquet_files]
    logger.info(
        "Sampling %d/%d parquet shards from %s with row cap=%d",
        len(sampled_files),
        len(parquet_files),
        input_path,
        max_rows,
    )

    output_file_paths: dict[str, str] = {}
    for partition in DiagnosticPartition:
        partition_dir = os.path.join(output_path, partition.value)
        fsspec_mkdirs(partition_dir, exist_ok=True)
        output_file_paths[partition.value] = os.path.join(partition_dir, "data-00000-of-00001.jsonl")

    with ExitStack() as stack:
        writers = {
            partition.value: stack.enter_context(fsspec.open(path, "wt", encoding="utf-8"))
            for partition, path in ((partition, output_file_paths[partition.value]) for partition in DiagnosticPartition)
        }

        row_budget_exhausted = False
        for parquet_file in sampled_files:
            for row in load_parquet_batched(parquet_file):
                if counters["seen_rows"] >= max_rows:
                    row_budget_exhausted = True
                    break
                counters["seen_rows"] += 1

                record = starcoder_fixture_row_to_record(row)
                if record is None:
                    continue

                counters["kept_rows"] += 1
                partition = record["partition"]
                partition_counts[partition] += 1
                writers[partition].write(json.dumps(record, ensure_ascii=False))
                writers[partition].write("\n")
            if row_budget_exhausted:
                break

    metadata = {
        "source": "bigcode/starcoderdata",
        "revision": STARCODERDATA_REVISION,
        "sample_limits": {"max_parquet_files": max_parquet_files, "max_rows": max_rows},
        "sampling": {"available_parquet_files": len(parquet_files), "sampled_parquet_files": len(sampled_files)},
        "counters": counters,
        "partition_counts": partition_counts,
        "training_ready_sources": [source.name for source in training_ready_sources()],
    }
    with fsspec.open(os.path.join(output_path, "metadata.json"), "wt", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def extract_starcoder_fixture_logs_step(
    *,
    source_path: str,
    max_parquet_files: int = DEFAULT_SAMPLE_MAX_PARQUET_FILES,
    max_rows: int = DEFAULT_SAMPLE_MAX_ROWS,
) -> StepSpec:
    """Return a StepSpec that materializes a capped sample of partitioned diagnostic logs."""
    return StepSpec(
        name="processed/diagnostic_logs/github_fixtures_sample",
        fn=lambda output_path: extract_starcoder_fixture_logs(
            source_path,
            output_path,
            max_parquet_files=max_parquet_files,
            max_rows=max_rows,
        ),
        hash_attrs={
            "version": "v1",
            "sample_only": True,
            "source_path": source_path,
            "max_parquet_files": max_parquet_files,
            "max_rows": max_rows,
            "split_policy": "97% train / 1% dev / 1% test / 1% issue_5093_holdout",
            "sanitization_rules": "gh token/aws key/secret kv/email/user path/internal gs path",
        },
    )
