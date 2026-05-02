# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Public diagnostic-log source inventory and GHALogs extraction helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import os.path
import re
import shutil
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterable
from contextlib import ExitStack
from dataclasses import dataclass
from enum import StrEnum
from io import BytesIO

import fsspec
import requests
from fray import ResourceConfig
from pydantic import BaseModel, ConfigDict
from rigging.filesystem import marin_prefix
from zephyr import Dataset, ZephyrContext, counters, load_zip_members

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
from marin.execution.step_spec import StepSpec
from marin.utils import fsspec_mkdirs

logger = logging.getLogger(__name__)

GHALOGS_RECORD_URL = "https://zenodo.org/records/14796970"
LOGCHUNKS_RECORD_URL = "https://zenodo.org/records/3632351"
LOGHUB_REPO_URL = "https://github.com/logpai/loghub"
LOGCHUNKS_DOWNLOAD_URL = "https://zenodo.org/records/3632351/files/LogChunks.zip?download=1"
LOGHUB_SNAPSHOT_URL = "https://github.com/logpai/loghub/archive/refs/heads/master.zip"
GHALOGS_ZIP_FILENAME = "github_run_logs.zip"
LOGCHUNKS_ZIP_FILENAME = "LogChunks.zip"
LOGHUB_DIRNAME = "loghub"
GHALOGS_STAGED_PREFIX = "raw/diagnostic_logs/ghalogs/zenodo-14796970"
LOGCHUNKS_STAGED_PREFIX = "raw/diagnostic_logs/logchunks/zenodo-3632351"
LOGHUB_STAGED_PREFIX = "raw/diagnostic_logs/loghub/logpai-loghub"
GHALOGS_STAGED_ARCHIVE_RELATIVE_PATH = os.path.join(
    "zenodo.org",
    "records",
    "14796970",
    "files",
    GHALOGS_ZIP_FILENAME,
)

GHALOGS_TOTAL_BYTES = 143_425_404_506
LOGCHUNKS_TOTAL_BYTES = 24_108_826
LOGHUB_REPO_SIZE_BYTES = 7_513_088
DEFAULT_GHALOGS_MAX_MEMBERS = 10_000
DEFAULT_LOGCHUNKS_MAX_EXAMPLES = 10_000
DEFAULT_LOGHUB_MAX_FILES = 100
DEFAULT_GHALOGS_MATERIALIZE_SHARDS = 128
DEFAULT_GHALOGS_PARTITION_SHARDS = 16
LONG_TAIL_PPL_EPIC_ISSUE = 5005
PUBLIC_DIAGNOSTIC_LOGS_ISSUE = 5094
_DOWNLOAD_CHUNK_BYTES = 1 << 20
_DOWNLOAD_TIMEOUT = 300

_PARTITION_BUCKETS = 10_000
_ISSUE_5093_HOLDOUT_BUCKETS = 100
_DEV_BUCKETS = 100
_TEST_BUCKETS = 100
_PARTITION_HASH_PERSON = b"diag-log-v1"

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
    (re.compile(r"gs://marin-[^)\s]+"), "gs://<REDACTED_INTERNAL_BUCKET>"),
)
_EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}\b")
_UNIX_USER_HOME_RE = re.compile(r"(?P<prefix>(?:/Users|/home)/)(?P<name>[^/\s]+)")
_WINDOWS_USER_HOME_RE = re.compile(r"(?P<prefix>\b[A-Za-z]:\\Users\\)(?P<name>[^\\\s]+)")
_USERNAME_RE_TEMPLATE = r"(?<![A-Za-z0-9_.@%+-]){username}(?![A-Za-z0-9_.@%+-])"
_MIN_USERNAME_LENGTH = 4
_USERNAME_DENYLIST = frozenset(
    {
        "admin",
        "build",
        "cache",
        "debug",
        "error",
        "false",
        "guest",
        "home",
        "local",
        "login",
        "logs",
        "none",
        "null",
        "root",
        "runner",
        "system",
        "test",
        "true",
        "user",
        "users",
    }
)


class DiagnosticPartition(StrEnum):
    """Stable split assignment for diagnostic logs."""

    TRAIN = "train"
    DEV = "dev"
    TEST = "test"
    ISSUE_5093_HOLDOUT = "issue_5093_holdout"


class DiagnosticLogsArtifact(BaseModel):
    """Strict typed artifact for a materialized diagnostic-log step."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ExtractedPartitionedDiagnosticLogs(DiagnosticLogsArtifact):
    """Materialized GHALogs sample with train/dev/test/holdout partitions."""

    source_label: str
    output_dir: str
    train_file: str
    dev_file: str
    test_file: str
    holdout_file: str
    metadata_path: str
    record_count: int
    bytes_written: int
    content_fingerprint: str


class ExtractedDiagnosticLogSlice(DiagnosticLogsArtifact):
    """Materialized single-file diagnostic-log slice."""

    source_label: str
    output_dir: str
    output_file: str
    metadata_path: str
    record_count: int
    bytes_written: int
    content_fingerprint: str


class ExtractedDiagnosticLogs(DiagnosticLogsArtifact):
    """Combined result for the public diagnostic-log extraction helpers."""

    ghalogs: ExtractedPartitionedDiagnosticLogs
    logchunks: ExtractedDiagnosticLogSlice
    loghub: ExtractedDiagnosticLogSlice


class MaterializedDiagnosticLogParquet(DiagnosticLogsArtifact):
    """Reusable parquet shards for a diagnostic-log corpus or partition."""

    source_label: str
    output_dir: str
    data_glob: str
    record_count: int
    counters: dict[str, int]
    content_fingerprint: str


def _source_policy(
    *,
    usage_policy: UsagePolicy,
    use_policy: str,
    contamination_risk: str,
    provenance_notes: str,
) -> IngestionPolicy:
    return IngestionPolicy(
        usage_policy=usage_policy,
        use_policy=use_policy,
        requires_sanitization=True,
        identity_treatment=IdentityTreatment.PSEUDONYMIZE,
        secret_redaction=SecretRedaction.REQUIRED,
        contamination_risk=contamination_risk,
        provenance_notes=provenance_notes,
    )


SOURCE_INVENTORY: tuple[IngestionSourceManifest, ...] = (
    IngestionSourceManifest(
        dataset_key="ghalogs/public",
        slice_key="diagnostic_logs/ghalogs/public_sample",
        source_label="ghalogs",
        source_urls=(GHALOGS_RECORD_URL,),
        source_license="Creative Commons Attribution Share Alike 4.0 International",
        source_format="runs.json.gz, repositories.json.gz, github_run_logs.zip",
        surface_form="sanitized_github_actions_logs",
        policy=_source_policy(
            usage_policy=UsagePolicy.TRAINING_ALLOWED,
            use_policy="Training and eval source after sanitization and sample-capped extraction.",
            contamination_risk="high: public CI logs can contain secrets and internal paths",
            provenance_notes="DOI 10.5281/zenodo.14796970, published 2025-02-03; Zenodo license id cc-by-sa-4.0.",
        ),
        staging=StagingMetadata(
            transform_name="extract_ghalogs",
            metadata={
                "output_layout": "train/dev/test/issue_5093_holdout jsonl partitions plus metadata.json",
                "provenance_fields": ["id", "archive_path", "partition"],
            },
        ),
        epic_issue=LONG_TAIL_PPL_EPIC_ISSUE,
        issue_numbers=(PUBLIC_DIAGNOSTIC_LOGS_ISSUE,),
        sample_caps=SampleCapConfig(max_members=DEFAULT_GHALOGS_MAX_MEMBERS),
        compressed_size_bytes=GHALOGS_TOTAL_BYTES,
        rough_tokens_b=None,
        source_metadata={"archive_filename": GHALOGS_ZIP_FILENAME},
    ),
    IngestionSourceManifest(
        dataset_key="logchunks/public",
        slice_key="diagnostic_logs/logchunks/eval_only",
        source_label="logchunks",
        source_urls=(LOGCHUNKS_RECORD_URL,),
        source_license="Creative Commons Attribution 4.0 International",
        source_format="LogChunks.zip (XML chunk annotations)",
        surface_form="sanitized_diagnostic_log_chunks",
        policy=_source_policy(
            usage_policy=UsagePolicy.EVAL_ONLY,
            use_policy="Eval-only diagnostic-log source. Do not mix into training.",
            contamination_risk="medium: labeled failure snippets may include local paths and user names",
            provenance_notes="DOI 10.5281/zenodo.3632351, published 2020-01-31; eval-only despite acceptable license.",
        ),
        staging=StagingMetadata(
            transform_name="extract_logchunks",
            metadata={"provenance_fields": ["id", "source_path", "category", "log_path"]},
        ),
        epic_issue=LONG_TAIL_PPL_EPIC_ISSUE,
        issue_numbers=(PUBLIC_DIAGNOSTIC_LOGS_ISSUE,),
        sample_caps=SampleCapConfig(max_examples=DEFAULT_LOGCHUNKS_MAX_EXAMPLES),
        compressed_size_bytes=LOGCHUNKS_TOTAL_BYTES,
        rough_tokens_b=None,
        source_metadata={"archive_filename": LOGCHUNKS_ZIP_FILENAME},
    ),
    IngestionSourceManifest(
        dataset_key="loghub/public",
        slice_key="diagnostic_logs/loghub/eval_only",
        source_label="loghub",
        source_urls=(LOGHUB_REPO_URL,),
        source_license="custom research/academic-only license",
        source_format="mixed plain-text log files grouped by dataset",
        surface_form="sanitized_raw_system_logs",
        policy=_source_policy(
            usage_policy=UsagePolicy.EVAL_ONLY,
            use_policy="Eval-only diagnostic-log source. Do not mix into training.",
            contamination_risk="medium: includes system identifiers and infrastructure paths",
            provenance_notes="LICENSE file restricts usage to research/academic work; acceptable only for eval use.",
        ),
        staging=StagingMetadata(
            transform_name="extract_loghub",
            metadata={"provenance_fields": ["id", "source_path"]},
        ),
        epic_issue=LONG_TAIL_PPL_EPIC_ISSUE,
        issue_numbers=(PUBLIC_DIAGNOSTIC_LOGS_ISSUE,),
        sample_caps=SampleCapConfig(max_files=DEFAULT_LOGHUB_MAX_FILES),
        compressed_size_bytes=LOGHUB_REPO_SIZE_BYTES,
        rough_tokens_b=None,
        source_metadata={"source_dirname": LOGHUB_DIRNAME},
    ),
    IngestionSourceManifest(
        dataset_key="marin_internal/ci_logs",
        slice_key="diagnostic_logs/marin_internal/eval_only",
        source_label="marin_owned_ci_iris_zephyr_logs",
        source_urls=("internal",),
        source_license="not public",
        source_format="internal run logs",
        surface_form="sanitized_internal_ci_logs",
        policy=_source_policy(
            usage_policy=UsagePolicy.EVAL_ONLY,
            use_policy="Eval-only until governance and sanitization policy is explicitly approved.",
            contamination_risk="high: internal infra identifiers and sensitive traces",
            provenance_notes="Eval-only until governance and sanitization policy is explicitly approved.",
        ),
        staging=StagingMetadata(transform_name="internal_only"),
        epic_issue=LONG_TAIL_PPL_EPIC_ISSUE,
        issue_numbers=(PUBLIC_DIAGNOSTIC_LOGS_ISSUE,),
    ),
    IngestionSourceManifest(
        dataset_key="marin_eval/diagnostic_logs_holdout",
        slice_key="diagnostic_logs/issue_5093_holdout/eval_only",
        source_label="issue_5093_eval_slices",
        source_urls=("https://github.com/marin-community/marin/issues/5093",),
        source_license="eval holdout policy",
        source_format="held-out eval slices",
        surface_form="held_out_diagnostic_log_slices",
        policy=_source_policy(
            usage_policy=UsagePolicy.EVAL_ONLY,
            use_policy="Eval-only holdout. Never include in training.",
            contamination_risk="high: direct eval contamination",
            provenance_notes="Never include in training.",
        ),
        staging=StagingMetadata(transform_name="holdout_only"),
        epic_issue=LONG_TAIL_PPL_EPIC_ISSUE,
        issue_numbers=(PUBLIC_DIAGNOSTIC_LOGS_ISSUE, 5093),
    ),
)
SOURCE_MANIFESTS = {source.source_label: source for source in SOURCE_INVENTORY}


def source_manifest(source_label: str) -> IngestionSourceManifest:
    """Return the manifest for one named diagnostic-log source."""

    return SOURCE_MANIFESTS[source_label]


def source_inventory() -> tuple[IngestionSourceManifest, ...]:
    """Return the immutable source inventory for public diagnostic logs."""

    return SOURCE_INVENTORY


def training_ready_sources() -> tuple[IngestionSourceManifest, ...]:
    """Return only source entries that are approved for training ingestion."""

    return tuple(source for source in SOURCE_INVENTORY if source.policy.training_allowed)


def blocked_sources() -> tuple[IngestionSourceManifest, ...]:
    """Return source entries blocked from both training and eval use."""

    return tuple(source for source in SOURCE_INVENTORY if source.policy.usage_policy == UsagePolicy.BLOCKED)


@dataclass
class _DocumentIdentityPseudonymizer:
    identity_ids: dict[str, str]
    username_ids: dict[str, str]

    @classmethod
    def from_text(cls, text: str) -> _DocumentIdentityPseudonymizer:
        pseudonymizer = cls(identity_ids={}, username_ids={})
        for match in _EMAIL_RE.finditer(text):
            pseudonymizer._register_email(match.group(0))
        for pattern in (_UNIX_USER_HOME_RE, _WINDOWS_USER_HOME_RE):
            for match in pattern.finditer(text):
                pseudonymizer._register_username(match.group("name"))
        return pseudonymizer

    def pseudonymize(self, text: str) -> str:
        pseudonymized = _EMAIL_RE.sub(self._replace_email, text)
        pseudonymized = _UNIX_USER_HOME_RE.sub(self._replace_home_path, pseudonymized)
        pseudonymized = _WINDOWS_USER_HOME_RE.sub(self._replace_home_path, pseudonymized)
        for username in sorted(self.username_ids, key=len, reverse=True):
            pattern = re.compile(_USERNAME_RE_TEMPLATE.format(username=re.escape(username)), re.IGNORECASE)
            pseudonymized = pattern.sub(self.username_ids[username], pseudonymized)
        return pseudonymized

    def _register_email(self, email: str) -> str:
        local_part = email.split("@", maxsplit=1)[0].split("+", maxsplit=1)[0]
        return self._register_username(local_part)

    def _register_username(self, username: str) -> str:
        canonical = username.casefold()
        if canonical not in self.identity_ids:
            self.identity_ids[canonical] = f"<USER_{len(self.identity_ids)}>"
        user_id = self.identity_ids[canonical]

        for candidate in _username_candidates(username):
            existing = self.username_ids.get(candidate)
            if existing is None:
                self.username_ids[candidate] = user_id
        return user_id

    def _replace_email(self, match: re.Match[str]) -> str:
        return self._register_email(match.group(0)).replace(">", "_EMAIL>")

    def _replace_home_path(self, match: re.Match[str]) -> str:
        return f"{match.group('prefix')}{self._register_username(match.group('name'))}"


def _username_candidates(username: str) -> tuple[str, ...]:
    candidates = {username}
    candidates.update(part for part in re.split(r"[._-]+", username) if part)
    return tuple(candidate for candidate in candidates if _is_safe_username_candidate(candidate))


def _is_safe_username_candidate(candidate: str) -> bool:
    normalized = candidate.casefold()
    return (
        len(candidate) >= _MIN_USERNAME_LENGTH
        and any(char.isalpha() for char in candidate)
        and normalized not in _USERNAME_DENYLIST
    )


def sanitize_diagnostic_log_text(text: str) -> str:
    """Redact secrets and per-document pseudonymize user identities in log text."""
    sanitized = text
    for pattern, replacement in _REDACTION_RULES:
        sanitized = pattern.sub(replacement, sanitized)
    return _DocumentIdentityPseudonymizer.from_text(sanitized).pseudonymize(sanitized)


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


def ghalogs_member_to_record(member_path: str, content: bytes) -> dict[str, str] | None:
    """Convert one GHALogs zip member into a sanitized diagnostic-log record."""
    text = content.decode("utf-8", errors="replace").strip()
    if not text:
        return None

    split_key = f"ghalogs:{member_path}"
    partition = assign_partition(split_key)
    row_id = hashlib.sha256(split_key.encode("utf-8")).hexdigest()

    return {
        "id": row_id,
        "text": sanitize_diagnostic_log_text(text),
        "source": "ghalogs",
        "archive_path": member_path,
        "partition": partition.value,
    }


def logchunks_example_to_record(source_path: str, example_index: int, example: ET.Element) -> dict[str, str] | None:
    """Convert one LogChunks XML example into a sanitized eval-only record."""
    chunk = example.findtext("Chunk")
    if chunk is None or not chunk.strip():
        return None

    log_path = example.findtext("Log") or ""
    keywords = example.findtext("Keywords") or ""
    category = example.findtext("Category") or ""
    split_key = f"logchunks:{source_path}:{example_index}"
    row_id = hashlib.sha256(split_key.encode("utf-8")).hexdigest()

    return {
        "id": row_id,
        "text": sanitize_diagnostic_log_text(chunk.strip()),
        "source": "logchunks",
        "source_path": source_path,
        "log_path": log_path,
        "keywords": keywords,
        "category": category,
    }


def loghub_file_to_record(source_path: str, content: bytes) -> dict[str, str] | None:
    """Convert one LogHub raw log file into a sanitized eval-only record."""
    text = content.decode("utf-8", errors="replace").strip()
    if not text:
        return None

    split_key = f"loghub:{source_path}"
    row_id = hashlib.sha256(split_key.encode("utf-8")).hexdigest()

    return {
        "id": row_id,
        "text": sanitize_diagnostic_log_text(text),
        "source": "loghub",
        "source_path": source_path,
    }


def _write_jsonl_records(output_file: str, records: Iterable[dict[str, str]]) -> tuple[int, int]:
    kept_records = 0
    bytes_written = 0
    output_dir = os.path.dirname(output_file)
    fsspec_mkdirs(output_dir, exist_ok=True)
    with fsspec.open(output_file, "wt", encoding="utf-8") as writer:
        for record in records:
            kept_records += 1
            payload = json.dumps(record, ensure_ascii=False)
            bytes_written += len(payload.encode("utf-8")) + 1
            writer.write(payload)
            writer.write("\n")
    return kept_records, bytes_written


def _write_source_metadata(
    *,
    manifest: IngestionSourceManifest,
    input_path: str,
    output_path: str,
    output_file: str,
    record_count: int,
    bytes_written: int,
    metadata: dict[str, JsonValue],
) -> str:
    return write_ingestion_metadata_json(
        manifest=manifest,
        materialized_output=MaterializedOutputMetadata(
            input_path=input_path,
            output_path=output_path,
            output_file=output_file,
            record_count=record_count,
            bytes_written=bytes_written,
            metadata=metadata,
        ),
    )


def _normalize_input_path(path: str) -> str:
    if path.startswith("/") or "://" in path:
        return path
    return os.path.join(marin_prefix(), path)


def _path_exists(path: str) -> bool:
    fs, relative_path = fsspec.core.url_to_fs(path)
    return fs.exists(relative_path)


def _download_to_path(url: str, destination_path: str) -> None:
    fsspec_mkdirs(os.path.dirname(destination_path), exist_ok=True)
    logger.info("Downloading %s to %s", url, destination_path)
    with requests.get(url, stream=True, timeout=_DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        with fsspec.open(destination_path, "wb") as writer:
            for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_BYTES):
                if chunk:
                    writer.write(chunk)


def _stage_logchunks_if_missing(destination_dir: str) -> str:
    archive_path = os.path.join(destination_dir, LOGCHUNKS_ZIP_FILENAME)
    if not _path_exists(archive_path):
        _download_to_path(LOGCHUNKS_DOWNLOAD_URL, archive_path)
    return destination_dir


def _stage_loghub_if_missing(destination_dir: str) -> str:
    loghub_root = os.path.join(destination_dir, LOGHUB_DIRNAME)
    if _list_loghub_files(loghub_root, 1):
        return destination_dir

    logger.info("Downloading %s to %s", LOGHUB_SNAPSHOT_URL, loghub_root)
    response = requests.get(LOGHUB_SNAPSHOT_URL, timeout=_DOWNLOAD_TIMEOUT)
    response.raise_for_status()
    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            _, _, relative_path = member.filename.partition("/")
            if not relative_path:
                continue
            destination_path = os.path.join(loghub_root, relative_path)
            fsspec_mkdirs(os.path.dirname(destination_path), exist_ok=True)
            with archive.open(member, "r") as reader, fsspec.open(destination_path, "wb") as writer:
                shutil.copyfileobj(reader, writer)
    return destination_dir


def _resolve_ghalogs_archive_path(input_path: str) -> tuple[str, str]:
    normalized_input_path = _normalize_input_path(input_path)
    archive_path = os.path.join(normalized_input_path, GHALOGS_STAGED_ARCHIVE_RELATIVE_PATH)
    if not _path_exists(archive_path):
        raise FileNotFoundError(
            "Missing staged GHALogs archive. "
            f"Expected {archive_path} "
            f"(relative path {GHALOGS_STAGED_ARCHIVE_RELATIVE_PATH} under {normalized_input_path})."
        )
    return normalized_input_path, archive_path


def _resolve_logchunks_input_path(input_path: str | None) -> str:
    normalized_input_path = _normalize_input_path(input_path or LOGCHUNKS_STAGED_PREFIX)
    archive_path = os.path.join(normalized_input_path, LOGCHUNKS_ZIP_FILENAME)
    if not _path_exists(archive_path):
        normalized_input_path = _stage_logchunks_if_missing(normalized_input_path)
    return normalized_input_path


def _resolve_loghub_input_path(input_path: str | None) -> str:
    normalized_input_path = _normalize_input_path(input_path or LOGHUB_STAGED_PREFIX)
    loghub_root = os.path.join(normalized_input_path, LOGHUB_DIRNAME)
    if not _list_loghub_files(loghub_root, 1):
        normalized_input_path = _stage_loghub_if_missing(normalized_input_path)
    return normalized_input_path


def _ghalogs_zip_member_to_records(member: dict[str, object]) -> list[dict[str, str]]:
    filename = member["filename"]
    content = member["content"]
    assert isinstance(filename, str), f"Expected zip member filename to be str, got {type(filename)}"
    assert isinstance(content, bytes), f"Expected zip member content to be bytes, got {type(content)}"

    record = ghalogs_member_to_record(filename, content)
    if record is None:
        counters.increment("ghalogs_materialize/dropped_empty")
        return []

    counters.increment("ghalogs_materialize/kept")
    counters.increment(f"ghalogs_materialize/partition_{record['partition']}")
    return [record]


def materialize_ghalogs_to_parquet(
    input_path: str,
    output_path: str,
    *,
    max_members: int | None = None,
    num_shards: int = DEFAULT_GHALOGS_MATERIALIZE_SHARDS,
    worker_resources: ResourceConfig | None = None,
    max_workers: int | None = None,
) -> MaterializedDiagnosticLogParquet:
    """Materialize sanitized GHALogs records into reusable parquet shards."""
    if max_members is not None and max_members <= 0:
        raise ValueError(f"max_members must be positive when set, got {max_members}")
    if num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {num_shards}")

    _, archive_path = _resolve_ghalogs_archive_path(input_path)
    pipeline = Dataset.from_list([archive_path]).flat_map(load_zip_members).flat_map(_ghalogs_zip_member_to_records)
    if max_members is not None:
        pipeline = pipeline.take_per_shard(max_members)

    pipeline = pipeline.reshard(num_shards).write_parquet(
        f"{output_path}/data-{{shard:05d}}-of-{{total:05d}}.parquet",
        skip_existing=True,
    )

    resources = worker_resources or ResourceConfig(cpu=1, ram="32g", disk="20g")
    ctx_kwargs: dict[str, object] = {"name": "materialize-ghalogs", "resources": resources}
    if max_workers is not None:
        ctx_kwargs["max_workers"] = max_workers
    outcome = ZephyrContext(**ctx_kwargs).execute(pipeline)
    counters_dict = dict(outcome.counters)
    manifest = source_manifest("ghalogs")
    return MaterializedDiagnosticLogParquet(
        source_label=manifest.source_label,
        output_dir=output_path,
        data_glob=f"{output_path}/*.parquet",
        record_count=counters_dict.get("zephyr/records_out", 0),
        counters=counters_dict,
        content_fingerprint=manifest.fingerprint(),
    )


def _count_partition_record(record: dict[str, str], partition: DiagnosticPartition) -> dict[str, str]:
    counters.increment(f"ghalogs_partition/{partition.value}_kept")
    return record


def materialize_ghalogs_partition_to_parquet(
    input_path: str,
    output_path: str,
    *,
    partition: DiagnosticPartition,
    num_shards: int = DEFAULT_GHALOGS_PARTITION_SHARDS,
    worker_resources: ResourceConfig | None = None,
    max_workers: int | None = None,
) -> MaterializedDiagnosticLogParquet:
    """Filter materialized GHALogs parquet shards down to one partition."""
    if num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {num_shards}")

    pipeline = (
        Dataset.from_files(f"{input_path}/*.parquet")
        .load_parquet()
        .filter(lambda record: record.get("partition") == partition.value)
        .map(lambda record: _count_partition_record(record, partition))
        .reshard(num_shards)
        .write_parquet(f"{output_path}/data-{{shard:05d}}-of-{{total:05d}}.parquet", skip_existing=True)
    )

    resources = worker_resources or ResourceConfig(cpu=1, ram="8g", disk="10g")
    ctx_kwargs: dict[str, object] = {"name": f"ghalogs-partition-{partition.value}", "resources": resources}
    if max_workers is not None:
        ctx_kwargs["max_workers"] = max_workers
    outcome = ZephyrContext(**ctx_kwargs).execute(pipeline)
    counters_dict = dict(outcome.counters)
    manifest = source_manifest("ghalogs")
    return MaterializedDiagnosticLogParquet(
        source_label=manifest.source_label,
        output_dir=output_path,
        data_glob=f"{output_path}/*.parquet",
        record_count=counters_dict.get("zephyr/records_out", 0),
        counters=counters_dict,
        content_fingerprint=manifest.fingerprint(),
    )


def extract_ghalogs(
    input_path: str,
    output_path: str,
    *,
    max_members: int = DEFAULT_GHALOGS_MAX_MEMBERS,
) -> ExtractedPartitionedDiagnosticLogs:
    """Extract a capped sample of partitioned, sanitized records from a staged GHALogs archive."""
    if max_members <= 0:
        raise ValueError(f"max_members must be positive, got {max_members}")

    input_path, archive_path = _resolve_ghalogs_archive_path(input_path)
    counters = {"seen_members": 0, "kept_records": 0}
    partition_counts = {partition.value: 0 for partition in DiagnosticPartition}
    total_bytes_written = 0

    output_file_paths: dict[str, str] = {}
    for partition in DiagnosticPartition:
        partition_dir = os.path.join(output_path, partition.value)
        fsspec_mkdirs(partition_dir, exist_ok=True)
        output_file_paths[partition.value] = os.path.join(partition_dir, "data-00000-of-00001.jsonl")

    logger.info("Extracting at most %d members from %s", max_members, archive_path)
    with fsspec.open(archive_path, "rb") as archive_handle, zipfile.ZipFile(archive_handle) as archive:
        with ExitStack() as stack:
            writers = {
                partition.value: stack.enter_context(fsspec.open(path, "wt", encoding="utf-8"))
                for partition, path in (
                    (partition, output_file_paths[partition.value]) for partition in DiagnosticPartition
                )
            }

            for member in archive.infolist():
                if counters["seen_members"] >= max_members:
                    break
                if member.is_dir():
                    continue

                counters["seen_members"] += 1
                with archive.open(member, "r") as member_handle:
                    record = ghalogs_member_to_record(member.filename, member_handle.read())

                if record is None:
                    continue

                counters["kept_records"] += 1
                partition = record["partition"]
                partition_counts[partition] += 1
                payload = json.dumps(record, ensure_ascii=False)
                total_bytes_written += len(payload.encode("utf-8")) + 1
                writers[partition].write(payload)
                writers[partition].write("\n")

    manifest = source_manifest("ghalogs")
    metadata_path = _write_source_metadata(
        manifest=manifest,
        input_path=input_path,
        output_path=output_path,
        output_file=output_file_paths[DiagnosticPartition.TRAIN.value],
        record_count=counters["kept_records"],
        bytes_written=total_bytes_written,
        metadata={
            "source_archive": archive_path,
            "sample_limits": {"max_members": max_members},
            "counters": counters,
            "partition_counts": partition_counts,
            "partition_files": output_file_paths,
            "training_ready_sources": [source.source_label for source in training_ready_sources()],
        },
    )
    return ExtractedPartitionedDiagnosticLogs(
        source_label=manifest.source_label,
        output_dir=output_path,
        train_file=output_file_paths[DiagnosticPartition.TRAIN.value],
        dev_file=output_file_paths[DiagnosticPartition.DEV.value],
        test_file=output_file_paths[DiagnosticPartition.TEST.value],
        holdout_file=output_file_paths[DiagnosticPartition.ISSUE_5093_HOLDOUT.value],
        metadata_path=metadata_path,
        record_count=counters["kept_records"],
        bytes_written=total_bytes_written,
        content_fingerprint=manifest.fingerprint(),
    )


def _iter_logchunks_records(archive_path: str, max_examples: int) -> Iterable[dict[str, str]]:
    seen_examples = 0
    with fsspec.open(archive_path, "rb") as archive_handle, zipfile.ZipFile(archive_handle) as archive:
        for member in archive.infolist():
            if seen_examples >= max_examples:
                break
            if member.is_dir() or not member.filename.endswith(".xml") or member.filename.startswith("__MACOSX/"):
                continue

            with archive.open(member, "r") as member_handle:
                root = ET.parse(member_handle).getroot()

            for example in root.findall("Example"):
                if seen_examples >= max_examples:
                    break
                record = logchunks_example_to_record(member.filename, seen_examples, example)
                seen_examples += 1
                if record is not None:
                    yield record


def extract_logchunks(
    input_path: str | None,
    output_path: str,
    *,
    max_examples: int = DEFAULT_LOGCHUNKS_MAX_EXAMPLES,
) -> ExtractedDiagnosticLogSlice:
    """Extract a capped sample of sanitized eval-only records from staged LogChunks."""
    if max_examples <= 0:
        raise ValueError(f"max_examples must be positive, got {max_examples}")

    input_path = _resolve_logchunks_input_path(input_path)
    archive_path = os.path.join(input_path, LOGCHUNKS_ZIP_FILENAME)
    output_file = os.path.join(output_path, "eval_only", "logchunks", "data-00000-of-00001.jsonl")
    kept_records, bytes_written = _write_jsonl_records(output_file, _iter_logchunks_records(archive_path, max_examples))
    manifest = source_manifest("logchunks")
    slice_output_dir = os.path.join(output_path, "eval_only", "logchunks")
    metadata_path = _write_source_metadata(
        manifest=manifest,
        input_path=input_path,
        output_path=slice_output_dir,
        output_file=output_file,
        record_count=kept_records,
        bytes_written=bytes_written,
        metadata={
            "source_archive": archive_path,
            "sample_limits": {"max_examples": max_examples},
        },
    )
    return ExtractedDiagnosticLogSlice(
        source_label=manifest.source_label,
        output_dir=slice_output_dir,
        output_file=output_file,
        metadata_path=metadata_path,
        record_count=kept_records,
        bytes_written=bytes_written,
        content_fingerprint=manifest.fingerprint(),
    )


def _source_path(fs: fsspec.AbstractFileSystem, relative_path: str) -> str:
    protocol = fs.protocol
    if isinstance(protocol, tuple):
        protocol = protocol[0]
    if protocol in (None, "", "file"):
        return relative_path
    return f"{protocol}://{relative_path}"


def _list_loghub_files(input_path: str, max_files: int) -> list[str]:
    fs, relative_root = fsspec.core.url_to_fs(input_path)
    pattern = os.path.join(relative_root.rstrip("/"), "**", "*_2k.log")
    paths = sorted(fs.glob(pattern, recursive=True))
    return [_source_path(fs, path) for path in paths[:max_files]]


def _iter_loghub_records(input_path: str, max_files: int) -> Iterable[dict[str, str]]:
    for log_path in _list_loghub_files(input_path, max_files):
        with fsspec.open(log_path, "rb") as handle:
            record = loghub_file_to_record(log_path, handle.read())
        if record is not None:
            yield record


def extract_loghub(
    input_path: str | None,
    output_path: str,
    *,
    max_files: int = DEFAULT_LOGHUB_MAX_FILES,
) -> ExtractedDiagnosticLogSlice:
    """Extract sanitized eval-only records from a staged LogHub checkout."""
    if max_files <= 0:
        raise ValueError(f"max_files must be positive, got {max_files}")

    input_path = _resolve_loghub_input_path(input_path)
    loghub_path = os.path.join(input_path, LOGHUB_DIRNAME)
    output_file = os.path.join(output_path, "eval_only", "loghub", "data-00000-of-00001.jsonl")
    kept_records, bytes_written = _write_jsonl_records(output_file, _iter_loghub_records(loghub_path, max_files))
    manifest = source_manifest("loghub")
    slice_output_dir = os.path.join(output_path, "eval_only", "loghub")
    metadata_path = _write_source_metadata(
        manifest=manifest,
        input_path=input_path,
        output_path=slice_output_dir,
        output_file=output_file,
        record_count=kept_records,
        bytes_written=bytes_written,
        metadata={
            "source_path": loghub_path,
            "sample_limits": {"max_files": max_files},
        },
    )
    return ExtractedDiagnosticLogSlice(
        source_label=manifest.source_label,
        output_dir=slice_output_dir,
        output_file=output_file,
        metadata_path=metadata_path,
        record_count=kept_records,
        bytes_written=bytes_written,
        content_fingerprint=manifest.fingerprint(),
    )


def extract_diagnostic_logs(
    ghalogs_input_path: str,
    output_path: str,
    *,
    logchunks_input_path: str | None = None,
    loghub_input_path: str | None = None,
    max_ghalogs_members: int = DEFAULT_GHALOGS_MAX_MEMBERS,
    max_logchunks_examples: int = DEFAULT_LOGCHUNKS_MAX_EXAMPLES,
    max_loghub_files: int = DEFAULT_LOGHUB_MAX_FILES,
) -> ExtractedDiagnosticLogs:
    """Extract GHALogs for training plus LogChunks/LogHub as eval-only records."""
    return ExtractedDiagnosticLogs(
        ghalogs=extract_ghalogs(ghalogs_input_path, output_path, max_members=max_ghalogs_members),
        logchunks=extract_logchunks(logchunks_input_path, output_path, max_examples=max_logchunks_examples),
        loghub=extract_loghub(loghub_input_path, output_path, max_files=max_loghub_files),
    )


def extract_ghalogs_step(
    *,
    source_path: str,
    max_members: int = DEFAULT_GHALOGS_MAX_MEMBERS,
    output_path_prefix: str | None = None,
) -> StepSpec:
    """Return a StepSpec that materializes the capped GHALogs sample."""
    source = source_manifest("ghalogs")
    return StepSpec(
        name="processed/diagnostic_logs/ghalogs_public_sample",
        output_path_prefix=output_path_prefix,
        fn=lambda output_path: extract_ghalogs(source_path, output_path, max_members=max_members),
        hash_attrs={
            "version": "v4",
            "source_path": source_path,
            "source_label": source.source_label,
            "max_members": max_members,
            "split_policy": "97% train / 1% dev / 1% test / 1% issue_5093_holdout",
            "source_content_fingerprint": source.fingerprint(),
            "sanitization_rules": "gh token/aws key/secret kv/email/user path/internal gs path",
        },
    )


def materialize_ghalogs_step(
    *,
    source_path: str = GHALOGS_STAGED_PREFIX,
    max_members: int | None = None,
    num_shards: int = DEFAULT_GHALOGS_MATERIALIZE_SHARDS,
    output_path_prefix: str | None = None,
) -> StepSpec:
    """Return a StepSpec that materializes GHALogs into reusable parquet shards."""
    source = source_manifest("ghalogs")
    return StepSpec(
        name="processed/diagnostic_logs/ghalogs_public_parquet",
        output_path_prefix=output_path_prefix,
        fn=lambda output_path: materialize_ghalogs_to_parquet(
            source_path,
            output_path,
            max_members=max_members,
            num_shards=num_shards,
        ),
        hash_attrs={
            "version": "v1",
            "source_path": source_path,
            "source_label": source.source_label,
            "max_members": max_members,
            "num_shards": num_shards,
            "source_content_fingerprint": source.fingerprint(),
            "sanitization_rules": "gh token/aws key/secret kv/email/user path/internal gs path",
            "output_format": "parquet",
        },
    )


def materialize_ghalogs_partition_step(
    *,
    materialized: StepSpec,
    partition: DiagnosticPartition,
    num_shards: int = DEFAULT_GHALOGS_PARTITION_SHARDS,
    output_path_prefix: str | None = None,
) -> StepSpec:
    """Return a StepSpec that filters materialized GHALogs parquet to one partition."""
    source = source_manifest("ghalogs")
    return StepSpec(
        name=f"processed/diagnostic_logs/ghalogs_public_{partition.value}_parquet",
        deps=[materialized],
        output_path_prefix=output_path_prefix,
        fn=lambda output_path: materialize_ghalogs_partition_to_parquet(
            materialized.output_path,
            output_path,
            partition=partition,
            num_shards=num_shards,
        ),
        hash_attrs={
            "version": "v1",
            "source_label": source.source_label,
            "materialized_input": materialized.output_path,
            "partition": partition.value,
            "num_shards": num_shards,
            "source_content_fingerprint": source.fingerprint(),
        },
    )


def extract_logchunks_step(
    *,
    source_path: str | None = None,
    max_examples: int = DEFAULT_LOGCHUNKS_MAX_EXAMPLES,
    output_path_prefix: str | None = None,
) -> StepSpec:
    """Return a StepSpec that materializes the capped LogChunks eval slice."""
    source = source_manifest("logchunks")
    return StepSpec(
        name="processed/diagnostic_logs/logchunks_eval_only",
        output_path_prefix=output_path_prefix,
        fn=lambda output_path: extract_logchunks(source_path, output_path, max_examples=max_examples),
        hash_attrs={
            "version": "v4",
            "source_path": source_path,
            "source_label": source.source_label,
            "max_examples": max_examples,
            "source_content_fingerprint": source.fingerprint(),
            "sanitization_rules": "gh token/aws key/secret kv/email/user path/internal gs path",
        },
    )


def extract_loghub_step(
    *,
    source_path: str | None = None,
    max_files: int = DEFAULT_LOGHUB_MAX_FILES,
    output_path_prefix: str | None = None,
) -> StepSpec:
    """Return a StepSpec that materializes the capped LogHub eval slice."""
    source = source_manifest("loghub")
    return StepSpec(
        name="processed/diagnostic_logs/loghub_eval_only",
        output_path_prefix=output_path_prefix,
        fn=lambda output_path: extract_loghub(source_path, output_path, max_files=max_files),
        hash_attrs={
            "version": "v4",
            "source_path": source_path,
            "source_label": source.source_label,
            "max_files": max_files,
            "source_content_fingerprint": source.fingerprint(),
            "sanitization_rules": "gh token/aws key/secret kv/email/user path/internal gs path",
        },
    )


def extract_diagnostic_logs_steps(
    *,
    ghalogs_source_path: str,
    logchunks_source_path: str | None = None,
    loghub_source_path: str | None = None,
    max_ghalogs_members: int = DEFAULT_GHALOGS_MAX_MEMBERS,
    max_logchunks_examples: int = DEFAULT_LOGCHUNKS_MAX_EXAMPLES,
    max_loghub_files: int = DEFAULT_LOGHUB_MAX_FILES,
    output_path_prefix: str | None = None,
) -> tuple[StepSpec, StepSpec, StepSpec]:
    """Return one materialization step per public diagnostic-log source."""
    return (
        extract_ghalogs_step(
            source_path=ghalogs_source_path,
            max_members=max_ghalogs_members,
            output_path_prefix=output_path_prefix,
        ),
        extract_logchunks_step(
            source_path=logchunks_source_path,
            max_examples=max_logchunks_examples,
            output_path_prefix=output_path_prefix,
        ),
        extract_loghub_step(
            source_path=loghub_source_path,
            max_files=max_loghub_files,
            output_path_prefix=output_path_prefix,
        ),
    )


def extract_diagnostic_logs_step(
    *,
    ghalogs_source_path: str,
    logchunks_source_path: str | None = None,
    loghub_source_path: str | None = None,
    max_ghalogs_members: int = DEFAULT_GHALOGS_MAX_MEMBERS,
    max_logchunks_examples: int = DEFAULT_LOGCHUNKS_MAX_EXAMPLES,
    max_loghub_files: int = DEFAULT_LOGHUB_MAX_FILES,
    output_path_prefix: str | None = None,
) -> StepSpec:
    """Return a StepSpec that materializes all public diagnostic-log slices together."""
    return StepSpec(
        name="processed/diagnostic_logs/public_sample",
        output_path_prefix=output_path_prefix,
        fn=lambda output_path: extract_diagnostic_logs(
            ghalogs_source_path,
            output_path,
            logchunks_input_path=logchunks_source_path,
            loghub_input_path=loghub_source_path,
            max_ghalogs_members=max_ghalogs_members,
            max_logchunks_examples=max_logchunks_examples,
            max_loghub_files=max_loghub_files,
        ),
        hash_attrs={
            "version": "v4",
            "sample_only": True,
            "ghalogs_source_path": ghalogs_source_path,
            "logchunks_source_path": logchunks_source_path,
            "loghub_source_path": loghub_source_path,
            "max_ghalogs_members": max_ghalogs_members,
            "max_logchunks_examples": max_logchunks_examples,
            "max_loghub_files": max_loghub_files,
            "split_policy": "97% train / 1% dev / 1% test / 1% issue_5093_holdout",
            "source_content_fingerprints": {
                source.source_label: source.fingerprint()
                for source in source_inventory()
                if source.source_label in {"ghalogs", "logchunks", "loghub"}
            },
            "sanitization_rules": "gh token/aws key/secret kv/email/user path/internal gs path",
        },
    )
