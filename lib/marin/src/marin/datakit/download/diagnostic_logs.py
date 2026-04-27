# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Public diagnostic-log source inventory and GHALogs extraction helpers."""

from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
import logging
import os.path
import re
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

import fsspec
from marin.utils import fsspec_mkdirs

from marin.execution.step_spec import StepSpec

logger = logging.getLogger(__name__)

GHALOGS_RECORD_URL = "https://zenodo.org/records/14796970"
LOGCHUNKS_RECORD_URL = "https://zenodo.org/records/3632351"
LOGHUB_REPO_URL = "https://github.com/logpai/loghub"
GHALOGS_ZIP_FILENAME = "github_run_logs.zip"
LOGCHUNKS_ZIP_FILENAME = "LogChunks.zip"
LOGHUB_DIRNAME = "loghub"

GHALOGS_TOTAL_BYTES = 143_425_404_506
LOGCHUNKS_TOTAL_BYTES = 24_108_826
LOGHUB_REPO_SIZE_BYTES = 7_513_088
DEFAULT_GHALOGS_MAX_MEMBERS = 10_000
DEFAULT_LOGCHUNKS_MAX_EXAMPLES = 10_000
DEFAULT_LOGHUB_MAX_FILES = 100

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
        source_license="Creative Commons Attribution Share Alike 4.0 International",
        source_format="runs.json.gz, repositories.json.gz, github_run_logs.zip",
        compressed_size_bytes=GHALOGS_TOTAL_BYTES,
        contamination_risk="high: public CI logs can contain secrets and internal paths",
        status=DiagnosticSourceStatus.TRAINING_READY,
        provenance_notes="DOI 10.5281/zenodo.14796970, published 2025-02-03; Zenodo license id cc-by-sa-4.0.",
        rough_tokens_b=None,
    ),
    DiagnosticLogSource(
        name="logchunks",
        source_url=LOGCHUNKS_RECORD_URL,
        source_license="Creative Commons Attribution 4.0 International",
        source_format="LogChunks.zip (XML chunk annotations)",
        compressed_size_bytes=LOGCHUNKS_TOTAL_BYTES,
        contamination_risk="medium: labeled failure snippets may include local paths and user names",
        status=DiagnosticSourceStatus.EVAL_ONLY,
        provenance_notes="DOI 10.5281/zenodo.3632351, published 2020-01-31; eval-only despite acceptable license.",
        rough_tokens_b=None,
    ),
    DiagnosticLogSource(
        name="loghub",
        source_url=LOGHUB_REPO_URL,
        source_license="custom research/academic-only license",
        source_format="mixed plain-text log files grouped by dataset",
        compressed_size_bytes=LOGHUB_REPO_SIZE_BYTES,
        contamination_risk="medium: includes system identifiers and infrastructure paths",
        status=DiagnosticSourceStatus.EVAL_ONLY,
        provenance_notes="LICENSE file restricts usage to research/academic work; acceptable only for eval use.",
        rough_tokens_b=None,
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


def _write_jsonl_records(output_file: str, records: Iterable[dict[str, str]]) -> int:
    kept_records = 0
    output_dir = os.path.dirname(output_file)
    fsspec_mkdirs(output_dir, exist_ok=True)
    with fsspec.open(output_file, "wt", encoding="utf-8") as writer:
        for record in records:
            kept_records += 1
            writer.write(json.dumps(record, ensure_ascii=False))
            writer.write("\n")
    return kept_records


def extract_ghalogs(
    input_path: str,
    output_path: str,
    *,
    max_members: int = DEFAULT_GHALOGS_MAX_MEMBERS,
) -> None:
    """Extract a capped sample of partitioned, sanitized records from a staged GHALogs archive."""
    if max_members <= 0:
        raise ValueError(f"max_members must be positive, got {max_members}")

    archive_path = os.path.join(input_path, GHALOGS_ZIP_FILENAME)
    counters = {"seen_members": 0, "kept_records": 0}
    partition_counts = {partition.value: 0 for partition in DiagnosticPartition}

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
                writers[partition].write(json.dumps(record, ensure_ascii=False))
                writers[partition].write("\n")

    metadata = {
        "source": "ghalogs",
        "source_url": GHALOGS_RECORD_URL,
        "source_archive": archive_path,
        "sample_limits": {"max_members": max_members},
        "counters": counters,
        "partition_counts": partition_counts,
        "training_ready_sources": [source.name for source in training_ready_sources()],
    }
    with fsspec.open(os.path.join(output_path, "metadata.json"), "wt", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


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
    input_path: str,
    output_path: str,
    *,
    max_examples: int = DEFAULT_LOGCHUNKS_MAX_EXAMPLES,
) -> None:
    """Extract a capped sample of sanitized eval-only records from staged LogChunks."""
    if max_examples <= 0:
        raise ValueError(f"max_examples must be positive, got {max_examples}")

    archive_path = os.path.join(input_path, LOGCHUNKS_ZIP_FILENAME)
    output_file = os.path.join(output_path, "eval_only", "logchunks", "data-00000-of-00001.jsonl")
    kept_records = _write_jsonl_records(output_file, _iter_logchunks_records(archive_path, max_examples))

    metadata = {
        "source": "logchunks",
        "source_url": LOGCHUNKS_RECORD_URL,
        "source_archive": archive_path,
        "sample_limits": {"max_examples": max_examples},
        "counters": {"kept_records": kept_records},
        "status": DiagnosticSourceStatus.EVAL_ONLY.value,
    }
    with fsspec.open(
        os.path.join(output_path, "eval_only", "logchunks", "metadata.json"), "wt", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


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
    input_path: str,
    output_path: str,
    *,
    max_files: int = DEFAULT_LOGHUB_MAX_FILES,
) -> None:
    """Extract sanitized eval-only records from a staged LogHub checkout."""
    if max_files <= 0:
        raise ValueError(f"max_files must be positive, got {max_files}")

    loghub_path = os.path.join(input_path, LOGHUB_DIRNAME)
    output_file = os.path.join(output_path, "eval_only", "loghub", "data-00000-of-00001.jsonl")
    kept_records = _write_jsonl_records(output_file, _iter_loghub_records(loghub_path, max_files))

    metadata = {
        "source": "loghub",
        "source_url": LOGHUB_REPO_URL,
        "source_path": loghub_path,
        "sample_limits": {"max_files": max_files},
        "counters": {"kept_records": kept_records},
        "status": DiagnosticSourceStatus.EVAL_ONLY.value,
    }
    with fsspec.open(
        os.path.join(output_path, "eval_only", "loghub", "metadata.json"), "wt", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def extract_diagnostic_logs(
    input_path: str,
    output_path: str,
    *,
    max_ghalogs_members: int = DEFAULT_GHALOGS_MAX_MEMBERS,
    max_logchunks_examples: int = DEFAULT_LOGCHUNKS_MAX_EXAMPLES,
    max_loghub_files: int = DEFAULT_LOGHUB_MAX_FILES,
) -> None:
    """Extract GHALogs for training plus LogChunks/LogHub as eval-only records."""
    extract_ghalogs(input_path, output_path, max_members=max_ghalogs_members)
    extract_logchunks(input_path, output_path, max_examples=max_logchunks_examples)
    extract_loghub(input_path, output_path, max_files=max_loghub_files)


def extract_diagnostic_logs_step(
    *,
    source_path: str,
    max_ghalogs_members: int = DEFAULT_GHALOGS_MAX_MEMBERS,
    max_logchunks_examples: int = DEFAULT_LOGCHUNKS_MAX_EXAMPLES,
    max_loghub_files: int = DEFAULT_LOGHUB_MAX_FILES,
) -> StepSpec:
    """Return a StepSpec that materializes diagnostic-log records."""
    return StepSpec(
        name="processed/diagnostic_logs/public_sample",
        fn=lambda output_path: extract_diagnostic_logs(
            source_path,
            output_path,
            max_ghalogs_members=max_ghalogs_members,
            max_logchunks_examples=max_logchunks_examples,
            max_loghub_files=max_loghub_files,
        ),
        hash_attrs={
            "version": "v1",
            "sample_only": True,
            "source_path": source_path,
            "max_ghalogs_members": max_ghalogs_members,
            "max_logchunks_examples": max_logchunks_examples,
            "max_loghub_files": max_loghub_files,
            "split_policy": "97% train / 1% dev / 1% test / 1% issue_5093_holdout",
            "eval_only_sources": "logchunks/loghub",
            "sanitization_rules": "gh token/aws key/secret kv/email/user path/internal gs path",
        },
    )
