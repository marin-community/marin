# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Public diagnostic-log source inventory and sanitization helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

GHALOGS_RECORD_URL = "https://zenodo.org/records/14796970"
LOGCHUNKS_RECORD_URL = "https://zenodo.org/records/3632351"
LOGHUB_REPO_URL = "https://github.com/logpai/loghub"

GHALOGS_TOTAL_BYTES = 143_425_404_506
LOGCHUNKS_TOTAL_BYTES = 24_108_826
LOGHUB_REPO_SIZE_BYTES = 7_513_088

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
