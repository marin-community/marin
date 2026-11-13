# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove normalized, order-insensitive shell-output comparison."""

import collections
import re

ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
ERROR = re.compile(r"(?i)\b(?:error|failed|failure|no such file|not found|permission denied|traceback)\b")
UNIT = re.compile(r"(?i)\s+(?:bytes?|kb|kib|mb|mib|gb|gib)\s*$")


def _record(line: str) -> str:
    value = ANSI.sub("", line).strip()
    value = re.sub(r"(?<!\S)/workspace/", "", value)
    value = re.sub(r"(?<!\S)\./", "", value)
    value = UNIT.sub("", value)
    return re.sub(r"\s+", " ", value).strip()


def _records(text: str) -> list[str]:
    return [record for line in text.replace("\0", "\n").splitlines() if (record := _record(line))]


def score_capture(actual: str, expected: str) -> tuple[int, list[str]]:
    """Compare newline- or NUL-delimited records, preserving duplicate counts."""
    expected_records = collections.Counter(_records(expected))
    actual_records = collections.Counter(_records(actual))
    if not expected_records:
        return (1, []) if not actual_records else (0, ["expected empty output"])
    missing = expected_records - actual_records
    if missing:
        return 0, [f"missing expected records: {dict(missing)}"]
    extras = actual_records - expected_records
    for record in extras:
        if ERROR.search(record):
            return 0, [f"unexpected error record: {record}"]
    return 1, []
