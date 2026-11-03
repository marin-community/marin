# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit exact empty SWH cache rows without certifying their advertised identities."""

import hashlib
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field

import pyarrow as pa
import pyarrow.parquet as pq
from marin.datakit.normalize import _text_from_value
from rigging.filesystem.buckets import filesystem_for

EMPTY_SHA1 = hashlib.sha1(b"").hexdigest()
EMPTY_GIT_SHA1 = hashlib.sha1(b"blob 0\0").hexdigest()
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
SCHEMA = pa.schema(
    [("sha1_git", pa.string()), ("sha1", pa.string()), ("content", pa.large_binary()), ("present", pa.bool_())]
)


def empty_reader_policy() -> dict:
    return {
        "rule": "exclude only present is True and type(content) is bytes and content == b''",
        "existing_absence": "present is False and exact empty bytes remains separate",
        "nonempty": "both SWH identities mandatory before decode, including unselected and whitespace rows",
        "normalizer_sha256": "f1cb9e48b4b9dd60c66d7a1bb520bf1060124f94c844848eb0c95827aafdc5c4",
        "historical_filter_sha256": "8b367a5f6b7e4a2dfe24245b60b281ca56aee379066ae949936823db0d16ffd0",
        "historical_order_sha256": "bbf24e931f0756baec72a63b71188fcec95800cdec2da70aa7aaba9e74f9779c",
        "historical_execution": "compatible; exact dirty execution not established",
        "advertised_identity_membership": "unknown for excluded records",
        "identity_repair": False,
    }


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _empty_record(row: dict, ordinal: int, group: int, within: int) -> dict:
    return {
        "raw_row": ordinal,
        "row_group": group,
        "row_within_group": within,
        "present": row["present"],
        "expected_sha1": row["sha1"],
        "expected_sha1_git": row["sha1_git"],
        "actual_sha1": EMPTY_SHA1,
        "actual_sha1_git": EMPTY_GIT_SHA1,
        "actual_sha256": EMPTY_SHA256,
        "payload_bytes": 0,
        "plain_identity_matches": row["sha1"] == EMPTY_SHA1,
        "git_identity_matches": row["sha1_git"] == EMPTY_GIT_SHA1,
        "advertised_identity_membership": "unknown",
    }


@dataclass
class SoftwareHeritageEmptyAudit:
    provider_snapshot: dict
    provider_path: str = ""
    expected_rows: int = 0
    rows_per_group: list[int] = field(default_factory=list)
    total_rows: int = 0
    returned_rows: int = 0
    nonempty_rows: int = 0
    verified_nonempty_rows: int = 0
    verified_nonempty_blank_rows: int = 0
    excluded_present_empty: list[dict] = field(default_factory=list)
    existing_absent_empty: list[dict] = field(default_factory=list)
    complete: bool = False

    def verified(self, row: dict) -> None:
        """Account for a nonempty input after the strict replay iterator finishes."""
        if row["content"]:
            self.verified_nonempty_rows += 1
            if not _text_from_value(row["content"]).strip():
                self.verified_nonempty_blank_rows += 1

    def report(self) -> dict:
        return {
            "policy": empty_reader_policy(),
            "policy_sha256": _digest(empty_reader_policy()),
            "provider_path": self.provider_path,
            "provider_snapshot": self.provider_snapshot,
            "expected_rows": self.expected_rows,
            "rows_per_group": self.rows_per_group,
            "total_rows": self.total_rows,
            "returned_rows": self.returned_rows,
            "nonempty_rows": self.nonempty_rows,
            "verified_nonempty_rows": self.verified_nonempty_rows,
            "verified_nonempty_blank_rows": self.verified_nonempty_blank_rows,
            "excluded_present_empty": self.excluded_present_empty,
            "existing_absent_empty": self.existing_absent_empty,
            "complete": self.complete,
        }


def read_software_heritage(path: str, audit: SoftwareHeritageEmptyAudit, batch_size: int) -> Iterator[dict]:
    """Yield original rows except audited exact present/empty inputs."""
    fs, key = filesystem_for(path)
    audit.provider_path = path
    with fs.open(key, "rb") as stream:
        parquet = pq.ParquetFile(stream, pre_buffer=False)
        if not parquet.schema_arrow.equals(SCHEMA):
            raise ValueError("SWH original schema differs from the pinned binary schema")
        audit.expected_rows = parquet.metadata.num_rows
        audit.rows_per_group = [parquet.metadata.row_group(i).num_rows for i in range(parquet.metadata.num_row_groups)]
        for group in range(parquet.metadata.num_row_groups):
            within = 0
            for batch in parquet.iter_batches(batch_size=batch_size, row_groups=[group], columns=SCHEMA.names):
                for row in batch.to_pylist():
                    if type(row["content"]) is not bytes or type(row["present"]) is not bool:
                        raise ValueError("SWH original present/content shape is invalid")
                    if any(
                        not isinstance(row[k], str) or not re.fullmatch("[0-9a-f]{40}", row[k])
                        for k in ("sha1", "sha1_git")
                    ):
                        raise ValueError("SWH advertised identity shape is invalid")
                    ordinal = audit.total_rows
                    audit.total_rows += 1
                    if row["content"] == b"":
                        record = _empty_record(row, ordinal, group, within)
                        within += 1
                        if row["present"] is True:
                            audit.excluded_present_empty.append(record)
                            continue
                        audit.existing_absent_empty.append(record)
                    else:
                        within += 1
                        if row["present"] is not True:
                            raise ValueError("SWH absent content must have an empty byte payload")
                        audit.nonempty_rows += 1
                    audit.returned_rows += 1
                    yield row
    if audit.total_rows != audit.expected_rows or audit.verified_nonempty_rows != audit.nonempty_rows:
        raise ValueError("SWH original reader did not complete all rows and strict nonempty verification")
    audit.complete = True


def validate_empty_audit(audit: dict | None, path: str, returned_rows: int, snapshot: dict) -> dict:
    """Require complete, bound EOF accounting and exact empty-record evidence."""
    if (
        audit is None
        or audit.get("policy") != empty_reader_policy()
        or audit.get("policy_sha256") != _digest(empty_reader_policy())
    ):
        raise ValueError("SWH empty reader audit has an absent or different policy")
    if not audit.get("complete") or audit["provider_path"] != path or audit["provider_snapshot"] != snapshot:
        raise ValueError("SWH empty reader audit is incomplete or belongs to a different original snapshot")
    counts = [
        audit[k]
        for k in (
            "expected_rows",
            "total_rows",
            "returned_rows",
            "nonempty_rows",
            "verified_nonempty_rows",
            "verified_nonempty_blank_rows",
        )
    ]
    groups = audit["rows_per_group"]
    if any(type(n) is not int or n < 0 for n in counts + groups):
        raise ValueError("SWH audit counters must be nonnegative integers")
    excluded, absent = audit["excluded_present_empty"], audit["existing_absent_empty"]
    if not (
        audit["expected_rows"] == audit["total_rows"] == sum(groups)
        and returned_rows == audit["returned_rows"] == audit["nonempty_rows"] + len(absent)
        and audit["returned_rows"] + len(excluded) == audit["total_rows"]
        and audit["verified_nonempty_rows"] == audit["nonempty_rows"]
        and audit["verified_nonempty_blank_rows"] <= audit["nonempty_rows"]
    ):
        raise ValueError("SWH EOF/empty/strict nonempty row accounting differs")
    seen: set[int] = set()
    offsets = [sum(groups[:i]) for i in range(len(groups))]
    for present, records in ((True, excluded), (False, absent)):
        previous = -1
        for record in records:
            row, group, within = record["raw_row"], record["row_group"], record["row_within_group"]
            if not (
                type(row) is int
                and row > previous
                and row not in seen
                and type(group) is int
                and 0 <= group < len(groups)
                and type(within) is int
                and 0 <= within < groups[group]
                and row == offsets[group] + within
            ):
                raise ValueError("SWH empty record row locator is duplicated or inconsistent")
            expected = _empty_record(
                {"present": present, "sha1": record["expected_sha1"], "sha1_git": record["expected_sha1_git"]},
                row,
                group,
                within,
            )
            if (
                any(type(record[k]) is not bool for k in ("present", "plain_identity_matches", "git_identity_matches"))
                or type(record["payload_bytes"]) is not int
                or record != expected
                or any(not re.fullmatch("[0-9a-f]{40}", record[k]) for k in ("expected_sha1", "expected_sha1_git"))
            ):
                raise ValueError("SWH empty record does not preserve its actual byte/hash faults")
            previous = row
            seen.add(row)
    return audit


def empty_source_audit(paths: list[str], audits: list[dict]) -> dict:
    if len(paths) != len(audits):
        raise ValueError("SWH source audit lacks complete file coverage")
    for path, audit in zip(paths, audits, strict=True):
        validate_empty_audit(audit, path, audit["returned_rows"], audit["provider_snapshot"])
    return {
        "policy": empty_reader_policy(),
        "policy_sha256": _digest(empty_reader_policy()),
        "complete": True,
        "provider_files": len(paths),
        "total_rows": sum(a["total_rows"] for a in audits),
        "strictly_verified_nonempty_rows": sum(a["verified_nonempty_rows"] for a in audits),
        "verified_nonempty_blank_rows": sum(a["verified_nonempty_blank_rows"] for a in audits),
        "excluded_present_empty_rows": sum(len(a["excluded_present_empty"]) for a in audits),
        "existing_absent_empty_rows": sum(len(a["existing_absent_empty"]) for a in audits),
        "file_audits_sha256": _digest(audits),
        "file_audits": audits,
        "advertised_identity_membership_for_excluded": "unknown",
        "identity_repair": False,
        "historical_execution": "compatible; exact dirty execution not established",
    }
