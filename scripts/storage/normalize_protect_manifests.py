#!/usr/bin/env -S uv run
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Normalize user-provided protect manifests into explicit bucket globs.

This script is local-only. It reads CSV manifests, expands wildcard bucket
selectors such as ``gs://marin-*`` into explicit regional buckets known to this
repo, converts a small set of regex-like patterns into shell-style globs, and
flags rows that are too broad to trust for a destructive purge.
"""

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from scripts.storage.generate_keep_globs import _is_overbroad_checkpoint_glob

MARIN_BUCKETS = [
    "marin-us-central1",
    "marin-us-central2",
    "marin-us-east1",
    "marin-us-east5",
    "marin-us-west4",
    "marin-eu-west4",
]

GLOB_CHARS = set("*?[]")


@dataclass(frozen=True)
class SourceRow:
    source_file: str
    row_number: int
    original_pattern: str
    owner: str
    reason: str
    artifact_kind: str
    priority: str
    source_kind: str
    module: str
    line: str


@dataclass(frozen=True)
class Issue:
    source_file: str
    row_number: int
    original_pattern: str
    normalized_glob: str
    severity: str
    code: str
    message: str


def _bucket_from_gs_url(pattern: str) -> str | None:
    if not pattern.startswith("gs://"):
        return None
    rest = pattern.removeprefix("gs://")
    return rest.split("/", 1)[0]


def _object_path_from_gs_url(pattern: str) -> str:
    rest = pattern.removeprefix("gs://")
    parts = rest.split("/", 1)
    return parts[1] if len(parts) == 2 else ""


def _has_glob(pattern: str) -> bool:
    return any(ch in pattern for ch in GLOB_CHARS) or "**" in pattern


def _looks_like_file_path(path: str) -> bool:
    basename = path.rstrip("/").split("/")[-1]
    return "." in basename and not basename.endswith(".*")


def _normalize_glob_object_path(path: str) -> tuple[str, list[tuple[str, str, str]]]:
    issues: list[tuple[str, str, str]] = []
    normalized = path.strip()

    if any(token in normalized for token in ("(?:", "(", ")", "|", "[", "]", "^", "$")):
        issues.append(
            ("error", "unsupported_regex_tokens", "Pattern contains regex tokens that were not converted automatically.")
        )

    if "/.*/" in normalized or normalized.endswith("/.*"):
        normalized = normalized.replace("/.*/", "/**/")
        if normalized.endswith("/.*"):
            normalized = normalized[:-3] + "/**"
        issues.append(
            ("warning", "regex_dotstar_prefix", "Converted `/. *`-style regex prefix to recursive glob `/**`.")
        )

    if ".*" in normalized:
        normalized = normalized.replace(".*", "*")
        issues.append(("warning", "regex_dotstar_suffix", "Converted regex-like `.*` suffix to glob `*`."))

    if normalized.endswith("/"):
        normalized = normalized + "**"
        issues.append(
            ("warning", "trailing_slash_recursive", "Assumed trailing slash means protect the full recursive prefix.")
        )
    elif not _has_glob(normalized) and not _looks_like_file_path(normalized):
        normalized = normalized + "/**"
        issues.append(
            ("warning", "assumed_prefix_recursive", "No glob characters found; treated path as a recursive prefix.")
        )

    normalized = re.sub(r"/{2,}", "/", normalized)
    return normalized, issues


def _expand_bucket_pattern(pattern: str) -> tuple[list[str], list[tuple[str, str, str]]]:
    issues: list[tuple[str, str, str]] = []
    if not pattern.startswith("gs://"):
        return [pattern], [("error", "unsupported_scheme", "Only `gs://...` patterns are supported.")]

    bucket = _bucket_from_gs_url(pattern)
    object_path = _object_path_from_gs_url(pattern)

    if bucket == "marin-*":
        object_glob, object_issues = _normalize_glob_object_path(object_path)
        issues.extend(object_issues)
        return [f"gs://{bucket_name}/{object_glob}" for bucket_name in MARIN_BUCKETS], issues

    if bucket.endswith("*"):
        bucket_prefix = bucket[:-1]
        matching_buckets = [bucket_name for bucket_name in MARIN_BUCKETS if bucket_name.startswith(bucket_prefix)]
        if matching_buckets:
            object_glob, object_issues = _normalize_glob_object_path(object_path)
            issues.extend(object_issues)
            issues.append(
                (
                    "info",
                    "bucket_prefix_expansion",
                    f"Expanded bucket wildcard `{bucket}` to {len(matching_buckets)} known repo buckets.",
                )
            )
            return [f"gs://{bucket_name}/{object_glob}" for bucket_name in matching_buckets], issues

    if bucket in MARIN_BUCKETS:
        object_glob, object_issues = _normalize_glob_object_path(object_path)
        issues.extend(object_issues)
        return [f"gs://{bucket}/{object_glob}"], issues

    object_glob, object_issues = _normalize_glob_object_path(object_path)
    issues.extend(object_issues)
    issues.append(("info", "nonstandard_bucket", "Bucket is not one of the repo's standard Marin regional buckets."))
    return [f"gs://{bucket}/{object_glob}"], issues


def _is_fatal_glob(normalized_glob: str) -> tuple[bool, tuple[str, str, str] | None]:
    bucket = _bucket_from_gs_url(normalized_glob)
    object_path = _object_path_from_gs_url(normalized_glob)

    if bucket is None or bucket == "":
        return True, ("error", "invalid_bucket", "Could not determine bucket from normalized glob.")

    if object_path in {"", "*", "**"}:
        return True, ("error", "bucket_wide_glob", "Pattern expands to essentially the whole bucket.")

    if any(token in object_path for token in ("(?:", "(", ")", "|", "[", "]", "^", "$")):
        return True, ("error", "unsupported_regex_tokens", "Pattern still contains regex tokens after normalization.")

    if _is_overbroad_checkpoint_glob(normalized_glob):
        return True, (
            "error",
            "overbroad_checkpoint_glob",
            "Checkpoint glob is too broad for a destructive purge allowlist.",
        )

    return False, None


def _read_high_value(path: Path) -> list[SourceRow]:
    rows: list[SourceRow] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row_number, row in enumerate(reader, start=2):
            pattern = (row.get("path_glob") or "").strip()
            if not pattern:
                continue
            rows.append(
                SourceRow(
                    source_file=str(path),
                    row_number=row_number,
                    original_pattern=pattern,
                    owner="",
                    reason=(row.get("reason") or "").strip(),
                    artifact_kind=(row.get("artifact_kind") or "").strip(),
                    priority=(row.get("priority") or "").strip(),
                    source_kind=(row.get("source_kind") or "").strip(),
                    module=(row.get("module") or "").strip(),
                    line=(row.get("line") or "").strip(),
                )
            )
    return rows


def _read_named(path: Path) -> list[SourceRow]:
    rows: list[SourceRow] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row_number, row in enumerate(reader, start=2):
            pattern = (row.get("Path") or "").strip()
            if not pattern:
                continue
            reason = (row.get("Reason (optional)") or "").strip()
            david_notes = (row.get("David Notes") or "").strip()
            combined_reason = reason if not david_notes else f"{reason} | David Notes: {david_notes}".strip(" |")
            rows.append(
                SourceRow(
                    source_file=str(path),
                    row_number=row_number,
                    original_pattern=pattern,
                    owner=(row.get("User") or "").strip(),
                    reason=combined_reason,
                    artifact_kind="",
                    priority="",
                    source_kind="named_manifest",
                    module="",
                    line="",
                )
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        if not rows:
            return
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def normalize_manifests(
    high_value_path: Path,
    named_path: Path,
    expanded_output: Path,
    deduped_output: Path,
    issues_output: Path,
) -> tuple[int, int, int]:
    source_rows = [*_read_high_value(high_value_path), *_read_named(named_path)]

    expanded_rows: list[dict[str, str]] = []
    issue_rows: list[dict[str, str]] = []

    for source in source_rows:
        normalized_globs, normalization_issues = _expand_bucket_pattern(source.original_pattern)
        for normalized_glob in normalized_globs:
            fatal, fatal_issue = _is_fatal_glob(normalized_glob)
            row_issues = list(normalization_issues)
            if fatal_issue is not None:
                row_issues.append(fatal_issue)

            for severity, code, message in row_issues:
                issue_rows.append(
                    {
                        "source_file": source.source_file,
                        "row_number": str(source.row_number),
                        "original_pattern": source.original_pattern,
                        "normalized_glob": normalized_glob,
                        "severity": severity,
                        "code": code,
                        "message": message,
                    }
                )

            expanded_rows.append(
                {
                    "normalized_glob": normalized_glob,
                    "bucket": _bucket_from_gs_url(normalized_glob) or "",
                    "included": "false" if fatal else "true",
                    "source_file": source.source_file,
                    "row_number": str(source.row_number),
                    "original_pattern": source.original_pattern,
                    "owner": source.owner,
                    "reason": source.reason,
                    "artifact_kind": source.artifact_kind,
                    "priority": source.priority,
                    "source_kind": source.source_kind,
                    "module": source.module,
                    "line": source.line,
                    "issue_codes": ";".join(
                        issue["code"]
                        for issue in [
                            {
                                "code": code,
                            }
                            for _, code, _ in row_issues
                        ]
                    ),
                }
            )

    included_rows = [row for row in expanded_rows if row["included"] == "true"]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in included_rows:
        grouped[row["normalized_glob"]].append(row)

    deduped_rows: list[dict[str, str]] = []
    for normalized_glob, rows in sorted(grouped.items()):
        deduped_rows.append(
            {
                "normalized_glob": normalized_glob,
                "bucket": rows[0]["bucket"],
                "owners": ";".join(sorted({row["owner"] for row in rows if row["owner"]})),
                "reasons": " || ".join(sorted({row["reason"] for row in rows if row["reason"]})),
                "sources": ";".join(sorted({f"{Path(row['source_file']).name}:{row['row_number']}" for row in rows})),
                "artifact_kinds": ";".join(sorted({row["artifact_kind"] for row in rows if row["artifact_kind"]})),
                "priority_max": max((int(row["priority"]) for row in rows if row["priority"].isdigit()), default=0),
            }
        )

    issue_rows.sort(
        key=lambda row: (row["severity"], row["source_file"], int(row["row_number"]), row["normalized_glob"])
    )
    expanded_rows.sort(
        key=lambda row: (row["included"] != "true", row["bucket"], row["normalized_glob"], row["source_file"])
    )
    deduped_rows.sort(key=lambda row: (row["bucket"], row["normalized_glob"]))

    _write_csv(expanded_output, expanded_rows)
    _write_csv(deduped_output, deduped_rows)
    _write_csv(
        issues_output,
        (
            issue_rows
            if issue_rows
            else [
                {
                    "source_file": "",
                    "row_number": "",
                    "original_pattern": "",
                    "normalized_glob": "",
                    "severity": "",
                    "code": "",
                    "message": "",
                }
            ]
        ),
    )

    return len(expanded_rows), len(deduped_rows), len(issue_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize protect manifests into explicit bucket globs.")
    parser.add_argument(
        "--high-value",
        type=Path,
        default=Path("high-value.csv"),
        help="Path to the generated high-value manifest.",
    )
    parser.add_argument(
        "--named",
        type=Path,
        default=Path("named.csv"),
        help="Path to the hand-maintained named manifest.",
    )
    parser.add_argument(
        "--expanded-output",
        type=Path,
        default=Path("scripts/storage/protect_manifest_expanded.csv"),
        help="Per-source expanded manifest output.",
    )
    parser.add_argument(
        "--deduped-output",
        type=Path,
        default=Path("scripts/storage/protect_manifest_deduped.csv"),
        help="Deduplicated manifest output.",
    )
    parser.add_argument(
        "--issues-output",
        type=Path,
        default=Path("scripts/storage/protect_manifest_issues.csv"),
        help="Normalization issues output.",
    )
    args = parser.parse_args()

    expanded_count, deduped_count, issue_count = normalize_manifests(
        high_value_path=args.high_value,
        named_path=args.named,
        expanded_output=args.expanded_output,
        deduped_output=args.deduped_output,
        issues_output=args.issues_output,
    )
    print(f"Wrote {expanded_count} expanded protect rows.")
    print(f"Wrote {deduped_count} deduplicated protect rows.")
    print(f"Recorded {issue_count} normalization issues.")


if __name__ == "__main__":
    main()
