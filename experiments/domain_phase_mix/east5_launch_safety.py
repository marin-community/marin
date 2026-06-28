# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate that data-mixing Iris parents are region-local.

The default gate is for east5 launches. Central1-by-design experiments such as
the StarCoder heteroskedastic SNR launcher can use the same checker with
``--expected-region us-central1 --expected-zone us-central1-a
--expected-bucket-prefix gs://marin-us-central1``.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

EAST5_REGION = "us-east5"
EAST5_ZONE = "us-east5-a"
EAST5_BUCKET_PREFIX = "gs://marin-us-east5"

_MARIN_BUCKET_RE = re.compile(r"gs://marin-us-[a-z0-9-]+[^\s\"']*")


@dataclass(frozen=True)
class LaunchSafetyResult:
    """Validation result for an east5 Iris launch command."""

    ok: bool
    errors: list[str]
    warnings: list[str]
    parent_regions: list[str]
    parent_zone: str | None
    allows_region_only_parent: bool
    child_tpu_regions: list[str]
    child_tpu_zone: str | None
    marin_gcs_paths: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "parent_regions": self.parent_regions,
            "parent_zone": self.parent_zone,
            "allows_region_only_parent": self.allows_region_only_parent,
            "child_tpu_regions": self.child_tpu_regions,
            "child_tpu_zone": self.child_tpu_zone,
            "marin_gcs_paths": self.marin_gcs_paths,
        }


def _option_values(tokens: Sequence[str], option: str) -> list[str]:
    values: list[str] = []
    prefix = f"{option}="
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == option:
            if index + 1 >= len(tokens) or tokens[index + 1].startswith("-"):
                values.append("")
            else:
                values.append(tokens[index + 1])
                index += 1
        elif token.startswith(prefix):
            values.append(token.removeprefix(prefix))
        index += 1
    return values


def _find_job_run_index(tokens: Sequence[str]) -> int:
    for index, token in enumerate(tokens[:-1]):
        if token == "job" and tokens[index + 1] == "run":
            return index
    raise ValueError("Expected an Iris job run command containing 'job run'.")


def _split_parent_and_child_tokens(tokens: Sequence[str], job_index: int) -> tuple[list[str], list[str]]:
    parent_tokens = list(tokens[job_index + 2 :])
    if "--" not in parent_tokens:
        return parent_tokens, []
    separator = parent_tokens.index("--")
    return parent_tokens[:separator], parent_tokens[separator + 1 :]


def _marin_gcs_paths(command: str) -> list[str]:
    paths: set[str] = set()
    for match in _MARIN_BUCKET_RE.findall(command):
        for path in match.split(","):
            if path.startswith("gs://marin-us-"):
                paths.add(path)
    return sorted(paths)


def validate_east5_iris_command(command: str) -> LaunchSafetyResult:
    """Validate a live east5 data-mixing Iris parent command.

    The parent process performs GCS reads/writes before child jobs run, so the
    Iris parent must be pinned to east5-a. Launcher-level child TPU placement is
    not sufficient.
    """

    return validate_regional_iris_command(
        command,
        expected_region=EAST5_REGION,
        expected_zone=EAST5_ZONE,
        expected_bucket_prefix=EAST5_BUCKET_PREFIX,
    )


def validate_regional_iris_command(
    command: str,
    *,
    expected_region: str,
    expected_zone: str,
    expected_bucket_prefix: str,
    allow_region_only_parent: bool = False,
) -> LaunchSafetyResult:
    """Validate a live data-mixing Iris parent command for one GCS region.

    The parent process performs GCS reads/writes before child jobs run, so the
    Iris parent must be pinned to the same region as the data it touches.
    By default this also requires a parent zone. Some launches intentionally
    leave the CPU parent region-only to avoid a saturated CPU zone while child
    TPU jobs remain zone-pinned; allow that only with
    ``allow_region_only_parent=True``.
    """

    tokens = shlex.split(command)
    job_index = _find_job_run_index(tokens)
    parent_tokens, child_tokens = _split_parent_and_child_tokens(tokens, job_index)

    parent_regions = _option_values(parent_tokens, "--region")
    parent_zone_values = _option_values(parent_tokens, "--zone")
    child_tpu_regions = _option_values(child_tokens, "--tpu-region")
    child_tpu_zone_values = _option_values(child_tokens, "--tpu-zone")
    gcs_paths = _marin_gcs_paths(command)

    errors: list[str] = []
    warnings: list[str] = []

    if not parent_regions:
        errors.append(f"Iris parent must include --region {expected_region}.")
    elif any(region != expected_region for region in parent_regions):
        errors.append(f"Iris parent --region values must all be {expected_region}: got {parent_regions}.")

    parent_zone = parent_zone_values[-1] if parent_zone_values else None
    if parent_zone is None:
        if allow_region_only_parent:
            warnings.append(
                f"Iris parent omits --zone; relying on --region {expected_region} "
                "for GCS locality and child --tpu-zone for TPU placement."
            )
        else:
            errors.append(f"Iris parent must include --zone {expected_zone}.")
    elif parent_zone != expected_zone:
        errors.append(f"Iris parent --zone must be {expected_zone}: got {parent_zone!r}.")

    if child_tpu_regions and any(region != expected_region for region in child_tpu_regions):
        errors.append(f"Child --tpu-region values must all be {expected_region}: got {child_tpu_regions}.")

    child_tpu_zone = child_tpu_zone_values[-1] if child_tpu_zone_values else None
    if child_tpu_zone is not None and child_tpu_zone != expected_zone:
        errors.append(f"Child --tpu-zone must be {expected_zone}: got {child_tpu_zone!r}.")

    bad_gcs_paths = [path for path in gcs_paths if not path.startswith(f"{expected_bucket_prefix}/")]
    if bad_gcs_paths:
        errors.append(
            "All Marin GCS paths in region-local launch commands must use "
            f"{expected_bucket_prefix}: got {bad_gcs_paths}."
        )

    return LaunchSafetyResult(
        ok=not errors,
        errors=errors,
        warnings=warnings,
        parent_regions=parent_regions,
        parent_zone=parent_zone,
        allows_region_only_parent=allow_region_only_parent,
        child_tpu_regions=child_tpu_regions,
        child_tpu_zone=child_tpu_zone,
        marin_gcs_paths=gcs_paths,
    )


def _read_command(args: argparse.Namespace) -> str:
    if args.command is not None:
        return args.command
    if args.command_file is not None:
        return Path(args.command_file).read_text()
    return sys.stdin.read()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--command", help="Iris job run command to validate.")
    source.add_argument("--command-file", help="File containing the Iris job run command.")
    source.add_argument("--stdin", action="store_true", help="Read the command from stdin.")
    parser.add_argument("--json", action="store_true", help="Emit a JSON validation result.")
    parser.add_argument("--expected-region", default=EAST5_REGION, help="Expected Iris parent and child TPU region.")
    parser.add_argument("--expected-zone", default=EAST5_ZONE, help="Expected Iris parent and child TPU zone.")
    parser.add_argument(
        "--expected-bucket-prefix",
        default=EAST5_BUCKET_PREFIX,
        help="Expected Marin GCS bucket prefix for all Marin GCS paths in the command.",
    )
    parser.add_argument(
        "--allow-region-only-parent",
        action="store_true",
        help=(
            "Allow the Iris CPU parent to omit --zone when --region matches the data region. "
            "Child TPU placement and Marin GCS paths are still validated."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    try:
        result = validate_regional_iris_command(
            _read_command(args).strip(),
            expected_region=args.expected_region,
            expected_zone=args.expected_zone,
            expected_bucket_prefix=args.expected_bucket_prefix,
            allow_region_only_parent=args.allow_region_only_parent,
        )
    except ValueError as exc:
        result = LaunchSafetyResult(
            ok=False,
            errors=[str(exc)],
            warnings=[],
            parent_regions=[],
            parent_zone=None,
            allows_region_only_parent=args.allow_region_only_parent,
            child_tpu_regions=[],
            child_tpu_zone=None,
            marin_gcs_paths=[],
        )

    if args.json:
        print(json.dumps(result.to_dict(), sort_keys=True))
    elif result.ok:
        print("region-local launch safety check passed")
    else:
        print("region-local launch safety check failed", file=sys.stderr)
        for error in result.errors:
            print(f"- {error}", file=sys.stderr)

    raise SystemExit(0 if result.ok else 1)


if __name__ == "__main__":
    main()
