# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "fsspec>=2025.3.0",
#   "gcsfs>=2025.3.0",
# ]
# ///

"""Classify frozen review-v9 training roots before launch or retry."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import urllib.parse
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import fsspec
from rigging.filesystem import prefix_join

from experiments.domain_phase_mix import launch_starcoder_wsd80_gradient_conflict_full as full

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/starcoder_wsd80_gradient_conflict_output_inventory_20260811"
OPERATIONAL_ROOT_FILES = frozenset({".executor_info", ".executor_status", ".executor_status.lock"})
CHECKPOINT_METADATA_PATTERN = re.compile(r"^checkpoints/step-(\d+)/metadata\.json$")


@dataclass(frozen=True)
class OutputInventory:
    """Occupancy summary for the exact frozen trajectory roots."""

    expected_root_count: int
    empty_root_count: int
    bookkeeping_root_count: int
    resumable_root_count: int
    completed_root_count: int
    partial_root_count: int
    unexpected_root_count: int
    nonempty_expected_roots: tuple[str, ...]
    bookkeeping_expected_roots: tuple[str, ...]
    resumable_expected_roots: tuple[str, ...]
    completed_expected_roots: tuple[str, ...]
    partial_expected_roots: tuple[str, ...]
    unexpected_roots: tuple[str, ...]


class RootState(StrEnum):
    """Fail-closed lifecycle classification for an exact owned root."""

    RESUMABLE = "resumable"
    COMPLETED = "completed"
    INVALID = "invalid"


def classify_objects(
    object_paths: tuple[str, ...],
    *,
    expected_terminal_steps: dict[str, int],
    version: str,
    owned_expected_roots: frozenset[str] = frozenset(),
    resumable_expected_roots: frozenset[str] = frozenset(),
    completed_expected_roots: frozenset[str] = frozenset(),
    invalid_expected_roots: frozenset[str] = frozenset(),
    additional_unexpected_roots: frozenset[str] = frozenset(),
) -> OutputInventory:
    """Classify bounded root evidence without traversing checkpoint tensor payloads."""
    objects_by_expected_root: dict[str, set[str]] = {}
    unexpected = set(additional_unexpected_roots)
    for object_path in object_paths:
        parts = tuple(part for part in object_path.strip("/").split("/") if part)
        if len(parts) < 2:
            unexpected.add("/".join(parts) or "<trajectory-parent>")
            continue
        trajectory_id, observed_version = parts[:2]
        root = f"{trajectory_id}/{observed_version}"
        if trajectory_id in expected_terminal_steps and observed_version == version:
            objects_by_expected_root.setdefault(root, set()).add("/".join(parts[2:]))
        else:
            unexpected.add(root)

    bookkeeping: set[str] = set()
    resumable: set[str] = set()
    completed: set[str] = set()
    partial: set[str] = set()
    for root, paths in objects_by_expected_root.items():
        if root in invalid_expected_roots or root not in owned_expected_roots:
            partial.add(root)
        elif root in completed_expected_roots:
            completed.add(root)
        elif root in resumable_expected_roots:
            resumable.add(root)
        elif paths and paths.issubset(OPERATIONAL_ROOT_FILES):
            bookkeeping.add(root)
        else:
            partial.add(root)

    occupied_expected = bookkeeping | resumable | completed | partial
    expected_count = len(expected_terminal_steps)
    return OutputInventory(
        expected_root_count=expected_count,
        empty_root_count=expected_count - len(occupied_expected),
        bookkeeping_root_count=len(bookkeeping),
        resumable_root_count=len(resumable),
        completed_root_count=len(completed),
        partial_root_count=len(partial),
        unexpected_root_count=len(unexpected),
        nonempty_expected_roots=tuple(sorted(occupied_expected)),
        bookkeeping_expected_roots=tuple(sorted(bookkeeping)),
        resumable_expected_roots=tuple(sorted(resumable)),
        completed_expected_roots=tuple(sorted(completed)),
        partial_expected_roots=tuple(sorted(partial)),
        unexpected_roots=tuple(sorted(unexpected)),
    )


def _relative_path(path: str, parent: str) -> str:
    prefix = parent.rstrip("/") + "/"
    return path.removeprefix(prefix) if path.startswith(prefix) else path


def _root_from_relative(path: str) -> str | None:
    parts = tuple(part for part in path.strip("/").split("/") if part)
    if len(parts) < 2:
        return None
    return "/".join(parts[:2])


def _load_json(fs: Any, path: str) -> dict[str, Any]:
    with fs.open(path, "r") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return value


def _read_text(fs: Any, path: str) -> str:
    with fs.open(path, "r") as handle:
        return handle.read().strip()


def _artifact_record_matches(artifact: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    """Require the exact frozen identity plus nonempty structured execution evidence."""
    observed = {key: artifact.get(key) for key in expected}
    return (
        observed == expected
        and isinstance(artifact.get("config"), dict)
        and isinstance(artifact.get("provenance"), dict)
    )


def _root_state(
    *,
    terminal_step: int,
    permanent_steps: set[int],
    temporary_steps: set[int],
    artifact_valid: bool,
    status_success: bool,
) -> RootState | None:
    """Classify durable evidence before StepRunner can treat SUCCESS as terminal."""
    all_steps = permanent_steps | temporary_steps
    if any(step > terminal_step for step in all_steps):
        return RootState.INVALID
    if status_success:
        if terminal_step in permanent_steps and artifact_valid:
            return RootState.COMPLETED
        return RootState.INVALID
    if artifact_valid:
        return RootState.INVALID
    if all_steps:
        return RootState.RESUMABLE
    return None


def _metadata_steps(
    fs: Any,
    metadata_paths: tuple[str, ...],
    *,
    parent: str,
    expected_roots: frozenset[str],
) -> tuple[dict[str, set[int]], set[str], set[str]]:
    steps_by_root: dict[str, set[int]] = {}
    invalid_roots: set[str] = set()
    unexpected_roots: set[str] = set()
    for metadata_path in metadata_paths:
        relative = _relative_path(metadata_path, parent)
        root = _root_from_relative(relative)
        if root is None or root not in expected_roots:
            unexpected_roots.add(f"checkpoint:{root or relative}")
            continue
        checkpoint_relative = relative.split("/", 2)[2]
        match = CHECKPOINT_METADATA_PATTERN.fullmatch(checkpoint_relative)
        if match is None:
            invalid_roots.add(root)
            continue
        path_step = int(match.group(1))
        try:
            metadata = _load_json(fs, metadata_path)
            metadata_step = int(metadata["step"])
            datetime.datetime.fromisoformat(str(metadata["timestamp"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
            invalid_roots.add(root)
            continue
        if metadata_step != path_step:
            invalid_roots.add(root)
            continue
        steps_by_root.setdefault(root, set()).add(metadata_step)
    return steps_by_root, invalid_roots, unexpected_roots


def _expected_root_owners(marin_prefix: str) -> Mapping[str, Mapping[str, Any]]:
    os.environ["MARIN_PREFIX"] = marin_prefix
    trajectories, steps = full.build_training_steps(
        marin_prefix=marin_prefix,
        tpu_type=full.base.DEFAULT_TPU_TYPE,
        tpu_region=full.base.DEFAULT_TPU_REGION,
        tpu_zone=full.base.DEFAULT_TPU_ZONE,
    )
    return full.expected_output_owners(trajectories, steps, marin_prefix=marin_prefix)


def audit_outputs(
    marin_prefix: str,
    *,
    expected_root_owners: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[OutputInventory, str, str]:
    """Audit shallow root state plus parseable permanent and temporary checkpoints."""
    _, trajectories, _ = full.load_design()
    expected_terminal_steps = {row.trajectory_id: row.total_steps - 1 for row in trajectories}
    expected_roots = frozenset(f"{trajectory_id}/{full.VERSION}" for trajectory_id in expected_terminal_steps)
    owners = expected_root_owners or _expected_root_owners(marin_prefix)
    if frozenset(owners) != expected_roots:
        raise ValueError("Expected executor-owner map does not cover the frozen trajectory roots")

    trajectory_parent = prefix_join(marin_prefix, f"checkpoints/{full.NAME}/trajectories")
    parsed_parent = urllib.parse.urlparse(trajectory_parent)
    parent_component = f"{parsed_parent.netloc}{parsed_parent.path}".strip("/")
    temporary_parent = prefix_join(marin_prefix, f"tmp/ttl=14d/checkpoints-temp/{parent_component}")
    fs, stripped_parent = fsspec.core.url_to_fs(trajectory_parent)
    _, stripped_temporary_parent = fsspec.core.url_to_fs(temporary_parent)
    fs.invalidate_cache(stripped_parent)
    fs.invalidate_cache(stripped_temporary_parent)

    root_entries: tuple[str, ...] = ()
    shallow_paths: tuple[str, ...] = ()
    permanent_metadata_paths: tuple[str, ...] = ()
    if fs.exists(stripped_parent):
        root_entries = tuple(str(path) for path in fs.find(stripped_parent, maxdepth=2, withdirs=True))
        shallow_paths = tuple(str(path) for path in fs.find(stripped_parent, maxdepth=3, withdirs=False))
        permanent_metadata_paths = tuple(
            str(path) for path in fs.glob(f"{stripped_parent}/*/*/checkpoints/*/metadata.json")
        )
    temporary_root_entries: tuple[str, ...] = ()
    temporary_metadata_paths: tuple[str, ...] = ()
    if fs.exists(stripped_temporary_parent):
        temporary_root_entries = tuple(
            str(path) for path in fs.find(stripped_temporary_parent, maxdepth=2, withdirs=True)
        )
        temporary_metadata_paths = tuple(
            str(path) for path in fs.glob(f"{stripped_temporary_parent}/*/*/checkpoints/*/metadata.json")
        )

    relative_objects = {
        _relative_path(path, stripped_parent)
        for path in (*shallow_paths, *permanent_metadata_paths)
        if path != stripped_parent
    }
    additional_unexpected: set[str] = set()
    observed_permanent_roots: set[str] = set()
    observed_temporary_roots: set[str] = set()
    for entry in root_entries:
        relative = _relative_path(entry, stripped_parent)
        parts = tuple(part for part in relative.strip("/").split("/") if part)
        if len(parts) != 2:
            continue
        root = "/".join(parts)
        if root in expected_roots:
            observed_permanent_roots.add(root)
        else:
            additional_unexpected.add(root)
    for entry in temporary_root_entries:
        relative = _relative_path(entry, stripped_temporary_parent)
        parts = tuple(part for part in relative.strip("/").split("/") if part)
        if len(parts) != 2:
            continue
        root = "/".join(parts)
        if root in expected_roots:
            observed_temporary_roots.add(root)
        else:
            additional_unexpected.add(f"temporary:{root}")

    permanent_steps, invalid_permanent, unexpected_permanent = _metadata_steps(
        fs,
        permanent_metadata_paths,
        parent=stripped_parent,
        expected_roots=expected_roots,
    )
    temporary_steps, invalid_temporary, unexpected_temporary = _metadata_steps(
        fs,
        temporary_metadata_paths,
        parent=stripped_temporary_parent,
        expected_roots=expected_roots,
    )
    additional_unexpected.update(unexpected_permanent)
    additional_unexpected.update(unexpected_temporary)

    owned_roots: set[str] = set()
    invalid_roots = set(invalid_permanent) | set(invalid_temporary)
    artifact_valid_roots: set[str] = set()
    status_success_roots: set[str] = set()
    paths_by_root: dict[str, set[str]] = {}
    for relative in relative_objects:
        root = _root_from_relative(relative)
        if root in expected_roots:
            paths_by_root.setdefault(root, set()).add(relative.split("/", 2)[2])
    for root in observed_permanent_roots:
        if root not in paths_by_root:
            relative_objects.add(f"{root}/__unclassified_state__")
            paths_by_root[root] = {"__unclassified_state__"}
    for root in observed_temporary_roots - observed_permanent_roots:
        relative_objects.add(f"{root}/__temporary_state__")
        paths_by_root[root] = {"__temporary_state__"}

    for root, paths in paths_by_root.items():
        expected_owner = owners[root]
        if ".executor_info" in paths:
            try:
                executor_info = _load_json(fs, f"{stripped_parent}/{root}/.executor_info")
            except (ValueError, json.JSONDecodeError, OSError):
                invalid_roots.add(root)
            else:
                expected_executor_info = expected_owner["executor_info"]
                observed_executor_info = {key: executor_info.get(key) for key in expected_executor_info}
                if observed_executor_info == expected_executor_info:
                    owned_roots.add(root)
                else:
                    invalid_roots.add(root)
        if ".artifact.json" in paths:
            try:
                artifact = _load_json(fs, f"{stripped_parent}/{root}/.artifact.json")
            except (ValueError, json.JSONDecodeError, OSError):
                invalid_roots.add(root)
            else:
                expected_artifact = expected_owner["artifact_record"]
                if _artifact_record_matches(artifact, expected_artifact):
                    artifact_valid_roots.add(root)
                else:
                    invalid_roots.add(root)
        if ".executor_status" in paths:
            try:
                if _read_text(fs, f"{stripped_parent}/{root}/.executor_status") == "SUCCESS":
                    status_success_roots.add(root)
            except OSError:
                invalid_roots.add(root)

    resumable_roots: set[str] = set()
    completed_roots: set[str] = set()
    for root in expected_roots:
        trajectory_id = root.split("/", 1)[0]
        state = _root_state(
            terminal_step=expected_terminal_steps[trajectory_id],
            permanent_steps=permanent_steps.get(root, set()),
            temporary_steps=temporary_steps.get(root, set()),
            artifact_valid=root in artifact_valid_roots,
            status_success=root in status_success_roots,
        )
        if state is RootState.INVALID:
            invalid_roots.add(root)
        elif state is RootState.COMPLETED:
            completed_roots.add(root)
        elif state is RootState.RESUMABLE:
            resumable_roots.add(root)

    inventory = classify_objects(
        tuple(sorted(relative_objects)),
        expected_terminal_steps=expected_terminal_steps,
        version=full.VERSION,
        owned_expected_roots=frozenset(owned_roots),
        resumable_expected_roots=frozenset(resumable_roots),
        completed_expected_roots=frozenset(completed_roots),
        invalid_expected_roots=frozenset(invalid_roots),
        additional_unexpected_roots=frozenset(additional_unexpected),
    )
    return inventory, trajectory_parent, temporary_parent


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marin-prefix", default="gs://marin-us-central1")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.marin_prefix != "gs://marin-us-central1":
        raise ValueError("Historical StarCoder output inventory must remain central1-local")
    inventory, trajectory_parent, temporary_parent = audit_outputs(args.marin_prefix)
    report = {
        "report_version": "2026-08-13-review-v9-output-inventory-v3",
        "scope": "launch_or_retry_owned_checkpoint_occupancy",
        "trajectory_parent": trajectory_parent,
        "temporary_checkpoint_parent": temporary_parent,
        "design_version": full.EXPECTED_DESIGN_VERSION,
        "design_sha256": full.EXPECTED_DESIGN_SHA256,
        "training_name": full.NAME,
        "training_version": full.VERSION,
        **asdict(inventory),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "output_inventory.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    report_sha256 = _sha256(output_path)
    print(json.dumps({"report": str(output_path), "sha256": report_sha256, **asdict(inventory)}))
    if inventory.partial_root_count or inventory.unexpected_root_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
