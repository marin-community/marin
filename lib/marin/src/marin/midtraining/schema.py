# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stable JSON schema for the per-cell run manifest.

This module exists so analysis scripts can read a manifest without
importing the launcher. The dataclasses in :mod:`marin.midtraining.spec`
are convenient when authoring a launch, but they pull in the rest of the
package; downstream consumers should depend on this schema instead.

Mirrors the artifact-as-contract pattern from
``experiments/downstream_scaling/evals/framework/schema.py``: filenames
are constants, the row shape is a ``TypedDict``, and the reader validates
shape before returning.
"""

import json
from pathlib import Path
from typing import Any, Required, TypedDict, TypeGuard

MANIFEST_FILENAME = "midtrain_manifest.json"
TRAIN_CONFIG_FILENAME = "train_lm_config.yaml"
LAUNCH_COMMAND_FILENAME = "launch_command.txt"
SCHEMA_VERSION = 2


class TokenizerRecord(TypedDict, total=False):
    key: Required[str]
    hf_repo: Required[str]
    revision: Required[str]
    bos_token_id: Required[int]
    eos_token_id: Required[int]
    vocab_size: Required[int]
    fingerprint: str | None


class CooldownStageRecord(TypedDict, total=False):
    source: Required[str]
    destination: Required[str]
    cross_region_copy: Required[bool]
    bytes_copied: Required[int]
    budget_gb: int
    reason: str


class RunManifestRow(TypedDict, total=False):
    schema_version: Required[int]
    written_at: Required[str]
    logical_cell_id: Required[str]
    attempt: Required[int]
    run_id: Required[str]
    mode: Required[str]
    output_path: Required[str]
    wandb_project: Required[str]
    wandb_entity: Required[str]
    base_flops_key: Required[str]
    tpu_type: Required[str]
    train_batch_size: Required[int]
    per_device_parallelism: Required[int]
    max_retries_failure: int
    max_task_failures: Required[int]
    data_manifest_uri: Required[str]
    data_manifest_fingerprint: Required[str]
    tokenizer: Required[TokenizerRecord]
    seq_len: Required[int]
    num_train_steps: Required[int]
    actual_tokens: Required[int]
    train_config_uri: Required[str]
    permanent_checkpoints_uri: Required[str]
    temp_checkpoints_uri: Required[str]
    init_checkpoint_uri: str | None
    staged_checkpoint_uri: str | None
    cooldown_stage_record: CooldownStageRecord | None
    preflight_failures: list[str]
    preflight_warnings: list[str]
    preflight_notes: list[str]
    extra_tags: list[str]
    status: Required[str]


VALID_MODES: frozenset[str] = frozenset({"cpt", "cooldown"})
VALID_STATUSES: frozenset[str] = frozenset(
    {"planned", "staged", "launched", "running", "succeeded", "failed", "stopped", "preempted"}
)


def is_run_manifest(row: Any) -> TypeGuard[RunManifestRow]:
    """Return ``True`` if ``row`` is shape-compatible with :class:`RunManifestRow`."""
    if not isinstance(row, dict):
        return False
    required_keys = (
        "schema_version",
        "written_at",
        "logical_cell_id",
        "attempt",
        "run_id",
        "mode",
        "output_path",
        "wandb_project",
        "wandb_entity",
        "base_flops_key",
        "tpu_type",
        "train_batch_size",
        "per_device_parallelism",
        "max_task_failures",
        "data_manifest_uri",
        "data_manifest_fingerprint",
        "tokenizer",
        "seq_len",
        "num_train_steps",
        "actual_tokens",
        "train_config_uri",
        "permanent_checkpoints_uri",
        "temp_checkpoints_uri",
        "status",
    )
    for key in required_keys:
        if key not in row:
            return False
    if row["mode"] not in VALID_MODES:
        return False
    if row["status"] not in VALID_STATUSES:
        return False
    if not isinstance(row["tokenizer"], dict):
        return False
    return True


def read_run_manifest(uri: str | Path) -> RunManifestRow:
    """Load a manifest from a local path or ``gs://`` URI and validate the shape."""
    uri_s = str(uri)
    if uri_s.startswith("gs://"):
        import fsspec

        with fsspec.open(uri_s, "r", encoding="utf-8") as f:
            raw = f.read()
    else:
        raw = Path(uri_s).read_text(encoding="utf-8")
    data = json.loads(raw)
    if not is_run_manifest(data):
        raise TypeError(f"Invalid run manifest at {uri_s}")
    return data


def write_run_manifest(row: RunManifestRow, uri: str | Path) -> None:
    """Write a manifest to a local path or ``gs://`` URI in canonical JSON form."""
    if not is_run_manifest(row):
        raise TypeError("Refusing to write malformed RunManifestRow")
    body = json.dumps(row, indent=2, sort_keys=True)
    uri_s = str(uri)
    if uri_s.startswith("gs://"):
        import fsspec

        with fsspec.open(uri_s, "w", encoding="utf-8") as f:
            f.write(body)
        return
    Path(uri_s).write_text(body, encoding="utf-8")


def manifest_uri(output_path: str) -> str:
    return f"{output_path.rstrip('/')}/{MANIFEST_FILENAME}"


def train_config_uri(output_path: str) -> str:
    return f"{output_path.rstrip('/')}/{TRAIN_CONFIG_FILENAME}"


def launch_command_uri(output_path: str) -> str:
    return f"{output_path.rstrip('/')}/{LAUNCH_COMMAND_FILENAME}"
