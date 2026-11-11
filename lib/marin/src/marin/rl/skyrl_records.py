# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Readers for native launch files and FineStore attempt records."""

from typing import Any, TypedDict

import yaml
from finestore.rl.run_records import (
    AttemptRecord,
    LaunchReceipt,
    read_attempt,
    read_attempt_metrics,
    read_attempts,
)
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join

RESOLVED_LAUNCH_FILENAME = "resolved-launch.yaml"
TERMINAL_RECEIPT_FILENAME = "terminal.json"
EXPORTS_SUBDIRECTORY = "exports"
ENGINE_CONFIG_KEYS = (
    "backend",
    "model_dtype",
    "vllm_attention_backend",
    "enable_prefix_caching",
    "max_num_seqs",
    "max_num_batched_tokens",
    "enable_chunked_prefill",
    "enforce_eager",
    "engine_init_kwargs",
)


class AttemptTracking(TypedDict, total=False):
    run_id: str
    attempt_id: str
    iris_job_id: str | None
    coordinator_iris_job_id: str | None
    started_at: str
    engine: dict[str, Any]
    wandb_url: str | None


def read_skyrl_launch_config(run_output: str) -> dict:
    """Read the native resolved-launch envelope's compiled configuration."""
    fs, key = filesystem_for(prefix_join(run_output, RESOLVED_LAUNCH_FILENAME))
    return yaml.safe_load(fs.cat(key))["config"]


def read_stored_skyrl_receipt(run_output: str) -> AttemptRecord | None:
    """Read the current native attempt, including its receipt and startup facts."""
    terminal = None
    fs, key = filesystem_for(prefix_join(run_output, TERMINAL_RECEIPT_FILENAME))
    if fs.exists(key):
        terminal = LaunchReceipt.model_validate_json(fs.cat(key))
    try:
        config = read_skyrl_launch_config(run_output)
    except FileNotFoundError:
        if terminal is None:
            return None
        config = terminal.config
    attempt_id = config["run"]["attempt_id"]
    record = read_attempt(config["artifacts"]["export_root"], attempt_id)
    if (
        terminal is not None
        and terminal.config["run"]["id"] == config["run"]["id"]
        and terminal.config["run"]["attempt_id"] == attempt_id
    ):
        if record is None:
            record = AttemptRecord(
                run_id=config["run"]["id"], attempt_id=attempt_id, config=terminal.config, result=terminal.result
            )
        elif record.result is None:
            record = record.model_copy(update={"result": terminal.result})
    if record is None:
        return None
    if record.run_id != config["run"]["id"]:
        raise ValueError(f"Stored attempt {attempt_id} belongs to a different native run")
    return record.model_copy(update={"config": record.config if record.config is not None else config})


def read_stored_skyrl_attempts(run_output: str) -> tuple[str | None, dict[str, AttemptRecord]]:
    """Read the current ID and native training attempts without querying Iris."""
    current = read_stored_skyrl_receipt(run_output)
    if current is None:
        try:
            config = read_skyrl_launch_config(run_output)
        except FileNotFoundError:
            return None, {}
    else:
        config = current.native.config
    attempts = {
        attempt_id: record
        for attempt_id, record in read_attempts(config["artifacts"]["export_root"]).items()
        if record.config is not None
        and record.run_id == config["run"]["id"]
        and record.native.config["run"]["mode"] != "checkpoint_export"
    }
    if current is not None:
        attempts[current.attempt_id] = current
    return config["run"]["attempt_id"], attempts


def read_skyrl_metrics(run_output: str, attempt_id: str | None = None) -> dict[int, dict]:
    """Read one native attempt's metrics and bounded checkpoint ancestors."""
    config = read_skyrl_launch_config(run_output)
    return read_attempt_metrics(config["artifacts"]["export_root"], attempt_id or config["run"]["attempt_id"])


def attempt_tracking(record: AttemptRecord) -> AttemptTracking:
    """Describe the native attempt's startup, engine configuration and W&B run."""
    if record.config is None or record.started_at is None:
        return {}
    config = record.config
    generator = config["skyrl"]["generator"]
    engine = {key: generator.get(key) for key in ENGINE_CONFIG_KEYS}
    engine.update(
        batch_invariant=config["skyrl"]["trainer"]["algorithm"]["batch_invariant"],
        vllm_env={key: value for key, value in config["runtime"]["task_env"].items() if key.startswith("VLLM_")},
        vllm_version=record.vllm_version,
    )
    entity = config["iris"]["wandb_entity"]
    trainer = config["skyrl"]["trainer"]
    wandb_url = (
        f"https://wandb.ai/{entity}/{trainer['project_name']}/runs/{record.attempt_id}"
        if entity and "wandb" in trainer["logger"]
        else None
    )
    return {
        "run_id": record.run_id,
        "attempt_id": record.attempt_id,
        "iris_job_id": record.native.result.iris_job_id,
        "coordinator_iris_job_id": config["run"].get("coordinator_iris_job_id"),
        "started_at": record.started_at,
        "engine": engine,
        "wandb_url": wandb_url,
    }
