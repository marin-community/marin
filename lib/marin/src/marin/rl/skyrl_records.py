# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Readers for saved SkyRL launch receipts, tracking and metrics."""

import json

import yaml
from pydantic import BaseModel
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join

RESOLVED_LAUNCH_FILENAME = "resolved-launch.yaml"
TERMINAL_RECEIPT_FILENAME = "terminal.json"
TRACKING_FILENAME = "tracking.json"
ATTEMPT_RECEIPT_FILENAME = "receipt.json"
EXPORTS_SUBDIRECTORY = "exports"
ATTEMPTS_SUBDIRECTORY = "attempts"
METRICS_PATTERN = "metrics/step-*.json"
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


class _SkyRLTerminalModel(BaseModel):
    policy_export_uri: str
    global_step: int
    tokenizer_uri: str
    tokenizer_revision: str
    checkpoint_root: str
    terminal_manifest_uri: str


class SkyRLLaunchResponse(BaseModel):
    state: str
    iris_job_id: str | None = None
    failure: str | None = None
    provenance: dict | None = None
    started_at: str | None = None
    finished_at: str | None = None
    model: _SkyRLTerminalModel | None = None


class SkyRLLaunchReceipt(BaseModel):
    config: dict
    result: SkyRLLaunchResponse


def read_skyrl_launch_config(run_output: str) -> dict:
    """Read the native resolved-launch envelope's compiled configuration."""
    fs, key = filesystem_for(prefix_join(run_output, RESOLVED_LAUNCH_FILENAME))
    return yaml.safe_load(fs.cat(key))["config"]


def read_skyrl_attempt_tracking(run_output: str, attempt_id: str) -> dict:
    """Read startup tracking with engine settings and the native W&B URL."""
    _, config = _stored_skyrl_receipt(run_output)
    if config is None:
        return {}
    if config["run"]["attempt_id"] != attempt_id:
        _, attempts = read_stored_skyrl_attempts(run_output)
        if attempt_id not in attempts:
            return {}
        config = attempts[attempt_id].config
    fs, key = filesystem_for(
        prefix_join(config["artifacts"]["export_root"], f"{ATTEMPTS_SUBDIRECTORY}/{attempt_id}/{TRACKING_FILENAME}")
    )
    if not fs.exists(key):
        return {}
    record = json.loads(fs.cat(key))
    generator = config["skyrl"]["generator"]
    engine = {key: generator.get(key) for key in ENGINE_CONFIG_KEYS}
    engine.update(
        batch_invariant=config["skyrl"]["trainer"]["algorithm"]["batch_invariant"],
        vllm_env={key: value for key, value in config["runtime"]["task_env"].items() if key.startswith("VLLM_")},
        vllm_version=record["engine"]["vllm_version"],
    )
    entity = config["iris"]["wandb_entity"]
    project = config["skyrl"]["trainer"]["project_name"]
    logger = config["skyrl"]["trainer"]["logger"]
    wandb_url = f"https://wandb.ai/{entity}/{project}/runs/{attempt_id}" if entity and "wandb" in logger else None
    return record | {"engine": engine, "wandb_url": wandb_url}


def _stored_skyrl_receipt(run_output: str) -> tuple[SkyRLLaunchReceipt | None, dict | None]:
    fs, key = filesystem_for(prefix_join(run_output, TERMINAL_RECEIPT_FILENAME))
    if fs.exists(key):
        receipt = SkyRLLaunchReceipt.model_validate_json(fs.cat(key))
        return receipt, receipt.config
    fs, key = filesystem_for(prefix_join(run_output, RESOLVED_LAUNCH_FILENAME))
    if not fs.exists(key):
        return None, None
    config = read_skyrl_launch_config(run_output)
    attempt_id = config["run"]["attempt_id"]
    # Pinned control receipts use the configured attempts_root.
    for uri in (
        prefix_join(
            config["artifacts"]["export_root"], f"{ATTEMPTS_SUBDIRECTORY}/{attempt_id}/{ATTEMPT_RECEIPT_FILENAME}"
        ),
        prefix_join(config["artifacts"]["attempts_root"], f"{attempt_id}.json"),
    ):
        fs, key = filesystem_for(uri)
        if fs.exists(key):
            return SkyRLLaunchReceipt.model_validate_json(fs.cat(key)), config
    return None, config


def read_stored_skyrl_receipt(run_output: str) -> SkyRLLaunchReceipt | None:
    """Read the current saved native receipt without querying Iris."""
    return _stored_skyrl_receipt(run_output)[0]


def read_stored_skyrl_attempts(run_output: str) -> tuple[str | None, dict[str, SkyRLLaunchReceipt]]:
    """Read the current attempt ID and saved native receipts without querying Iris."""
    receipt, config = _stored_skyrl_receipt(run_output)
    if config is None:
        return None, {}
    current = config["run"]["attempt_id"]
    attempts = {current: receipt} if receipt is not None else {}
    for pattern in (
        prefix_join(config["artifacts"]["export_root"], f"{ATTEMPTS_SUBDIRECTORY}/*/{ATTEMPT_RECEIPT_FILENAME}"),
        prefix_join(config["artifacts"]["attempts_root"], "*.json"),
    ):
        fs, key = filesystem_for(pattern)
        for path in fs.glob(key):
            saved = SkyRLLaunchReceipt.model_validate_json(fs.cat(path))
            attempts.setdefault(saved.config["run"]["attempt_id"], saved)
    return current, attempts


def read_skyrl_metrics(run_output: str) -> dict[int, dict]:
    """Read each saved step's metrics from a native run."""
    config = read_skyrl_launch_config(run_output)
    fs, key = filesystem_for(prefix_join(config["artifacts"]["export_root"], METRICS_PATTERN))
    rows = [json.loads(fs.cat(path)) for path in sorted(fs.glob(key))]
    return {int(row["step"]): row for row in rows}
