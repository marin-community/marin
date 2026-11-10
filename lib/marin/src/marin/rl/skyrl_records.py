# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Readers for saved SkyRL launch receipts, tracking and metrics."""

import json
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import yaml
from iris.cli.connect import connect_controller, rpc_client
from iris.cluster.client.job_info import get_job_info
from iris.cluster.types import JobName, is_job_finished
from iris.rpc import controller_pb2, job_pb2
from pydantic import BaseModel
from rigging.config_discovery import find_project_root
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join

IRIS_HUB_CLUSTER_CONFIG = "lib/iris/config/marin.yaml"
RESOLVED_LAUNCH_FILENAME = "resolved-launch.yaml"
TERMINAL_RECEIPT_FILENAME = "terminal.json"
TRACKING_FILENAME = "tracking.json"
ATTEMPT_RECEIPT_FILENAME = "receipt.json"
METRICS_PATTERN = "exports/metrics/step-*.json"
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


class SkyRLReceiptLookupError(ValueError):
    """The recorded Iris job cannot be identified uniquely."""


class SkyRLLaunchReceipt(BaseModel):
    config: dict
    result: SkyRLLaunchResponse


def named_evaluation_profiles(config: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        name: profile
        for callback in config.get("trainer", {}).get("callbacks") or []
        if callback.get("type") == "evaluation"
        for name, profile in (callback.get("additional_evaluations") or {}).items()
    }


def read_skyrl_launch_config(run_output: str) -> dict:
    """Read the native resolved-launch envelope's compiled configuration."""
    fs, key = filesystem_for(prefix_join(run_output, RESOLVED_LAUNCH_FILENAME))
    return yaml.safe_load(fs.cat(key))["config"]


def read_skyrl_attempt_tracking(run_output: str, attempt_id: str) -> dict:
    """Read startup tracking with engine settings and the native W&B URL."""
    fs, key = filesystem_for(prefix_join(run_output, f"exports/attempts/{attempt_id}/{TRACKING_FILENAME}"))
    if not fs.exists(key):
        return {}
    record = json.loads(fs.cat(key))
    _, attempts = read_stored_skyrl_attempts(run_output)
    config = attempts[attempt_id].config if attempt_id in attempts else read_skyrl_launch_config(run_output)
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
    for uri in (
        prefix_join(run_output, f"exports/attempts/{attempt_id}/{ATTEMPT_RECEIPT_FILENAME}"),
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
        prefix_join(run_output, f"exports/attempts/*/{ATTEMPT_RECEIPT_FILENAME}"),
        prefix_join(config["artifacts"]["attempts_root"], "*.json"),
    ):
        fs, key = filesystem_for(pattern)
        for path in fs.glob(key):
            saved = SkyRLLaunchReceipt.model_validate_json(fs.cat(path))
            attempts.setdefault(saved.config["run"]["attempt_id"], saved)
    return current, attempts


def read_skyrl_metrics(run_output: str) -> dict[int, dict]:
    """Read each saved step's metrics from a native run."""
    fs, key = filesystem_for(prefix_join(run_output, METRICS_PATTERN))
    rows = [json.loads(fs.cat(path)) for path in sorted(fs.glob(key))]
    return {int(row["step"]): row for row in rows}


def read_skyrl_launch_receipt(run_output: str) -> SkyRLLaunchReceipt | None:
    """Read a launch's terminal or attempt receipt, or build one from its Iris job's state.

    Returns None when the output has no resolved launch. Without a receipt file, the job state comes from the Iris
    controller and may be running. Raises SkyRLReceiptLookupError when the job name matches no job or several.
    """
    receipt, config = _stored_skyrl_receipt(run_output)
    if receipt is not None:
        return receipt
    if config is None:
        return None
    name = config["iris"]["job_name"]
    info = get_job_info()
    with ExitStack() as stack:
        if info is not None and info.controller_address:
            address, credentials = info.controller_address, info.credentials
        else:
            root = find_project_root(Path(__file__))
            if root is None:
                raise RuntimeError("Iris receipt discovery requires a Marin checkout")
            endpoint = stack.enter_context(connect_controller(config_file=root / IRIS_HUB_CLUSTER_CONFIG))
            address, credentials = endpoint.url, endpoint.credentials
        client = stack.enter_context(rpc_client(address, credentials))
        response = client.list_jobs(
            controller_pb2.Controller.ListJobsRequest(
                query=controller_pb2.Controller.JobQuery(name_filter=name, limit=500)
            )
        )
        jobs = response.jobs
    matches = [job for job in jobs if str(JobName.from_wire(job.job_id)).rsplit("/", 1)[-1] == name]
    if len(matches) != 1:
        raise SkyRLReceiptLookupError(f"cannot identify the Iris job {name}; its state is unavailable")
    job = matches[0]
    state = "submitted"
    if is_job_finished(job.state):
        state = "succeeded" if job.state == job_pb2.JOB_STATE_SUCCEEDED else "failed"
    return SkyRLLaunchReceipt.model_validate(
        {
            "config": config,
            "result": {
                "state": state,
                "iris_job_id": str(JobName.from_wire(job.job_id)),
                "model": None,
            },
        }
    )
