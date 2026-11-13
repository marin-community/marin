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
from pydantic import AliasPath, BaseModel, ConfigDict, Field, ValidationError, field_validator
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


class RecordedSource(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    uri: str
    identity: str
    local_path: str
    relative_path: str


class RecordedEvaluation(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    sampling_params: dict[str, Any]
    n_samples_per_prompt: int
    max_generate_length: int = Field(validation_alias=AliasPath("sampling_params", "max_generate_length"))


class RecordedCallback(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    type: str
    additional_evaluations: dict[str, RecordedEvaluation] | None = None


class RecordedLaunch(BaseModel):
    """Fields consumed by native attempt, evaluation and progress readers."""

    model_config = ConfigDict(extra="ignore", strict=True)
    run_id: str = Field(validation_alias=AliasPath("run", "id"))
    attempt_id: str = Field(validation_alias=AliasPath("run", "attempt_id"))
    run_mode: str = Field(validation_alias=AliasPath("run", "mode"))
    coordinator_iris_job_id: str | None = Field(validation_alias=AliasPath("run", "coordinator_iris_job_id"))
    export_root: str = Field(validation_alias=AliasPath("artifacts", "export_root"))
    wandb_entity: str | None = Field(validation_alias=AliasPath("iris", "wandb_entity"))
    num_nodes: int = Field(validation_alias=AliasPath("iris", "allocation", "num_nodes"))
    gpus_per_node: int = Field(validation_alias=AliasPath("iris", "allocation", "gpus_per_node"))
    training_type: str | None = Field(validation_alias=AliasPath("runtime", "training_type"))
    task_env: dict[str, str] = Field(validation_alias=AliasPath("runtime", "task_env"))
    train_data: list[RecordedSource] = Field(validation_alias=AliasPath("inputs", "train_data"))
    validation_data: list[RecordedSource] = Field(validation_alias=AliasPath("inputs", "validation_data"))
    evaluation_data: dict[str, list[RecordedSource]] = Field(
        default_factory=dict, validation_alias=AliasPath("inputs", "evaluation_data")
    )
    model_uri: str = Field(validation_alias=AliasPath("inputs", "model", "uri"))
    model_identity: str = Field(validation_alias=AliasPath("inputs", "model", "identity"))
    model_local_path: str = Field(validation_alias=AliasPath("inputs", "model", "local_path"))
    max_steps: int | None = Field(validation_alias=AliasPath("skyrl", "trainer", "max_steps"))
    max_prompt_length: int = Field(validation_alias=AliasPath("skyrl", "trainer", "max_prompt_length"))
    project_name: str = Field(validation_alias=AliasPath("skyrl", "trainer", "project_name"))
    logger: list[str] = Field(validation_alias=AliasPath("skyrl", "trainer", "logger"))
    callbacks: list[RecordedCallback] | None = Field(validation_alias=AliasPath("skyrl", "trainer", "callbacks"))
    batch_invariant: bool = Field(validation_alias=AliasPath("skyrl", "trainer", "algorithm", "batch_invariant"))
    max_turns: int = Field(validation_alias=AliasPath("skyrl", "generator", "max_turns"))
    eval_sampling_params: dict[str, Any] = Field(
        validation_alias=AliasPath("skyrl", "generator", "eval_sampling_params")
    )
    eval_max_generate_length: int = Field(
        validation_alias=AliasPath("skyrl", "generator", "eval_sampling_params", "max_generate_length")
    )
    eval_n_samples_per_prompt: int = Field(validation_alias=AliasPath("skyrl", "generator", "eval_n_samples_per_prompt"))
    chat_template: dict[str, Any] | None = Field(validation_alias=AliasPath("skyrl", "generator", "chat_template"))
    backend: str = Field(validation_alias=AliasPath("skyrl", "generator", "backend"))
    model_dtype: str = Field(validation_alias=AliasPath("skyrl", "generator", "model_dtype"))
    vllm_attention_backend: str | None = Field(
        validation_alias=AliasPath("skyrl", "generator", "vllm_attention_backend")
    )
    enable_prefix_caching: bool = Field(validation_alias=AliasPath("skyrl", "generator", "enable_prefix_caching"))
    max_num_seqs: int = Field(validation_alias=AliasPath("skyrl", "generator", "max_num_seqs"))
    max_num_batched_tokens: int | None = Field(
        validation_alias=AliasPath("skyrl", "generator", "max_num_batched_tokens")
    )
    enable_chunked_prefill: bool | None = Field(
        validation_alias=AliasPath("skyrl", "generator", "enable_chunked_prefill")
    )
    enforce_eager: bool = Field(validation_alias=AliasPath("skyrl", "generator", "enforce_eager"))
    engine_init_kwargs: dict[str, Any] = Field(validation_alias=AliasPath("skyrl", "generator", "engine_init_kwargs"))

    @field_validator("logger", mode="before")
    @classmethod
    def logger_collection(cls, value):
        return [value] if isinstance(value, str) else value

    @classmethod
    def from_document(cls, document: dict, *, attempt: str) -> "RecordedLaunch":
        try:
            return cls.model_validate(document)
        except ValidationError as error:
            fields = "; ".join(
                f"{'.'.join(map(str, item['loc']))}: {item['type']}"
                for item in error.errors(include_input=False, include_context=False, include_url=False)
            )
            raise ValueError(f"Attempt {attempt} has malformed recorded configuration: {fields}") from None


def recorded_launch(record: AttemptRecord) -> RecordedLaunch:
    if record.config is None:
        raise ValueError(f"Attempt {record.attempt_id} has no native launch configuration")
    return RecordedLaunch.from_document(record.config, attempt=record.attempt_id)


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


def attempt_tracking(record: AttemptRecord, config: RecordedLaunch) -> AttemptTracking:
    """Describe the native attempt's startup, engine configuration and W&B run."""
    if record.started_at is None:
        return {}
    engine = {key: getattr(config, key) for key in ENGINE_CONFIG_KEYS}
    engine.update(
        batch_invariant=config.batch_invariant,
        vllm_env={key: value for key, value in config.task_env.items() if key.startswith("VLLM_")},
        vllm_version=record.vllm_version,
    )
    wandb_url = (
        f"https://wandb.ai/{config.wandb_entity}/{config.project_name}/runs/{record.attempt_id}"
        if config.wandb_entity and "wandb" in config.logger
        else None
    )
    return {
        "run_id": record.run_id,
        "attempt_id": record.attempt_id,
        "iris_job_id": record.native.result.iris_job_id,
        "coordinator_iris_job_id": config.coordinator_iris_job_id,
        "started_at": record.started_at,
        "engine": engine,
        "wandb_url": wandb_url,
    }
