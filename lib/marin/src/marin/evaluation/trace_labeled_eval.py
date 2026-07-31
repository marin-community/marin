# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run Levanter labeled loss evaluation on OpenAI-style agent traces."""

import json
import logging
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

import fsspec
import jax
import jmp
import levanter
import levanter.tracker
from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning
from levanter.data.sharded_datasource import FirstRowsShardedDataSource, ShardedDataSource
from levanter.data.text.datasets import LmDatasetSourceConfigBase
from levanter.data.text.trace_chat import (
    TraceChatEvaluationFormat,
    build_trace_chat_dataset_cache,
    dataset_for_trace_chat_format,
)
from levanter.eval import LabeledEvaluator, eval_labeled_model
from levanter.models.llama import LlamaConfig
from levanter.models.lm_model import LmConfig
from levanter.tokenizers import load_tokenizer as load_marin_tokenizer
from levanter.trainer import TrainerConfig

from marin.evaluation.model_loading import load_eval_model
from marin.training.run_environment import extras_for_resources

logger = logging.getLogger(__name__)

T = TypeVar("T")
DEFAULT_TRACE_LABELED_EVAL_WANDB_PROJECT = "marin-analysis"
DEFAULT_TRACE_LABELED_EVAL_WANDB_TAGS = ("trace_labeled_eval",)
CORRECT_OUTCOME_LABEL = "CORRECT"
INCORRECT_OUTCOME_LABEL = "INCORRECT"
DEFAULT_OUTCOME_JUDGE_PROMPT = (
    "Given the trace above and any final patch shown, predict whether the attempted solution would pass the task.\n"
    "Answer exactly one token: CORRECT or INCORRECT."
)
RESULTS_FILENAME = "results.json"
TRACE_LABELED_EVAL_STATUS_PARTIAL = "partial"
TRACE_LABELED_EVAL_STATUS_COMPLETED = "completed"
TRACE_LABELED_EVAL_DATASET_STATUS_COMPLETED = "completed"
DEFAULT_DATASET_EVAL_MAX_ATTEMPTS = 3
DEFAULT_DATASET_EVAL_RETRY_INITIAL_DELAY = 30.0
DEFAULT_DATASET_EVAL_RETRY_MAX_DELAY = 300.0
DEFAULT_HF_HUB_TIMEOUT = "60"
DEFAULT_TRACE_LABELED_EVAL_TOKENIZER = "marin-community/marin-tokenizer"
HF_HUB_TIMEOUT_ENV_VARS = (
    "HF_HUB_ETAG_TIMEOUT",
    "HF_HUB_DOWNLOAD_TIMEOUT",
    "HF_HUB_REQUEST_TIMEOUT",
)


@dataclass(frozen=True)
class TraceRowAdapterConfig:
    """Normalize trace rows and add derived patch/outcome targets."""

    input_messages_field: str | None = None
    patch_field: str | None = None
    outcome_field: str | None = None
    task_id_field: str | None = None
    record_id_field: str | None = None
    patch_loss_tag: str = "patch"
    outcome_loss_tag: str = "outcome"
    patch_prefix: str = "Final Patch:\n"
    positive_outcome_label: str = CORRECT_OUTCOME_LABEL
    negative_outcome_label: str = INCORRECT_OUTCOME_LABEL
    outcome_prompt: str = DEFAULT_OUTCOME_JUDGE_PROMPT


@dataclass(frozen=True)
class TraceLabeledEvalDatasetConfig:
    """A trace dataset and label format to evaluate."""

    source: LmDatasetSourceConfigBase
    split: str
    trace_format: TraceChatEvaluationFormat = field(default_factory=TraceChatEvaluationFormat)
    max_examples: int | None = None
    row_adapter: TraceRowAdapterConfig | None = None
    row_prefix_fraction: float | None = None


@dataclass
class TraceLabeledEvalConfig:
    """Configuration for trace-labeled evaluation. Also serves as Levanter init config."""

    name: str | None = None
    checkpoint_path: str | None = None
    checkpoint_is_hf: bool = False
    tokenizer: str = DEFAULT_TRACE_LABELED_EVAL_TOKENIZER
    model: LmConfig = field(default_factory=LlamaConfig)
    datasets: dict[str, TraceLabeledEvalDatasetConfig] = field(default_factory=dict)
    trainer: TrainerConfig = field(default_factory=lambda: TrainerConfig(mp=jmp.get_policy("c=bf16")))
    max_eval_length: int = 4096
    output_path: str = ""
    dataset_eval_max_attempts: int = DEFAULT_DATASET_EVAL_MAX_ATTEMPTS
    dataset_eval_retry_initial_delay: float = DEFAULT_DATASET_EVAL_RETRY_INITIAL_DELAY
    dataset_eval_retry_max_delay: float = DEFAULT_DATASET_EVAL_RETRY_MAX_DELAY
    job_failure_max_retries: int = 1


@dataclass(frozen=True)
class TraceLabeledEvalOutput:
    """Output produced by a completed trace-labeled evaluation run."""

    results_path: str


@dataclass(frozen=True)
class TraceLabeledEvalOnPodConfig:
    """Wrapper config for running trace-labeled evaluation on a TPU pod via fray."""

    trace_labeled_eval_config: TraceLabeledEvalConfig
    resources: ResourceConfig


def _lookup_field(row: Mapping[str, Any], field_path: str) -> Any:
    if field_path in row:
        return row[field_path]

    value: Any = row
    for part in field_path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _normalize_text(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def _normalize_message_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return _normalize_text(value)
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, Mapping):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(_normalize_text(text))
                    continue
            parts.append(json.dumps(item, sort_keys=True))
        return "\n".join(part for part in parts if part)
    return json.dumps(value, sort_keys=True)


def _normalize_reasoning_content(raw_message: Mapping[str, Any]) -> str | None:
    for field_name in ("reasoning_content", "thinking", "thought", "reasoning"):
        value = raw_message.get(field_name)
        if value is not None:
            return _normalize_message_content(value)
    return None


def _canonical_trace_role(role: Any) -> str:
    role_text = str(role or "user").strip().lower()
    if role_text in {"ai", "agent", "assistant"}:
        return "assistant"
    if role_text in {"human", "user"}:
        return "user"
    if role_text in {"tool", "function", "ipython", "observation"}:
        return "tool"
    return role_text


def _normalize_trace_messages(raw_messages: Any) -> list[dict[str, Any]]:
    if isinstance(raw_messages, str):
        raw_messages = json.loads(raw_messages)
    if not isinstance(raw_messages, Sequence):
        raise ValueError(f"Expected trace messages to be a sequence, got {type(raw_messages).__name__}.")

    normalized: list[dict[str, Any]] = []
    for raw_message in raw_messages:
        if not isinstance(raw_message, Mapping):
            raise ValueError(f"Expected trace message to be a mapping, got {type(raw_message).__name__}.")

        role = _canonical_trace_role(raw_message.get("role"))
        content = raw_message.get("content")
        if content is None:
            content = raw_message.get("text")
        if content is None and role == "system":
            content = raw_message.get("system_prompt")
        if content is None and not raw_message.get("tool_calls"):
            raise ValueError("Trace message must include content, text, system_prompt, or tool_calls.")

        message: dict[str, Any] = {
            "role": role,
            "content": _normalize_message_content(content),
        }
        reasoning_content = _normalize_reasoning_content(raw_message)
        if reasoning_content:
            message["reasoning_content"] = reasoning_content
        for key in ("name", "tool_call_id", "tool_calls", "loss_tags"):
            value = raw_message.get(key)
            if value is not None:
                message[key] = value

        if message["content"] or message.get("tool_calls"):
            normalized.append(message)

    return normalized


def _prefixed_trace_messages(
    messages: Sequence[Mapping[str, Any]], prefix_fraction: float | None
) -> list[dict[str, Any]]:
    if prefix_fraction is None:
        return [dict(message) for message in messages]
    if not 0.0 <= prefix_fraction <= 1.0:
        raise ValueError(f"prefix_fraction must be in [0, 1], got {prefix_fraction!r}")
    if prefix_fraction <= 0.0:
        return []
    if prefix_fraction >= 1.0:
        return [dict(message) for message in messages]
    if not messages:
        return []

    keep_messages = max(1, math.ceil(len(messages) * prefix_fraction))
    return [dict(message) for message in messages[:keep_messages]]


def _row_identifier(row: Mapping[str, Any], field_path: str | None, *, default: str) -> str:
    if field_path is None:
        return default
    value = _lookup_field(row, field_path)
    if value is None:
        return default
    return str(value)


def _outcome_label(value: Any, row_adapter: TraceRowAdapterConfig) -> str:
    if isinstance(value, bool):
        return row_adapter.positive_outcome_label if value else row_adapter.negative_outcome_label
    if isinstance(value, int | float):
        return row_adapter.positive_outcome_label if value > 0 else row_adapter.negative_outcome_label
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "1.0", "true", "yes", "y", "correct", "resolved", "success", "succeeded"}:
            return row_adapter.positive_outcome_label
        if normalized in {"0", "0.0", "false", "no", "n", "incorrect", "unresolved", "failure", "failed"}:
            return row_adapter.negative_outcome_label

    raise ValueError(f"Cannot convert outcome value {value!r} to {CORRECT_OUTCOME_LABEL}/{INCORRECT_OUTCOME_LABEL}.")


def _adapt_trace_row(
    row: Mapping[str, Any],
    trace_format: TraceChatEvaluationFormat,
    row_adapter: TraceRowAdapterConfig,
    *,
    prefix_fraction: float | None = None,
) -> dict[str, Any]:
    input_messages_field = row_adapter.input_messages_field or trace_format.messages_field
    messages = _normalize_trace_messages(_lookup_field(row, input_messages_field))
    messages = _prefixed_trace_messages(messages, prefix_fraction)

    patch = _lookup_field(row, row_adapter.patch_field) if row_adapter.patch_field is not None else None
    patch_text = _normalize_message_content(patch)
    if patch_text:
        messages.append(
            {
                "role": "assistant",
                "content": f"{row_adapter.patch_prefix}{patch_text}",
                "loss_tags": [row_adapter.patch_loss_tag],
            }
        )

    adapted = dict(row)
    adapted["trace_task_id"] = _row_identifier(row, row_adapter.task_id_field, default="")
    adapted["trace_record_id"] = _row_identifier(row, row_adapter.record_id_field, default="")
    if row_adapter.outcome_field is not None:
        label = _outcome_label(_lookup_field(row, row_adapter.outcome_field), row_adapter)
        if row_adapter.outcome_prompt:
            messages.append({"role": "user", "content": row_adapter.outcome_prompt})
        messages.append(
            {
                "role": "assistant",
                "content": label,
                "loss_tags": [row_adapter.outcome_loss_tag],
            }
        )
        adapted["trace_outcome_label"] = label

    adapted[trace_format.messages_field] = messages
    return adapted


def _safe_name(name: str) -> str:
    return name.replace("/", "--").replace(" ", "_")


def _base_source_for_dataset(dataset_config: TraceLabeledEvalDatasetConfig) -> ShardedDataSource[dict]:
    source = dataset_config.source.get_shard_source(dataset_config.split)
    if source is None:
        raise ValueError(f"No shard source for split {dataset_config.split!r} in {dataset_config.source!r}")
    if dataset_config.max_examples is not None:
        return FirstRowsShardedDataSource(source, dataset_config.max_examples)
    return source


def _source_for_dataset(dataset_config: TraceLabeledEvalDatasetConfig) -> ShardedDataSource[dict]:
    limited_source = _base_source_for_dataset(dataset_config)
    if dataset_config.row_adapter is None:
        return limited_source
    return limited_source.map(
        lambda row: _adapt_trace_row(
            row,
            dataset_config.trace_format,
            dataset_config.row_adapter,
            prefix_fraction=dataset_config.row_prefix_fraction,
        )
    )


def _dataset_metadata(dataset_config: TraceLabeledEvalDatasetConfig) -> dict[str, object]:
    source = dataset_config.source
    metadata: dict[str, object] = {
        "source_type": type(source).__name__,
        "split": dataset_config.split,
        "max_examples": dataset_config.max_examples,
        "loss_tags": list(dataset_config.trace_format.loss_tags),
        "row_prefix_fraction": dataset_config.row_prefix_fraction,
    }
    if dataset_config.row_adapter is not None:
        metadata["row_adapter"] = {
            "input_messages_field": dataset_config.row_adapter.input_messages_field,
            "patch_field": dataset_config.row_adapter.patch_field,
            "outcome_field": dataset_config.row_adapter.outcome_field,
            "task_id_field": dataset_config.row_adapter.task_id_field,
            "record_id_field": dataset_config.row_adapter.record_id_field,
            "patch_loss_tag": dataset_config.row_adapter.patch_loss_tag,
            "outcome_loss_tag": dataset_config.row_adapter.outcome_loss_tag,
            "outcome_prompt": dataset_config.row_adapter.outcome_prompt,
        }
    for field_name in ("id", "name", "stream", "splits"):
        if hasattr(source, field_name):
            metadata[field_name] = getattr(source, field_name)
    return metadata


def _results_path(output_path: str) -> str:
    return os.path.join(output_path, RESULTS_FILENAME)


def _new_results(config: TraceLabeledEvalConfig) -> dict[str, object]:
    return {
        "checkpoint_path": config.checkpoint_path,
        "checkpoint_is_hf": config.checkpoint_is_hf,
        "tokenizer": config.tokenizer,
        "max_eval_length": config.max_eval_length,
        "status": TRACE_LABELED_EVAL_STATUS_PARTIAL,
        "completed_datasets": 0,
        "datasets": {},
    }


def _read_results(output_path: str) -> dict[str, object] | None:
    results_path = _results_path(output_path)
    fs, _, _ = fsspec.get_fs_token_paths(results_path)
    if not fs.exists(results_path):
        return None
    with fs.open(results_path) as f:
        results = json.load(f)
    if not isinstance(results, dict):
        raise ValueError(f"Expected {results_path} to contain a JSON object")
    return results


def _dataset_results(results: dict[str, object]) -> dict[str, object]:
    datasets = results.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError("Trace labeled eval results must contain a datasets object")
    return datasets


def _load_or_create_results(config: TraceLabeledEvalConfig) -> dict[str, object]:
    results = _read_results(config.output_path)
    if results is None:
        return _new_results(config)

    expected = _new_results(config)
    for field_name in ("checkpoint_path", "checkpoint_is_hf", "tokenizer", "max_eval_length"):
        if field_name in results and results[field_name] != expected[field_name]:
            raise ValueError(
                f"Existing trace labeled eval results at {config.output_path} have {field_name}="
                f"{results[field_name]!r}, expected {expected[field_name]!r}"
            )

    results.setdefault("status", TRACE_LABELED_EVAL_STATUS_PARTIAL)
    results.setdefault("completed_datasets", 0)
    results.setdefault("datasets", {})
    _dataset_results(results)
    return results


def _is_completed_dataset_result(dataset_result: object) -> bool:
    if not isinstance(dataset_result, Mapping):
        return False
    return isinstance(dataset_result.get("metrics"), Mapping)


def _completed_dataset_count(results: dict[str, object]) -> int:
    return sum(
        1 for dataset_result in _dataset_results(results).values() if _is_completed_dataset_result(dataset_result)
    )


def _completed_dataset_metrics(results: dict[str, object]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for dataset_result in _dataset_results(results).values():
        if not isinstance(dataset_result, Mapping):
            continue
        dataset_metrics = dataset_result.get("metrics")
        if not isinstance(dataset_metrics, Mapping):
            continue
        for metric_name, metric_value in dataset_metrics.items():
            if not isinstance(metric_name, str):
                raise ValueError(f"Trace labeled eval metric names must be strings, got {metric_name!r}")
            if not isinstance(metric_value, int | float):
                raise ValueError(f"Trace labeled eval metric {metric_name!r} must be numeric, got {metric_value!r}")
            metrics[metric_name] = float(metric_value)
    return metrics


def _record_dataset_result(
    results: dict[str, object],
    dataset_name: str,
    dataset_config: TraceLabeledEvalDatasetConfig,
    metrics: Mapping[str, float],
) -> None:
    _dataset_results(results)[dataset_name] = {
        "status": TRACE_LABELED_EVAL_DATASET_STATUS_COMPLETED,
        "metadata": _dataset_metadata(dataset_config),
        "metrics": dict(metrics),
    }
    results["status"] = TRACE_LABELED_EVAL_STATUS_PARTIAL
    results["completed_datasets"] = _completed_dataset_count(results)


def _write_results(output_path: str, results: dict[str, object]) -> None:
    results_path = _results_path(output_path)
    tmp_path = f"{results_path}.tmp.{os.getpid()}"
    fs, _, _ = fsspec.get_fs_token_paths(output_path)
    fs.makedirs(output_path, exist_ok=True)
    with fs.open(tmp_path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
        f.write("\n")
    fs.mv(tmp_path, results_path)


def _run_with_retries(
    operation_name: str,
    operation: Callable[[], T],
    *,
    max_attempts: int,
    initial_delay: float,
    max_delay: float,
) -> T:
    attempt = 1
    delay = initial_delay
    while True:
        try:
            return operation()
        except Exception:
            if attempt >= max_attempts:
                logger.exception("%s failed after %d attempt(s)", operation_name, attempt)
                raise
            logger.warning(
                "%s failed on attempt %d/%d; retrying in %.1f seconds",
                operation_name,
                attempt,
                max_attempts,
                delay,
                exc_info=True,
            )
            time.sleep(delay)
            attempt += 1
            delay = min(delay * 2, max_delay)


def _validate_eval_config(config: TraceLabeledEvalConfig) -> None:
    if not config.datasets:
        raise ValueError("Trace labeled evaluation requires at least one dataset")
    if config.checkpoint_path is None:
        raise ValueError("Trace labeled evaluation requires checkpoint_path")
    if config.dataset_eval_max_attempts < 1:
        raise ValueError("dataset_eval_max_attempts must be at least 1")
    if config.dataset_eval_retry_initial_delay < 0:
        raise ValueError("dataset_eval_retry_initial_delay must be non-negative")
    if config.dataset_eval_retry_max_delay < config.dataset_eval_retry_initial_delay:
        raise ValueError("dataset_eval_retry_max_delay must be at least dataset_eval_retry_initial_delay")
    if config.job_failure_max_retries < 0:
        raise ValueError("job_failure_max_retries must be non-negative")
    for dataset_name, dataset_config in config.datasets.items():
        if dataset_config.row_prefix_fraction is not None and dataset_config.row_adapter is None:
            raise ValueError(f"Dataset {dataset_name!r} requires row_adapter when row_prefix_fraction is set")
        if dataset_config.row_prefix_fraction is not None and not 0.0 <= dataset_config.row_prefix_fraction <= 1.0:
            raise ValueError(
                f"Dataset {dataset_name!r} has invalid row_prefix_fraction {dataset_config.row_prefix_fraction!r}; "
                "expected a value in [0, 1]."
            )


def trace_labeled_eval(config: TraceLabeledEvalConfig) -> None:
    """Compute labeled losses over configured trace datasets."""

    _validate_eval_config(config)

    levanter.trainer.initialize(config)
    try:
        tokenizer = load_marin_tokenizer(config.tokenizer)

        EvalBatch = config.trainer.EvalBatch
        Pos = config.model.max_Pos.resize(config.max_eval_length)

        compute_axis_mapping = config.trainer.compute_axis_mapping
        parameter_axis_mapping = config.trainer.parameter_axis_mapping

        with config.trainer.use_device_mesh():
            key = jax.random.PRNGKey(0)

            vocab_size = len(tokenizer)
            Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), compute_axis_mapping)
            if vocab_size != Vocab.size:
                logger.info("Rounding vocab size from %d to %d for partitioning", vocab_size, Vocab.size)

            mp: jmp.Policy = config.trainer.mp

            assert config.checkpoint_path is not None  # ensured by _validate_eval_config
            model = load_eval_model(
                config.model,
                config.checkpoint_path,
                checkpoint_is_hf=config.checkpoint_is_hf,
                Vocab=Vocab,
                axis_mapping=parameter_axis_mapping,
                tokenizer=tokenizer,
                mp=mp,
                key=key,
            )

            results = _load_or_create_results(config)
            dataset_results = _dataset_results(results)
            tracker_metrics = _completed_dataset_metrics(results)
            if tracker_metrics:
                levanter.tracker.log(tracker_metrics, step=_completed_dataset_count(results))

            for dataset_name, dataset_config in config.datasets.items():
                if _is_completed_dataset_result(dataset_results.get(dataset_name)):
                    logger.info("Skipping completed trace dataset %s", dataset_name)
                    continue

                def evaluate_dataset(
                    current_dataset_name: str = dataset_name,
                    current_dataset_config: TraceLabeledEvalDatasetConfig = dataset_config,
                ) -> dict[str, float]:
                    source = _source_for_dataset(current_dataset_config)
                    cache_dir = os.path.join(config.output_path, "cache", _safe_name(current_dataset_name))
                    cache = build_trace_chat_dataset_cache(
                        cache_dir, source, current_dataset_config.trace_format, tokenizer
                    )
                    dataset = dataset_for_trace_chat_format(
                        current_dataset_config.trace_format,
                        Pos,
                        cache,
                        block_cross_document_attention=True,
                    )

                    evaluator = LabeledEvaluator.for_labeled_examples(
                        EvalBatch,
                        dataset,
                        label_spec=current_dataset_config.trace_format.loss_label_spec(),
                        tokenizer=tokenizer,
                        device_mesh=config.trainer.device_mesh,
                        axis_mapping=compute_axis_mapping,
                        mp=mp,
                    )
                    prefix = f"trace_labeled_eval/{current_dataset_name}"
                    return eval_labeled_model(evaluator, model, prefix=prefix)

                log_dict = _run_with_retries(
                    f"Trace dataset {dataset_name}",
                    evaluate_dataset,
                    max_attempts=config.dataset_eval_max_attempts,
                    initial_delay=config.dataset_eval_retry_initial_delay,
                    max_delay=config.dataset_eval_retry_max_delay,
                )
                _record_dataset_result(results, dataset_name, dataset_config, log_dict)
                tracker_metrics.update(log_dict)

                if jax.process_index() == 0:
                    _write_results(config.output_path, results)
                levanter.tracker.log(log_dict, step=_completed_dataset_count(results))

            results["status"] = TRACE_LABELED_EVAL_STATUS_COMPLETED
            results["completed_datasets"] = _completed_dataset_count(results)
            levanter.tracker.log(tracker_metrics, step=_completed_dataset_count(results))

            if jax.process_index() == 0:
                _write_results(config.output_path, results)
    finally:
        levanter.tracker.current_tracker().finish()


def run_trace_labeled_eval_on_pod(config: TraceLabeledEvalOnPodConfig) -> TraceLabeledEvalOutput:
    """Submit trace-labeled evaluation as a fray job and wait for completion."""

    client = current_client()

    extras = extras_for_resources(config.resources)

    job_name = "trace-labeled-eval"
    if config.trace_labeled_eval_config.name:
        job_name = f"{job_name}-{_safe_name(config.trace_labeled_eval_config.name)}"

    job_request = JobRequest(
        name=job_name,
        entrypoint=Entrypoint.from_callable(trace_labeled_eval, args=[config.trace_labeled_eval_config]),
        resources=config.resources,
        environment=create_environment(
            env_vars={name: os.getenv(name, DEFAULT_HF_HUB_TIMEOUT) for name in HF_HUB_TIMEOUT_ENV_VARS},
            extras=extras,
        ),
        max_retries_failure=config.trace_labeled_eval_config.job_failure_max_retries,
        max_task_failures=config.trace_labeled_eval_config.job_failure_max_retries,
    )
    job = client.submit(job_request)
    job.wait(raise_on_failure=True)
    return TraceLabeledEvalOutput(
        results_path=os.path.join(config.trace_labeled_eval_config.output_path, RESULTS_FILENAME),
    )
