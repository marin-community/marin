# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run Levanter masked loss evaluation on OpenAI-style agent traces."""

import json
import logging
import os
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

import equinox as eqx
import fsspec
import jax
import jmp

import haliax as hax
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning

import levanter
from levanter.checkpoint import load_checkpoint
from levanter.compat.hf_checkpoints import HFCheckpointConverter, RepoRef
from levanter.data.sharded_datasource import ShardedDataSource
from levanter.data.text import (
    LmDatasetSourceConfigBase,
    TraceChatEvaluationFormat,
    build_trace_chat_dataset_cache,
    dataset_for_trace_chat_format,
)
from levanter.eval import MaskedEvaluator, eval_masked_model
from levanter.models.llama import LlamaConfig
from levanter.models.lm_model import LmConfig
from levanter.tokenizers import load_tokenizer as load_marin_tokenizer
from levanter.tracker.json_file import JsonFileTrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import use_cpu_device

from fray.v2 import current_client
from fray.v2.types import Entrypoint, JobRequest, ResourceConfig, TpuConfig, create_environment

from marin.execution.executor import ExecutorStep, InputName, this_output_path
from marin.utilities.executor_utils import ckpt_path_to_step_name

logger = logging.getLogger(__name__)

T = TypeVar("T")
DEFAULT_TRACE_MASKED_EVAL_WANDB_PROJECT = "marin-analysis"
DEFAULT_TRACE_MASKED_EVAL_WANDB_TAGS = ("trace_masked_eval",)
CORRECT_OUTCOME_LABEL = "CORRECT"
INCORRECT_OUTCOME_LABEL = "INCORRECT"
RESULTS_FILENAME = "results.json"
TRACE_MASKED_EVAL_STATUS_PARTIAL = "partial"
TRACE_MASKED_EVAL_STATUS_COMPLETED = "completed"
TRACE_MASKED_EVAL_DATASET_STATUS_COMPLETED = "completed"
DEFAULT_DATASET_EVAL_MAX_ATTEMPTS = 3
DEFAULT_DATASET_EVAL_RETRY_INITIAL_DELAY = 30.0
DEFAULT_DATASET_EVAL_RETRY_MAX_DELAY = 300.0
DEFAULT_HF_HUB_TIMEOUT = "60"
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
    patch_loss_tag: str = "patch"
    outcome_loss_tag: str = "outcome"
    patch_prefix: str = "Final Patch:\n"
    positive_outcome_label: str = CORRECT_OUTCOME_LABEL
    negative_outcome_label: str = INCORRECT_OUTCOME_LABEL
    max_trace_messages: int | None = None
    preserve_initial_trace_messages: int = 0
    max_message_chars: int | None = None
    max_patch_chars: int | None = None
    truncation_side: Literal["left", "right"] = "right"


@dataclass(frozen=True)
class TraceMaskedEvalDatasetConfig:
    """A trace dataset and mask format to evaluate."""

    source: LmDatasetSourceConfigBase
    split: str
    trace_format: TraceChatEvaluationFormat = field(default_factory=TraceChatEvaluationFormat)
    max_examples: int | None = None
    row_adapter: TraceRowAdapterConfig | None = None


@dataclass
class TraceMaskedEvalConfig:
    """Configuration for trace-masked evaluation. Also serves as Levanter init config."""

    name: str | None = None
    checkpoint_path: str | None = None
    checkpoint_is_hf: bool = False
    tokenizer: str = "gpt2"
    model: LmConfig = field(default_factory=LlamaConfig)
    datasets: dict[str, TraceMaskedEvalDatasetConfig] = field(default_factory=dict)
    trainer: TrainerConfig = field(default_factory=lambda: TrainerConfig(mp=jmp.get_policy("c=bf16")))
    max_eval_length: int = 4096
    output_path: str = ""
    dataset_eval_max_attempts: int = DEFAULT_DATASET_EVAL_MAX_ATTEMPTS
    dataset_eval_retry_initial_delay: float = DEFAULT_DATASET_EVAL_RETRY_INITIAL_DELAY
    dataset_eval_retry_max_delay: float = DEFAULT_DATASET_EVAL_RETRY_MAX_DELAY
    job_failure_max_retries: int = 1


@dataclass(frozen=True)
class TraceMaskedEvalOnPodConfig:
    """Wrapper config for running trace-masked evaluation on a TPU pod via fray."""

    trace_masked_eval_config: TraceMaskedEvalConfig
    resources: ResourceConfig


class FirstRowsShardedDataSource(ShardedDataSource[T]):
    """A single-shard view over the first rows of another sharded source."""

    def __init__(self, source: ShardedDataSource[T], max_rows: int):
        if max_rows <= 0:
            raise ValueError("max_rows must be positive")
        self.source = source
        self.max_rows = max_rows

    @property
    def shard_names(self) -> Sequence[str]:
        return ["data"]

    def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[T]:
        if shard_name != "data":
            raise ValueError(f"Unknown shard {shard_name!r}")
        if row >= self.max_rows:
            return

        emitted = 0
        for item in self.source:
            if emitted >= row:
                emitted += 1
                yield item
                if emitted >= self.max_rows:
                    return
            else:
                emitted += 1


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

        message: dict[str, Any] = {
            "role": role,
            "content": _normalize_message_content(content),
        }
        for key in ("name", "tool_call_id", "tool_calls", "reasoning_content", "loss_tags"):
            value = raw_message.get(key)
            if value is not None:
                message[key] = value

        if message["content"] or message.get("tool_calls"):
            normalized.append(message)

    return normalized


def _limited_text(value: str, max_chars: int | None, side: Literal["left", "right"]) -> str:
    if max_chars is None or len(value) <= max_chars:
        return value
    if max_chars <= 0:
        raise ValueError("max_chars must be positive when set")
    if side == "left":
        return value[:max_chars]
    if side == "right":
        return value[-max_chars:]
    raise ValueError(f"Unsupported truncation side {side!r}.")


def _limited_message(
    message: Mapping[str, Any], max_chars: int | None, side: Literal["left", "right"]
) -> dict[str, Any]:
    if max_chars is None:
        return dict(message)

    limited = dict(message)
    for field_name in ("content", "reasoning_content"):
        value = limited.get(field_name)
        if isinstance(value, str):
            limited[field_name] = _limited_text(value, max_chars, side)
    return limited


def _limited_trace_messages(
    messages: Sequence[Mapping[str, Any]],
    row_adapter: TraceRowAdapterConfig,
) -> list[dict[str, Any]]:
    if row_adapter.truncation_side not in {"left", "right"}:
        raise ValueError(f"Unsupported truncation_side {row_adapter.truncation_side!r}.")
    if row_adapter.max_message_chars is not None and row_adapter.max_message_chars <= 0:
        raise ValueError("max_message_chars must be positive when set")
    if row_adapter.max_patch_chars is not None and row_adapter.max_patch_chars <= 0:
        raise ValueError("max_patch_chars must be positive when set")
    if row_adapter.preserve_initial_trace_messages < 0:
        raise ValueError("preserve_initial_trace_messages must be non-negative")
    if row_adapter.max_trace_messages is not None and row_adapter.max_trace_messages <= 0:
        raise ValueError("max_trace_messages must be positive when set")
    if (
        row_adapter.max_trace_messages is not None
        and row_adapter.preserve_initial_trace_messages > row_adapter.max_trace_messages
    ):
        raise ValueError("preserve_initial_trace_messages cannot exceed max_trace_messages")

    limited_messages = [
        _limited_message(message, row_adapter.max_message_chars, row_adapter.truncation_side) for message in messages
    ]
    max_trace_messages = row_adapter.max_trace_messages
    if max_trace_messages is None or len(limited_messages) <= max_trace_messages:
        return limited_messages

    if row_adapter.truncation_side == "left":
        return limited_messages[:max_trace_messages]

    preserve_initial = row_adapter.preserve_initial_trace_messages
    tail_count = max_trace_messages - preserve_initial
    if tail_count == 0:
        return limited_messages[:preserve_initial]
    if preserve_initial == 0:
        return limited_messages[-tail_count:]
    return [*limited_messages[:preserve_initial], *limited_messages[-tail_count:]]


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
) -> dict[str, Any]:
    input_messages_field = row_adapter.input_messages_field or trace_format.messages_field
    messages = _normalize_trace_messages(_lookup_field(row, input_messages_field))
    messages = _limited_trace_messages(messages, row_adapter)

    patch = _lookup_field(row, row_adapter.patch_field) if row_adapter.patch_field is not None else None
    patch_text = _normalize_message_content(patch)
    if patch_text:
        max_patch_chars = row_adapter.max_patch_chars
        if max_patch_chars is None:
            max_patch_chars = row_adapter.max_message_chars
        patch_text = _limited_text(patch_text, max_patch_chars, row_adapter.truncation_side)
        messages.append(
            {
                "role": "assistant",
                "content": f"{row_adapter.patch_prefix}{patch_text}",
                "loss_tags": [row_adapter.patch_loss_tag],
            }
        )

    adapted = dict(row)
    if row_adapter.outcome_field is not None:
        label = _outcome_label(_lookup_field(row, row_adapter.outcome_field), row_adapter)
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


def _source_for_dataset(dataset_config: TraceMaskedEvalDatasetConfig) -> ShardedDataSource[dict]:
    source = dataset_config.source.get_shard_source(dataset_config.split)
    if source is None:
        raise ValueError(f"No shard source for split {dataset_config.split!r} in {dataset_config.source!r}")
    if dataset_config.max_examples is None:
        limited_source = source
    else:
        limited_source = FirstRowsShardedDataSource(source, dataset_config.max_examples)

    if dataset_config.row_adapter is None:
        return limited_source
    return limited_source.map(lambda row: _adapt_trace_row(row, dataset_config.trace_format, dataset_config.row_adapter))


def _safe_name(name: str) -> str:
    return name.replace("/", "--").replace(" ", "_")


def _dataset_metadata(dataset_config: TraceMaskedEvalDatasetConfig) -> dict[str, object]:
    source = dataset_config.source
    metadata: dict[str, object] = {
        "source_type": type(source).__name__,
        "split": dataset_config.split,
        "max_examples": dataset_config.max_examples,
        "loss_tags": list(dataset_config.trace_format.loss_tags),
    }
    if dataset_config.row_adapter is not None:
        metadata["row_adapter"] = {
            "input_messages_field": dataset_config.row_adapter.input_messages_field,
            "patch_field": dataset_config.row_adapter.patch_field,
            "outcome_field": dataset_config.row_adapter.outcome_field,
            "patch_loss_tag": dataset_config.row_adapter.patch_loss_tag,
            "outcome_loss_tag": dataset_config.row_adapter.outcome_loss_tag,
            "max_trace_messages": dataset_config.row_adapter.max_trace_messages,
            "preserve_initial_trace_messages": dataset_config.row_adapter.preserve_initial_trace_messages,
            "max_message_chars": dataset_config.row_adapter.max_message_chars,
            "max_patch_chars": dataset_config.row_adapter.max_patch_chars,
            "truncation_side": dataset_config.row_adapter.truncation_side,
        }
    for field_name in ("id", "name", "stream", "splits"):
        if hasattr(source, field_name):
            metadata[field_name] = getattr(source, field_name)
    return metadata


def _results_path(output_path: str) -> str:
    return os.path.join(output_path, RESULTS_FILENAME)


def _new_results(config: TraceMaskedEvalConfig) -> dict[str, object]:
    return {
        "checkpoint_path": config.checkpoint_path,
        "checkpoint_is_hf": config.checkpoint_is_hf,
        "tokenizer": config.tokenizer,
        "max_eval_length": config.max_eval_length,
        "status": TRACE_MASKED_EVAL_STATUS_PARTIAL,
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
        raise ValueError("Trace masked eval results must contain a datasets object")
    return datasets


def _validate_loaded_results(config: TraceMaskedEvalConfig, results: dict[str, object]) -> None:
    expected = _new_results(config)
    for field_name in ("checkpoint_path", "checkpoint_is_hf", "tokenizer", "max_eval_length"):
        if field_name in results and results[field_name] != expected[field_name]:
            raise ValueError(
                f"Existing trace masked eval results at {config.output_path} have {field_name}="
                f"{results[field_name]!r}, expected {expected[field_name]!r}"
            )


def _load_or_create_results(config: TraceMaskedEvalConfig) -> dict[str, object]:
    results = _read_results(config.output_path)
    if results is None:
        return _new_results(config)

    _validate_loaded_results(config, results)
    results.setdefault("checkpoint_path", config.checkpoint_path)
    results.setdefault("checkpoint_is_hf", config.checkpoint_is_hf)
    results.setdefault("tokenizer", config.tokenizer)
    results.setdefault("max_eval_length", config.max_eval_length)
    results.setdefault("status", TRACE_MASKED_EVAL_STATUS_PARTIAL)
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
        if not _is_completed_dataset_result(dataset_result):
            continue
        dataset_metrics = dataset_result["metrics"]
        assert isinstance(dataset_metrics, Mapping)
        for metric_name, metric_value in dataset_metrics.items():
            if not isinstance(metric_name, str):
                raise ValueError(f"Trace masked eval metric names must be strings, got {metric_name!r}")
            if not isinstance(metric_value, int | float):
                raise ValueError(f"Trace masked eval metric {metric_name!r} must be numeric, got {metric_value!r}")
            metrics[metric_name] = float(metric_value)
    return metrics


def _record_dataset_result(
    results: dict[str, object],
    dataset_name: str,
    dataset_config: TraceMaskedEvalDatasetConfig,
    metrics: Mapping[str, float],
) -> None:
    _dataset_results(results)[dataset_name] = {
        "status": TRACE_MASKED_EVAL_DATASET_STATUS_COMPLETED,
        "metadata": _dataset_metadata(dataset_config),
        "metrics": dict(metrics),
    }
    results["status"] = TRACE_MASKED_EVAL_STATUS_PARTIAL
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


def _validate_eval_config(config: TraceMaskedEvalConfig) -> None:
    if not config.datasets:
        raise ValueError("Trace masked evaluation requires at least one dataset")
    if config.checkpoint_path is None:
        raise ValueError("Trace masked evaluation requires checkpoint_path")
    if config.dataset_eval_max_attempts < 1:
        raise ValueError("dataset_eval_max_attempts must be at least 1")
    if config.dataset_eval_retry_initial_delay < 0:
        raise ValueError("dataset_eval_retry_initial_delay must be non-negative")
    if config.dataset_eval_retry_max_delay < config.dataset_eval_retry_initial_delay:
        raise ValueError("dataset_eval_retry_max_delay must be at least dataset_eval_retry_initial_delay")
    if config.job_failure_max_retries < 0:
        raise ValueError("job_failure_max_retries must be non-negative")


def trace_masked_eval(config: TraceMaskedEvalConfig) -> None:
    """Compute masked losses over configured trace datasets."""

    _validate_eval_config(config)

    levanter.initialize(config)
    try:
        tokenizer = load_marin_tokenizer(config.tokenizer)

        hf_checkpoint = RepoRef.from_string(config.checkpoint_path) if config.checkpoint_is_hf else None
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

            if config.checkpoint_path is not None and not config.checkpoint_is_hf:
                with use_cpu_device():
                    model = eqx.filter_eval_shape(config.model.build, Vocab, key=key)
                    model = load_checkpoint(model, config.checkpoint_path, subpath="model")
                model = hax.shard_with_axis_mapping(model, parameter_axis_mapping)
            elif hf_checkpoint is not None:
                model_config = config.model
                if not hasattr(model_config, "hf_checkpoint_converter"):
                    raise ValueError("Model config does not have an HF checkpoint converter. Can't load HF checkpoint.")
                converter: HFCheckpointConverter = model_config.hf_checkpoint_converter()
                converter = converter.replaced(reference_checkpoint=hf_checkpoint, tokenizer=tokenizer)
                model = converter.load_pretrained(
                    model_config.model_type,
                    ref=hf_checkpoint,
                    axis_mapping=parameter_axis_mapping,
                    dtype=mp.compute_dtype,
                )
            else:
                raise AssertionError("Should not get here")

            results = _load_or_create_results(config)
            dataset_results = _dataset_results(results)
            tracker_metrics = _completed_dataset_metrics(results)
            if tracker_metrics:
                logger.info(
                    "Loaded %d completed trace dataset(s) from %s",
                    _completed_dataset_count(results),
                    config.output_path,
                )
                levanter.tracker.log(tracker_metrics, step=_completed_dataset_count(results))

            for dataset_name, dataset_config in config.datasets.items():
                if _is_completed_dataset_result(dataset_results.get(dataset_name)):
                    logger.info("Skipping completed trace dataset %s", dataset_name)
                    continue

                logger.info("Evaluating trace dataset %s", dataset_name)

                def evaluate_dataset(
                    current_dataset_name: str = dataset_name,
                    current_dataset_config: TraceMaskedEvalDatasetConfig = dataset_config,
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

                    evaluator = MaskedEvaluator.for_trace_lm(
                        EvalBatch,
                        dataset,
                        target_names=current_dataset_config.trace_format.loss_tags,
                        tokenizer=tokenizer,
                        device_mesh=config.trainer.device_mesh,
                        axis_mapping=compute_axis_mapping,
                        mp=mp,
                    )
                    return eval_masked_model(evaluator, model, prefix=f"trace_masked_eval/{current_dataset_name}")

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

            results["status"] = TRACE_MASKED_EVAL_STATUS_COMPLETED
            results["completed_datasets"] = _completed_dataset_count(results)
            levanter.tracker.log(tracker_metrics, step=_completed_dataset_count(results))

            if jax.process_index() == 0:
                _write_results(config.output_path, results)
    finally:
        levanter.tracker.current_tracker().finish()


def run_trace_masked_eval_on_pod(config: TraceMaskedEvalOnPodConfig) -> None:
    """Submit trace-masked evaluation as a fray job and wait for completion."""

    client = current_client()

    extras = []
    if isinstance(config.resources.device, TpuConfig):
        extras.append("tpu")

    job_name = "trace-masked-eval"
    if config.trace_masked_eval_config.name:
        job_name = f"{job_name}-{_safe_name(config.trace_masked_eval_config.name)}"

    job_request = JobRequest(
        name=job_name,
        entrypoint=Entrypoint.from_callable(trace_masked_eval, args=[config.trace_masked_eval_config]),
        resources=config.resources,
        environment=create_environment(
            env_vars={name: os.getenv(name, DEFAULT_HF_HUB_TIMEOUT) for name in HF_HUB_TIMEOUT_ENV_VARS},
            extras=extras,
        ),
        max_retries_failure=config.trace_masked_eval_config.job_failure_max_retries,
    )
    job = client.submit(job_request)
    job.wait(raise_on_failure=True)


def default_trace_masked_eval(
    *,
    checkpoint: str | InputName,
    model: LmConfig,
    tokenizer: str,
    datasets: dict[str, TraceMaskedEvalDatasetConfig],
    resource_config: ResourceConfig,
    checkpoint_is_hf: bool,
    per_device_batch_size: int = 1,
    max_eval_length: int = 4096,
    name: str | None = None,
    wandb_project: str = DEFAULT_TRACE_MASKED_EVAL_WANDB_PROJECT,
    wandb_tags: Sequence[str] = DEFAULT_TRACE_MASKED_EVAL_WANDB_TAGS,
    wandb_group: str | None = None,
) -> ExecutorStep:
    """Create an ExecutorStep that evaluates named trace masks."""

    if not name:
        name = ckpt_path_to_step_name(checkpoint)

    return ExecutorStep(
        name=f"analysis/trace_masked_eval/{name}",
        fn=run_trace_masked_eval_on_pod,
        config=TraceMaskedEvalOnPodConfig(
            trace_masked_eval_config=TraceMaskedEvalConfig(
                name=name,
                checkpoint_path=checkpoint,  # type: ignore[arg-type]
                checkpoint_is_hf=checkpoint_is_hf,
                tokenizer=tokenizer,
                model=model,
                datasets=datasets,
                trainer=TrainerConfig(
                    tracker=(
                        WandbConfig(
                            project=wandb_project,
                            name=name,
                            tags=list(wandb_tags),
                            group=wandb_group,
                        ),
                        JsonFileTrackerConfig(output_path=this_output_path()),
                    ),
                    per_device_eval_parallelism=per_device_batch_size,
                    mp=jmp.get_policy("c=bf16"),
                ),
                max_eval_length=max_eval_length,
                output_path=this_output_path(),
            ),
            resources=resource_config,
        ),
    )
