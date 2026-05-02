# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run Levanter masked loss evaluation on OpenAI-style agent traces."""

import json
import logging
import math
import os
import time
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, TypeVar

import equinox as eqx
import fsspec
import haliax as hax
import jax
import jax.numpy as jnp
import jmp
import levanter
import numpy as np
from fray import current_client
from fray.types import Entrypoint, JobRequest, ResourceConfig, TpuConfig, create_environment
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning
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
from levanter.models.lm_model import LmConfig, LmExample
from levanter.tokenizers import load_tokenizer as load_marin_tokenizer
from levanter.tracker.json_file import JsonFileTrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import use_cpu_device
from levanter.utils.tree_utils import inference_mode
from tqdm_loggable.auto import tqdm

from marin.execution.executor import ExecutorStep, InputName, this_output_path
from marin.utilities.executor_utils import ckpt_path_to_step_name

logger = logging.getLogger(__name__)

T = TypeVar("T")
DEFAULT_TRACE_MASKED_EVAL_WANDB_PROJECT = "marin-analysis"
DEFAULT_TRACE_MASKED_EVAL_WANDB_TAGS = ("trace_masked_eval",)
CORRECT_OUTCOME_LABEL = "CORRECT"
INCORRECT_OUTCOME_LABEL = "INCORRECT"
DEFAULT_OUTCOME_JUDGE_PROMPT = (
    "Given the trace above and any final patch shown, predict whether the attempted solution would pass the task.\n"
    "Answer exactly one token: CORRECT or INCORRECT."
)
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
    task_id_field: str | None = None
    record_id_field: str | None = None
    patch_loss_tag: str = "patch"
    outcome_loss_tag: str = "outcome"
    patch_prefix: str = "Final Patch:\n"
    positive_outcome_label: str = CORRECT_OUTCOME_LABEL
    negative_outcome_label: str = INCORRECT_OUTCOME_LABEL
    outcome_prompt: str = DEFAULT_OUTCOME_JUDGE_PROMPT
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
    row_prefix_fraction: float | None = None
    contrastive_outcome: bool = False
    outcome_prefix_fractions: tuple[float, ...] = ()
    write_outcome_example_scores: bool = False


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
    include_outcome_label: bool = True,
    include_patch: bool = True,
    prefix_fraction: float | None = None,
) -> dict[str, Any]:
    input_messages_field = row_adapter.input_messages_field or trace_format.messages_field
    messages = _normalize_trace_messages(_lookup_field(row, input_messages_field))
    messages = _limited_trace_messages(messages, row_adapter)
    messages = _prefixed_trace_messages(messages, prefix_fraction)

    patch = (
        _lookup_field(row, row_adapter.patch_field) if include_patch and row_adapter.patch_field is not None else None
    )
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
    adapted["trace_task_id"] = _row_identifier(row, row_adapter.task_id_field, default="")
    adapted["trace_record_id"] = _row_identifier(row, row_adapter.record_id_field, default="")
    if row_adapter.outcome_field is not None:
        label = _outcome_label(_lookup_field(row, row_adapter.outcome_field), row_adapter)
        if include_outcome_label:
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


def _base_source_for_dataset(dataset_config: TraceMaskedEvalDatasetConfig) -> ShardedDataSource[dict]:
    source = dataset_config.source.get_shard_source(dataset_config.split)
    if source is None:
        raise ValueError(f"No shard source for split {dataset_config.split!r} in {dataset_config.source!r}")
    if dataset_config.max_examples is not None:
        return FirstRowsShardedDataSource(source, dataset_config.max_examples)
    return source


def _source_for_dataset(
    dataset_config: TraceMaskedEvalDatasetConfig,
    *,
    include_outcome_label: bool = True,
) -> ShardedDataSource[dict]:
    limited_source = _base_source_for_dataset(dataset_config)
    if dataset_config.row_adapter is None:
        return limited_source
    return limited_source.map(
        lambda row: _adapt_trace_row(
            row,
            dataset_config.trace_format,
            dataset_config.row_adapter,
            include_outcome_label=include_outcome_label,
            prefix_fraction=dataset_config.row_prefix_fraction,
        )
    )


def _adapt_contrastive_outcome_row(
    row: Mapping[str, Any],
    trace_format: TraceChatEvaluationFormat,
    row_adapter: TraceRowAdapterConfig,
    candidate_label: str,
    *,
    include_patch: bool = True,
    prefix_fraction: float | None = None,
) -> dict[str, Any]:
    adapted = _adapt_trace_row(
        row,
        trace_format,
        row_adapter,
        include_outcome_label=False,
        include_patch=include_patch,
        prefix_fraction=prefix_fraction,
    )
    messages = [dict(message) for message in adapted[trace_format.messages_field]]
    messages.append({"role": "user", "content": row_adapter.outcome_prompt})
    messages.append(
        {
            "role": "assistant",
            "content": candidate_label,
            "loss_tags": [row_adapter.outcome_loss_tag],
        }
    )
    adapted[trace_format.messages_field] = messages
    adapted["trace_outcome_candidate_label"] = candidate_label
    return adapted


def _safe_name(name: str) -> str:
    return name.replace("/", "--").replace(" ", "_")


def _dataset_metadata(dataset_config: TraceMaskedEvalDatasetConfig) -> dict[str, object]:
    source = dataset_config.source
    metadata: dict[str, object] = {
        "source_type": type(source).__name__,
        "split": dataset_config.split,
        "max_examples": dataset_config.max_examples,
        "loss_tags": list(dataset_config.trace_format.loss_tags),
        "row_prefix_fraction": dataset_config.row_prefix_fraction,
        "contrastive_outcome": dataset_config.contrastive_outcome,
        "outcome_prefix_fractions": list(dataset_config.outcome_prefix_fractions),
        "write_outcome_example_scores": dataset_config.write_outcome_example_scores,
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
    *,
    artifacts: Mapping[str, str] | None = None,
) -> None:
    dataset_result: dict[str, object] = {
        "status": TRACE_MASKED_EVAL_DATASET_STATUS_COMPLETED,
        "metadata": _dataset_metadata(dataset_config),
        "metrics": dict(metrics),
    }
    if artifacts:
        dataset_result["artifacts"] = dict(artifacts)
    _dataset_results(results)[dataset_name] = dataset_result
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


def _tokenizer_pad_id(tokenizer: Any) -> int:
    for token_id in (tokenizer.pad_token_id, tokenizer.eos_token_id):
        if token_id is not None:
            return int(token_id)
    return 0


def _slice_array(array: np.ndarray, max_length: int, strategy: Literal["left", "right", "raise"]) -> np.ndarray:
    if len(array) <= max_length:
        return array
    if strategy == "left":
        return array[:max_length]
    if strategy == "right":
        return array[-max_length:]
    if strategy == "raise":
        raise ValueError(f"Contrastive outcome example has {len(array)} tokens, exceeding max length {max_length}.")
    raise ValueError(f"Unsupported slice strategy {strategy!r}.")


def _pad_array(array: np.ndarray, max_length: int, pad_value: int | float) -> np.ndarray:
    if len(array) > max_length:
        raise ValueError(f"Cannot pad array of length {len(array)} to shorter length {max_length}.")
    if len(array) == max_length:
        return array
    pad_width = max_length - len(array)
    return np.pad(array, (0, pad_width), constant_values=pad_value)


def _shift_loss_mask(mask: np.ndarray) -> np.ndarray:
    shifted = np.roll(mask.astype(np.float32), -1)
    shifted[-1] = 0.0
    return shifted


def _prepare_contrastive_candidate(
    row: Mapping[str, Any],
    trace_format: TraceChatEvaluationFormat,
    row_adapter: TraceRowAdapterConfig,
    tokenizer: Any,
    max_eval_length: int,
    candidate_label: str,
    *,
    include_patch: bool = True,
    prefix_fraction: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    contrastive_format = replace(
        trace_format,
        loss_tags=(row_adapter.outcome_loss_tag,),
        pack=None,
        include_role_tags=False,
        include_final_assistant_tag=False,
    )
    candidate_row = _adapt_contrastive_outcome_row(
        row,
        contrastive_format,
        row_adapter,
        candidate_label,
        include_patch=include_patch,
        prefix_fraction=prefix_fraction,
    )
    processed = contrastive_format.build_preprocessor(tokenizer)([candidate_row])[0]

    input_ids = np.asarray(processed["input_ids"], dtype=np.int32)
    outcome_mask = np.asarray(processed["trace_masks"][row_adapter.outcome_loss_tag], dtype=np.float32)
    input_ids = _slice_array(input_ids, max_eval_length, trace_format.slice_strategy)
    outcome_mask = _slice_array(outcome_mask, max_eval_length, trace_format.slice_strategy)
    loss_weight = _shift_loss_mask(outcome_mask)

    if np.sum(loss_weight) == 0:
        raise ValueError(f"Contrastive outcome candidate {candidate_label!r} produced no scored label tokens.")

    input_ids = _pad_array(input_ids, max_eval_length, _tokenizer_pad_id(tokenizer)).astype(np.int32)
    loss_weight = _pad_array(loss_weight, max_eval_length, 0.0).astype(np.float32)
    return input_ids, loss_weight


def _binary_auroc(scores: Sequence[float], labels: Sequence[bool]) -> tuple[float, bool]:
    positives = [score for score, label in zip(scores, labels, strict=True) if label]
    negatives = [score for score, label in zip(scores, labels, strict=True) if not label]
    if not positives or not negatives:
        return 0.5, False

    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives)), True


def _grouped_binary_auroc(
    scores: Sequence[float], labels: Sequence[bool], group_ids: Sequence[str]
) -> tuple[float, bool, int, int, int, float]:
    grouped_examples: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    for score, label, group_id in zip(scores, labels, group_ids, strict=True):
        if not group_id:
            continue
        grouped_examples[group_id].append((score, label))

    if not grouped_examples:
        return 0.5, False, 0, 0, 0, 0.5

    total_wins = 0.0
    total_pairs = 0
    comparable_groups = 0
    mean_group_aurocs: list[float] = []
    for examples in grouped_examples.values():
        positive_scores = [score for score, label in examples if label]
        negative_scores = [score for score, label in examples if not label]
        if not positive_scores or not negative_scores:
            continue

        comparable_groups += 1
        group_wins = 0.0
        for positive_score in positive_scores:
            for negative_score in negative_scores:
                if positive_score > negative_score:
                    group_wins += 1.0
                elif positive_score == negative_score:
                    group_wins += 0.5
        group_pairs = len(positive_scores) * len(negative_scores)
        total_wins += group_wins
        total_pairs += group_pairs
        mean_group_aurocs.append(group_wins / group_pairs)

    if total_pairs == 0:
        return 0.5, False, len(grouped_examples), 0, 0, 0.5

    return (
        total_wins / total_pairs,
        True,
        len(grouped_examples),
        comparable_groups,
        total_pairs,
        float(np.mean(mean_group_aurocs)),
    )


def _prefix_metric_name(prefix_fraction: float) -> str:
    if prefix_fraction >= 1.0:
        return "prefix_100"
    percent = round(prefix_fraction * 100)
    return f"prefix_{percent}"


def _outcome_example_scores_path(output_path: str, dataset_name: str) -> str:
    return os.path.join(output_path, "examples", f"{_safe_name(dataset_name)}.outcome_examples.jsonl")


def _write_jsonl_records(path: str, records: Sequence[Mapping[str, Any]]) -> None:
    fs, _, _ = fsspec.get_fs_token_paths(path)
    fs.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with fs.open(tmp_path, "w") as f:
        for record in records:
            json.dump(record, f, sort_keys=True)
            f.write("\n")
    fs.mv(tmp_path, path)


def _sigmoid(value: float) -> float:
    if value >= 0:
        exp_neg = np.exp(-value)
        return float(1.0 / (1.0 + exp_neg))
    exp_value = np.exp(value)
    return float(exp_value / (1.0 + exp_value))


def _contrastive_outcome_summary(
    *,
    margins: Sequence[float],
    normalized_margins: Sequence[float],
    gold_is_correct: Sequence[bool],
    group_ids: Sequence[str] | None,
    correct_logprobs: Sequence[float],
    incorrect_logprobs: Sequence[float],
    correct_token_counts: Sequence[float],
    incorrect_token_counts: Sequence[float],
) -> dict[str, float]:
    if not margins:
        raise ValueError("Contrastive outcome metrics require at least one example.")

    predictions = [margin > 0 for margin in margins]
    accuracy = np.mean([prediction == label for prediction, label in zip(predictions, gold_is_correct, strict=True)])
    normalized_predictions = [margin > 0 for margin in normalized_margins]
    normalized_accuracy = np.mean(
        [prediction == label for prediction, label in zip(normalized_predictions, gold_is_correct, strict=True)]
    )
    auroc, auroc_defined = _binary_auroc(margins, gold_is_correct)
    normalized_auroc, normalized_auroc_defined = _binary_auroc(normalized_margins, gold_is_correct)
    probabilities = [_sigmoid(margin) for margin in margins]
    brier = np.mean(
        [(probability - float(label)) ** 2 for probability, label in zip(probabilities, gold_is_correct, strict=True)]
    )
    normalized_probabilities = [_sigmoid(margin) for margin in normalized_margins]
    normalized_brier = np.mean(
        [
            (probability - float(label)) ** 2
            for probability, label in zip(normalized_probabilities, gold_is_correct, strict=True)
        ]
    )
    same_task_auroc = 0.5
    same_task_auroc_defined = False
    same_task_groups = 0
    same_task_groups_with_pairs = 0
    same_task_pairs = 0
    same_task_mean_group_auroc = 0.5
    same_task_normalized_auroc = 0.5
    same_task_normalized_auroc_defined = False
    same_task_normalized_mean_group_auroc = 0.5
    if group_ids is not None:
        (
            same_task_auroc,
            same_task_auroc_defined,
            same_task_groups,
            same_task_groups_with_pairs,
            same_task_pairs,
            same_task_mean_group_auroc,
        ) = _grouped_binary_auroc(margins, gold_is_correct, group_ids)
        (
            same_task_normalized_auroc,
            same_task_normalized_auroc_defined,
            _same_task_norm_groups,
            _same_task_norm_groups_with_pairs,
            _same_task_norm_pairs,
            same_task_normalized_mean_group_auroc,
        ) = _grouped_binary_auroc(normalized_margins, gold_is_correct, group_ids)
        assert _same_task_norm_groups == same_task_groups
        assert _same_task_norm_groups_with_pairs == same_task_groups_with_pairs
        assert _same_task_norm_pairs == same_task_pairs

    return {
        "accuracy": float(accuracy),
        "auroc": float(auroc),
        "auroc_defined": float(auroc_defined),
        "brier": float(brier),
        "normalized_accuracy": float(normalized_accuracy),
        "normalized_auroc": float(normalized_auroc),
        "normalized_auroc_defined": float(normalized_auroc_defined),
        "normalized_brier": float(normalized_brier),
        "examples": float(len(margins)),
        "positive_examples": float(sum(gold_is_correct)),
        "negative_examples": float(len(gold_is_correct) - sum(gold_is_correct)),
        "positive_rate": float(np.mean(gold_is_correct)),
        "same_task_auroc": float(same_task_auroc),
        "same_task_auroc_defined": float(same_task_auroc_defined),
        "same_task_groups": float(same_task_groups),
        "same_task_groups_with_pairs": float(same_task_groups_with_pairs),
        "same_task_pairs": float(same_task_pairs),
        "same_task_mean_group_auroc": float(same_task_mean_group_auroc),
        "same_task_normalized_auroc": float(same_task_normalized_auroc),
        "same_task_normalized_auroc_defined": float(same_task_normalized_auroc_defined),
        "same_task_normalized_mean_group_auroc": float(same_task_normalized_mean_group_auroc),
        "mean_margin": float(np.mean(margins)),
        "mean_normalized_margin": float(np.mean(normalized_margins)),
        "mean_correct_logprob": float(np.mean(correct_logprobs)),
        "mean_incorrect_logprob": float(np.mean(incorrect_logprobs)),
        "mean_correct_label_tokens": float(np.mean(correct_token_counts)),
        "mean_incorrect_label_tokens": float(np.mean(incorrect_token_counts)),
    }


def _prefixed_metrics(prefix: str, metrics: Mapping[str, float]) -> dict[str, float]:
    return {f"{prefix}/{name}": value for name, value in metrics.items()}


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
    for dataset_name, dataset_config in config.datasets.items():
        if dataset_config.row_prefix_fraction is not None and dataset_config.row_adapter is None:
            raise ValueError(f"Dataset {dataset_name!r} requires row_adapter when row_prefix_fraction is set")
        if dataset_config.row_prefix_fraction is not None and not 0.0 <= dataset_config.row_prefix_fraction <= 1.0:
            raise ValueError(
                f"Dataset {dataset_name!r} has invalid row_prefix_fraction {dataset_config.row_prefix_fraction!r}; "
                "expected a value in [0, 1]."
            )
        for prefix_fraction in dataset_config.outcome_prefix_fractions:
            if not 0.0 < prefix_fraction <= 1.0:
                raise ValueError(
                    f"Dataset {dataset_name!r} has invalid outcome prefix fraction {prefix_fraction!r}; "
                    "expected values in (0, 1]."
                )
        if not dataset_config.contrastive_outcome:
            continue
        if dataset_config.row_adapter is None:
            raise ValueError(f"Dataset {dataset_name!r} requires row_adapter for contrastive outcome evaluation")
        if dataset_config.row_adapter.outcome_field is None:
            raise ValueError(f"Dataset {dataset_name!r} requires outcome_field for contrastive outcome evaluation")


def _score_contrastive_outcomes(
    *,
    model: Any,
    dataset_config: TraceMaskedEvalDatasetConfig,
    tokenizer: Any,
    max_eval_length: int,
    compute_axis_mapping: Mapping[str, Any] | None,
    mp: jmp.Policy,
    prefix: str,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    row_adapter = dataset_config.row_adapter
    if row_adapter is None or row_adapter.outcome_field is None:
        raise ValueError("Contrastive outcome evaluation requires a row adapter with an outcome field.")

    Pos = model.Pos.resize(max_eval_length)
    positive_label = row_adapter.positive_outcome_label
    negative_label = row_adapter.negative_outcome_label
    prefix_fractions = tuple(sorted(set(dataset_config.outcome_prefix_fractions)))
    score_model = inference_mode(model, True)
    score_model = mp.cast_to_compute(score_model)

    @hax.named_jit(axis_resources=compute_axis_mapping)
    def score_candidate(candidate_model, tokens, loss_weight):
        example = LmExample.causal(
            hax.named(tokens, Pos),
            loss_weight=hax.named(loss_weight, Pos),
            block_cross_document_attention=False,
        )
        per_pos_loss = candidate_model.compute_next_token_loss(example, reduction=None, reduction_axis=()).array
        weights = example.loss_weight.array
        loss_sum = jnp.sum(per_pos_loss * weights)
        token_count = jnp.sum(weights)
        return -loss_sum, token_count

    start = time.perf_counter()
    margins: list[float] = []
    normalized_margins: list[float] = []
    gold_is_correct: list[bool] = []
    correct_logprobs: list[float] = []
    incorrect_logprobs: list[float] = []
    correct_token_counts: list[float] = []
    incorrect_token_counts: list[float] = []
    group_ids: list[str] = []
    example_records: list[dict[str, Any]] = []
    prefix_metrics_data = {
        prefix_fraction: {
            "margins": [],
            "normalized_margins": [],
            "gold_is_correct": [],
            "group_ids": [],
            "correct_logprobs": [],
            "incorrect_logprobs": [],
            "correct_token_counts": [],
            "incorrect_token_counts": [],
        }
        for prefix_fraction in prefix_fractions
    }

    source = _base_source_for_dataset(dataset_config)
    progress_total = dataset_config.max_examples
    logger.info(
        "Scoring contrastive outcomes for %s%s",
        prefix,
        f" over {progress_total} example(s)" if progress_total is not None else "",
    )
    for example_index, row in enumerate(
        tqdm(
            source,
            desc=f"{prefix}/outcome_contrastive",
            total=progress_total,
            unit="example",
        )
    ):
        gold_label = _outcome_label(_lookup_field(row, row_adapter.outcome_field), row_adapter)
        task_id = _row_identifier(row, row_adapter.task_id_field, default="")
        record_id = _row_identifier(row, row_adapter.record_id_field, default=str(example_index))
        correct_tokens, correct_loss_weight = _prepare_contrastive_candidate(
            row,
            dataset_config.trace_format,
            row_adapter,
            tokenizer,
            max_eval_length,
            positive_label,
            include_patch=True,
        )
        incorrect_tokens, incorrect_loss_weight = _prepare_contrastive_candidate(
            row,
            dataset_config.trace_format,
            row_adapter,
            tokenizer,
            max_eval_length,
            negative_label,
            include_patch=True,
        )

        correct_logprob, correct_token_count = score_candidate(
            score_model, jnp.asarray(correct_tokens), jnp.asarray(correct_loss_weight)
        )
        incorrect_logprob, incorrect_token_count = score_candidate(
            score_model, jnp.asarray(incorrect_tokens), jnp.asarray(incorrect_loss_weight)
        )
        correct_logprob = float(jax.device_get(correct_logprob))
        incorrect_logprob = float(jax.device_get(incorrect_logprob))
        correct_token_count = float(jax.device_get(correct_token_count))
        incorrect_token_count = float(jax.device_get(incorrect_token_count))

        correct_logprobs.append(correct_logprob)
        incorrect_logprobs.append(incorrect_logprob)
        correct_token_counts.append(correct_token_count)
        incorrect_token_counts.append(incorrect_token_count)
        margins.append(correct_logprob - incorrect_logprob)
        normalized_margins.append(
            (correct_logprob / max(correct_token_count, 1.0)) - (incorrect_logprob / max(incorrect_token_count, 1.0))
        )
        gold_is_correct.append(gold_label == positive_label)
        group_ids.append(task_id)

        example_record: dict[str, Any] = {
            "example_index": example_index,
            "record_id": record_id,
            "task_id": task_id,
            "gold_label": gold_label,
            "gold_is_correct": gold_label == positive_label,
            "correct_logprob": correct_logprob,
            "incorrect_logprob": incorrect_logprob,
            "correct_token_count": correct_token_count,
            "incorrect_token_count": incorrect_token_count,
            "margin": margins[-1],
            "normalized_margin": normalized_margins[-1],
        }

        if prefix_fractions:
            prefix_record: dict[str, dict[str, float]] = {}
            for prefix_fraction in prefix_fractions:
                prefix_name = _prefix_metric_name(prefix_fraction)
                if prefix_fraction >= 1.0:
                    prefix_correct_logprob = correct_logprob
                    prefix_incorrect_logprob = incorrect_logprob
                    prefix_correct_token_count = correct_token_count
                    prefix_incorrect_token_count = incorrect_token_count
                else:
                    prefix_correct_tokens, prefix_correct_loss_weight = _prepare_contrastive_candidate(
                        row,
                        dataset_config.trace_format,
                        row_adapter,
                        tokenizer,
                        max_eval_length,
                        positive_label,
                        include_patch=False,
                        prefix_fraction=prefix_fraction,
                    )
                    prefix_incorrect_tokens, prefix_incorrect_loss_weight = _prepare_contrastive_candidate(
                        row,
                        dataset_config.trace_format,
                        row_adapter,
                        tokenizer,
                        max_eval_length,
                        negative_label,
                        include_patch=False,
                        prefix_fraction=prefix_fraction,
                    )
                    prefix_correct_logprob, prefix_correct_token_count = score_candidate(
                        score_model,
                        jnp.asarray(prefix_correct_tokens),
                        jnp.asarray(prefix_correct_loss_weight),
                    )
                    prefix_incorrect_logprob, prefix_incorrect_token_count = score_candidate(
                        score_model,
                        jnp.asarray(prefix_incorrect_tokens),
                        jnp.asarray(prefix_incorrect_loss_weight),
                    )
                    prefix_correct_logprob = float(jax.device_get(prefix_correct_logprob))
                    prefix_incorrect_logprob = float(jax.device_get(prefix_incorrect_logprob))
                    prefix_correct_token_count = float(jax.device_get(prefix_correct_token_count))
                    prefix_incorrect_token_count = float(jax.device_get(prefix_incorrect_token_count))

                prefix_margin = prefix_correct_logprob - prefix_incorrect_logprob
                prefix_normalized_margin = (prefix_correct_logprob / max(prefix_correct_token_count, 1.0)) - (
                    prefix_incorrect_logprob / max(prefix_incorrect_token_count, 1.0)
                )
                prefix_metrics_data[prefix_fraction]["margins"].append(prefix_margin)
                prefix_metrics_data[prefix_fraction]["normalized_margins"].append(prefix_normalized_margin)
                prefix_metrics_data[prefix_fraction]["gold_is_correct"].append(gold_label == positive_label)
                prefix_metrics_data[prefix_fraction]["group_ids"].append(task_id)
                prefix_metrics_data[prefix_fraction]["correct_logprobs"].append(prefix_correct_logprob)
                prefix_metrics_data[prefix_fraction]["incorrect_logprobs"].append(prefix_incorrect_logprob)
                prefix_metrics_data[prefix_fraction]["correct_token_counts"].append(prefix_correct_token_count)
                prefix_metrics_data[prefix_fraction]["incorrect_token_counts"].append(prefix_incorrect_token_count)
                prefix_record[prefix_name] = {
                    "correct_logprob": prefix_correct_logprob,
                    "incorrect_logprob": prefix_incorrect_logprob,
                    "correct_token_count": prefix_correct_token_count,
                    "incorrect_token_count": prefix_incorrect_token_count,
                    "margin": prefix_margin,
                    "normalized_margin": prefix_normalized_margin,
                }

            example_record["prefixes"] = prefix_record

        example_records.append(example_record)

    metrics = _contrastive_outcome_summary(
        margins=margins,
        normalized_margins=normalized_margins,
        gold_is_correct=gold_is_correct,
        group_ids=group_ids,
        correct_logprobs=correct_logprobs,
        incorrect_logprobs=incorrect_logprobs,
        correct_token_counts=correct_token_counts,
        incorrect_token_counts=incorrect_token_counts,
    )
    metrics["total_time"] = time.perf_counter() - start
    prefixed_metrics = _prefixed_metrics(f"{prefix}/outcome_contrastive", metrics)

    for prefix_fraction, prefix_data in prefix_metrics_data.items():
        prefix_summary = _contrastive_outcome_summary(
            margins=prefix_data["margins"],
            normalized_margins=prefix_data["normalized_margins"],
            gold_is_correct=prefix_data["gold_is_correct"],
            group_ids=prefix_data["group_ids"],
            correct_logprobs=prefix_data["correct_logprobs"],
            incorrect_logprobs=prefix_data["incorrect_logprobs"],
            correct_token_counts=prefix_data["correct_token_counts"],
            incorrect_token_counts=prefix_data["incorrect_token_counts"],
        )
        prefixed_metrics.update(
            _prefixed_metrics(
                f"{prefix}/outcome_contrastive/{_prefix_metric_name(prefix_fraction)}",
                prefix_summary,
            )
        )

    return prefixed_metrics, example_records


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
                ) -> tuple[dict[str, float], dict[str, str]]:
                    source = _source_for_dataset(
                        current_dataset_config,
                        include_outcome_label=not current_dataset_config.contrastive_outcome,
                    )
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
                    prefix = f"trace_masked_eval/{current_dataset_name}"
                    metrics = eval_masked_model(evaluator, model, prefix=prefix)
                    artifacts: dict[str, str] = {}
                    if current_dataset_config.contrastive_outcome:
                        outcome_metrics, example_records = _score_contrastive_outcomes(
                            model=model,
                            dataset_config=current_dataset_config,
                            tokenizer=tokenizer,
                            max_eval_length=config.max_eval_length,
                            compute_axis_mapping=compute_axis_mapping,
                            mp=mp,
                            prefix=prefix,
                        )
                        metrics.update(outcome_metrics)
                        if current_dataset_config.write_outcome_example_scores and jax.process_index() == 0:
                            examples_path = _outcome_example_scores_path(config.output_path, current_dataset_name)
                            _write_jsonl_records(examples_path, example_records)
                            artifacts["outcome_example_scores"] = examples_path
                    return metrics, artifacts

                log_dict, artifacts = _run_with_retries(
                    f"Trace dataset {dataset_name}",
                    evaluate_dataset,
                    max_attempts=config.dataset_eval_max_attempts,
                    initial_delay=config.dataset_eval_retry_initial_delay,
                    max_delay=config.dataset_eval_retry_max_delay,
                )
                _record_dataset_result(results, dataset_name, dataset_config, log_dict, artifacts=artifacts)
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
