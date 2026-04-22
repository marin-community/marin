# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run Levanter masked loss evaluation on OpenAI-style agent traces."""

import json
import logging
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar
from collections.abc import Mapping

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
            if emitted >= self.max_rows:
                break
            if emitted >= row:
                yield item
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
        }
    for field_name in ("id", "name", "stream", "splits"):
        if hasattr(source, field_name):
            metadata[field_name] = getattr(source, field_name)
    return metadata


def _write_results(output_path: str, results: dict[str, object]) -> None:
    fs, _, _ = fsspec.get_fs_token_paths(output_path)
    fs.makedirs(output_path, exist_ok=True)
    with fs.open(os.path.join(output_path, "results.json"), "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
        f.write("\n")


def trace_masked_eval(config: TraceMaskedEvalConfig) -> None:
    """Compute masked losses over configured trace datasets."""

    if not config.datasets:
        raise ValueError("Trace masked evaluation requires at least one dataset")
    if config.checkpoint_path is None:
        raise ValueError("Trace masked evaluation requires checkpoint_path")

    levanter.initialize(config)
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

        results: dict[str, object] = {
            "checkpoint_path": config.checkpoint_path,
            "checkpoint_is_hf": config.checkpoint_is_hf,
            "tokenizer": config.tokenizer,
            "max_eval_length": config.max_eval_length,
            "datasets": {},
        }
        dataset_results: dict[str, object] = {}

        tracker_metrics: dict[str, float] = {}
        for dataset_name, dataset_config in config.datasets.items():
            logger.info("Evaluating trace dataset %s", dataset_name)
            source = _source_for_dataset(dataset_config)
            cache_dir = os.path.join(config.output_path, "cache", _safe_name(dataset_name))
            cache = build_trace_chat_dataset_cache(cache_dir, source, dataset_config.trace_format, tokenizer)
            dataset = dataset_for_trace_chat_format(
                dataset_config.trace_format,
                Pos,
                cache,
                block_cross_document_attention=True,
            )

            evaluator = MaskedEvaluator.for_trace_lm(
                EvalBatch,
                dataset,
                target_names=dataset_config.trace_format.loss_tags,
                tokenizer=tokenizer,
                device_mesh=config.trainer.device_mesh,
                axis_mapping=compute_axis_mapping,
                mp=mp,
            )
            log_dict = eval_masked_model(evaluator, model, prefix=f"trace_masked_eval/{dataset_name}")
            tracker_metrics.update(log_dict)
            dataset_results[dataset_name] = {
                "metadata": _dataset_metadata(dataset_config),
                "metrics": log_dict,
            }

        results["datasets"] = dataset_results
        levanter.tracker.log(tracker_metrics, step=0)

        if jax.process_index() == 0:
            _write_results(config.output_path, results)

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
        environment=create_environment(extras=extras),
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
