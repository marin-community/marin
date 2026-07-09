# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Optional

import jmp
from jax.sharding import Mesh

import haliax as hax
from haliax.partitioning import ResourceMapping

from rigging.filesystem import prefix_join

import levanter.tracker
from levanter.callbacks._core import StepInfo
from levanter.data import AsyncDataset
from levanter.data.sharded_datasource import FirstRowsShardedDataSource, ShardedDataSource
from levanter.data.text import (
    LmDataConfig,
    LmDatasetSourceConfigBase,
    TraceChatEvaluationFormat,
    build_trace_chat_dataset_cache,
    dataset_for_trace_chat_format,
)
from levanter.data.text.examples import LabeledLmExample, LossLabelSpec
from levanter.eval import LabeledEvaluator, eval_labeled_model
from levanter.tokenizers import MarinTokenizer
from levanter.trainer import Trainer, TrainerConfig


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LabeledLmEvalDatasetConfig:
    """A trace-chat dataset to evaluate with per-label LM loss during training."""

    source: LmDatasetSourceConfigBase
    split: str
    trace_format: TraceChatEvaluationFormat = field(default_factory=TraceChatEvaluationFormat)
    cache_dir: str | None = None
    max_examples: int | None = None


@dataclass(frozen=True)
class LabeledLmEvalConfig:
    """Configuration for periodic labeled LM eval callbacks."""

    datasets: dict[str, LabeledLmEvalDatasetConfig] = field(default_factory=dict)
    cache_dir: str | None = None
    prefix: str = "labeled_eval"
    steps: int | None = None
    eval_current: bool = True
    eval_model: bool = True


def labeled_eval_cache_dir(
    data_config: LmDataConfig,
    labeled_eval_config: LabeledLmEvalConfig,
    dataset_name: str,
    dataset_config: LabeledLmEvalDatasetConfig,
) -> str:
    if dataset_config.cache_dir is not None:
        return dataset_config.cache_dir

    cache_root = labeled_eval_config.cache_dir
    if cache_root is None and data_config.cache_dir is not None:
        cache_root = prefix_join(data_config.cache_dir, "labeled_eval")
    if cache_root is None:
        raise ValueError(f"No cache_dir provided for labeled eval dataset {dataset_name}")
    return prefix_join(cache_root, dataset_name)


def source_for_labeled_eval_dataset(dataset_config: LabeledLmEvalDatasetConfig) -> ShardedDataSource[dict]:
    source = dataset_config.source.get_shard_source(dataset_config.split)
    if source is None:
        raise ValueError(f"No shard source for split {dataset_config.split!r} in {dataset_config.source!r}")
    if dataset_config.max_examples is None:
        return source
    return FirstRowsShardedDataSource(source, dataset_config.max_examples)


def cb_labeled_evaluate(
    evaluator: LabeledEvaluator,
    *,
    prefix: str = "labeled_eval",
    eval_current: bool = True,
    eval_model: bool = True,
) -> Callable[[StepInfo], None]:
    """Build a callback that logs labeled eval metrics for current and/or eval-mode model."""
    if not eval_current and not eval_model:
        raise ValueError("At least one of eval_current or eval_model should be True")

    last_eval_step: int | None = None

    def eval_callback(step: StepInfo, force: bool = False):
        del force
        nonlocal last_eval_step

        step_count = step.step
        if step_count < 0:
            return
        if last_eval_step == step_count:
            return

        if eval_current:
            log_dict = eval_labeled_model(evaluator, step.model, prefix=prefix)
            levanter.tracker.log(log_dict, step=step_count)

        if eval_model:
            log_dict = eval_labeled_model(evaluator, step.eval_model, prefix=os.path.join(prefix, "eval_model"))
            levanter.tracker.log(log_dict, step=step_count)

        last_eval_step = step_count

    return eval_callback


def cb_labeled_lm_evaluate(
    EvalBatch: hax.Axis,
    eval_set: AsyncDataset[LabeledLmExample],
    label_spec: LossLabelSpec,
    tokenizer: Optional[MarinTokenizer] = None,
    device_mesh: Optional[Mesh] = None,
    axis_mapping: ResourceMapping | None = None,
    *,
    prefix: str = "labeled_eval",
    eval_current: bool = True,
    eval_model: bool = True,
    mp: jmp.Policy | None = None,
) -> Callable[[StepInfo], None]:
    """Build a training callback for periodic labeled LM loss evaluation."""
    evaluator = LabeledEvaluator.for_labeled_examples(
        EvalBatch=EvalBatch,
        eval_set=eval_set,
        label_spec=label_spec,
        tokenizer=tokenizer,
        device_mesh=device_mesh,
        axis_mapping=axis_mapping,
        mp=mp,
    )
    return cb_labeled_evaluate(
        evaluator,
        prefix=prefix,
        eval_current=eval_current,
        eval_model=eval_model,
    )


def add_labeled_lm_eval_callbacks(
    trainer: Trainer,
    *,
    labeled_eval_config: LabeledLmEvalConfig,
    data_config: LmDataConfig,
    trainer_config: TrainerConfig,
    EvalBatch: hax.Axis,
    Pos: hax.Axis,
    tokenizer: MarinTokenizer,
    device_mesh: Mesh,
    axis_mapping: ResourceMapping,
    max_eval_examples_per_dataset: int | None,
) -> None:
    """Build trace-chat labeled eval caches and register periodic training hooks."""
    if not labeled_eval_config.datasets:
        logger.warning("labeled_eval was configured without any datasets.")

    for dataset_name, dataset_config in labeled_eval_config.datasets.items():
        source = source_for_labeled_eval_dataset(dataset_config)
        cache_dir = labeled_eval_cache_dir(data_config, labeled_eval_config, dataset_name, dataset_config)
        cache = build_trace_chat_dataset_cache(
            cache_dir,
            source,
            dataset_config.trace_format,
            tokenizer,
            data_config.cache_options,
        )
        dataset = dataset_for_trace_chat_format(
            dataset_config.trace_format,
            Pos,
            cache,
            block_cross_document_attention=data_config.block_cross_document_attention,
        )
        if max_eval_examples_per_dataset is not None:
            dataset = dataset.take(max_eval_examples_per_dataset)

        trainer.add_hook(
            cb_labeled_lm_evaluate(
                EvalBatch,
                dataset,
                dataset_config.trace_format.loss_label_spec(),
                tokenizer,
                device_mesh,
                axis_mapping,
                prefix=os.path.join(labeled_eval_config.prefix, dataset_name),
                eval_current=labeled_eval_config.eval_current,
                eval_model=labeled_eval_config.eval_model,
                mp=trainer_config.mp,
            ),
            every=labeled_eval_config.steps or trainer_config.steps_per_eval,
        )
