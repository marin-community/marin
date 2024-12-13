"""
This file represents the best practices for each stage of the pipeline.
"""

import dataclasses
import os
from collections.abc import Sequence
from datetime import timedelta
from functools import lru_cache

import jmp
from levanter.checkpoint import CheckpointerConfig
from levanter.compat.hf_checkpoints import load_tokenizer
from levanter.data.text import LMMixtureDatasetConfig, SupervisedSourceConfig, SupervisedUrlSourceConfig
from levanter.eval_harness import LmEvalHarnessConfig
from levanter.models.llama import LlamaConfig
from levanter.models.lm_model import LmConfig
from levanter.optim import AdamConfig
from levanter.store.cache import CacheOptions
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig

from experiments.eval_datasets import (
    eval_datasets,
)
from experiments.evals.task_configs import CORE_TASKS, convert_to_levanter_task_config
from experiments.llama import compute_num_parameters
from experiments.paloma import paloma_tokenized
from experiments.simple_train_config import SimpleTrainConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import (
    ExecutorStep,
    InputName,
    output_path_of,
    this_output_path,
    unwrap_versioned_value,
    versioned,
)
from marin.processing.tokenize import (
    TokenizeConfig,
    TokenizerStep,
    add_validation_sets_to_mixture,
    levanter_tokenize_supervised,
    lm_data_config,
    tokenize,
)
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm


def default_tokenize(
    name: str,
    dataset: InputName | ExecutorStep,
    tokenizer: str,
    options: CacheOptions | None = None,
    text_key: str = "text",
) -> ExecutorStep:
    config = TokenizeConfig(
        train_paths=[dataset],
        validation_paths=[],
        cache_path=this_output_path(),
        tokenizer=versioned(tokenizer),
        text_key=text_key,
    )
    if options is not None:
        config = dataclasses.replace(config, cache_options=options)

    return ExecutorStep(
        name=os.path.join("tokenized", name),
        description=f"Tokenize raw text using the {tokenizer} tokenizer.",
        fn=tokenize,
        config=config,
    )


@lru_cache  # LRU to make the executor happier
def default_validation_sets(tokenizer: str, base_path: str = "tokenized/") -> dict[str, TokenizerStep]:
    return paloma_tokenized(base_path=base_path, tokenizer=tokenizer)


@lru_cache  # LRU to make the executor happier
def default_evaluation_data(tokenizer: str) -> dict[str, SupervisedSourceConfig]:

    eval_dataconfigs = {}

    for dataset in eval_datasets:
        validation_paths = [output_path_of(step).cd(f"{dataset.org}/*.jsonl.gz") for step in dataset.steps]

        cache = ExecutorStep(
            name=f"tokenized/evaluation/{dataset.name}",
            fn=levanter_tokenize_supervised,
            config=TokenizeConfig(
                train_paths=[],
                validation_paths=validation_paths,
                cache_path=this_output_path(),
                input_field="prompt",
                output_field="response",
                tokenizer=tokenizer,
                tags=dataset.tags,
            ),
        )

        eval_dataconfigs[dataset.name] = SupervisedUrlSourceConfig(
            validation_urls=validation_paths,
            cache_dir=output_path_of(cache),
            input_field="prompt",
            output_field="response",
            tags=dataset.tags,
        )
    return eval_dataconfigs


def default_train(
    name: str,
    tokenized: InputName | ExecutorStep | LMMixtureDatasetConfig,
    model_config: LmConfig,
    train_config: SimpleTrainConfig,
    tags: Sequence[str] = (),
    use_default_validation: bool = True,
    use_default_evaluation: bool = True,
    eval_harness_tasks: Sequence[EvalTaskConfig] = CORE_TASKS,
) -> ExecutorStep:
    """
    Train a language model using the default configuration.

    Args:
        name:  The name of the training run. Will form the basis of the output path for the executor step.
        tokenized:  The tokenized data to train on. This can be an InputName, ExecutorStep, or LMMixtureDatasetConfig.
        model_config: Levanter LmConfig for the model to train.
        train_config: SimpleTrainConfig for the training run.
        tags: Any additional tags to add to the Wandb tracker.
        use_default_validation: Whether to use the default validation sets (currently Paloma).
        use_default_evaluation: Whether to use the default supervised validation data (currently MMLU).
        eval_harness_tasks: List of evaluation harness tasks. Defaults to the CORE set of tasks. Use () or [] to disable.
    """

    pretraining_data, evaluation_data = _prepare_data_config(tokenized, use_default_validation, use_default_evaluation)

    vocab_size = _get_vocab_size(pretraining_data)

    steps_per_export = train_config.steps_per_export if train_config.steps_per_export is not None else 1000

    # Max length of 64 characters for WANDB run is 64 characters
    name = name[:64]

    # TODO: right now, assume architecture is a LlamaConfig, generalize this
    assert isinstance(model_config, LlamaConfig)
    return ExecutorStep(
        name=os.path.join("checkpoints", name),
        description=(
            f"Train a {compute_num_parameters(model_config, vocab_size) :,} parameter model for "
            f"{train_config.num_train_steps} (steps) * "
            f"{train_config.train_batch_size} (batch_size) * "
            f"{model_config.seq_len} (seq_len) "
            f"= {train_config.num_train_steps * train_config.train_batch_size * model_config.seq_len:,} tokens."
        ),
        fn=run_levanter_train_lm,
        config=TrainLmOnPodConfig(
            output_path=this_output_path(),
            tpu_type=train_config.tpu_type,
            node_count=train_config.node_count,
            data=pretraining_data,
            supervised_data=evaluation_data,
            trainer=TrainerConfig(
                tracker=WandbConfig(
                    project="marin",
                    tags=[name, *tags],
                ),
                mp=jmp.get_policy("p=f32,c=bfloat16"),
                train_batch_size=train_config.train_batch_size,
                num_train_steps=train_config.num_train_steps,
                steps_per_eval=train_config.steps_per_eval if train_config.steps_per_eval is not None else 1000,
                checkpointer=CheckpointerConfig(
                    save_interval=timedelta(minutes=10),
                    keep=[dict(every=steps_per_export)],
                ),
                replica_dcn_axis_size=-1,
            ),
            z_loss_weight=train_config.z_loss_weight,
            model=model_config,
            optimizer=AdamConfig(
                learning_rate=train_config.learning_rate,
                weight_decay=(
                    train_config.weight_decay if train_config.weight_decay is not None else AdamConfig().weight_decay
                ),
                warmup=train_config.warmup if train_config.warmup is not None else AdamConfig().warmup,
                decay=train_config.decay if train_config.decay is not None else AdamConfig().decay,
                lr_schedule=train_config.lr_schedule if train_config.lr_schedule is not None else AdamConfig.lr_schedule,
                cycle_length=train_config.cycle_length,
                min_lr_ratio=(
                    train_config.min_lr_ratio if train_config.min_lr_ratio is not None else AdamConfig().min_lr_ratio
                ),
            ),
            hf_save_steps=steps_per_export,
            data_seed=train_config.data_seed,
            eval_harness_steps=train_config.steps_per_task_eval or 10000,
            eval_harness=LmEvalHarnessConfig(
                task_spec=convert_to_levanter_task_config(eval_harness_tasks),
            ),
        ),
        pip_dependency_groups=["tokenize_train"],
    )


def _get_vocab_size(pretraining_data):
    tokenizer = unwrap_versioned_value(pretraining_data.tokenizer)
    vocab_size = load_tokenizer(tokenizer).vocab_size
    return vocab_size


def _prepare_data_config(
    tokenized: InputName | ExecutorStep | LMMixtureDatasetConfig,
    use_default_validation: bool,
    use_default_evaluation: bool,
) -> tuple[LMMixtureDatasetConfig, dict[str, SupervisedSourceConfig] | None]:
    """
    Prepare a tokenized dataset for training. This is mostly just combining the tokenized data with the validation sets.

    Returns:
        The data config to use for training with any validation sets added.
        The evaluation data config for internal evaluation.

    """
    tokenizer = _get_tokenizer_for_train(tokenized)
    if use_default_validation:
        validation_sets = default_validation_sets(tokenizer=tokenizer)
    else:
        validation_sets = []
    if use_default_evaluation:
        evaluation_data = default_evaluation_data(tokenizer)
    else:
        evaluation_data = None
    if isinstance(tokenized, InputName | ExecutorStep):
        pretraining_data = lm_data_config(training_set=tokenized, validation_sets=validation_sets)
    else:
        # TODO: would be better to expose hooks in levanter instead of relying on mixtures
        pretraining_data = tokenized
        if validation_sets:
            pretraining_data = add_validation_sets_to_mixture(pretraining_data, validation_sets)
    return pretraining_data, evaluation_data


def _get_tokenizer_for_train(tokenized: InputName | ExecutorStep | LMMixtureDatasetConfig) -> str:
    match tokenized:
        case LMMixtureDatasetConfig(tokenizer=tokenizer):
            pass
        case ExecutorStep(config=TokenizeConfig(tokenizer=tokenizer)):
            pass
        case InputName(step=ExecutorStep(config=TokenizeConfig(tokenizer=tokenizer))):
            pass
        case _:
            raise ValueError(f"Could not determine tokenizer from {tokenized}")

    return tokenizer
