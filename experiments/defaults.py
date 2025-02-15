"""
This file represents the best practices for each stage of the pipeline.
"""

import dataclasses
import logging
import os
from collections.abc import Sequence
from datetime import timedelta
from functools import lru_cache

import jmp
from levanter.checkpoint import CheckpointerConfig
from levanter.compat.hf_checkpoints import load_tokenizer
from levanter.data.text import LMMixtureDatasetConfig
from levanter.eval_harness import LmEvalHarnessConfig
from levanter.models.llama import LlamaConfig
from levanter.models.lm_model import LmConfig
from levanter.optim import AdamConfig
from levanter.schedule import BatchSchedule
from levanter.store.cache import CacheOptions
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig

from experiments.evals.task_configs import CORE_TASKS, convert_to_levanter_task_config, convert_to_task_metrics
from experiments.llama import compute_num_parameters
from experiments.paloma import paloma_tokenized
from experiments.simple_train_config import SimpleTrainConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import (
    ExecutorStep,
    InputName,
    get_executor_step,
    this_output_path,
    unwrap_versioned_value,
    versioned,
)
from marin.processing.tokenize import (
    TokenizeConfig,
    TokenizerStep,
    add_validation_sets_to_mixture,
    lm_data_config,
    tokenize,
)
from marin.scaling_laws.scaling_laws import ScalingLawConfig, run_scaling_law_analysis
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm

logger = logging.getLogger("ray")


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


def simulated_epoching_train(
    name: str,
    tokenized: InputName | ExecutorStep | LMMixtureDatasetConfig,
    model_config: LmConfig,
    train_config: SimpleTrainConfig,
    target_budget: int,
    tags: Sequence[str] = (),
    use_default_validation: bool = True,
    eval_harness_tasks: Sequence[EvalTaskConfig] = CORE_TASKS,
) -> ExecutorStep:
    """
    Simulates the number of epochs seen in a full training run by sub-sampling individual datasets.
    Otherwise, operates the same as default_train.

    Args:
        name:  The name of the training run. Will form the basis of the output path for the executor step.
        tokenized:  The tokenized data to train on. This can be an InputName, ExecutorStep, or LMMixtureDatasetConfig.
        model_config: Levanter LmConfig for the model to train.
        train_config: SimpleTrainConfig for the training run.
        target_budget: Target token budget to simulate.
        tags: Any additional tags to add to the Wandb tracker.
        use_default_validation: Whether to use the default validation sets (currently Paloma).
        eval_harness_tasks: List of evaluation harness tasks. Defaults to the CORE set of tasks. Use () or [] to disable.
    """
    pretraining_data = _prepare_data_config(tokenized, use_default_validation)

    # Extract sequence length from model configuration
    seq_len = model_config.Pos.size

    # Calculate the experiment token budget
    experiment_budget = train_config.train_batch_size * train_config.num_train_steps * seq_len

    simulated_pretraining_data = dataclasses.replace(
        pretraining_data, target_budget=target_budget, experiment_budget=experiment_budget
    )

    logger.info(
        f"Simulating Epoching Behavior, Experiment Tokens {experiment_budget}, "
        + "Simulated Target Tokens {target_budget}"
    )

    return default_train(
        name, simulated_pretraining_data, model_config, train_config, tags, use_default_validation, eval_harness_tasks
    )


def default_train(
    name: str,
    tokenized: InputName | ExecutorStep | LMMixtureDatasetConfig,
    model_config: LmConfig,
    train_config: SimpleTrainConfig,
    tags: Sequence[str] = (),
    use_default_validation: bool = True,
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
        eval_harness_tasks: List of evaluation harness tasks. Defaults to the CORE set of tasks. Use () or [] to disable.
    """

    pretraining_data = _prepare_data_config(tokenized, use_default_validation)

    vocab_size = _get_vocab_size(pretraining_data)

    steps_per_export = train_config.steps_per_export

    # Max length of 64 characters for WANDB run is 64 characters
    # we don't want to use the first 64 because the UID bit goes at the end. instead, grab the trailing -XXX
    # and add whatever we can fit in the remaining space.
    if len(name) > 64:
        old_name = name
        if "-" not in name:
            name = name[:64]
        else:
            prefix, suffix = name.rsplit("-", 1)
            if len(suffix) >= 64:
                suffix = suffix[:64]
                name = suffix
            else:
                name = prefix[: 63 - len(suffix)] + "-" + suffix
        logger.warning(f"Truncated name from {old_name} to {name} to fit within WANDB limits.")

    # TODO: right now, assume architecture is a LlamaConfig, generalize this
    assert isinstance(model_config, LlamaConfig)
    if eval_harness_tasks:
        harness_config = LmEvalHarnessConfig(task_spec=convert_to_levanter_task_config(eval_harness_tasks))
    else:
        harness_config = None

    if train_config.steps_per_hf_export is None:
        steps_per_export_hf = steps_per_export
    elif train_config.steps_per_hf_export == -1:
        steps_per_export_hf = None
    else:
        steps_per_export_hf = train_config.steps_per_hf_export

    model_averaging = None
    if train_config.ema_beta is not None:
        from levanter.optim.model_averaging import EmaModelAveragingConfig

        model_averaging = EmaModelAveragingConfig(beta=train_config.ema_beta)

    schedule = BatchSchedule(train_config.train_batch_size)
    total_examples = schedule.global_data_offset_by_step(train_config.num_train_steps)

    return ExecutorStep(
        name=os.path.join("checkpoints", name),
        description=(
            f"Train a {compute_num_parameters(model_config, vocab_size) :,} parameter model for "
            f"{train_config.num_train_steps} (steps) * "
            f"{train_config.train_batch_size} (batch_size) * "
            f"{model_config.seq_len} (seq_len) "
            f"= {total_examples * model_config.seq_len:,} tokens."
        ),
        fn=run_levanter_train_lm,
        config=TrainLmOnPodConfig(
            output_path=this_output_path(),
            tpu_type=train_config.tpu_type,
            node_count=train_config.node_count,
            allow_out_of_region_reads=train_config.allow_out_of_region_reads,
            allow_out_of_region_writes=train_config.allow_out_of_region_writes,
            data=pretraining_data,
            trainer=TrainerConfig(
                tracker=WandbConfig(
                    project="marin",
                    tags=[*tags],
                ),
                mp=jmp.get_policy("p=f32,c=bfloat16"),
                train_batch_size=train_config.train_batch_size,
                num_train_steps=train_config.num_train_steps,
                steps_per_eval=train_config.steps_per_eval if train_config.steps_per_eval is not None else 1000,
                checkpointer=CheckpointerConfig(
                    save_interval=timedelta(minutes=10),
                    keep=[dict(every=steps_per_export)],
                ),
                model_averaging=model_averaging,
                replica_dcn_axis_size=-1,
                allow_partial_checkpoint=train_config.allow_partial_checkpoint,
            ),
            z_loss_weight=train_config.z_loss_weight,
            model=model_config,
            optimizer=AdamConfig(
                learning_rate=train_config.learning_rate,
                weight_decay=(
                    train_config.weight_decay if train_config.weight_decay is not None else AdamConfig().weight_decay
                ),
                beta1=(train_config.beta1 if train_config.beta1 is not None else AdamConfig().beta1),
                beta2=(train_config.beta2 if train_config.beta2 is not None else AdamConfig().beta2),
                epsilon=(train_config.epsilon if train_config.epsilon is not None else AdamConfig().epsilon),
                max_grad_norm=(
                    train_config.max_grad_norm if train_config.max_grad_norm is not None else AdamConfig().max_grad_norm
                ),
                warmup=(train_config.warmup if train_config.warmup is not None else AdamConfig().warmup),
                decay=(train_config.decay if train_config.decay is not None else AdamConfig().decay),
                lr_schedule=(
                    train_config.lr_schedule if train_config.lr_schedule is not None else AdamConfig().lr_schedule
                ),
                cycle_length=train_config.cycle_length,  # can be int, list[int], or None
                min_lr_ratio=(
                    train_config.min_lr_ratio if train_config.min_lr_ratio is not None else AdamConfig().min_lr_ratio
                ),
            ),
            hf_save_steps=steps_per_export_hf,
            data_seed=train_config.data_seed,
            eval_harness_steps=train_config.steps_per_task_eval or 10000,
            eval_harness=harness_config,
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
) -> LMMixtureDatasetConfig:
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

    if isinstance(tokenized, InputName | ExecutorStep):
        pretraining_data = lm_data_config(training_set=tokenized, validation_sets=validation_sets)
    else:
        # TODO: would be better to expose hooks in levanter instead of relying on mixtures
        pretraining_data = tokenized
        if validation_sets:
            pretraining_data = add_validation_sets_to_mixture(pretraining_data, validation_sets)
    return pretraining_data


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


def default_scaling_law_pred(
    ladder_runs: Sequence[ExecutorStep | InputName | str],
    pred_run: ExecutorStep | InputName | None = None,
    task_losses: Sequence[str] = ("eval/paloma/c4_en/bpb"),
    task_accuracies: Sequence[str] | Sequence[EvalTaskConfig] = ("lm_eval/hellaswag_10shot/acc"),
):
    """
    Given a suite of small models, predict the performance on a number of (N, D) values.
    """
    # get the executor steps or run IDs for the ladder runs and the pred run
    ladder_steps_or_ids = [get_executor_step(run) if not isinstance(run, str) else run for run in ladder_runs]

    pred_run_or_id = None
    if pred_run:
        pred_run_or_id = get_executor_step(pred_run) if not isinstance(pred_run, str) else pred_run

    # convert the task accuracies to strings if they are `EvalTaskConfig`s
    if isinstance(task_accuracies[0], EvalTaskConfig):
        task_accuracies = convert_to_task_metrics(task_accuracies, metric="acc")

    if pred_run_or_id:
        name = pred_run_or_id if isinstance(pred_run_or_id, str) else pred_run_or_id.name
    else:
        name = "projection"

    return ExecutorStep(
        name=f"""scaling_laws/{name}""",
        fn=run_scaling_law_analysis,
        config=ScalingLawConfig(
            ladder_model_steps=ladder_steps_or_ids,
            pred_model_step=pred_run_or_id,
            task_losses=task_losses,
            task_accuracies=task_accuracies,
        ),
    )
