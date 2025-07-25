"""
This file represents the best practices for each stage of the pipeline.
"""

import dataclasses
import logging
import os
from collections.abc import Sequence
from datetime import timedelta
from functools import lru_cache
from typing import Any

import jmp
from haliax.partitioning import ResourceAxis
from haliax.quantization import QuantizationConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.compat.hf_checkpoints import load_tokenizer
from levanter.data.text import LmDatasetFormatBase, LMMixtureDatasetConfig, TextLmDatasetFormat
from levanter.eval_harness import LmEvalHarnessConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.models.llama import LlamaConfig
from levanter.models.lm_model import LmConfig
from levanter.optim import AdamConfig
from levanter.schedule import BatchSchedule
from levanter.store.cache import CacheOptions
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils import fsspec_utils

from experiments.anneal_config import AnnealConfig
from experiments.evals.task_configs import (
    CORE_TASKS,
    MMLU_TASKS,
    convert_to_levanter_task_config,
    convert_to_task_metrics,
)
from experiments.llama import compute_num_parameters, llama_8b
from experiments.paloma import paloma_tokenized
from experiments.simple_sft_config import SimpleSFTConfig
from experiments.simple_train_config import SimpleTrainConfig
from marin.download.huggingface.download import DownloadConfig
from marin.download.huggingface.download_hf import download_hf
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import (
    ExecutorStep,
    InputName,
    ensure_versioned,
    get_executor_step,
    this_output_path,
    unwrap_versioned_value,
)
from marin.processing.tokenize import (
    TokenizeConfig,
    TokenizerStep,
    add_validation_sets_to_mixture,
    lm_data_config,
    tokenize,
)
from marin.processing.tokenize.tokenize import HfTokenizeConfig, TokenizeConfigBase
from marin.scaling_laws.scaling_laws import ScalingLawConfig, run_scaling_law_analysis
from marin.training.training import (
    TrainLmOnPodConfig,
    run_levanter_train_lm,
)

logger = logging.getLogger("ray")


def default_download(
    name: str,
    hf_dataset_id: str,
    revision: str,
    override_output_path: str | None = None,
    **kwargs: Any,
) -> InputName:
    """
    Download a HuggingFace dataset and upload it to a specified path with default configuration.

    Args:
        name: The name of the Download step. It forms the basis of the output path
            unless override_output_path is explicitly specified.
        hf_dataset_id: The HuggingFace dataset ID to download. As `$ORG/$DATASET` on HF Hub
        revision: The revision of the dataset to download.
            Short Commit Hash from HF Dataset Repo (7 characters)
        override_output_path: Optional. The output path for the dataset.
        **kwargs: Additional keyword arguments that are passed to the download config.

    The final output data will reside in '{output_path}/{revision}'.
    """

    step = ExecutorStep(
        name=name,
        description=f"Download {hf_dataset_id} revision {revision}",
        fn=download_hf,
        config=DownloadConfig(
            hf_dataset_id=hf_dataset_id,
            revision=revision,
            gcs_output_path=this_output_path(),
            wait_for_completion=True,
            **kwargs,
        ),
        override_output_path=override_output_path,
    )

    return step.as_input_name()


def default_tokenize(
    name: str,
    dataset: InputName | ExecutorStep | str,
    tokenizer: str,
    options: CacheOptions | None = None,
    format: LmDatasetFormatBase = TextLmDatasetFormat(),  # noqa
    *,
    is_validation: bool = False,
) -> ExecutorStep:
    """
    Tokenizes a dataset using the specified tokenizer and Levanter's tokenization infrastructure.

    Args:
        name: The name of the tokenized dataset. This is used to form the output path for the executor step.
            `tokenized/` will be prepended to the name.
        dataset:  The dataset to tokenize. This can be an InputName, ExecutorStep, or a string as a
            path to the dataset or a HuggingFace dataset ID.
        tokenizer: string HuggingFace tokenizer name. Should be the same as you intend to use in the tokenizer
            spec for the training run.
        options: CacheOptions to use for tokenization. You typically don't need to set this.
        format: The format of the dataset. This is used to determine how to tokenize the data.

            See [Levanter's documentation](https://levanter.readthedocs.io/en/latest/reference/Data-Formats/)
            for more details.
        is_validation: Whether the dataset is a validation set. Doesn't do anything for HF datasets.
    Returns:
        An ExecutorStep that represents the tokenized dataset.
    """

    # sniff out if it's a HuggingFace dataset
    if isinstance(dataset, str) and dataset.count("/") == 1 and not fsspec_utils.exists(dataset):
        config = HfTokenizeConfig(
            id=dataset,
            cache_path=this_output_path(),
            tokenizer=ensure_versioned(tokenizer),
            format=format,
        )
    else:
        config = TokenizeConfig(
            train_paths=[dataset] if not is_validation else [],
            validation_paths=[dataset] if is_validation else [],
            cache_path=this_output_path(),
            tokenizer=ensure_versioned(tokenizer),
            format=format,
        )

    if options is not None:
        config = dataclasses.replace(config, cache_options=options)

    return ExecutorStep(
        name=os.path.join("tokenized", name),
        description=f"Tokenize raw text using the {tokenizer} tokenizer.",
        fn=tokenize,
        config=config,
        pip_dependency_groups=["tokenize_train"],
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
        eval_harness_tasks: List of evaluation harness tasks. Defaults to the CORE set of tasks. Use () or [] to disable
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
    override_output_path: str | None = None,
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
        eval_harness_tasks: List of evaluation harness tasks. Defaults to the CORE set of tasks. Use () or [] to disable
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

    if train_config.per_device_eval_parallelism is None:
        per_device_eval_parallelism = -1
    else:
        per_device_eval_parallelism = train_config.per_device_eval_parallelism

    schedule = BatchSchedule(unwrap_versioned_value(train_config.train_batch_size))
    total_examples = schedule.global_data_offset_by_step(train_config.num_train_steps)

    checkpoint_path_to_load_from = train_config.initialize_from_checkpoint_path
    hf_checkpoint_path_to_load_from = train_config.initialize_from_hf

    if hf_checkpoint_path_to_load_from is not None and checkpoint_path_to_load_from is not None:
        raise ValueError("Cannot specify both initialize_from_checkpoint_path and initialize_from_hf")

    # Create the inner config
    inner_config = TrainLmConfig(
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
            per_device_eval_parallelism=per_device_eval_parallelism,
            max_eval_batches=train_config.max_eval_batches,
            allow_nondivisible_batch_size=True,
            quantization=QuantizationConfig(int8=train_config.int8) if train_config.int8 else None,
            initialize_from=None if train_config.reset_data_loader_on_init else checkpoint_path_to_load_from,
            watch=train_config.watch,
            axis_resources={
                # Special axes for MoEs
                "token": (ResourceAxis.REPLICA, ResourceAxis.DATA),
                "token_repeat": (ResourceAxis.REPLICA, ResourceAxis.DATA),
            },
        ),
        initialize_from_checkpoint_path=(
            checkpoint_path_to_load_from if train_config.reset_data_loader_on_init else None
        ),
        initialize_from_hf=hf_checkpoint_path_to_load_from or False,
        z_loss_weight=train_config.z_loss_weight,
        model=model_config,
        optimizer=(
            train_config.optimizer_config
            if getattr(train_config, "optimizer_config", None) is not None
            else AdamConfig(
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
                rewarmup=(train_config.rewarmup if train_config.rewarmup is not None else AdamConfig().rewarmup),
                decay=(train_config.decay if train_config.decay is not None else AdamConfig().decay),
                lr_schedule=(
                    train_config.lr_schedule if train_config.lr_schedule is not None else AdamConfig().lr_schedule
                ),
                cycle_length=train_config.cycle_length,  # can be int, list[int], or None
                min_lr_ratio=(
                    train_config.min_lr_ratio if train_config.min_lr_ratio is not None else AdamConfig().min_lr_ratio
                ),
                skip_bad_steps=train_config.skip_bad_steps,
            )
        ),
        hf_save_steps=steps_per_export_hf,
        data_seed=train_config.data_seed,
        eval_harness_steps=train_config.steps_per_task_eval or 10000,
        eval_harness=harness_config,
    )

    # Create the pod config
    pod_config = train_config.resources

    # Create the full config
    config = TrainLmOnPodConfig(
        train_config=inner_config,
        resources=pod_config,
        output_path=this_output_path(),
    )

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
        config=config,
        pip_dependency_groups=["tokenize_train"],
        override_output_path=override_output_path,
    )


def default_sft(
    name: str,
    tokenized: InputName | ExecutorStep | LMMixtureDatasetConfig,
    model_config: LlamaConfig,
    sft_config: SimpleSFTConfig,
    tags: Sequence[str] = (),
) -> ExecutorStep:
    """
    Creates an ExecutorStep for supervised fine-tuning of a language model.

    This function provides a unified interface for both single-dataset SFT and mixture-based
    SFT with a simplified configuration approach.

    Args:
        name: The name of the training run, forms the basis of the output path.
        tokenized: The tokenized data to train on:
                  - For single dataset: an InputName or ExecutorStep for a tokenized dataset.
                  - For mixture: a LMMixtureDatasetConfig with multiple datasets.
        model_config: Levanter LlamaConfig for the model architecture to train.
        sft_config: Configuration for the SFT training process.
        tags: Additional tags for WandB logging. Default: ().

    Returns:
        An ExecutorStep configured for supervised fine-tuning.
    """
    # Set up common configurations
    if "sft" not in tags:
        tags = [*tags, "sft"]

    initialize_from_hf = sft_config.initialize_from_hf

    if initialize_from_hf is None:
        initialize_from_hf = (
            sft_config.model_name_or_path is not None and sft_config.initialize_from_checkpoint_path is None
        )
    elif initialize_from_hf is True and sft_config.model_name_or_path is None:
        raise ValueError("initialize_from_hf is True but model_name_or_path is not set")
    elif initialize_from_hf is False and sft_config.initialize_from_checkpoint_path is None:
        raise ValueError("initialize_from_hf is False but initialize_from_checkpoint_path is not set")

    # now we just shell out to default_train
    normal_train_config = SimpleTrainConfig(
        resources=sft_config.resources,
        train_batch_size=sft_config.train_batch_size,
        num_train_steps=sft_config.num_train_steps,
        learning_rate=sft_config.learning_rate,
        lr_schedule=sft_config.lr_schedule,
        decay=sft_config.cooldown,
        weight_decay=sft_config.weight_decay,
        min_lr_ratio=sft_config.min_lr_ratio,
        max_grad_norm=sft_config.max_grad_norm,
        warmup=sft_config.warmup,
        steps_per_eval=sft_config.steps_per_eval,
        steps_per_export=sft_config.steps_per_checkpoint,
        int8=sft_config.int8,
        steps_per_hf_export=sft_config.steps_per_hf_export,
        initialize_from_hf=sft_config.model_name_or_path if initialize_from_hf else None,
        initialize_from_checkpoint_path=sft_config.initialize_from_checkpoint_path,
        data_seed=sft_config.seed,
        z_loss_weight=sft_config.z_loss_weight,
    )

    if sft_config.reinit_tokens:
        raise NotImplementedError("reinit_tokens is not supported by default_train")

    # Create and return the ExecutorStep
    return default_train(
        name=name,
        tokenized=tokenized,
        model_config=model_config,
        train_config=normal_train_config,
        tags=tags,
        eval_harness_tasks=[],
        use_default_validation=False,
    )


def default_anneal(name: str, anneal_config: AnnealConfig) -> ExecutorStep:
    """

    Runs an annealing training run. This is a kind of continued pre-training intended
    to replicate Llama 3-style data ablations (or XXX databricks microannealing)

    Args:
        name: The name of the training run. Will form the basis of the output path for the executor step.
              `checkpoints/` will be prepended to the name.
        anneal_config: Configuration for the annealing run.
    Returns:

        An ExecutorStep configured for annealing.

    """
    checkpoint_path = anneal_config.initialize_from_checkpoint_path
    imputed_checkpoint_step = _impute_checkpoint_step(checkpoint_path)

    num_anneal_steps = anneal_config.num_anneal_training_tokens / (
        anneal_config.train_batch_size * AnnealConfig.LLAMA_MAX_SEQ_LEN
    )
    num_train_steps = imputed_checkpoint_step + num_anneal_steps

    # We need to simulate having a learning rate that decays from anneal_config.learning rate to 0
    # over the course of the training. However, we have already taken anneal_config.checkpoint_step steps,
    # so we need to calculate what the max lr would've been if we had started training with a linear schedule
    # and then decayed it to 0 over the course of the training.
    # The formula for the max lr is:
    # max_lr = num_train_steps * slope
    # slope = anneal_config.learning_rate / num_anneal_steps

    learning_rate = num_train_steps * (anneal_config.learning_rate / num_anneal_steps)

    anneal_stage_train_config = SimpleTrainConfig(
        resources=anneal_config.resources,
        train_batch_size=anneal_config.train_batch_size,
        num_train_steps=num_train_steps,
        learning_rate=learning_rate,
        weight_decay=anneal_config.weight_decay,
        min_lr_ratio=anneal_config.min_lr_ratio,
        steps_per_export=anneal_config.steps_per_export,
        lr_schedule=anneal_config.lr_schedule,
        initialize_from_checkpoint_path=checkpoint_path,
    )

    return default_train(
        name=name,
        tokenized=anneal_config.dataset_config,
        model_config=llama_8b,
        train_config=anneal_stage_train_config,
        use_default_validation=anneal_config.use_default_validation,
        eval_harness_tasks=MMLU_TASKS,
    )


def _impute_checkpoint_step(checkpoint_path: str | InputName) -> int:
    """
    Extracts the checkpoint step from a checkpoint path.
    Args:
        checkpoint_path:

    Returns:

    """
    if isinstance(checkpoint_path, InputName):
        checkpoint_path = checkpoint_path.name
    imputed_checkpoint_steps = checkpoint_path.index("step-")
    imputed_checkpoint_step = int(checkpoint_path[imputed_checkpoint_steps + len("step-") :])
    return imputed_checkpoint_step


@lru_cache
def _cached_load_tokenizer(tokenizer_name: str):
    return load_tokenizer(tokenizer_name)


def _get_vocab_size(pretraining_data):
    tokenizer = unwrap_versioned_value(pretraining_data.tokenizer)
    vocab_size = _cached_load_tokenizer(tokenizer).vocab_size
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
        validation_sets = {}

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
        case ExecutorStep(config=config) if isinstance(config, TokenizeConfigBase):
            tokenizer = config.tokenizer
        case ExecutorStep(config=HfTokenizeConfig(tokenizer=tokenizer)):
            pass
        case InputName(step=ExecutorStep(config)) if isinstance(config, TokenizeConfigBase):
            tokenizer = config.tokenizer
        case _:
            raise ValueError(f"Could not determine tokenizer from {tokenized}")

    return tokenizer


def default_scaling_law_pred(
    ladder_runs: Sequence[ExecutorStep | InputName | str],
    pred_run: ExecutorStep | InputName | str | None = None,
    task_losses: Sequence[str] = ("eval/paloma/c4_en/bpb",),
    task_accuracies: Sequence[str] | Sequence[EvalTaskConfig] | None = None,
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
    if task_accuracies is not None:
        task_accuracies = convert_to_task_metrics(task_accuracies, metric="acc")

    if pred_run_or_id:
        name = pred_run_or_id if isinstance(pred_run_or_id, str) else pred_run_or_id.name
    else:
        name = "projection"

    return ExecutorStep(
        name=f"""scaling_laws/{name}""",
        fn=run_scaling_law_analysis,
        config=ScalingLawConfig(
            name=name,
            ladder_model_steps=ladder_steps_or_ids,
            pred_model_step=pred_run_or_id,
            task_losses=task_losses,
            task_accuracies=task_accuracies,
        ),
    )
