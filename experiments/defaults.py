# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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
from fray.v2 import ResourceConfig
from marin.execution.remote import remote
from haliax.partitioning import ResourceAxis
from haliax.quantization import QuantizationConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import (
    BlockShuffleConfig,
    LmDatasetFormatBase,
    LMMixtureDatasetConfig,
    PreferenceLmDataConfig,
    TextLmDatasetFormat,
)
from levanter.eval_harness import LmEvalHarnessConfig
from levanter.main.train_dpo import TrainDpoConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.models.llama import LlamaConfig
from levanter.models.lm_model import LmConfig
from levanter.optim import AdamConfig
from levanter.schedule import BatchSchedule
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils import fsspec_utils

from experiments.evals.task_configs import CORE_TASKS
from marin.evaluation.evaluation_config import convert_to_levanter_task_config
from experiments.paloma import paloma_tokenized
from experiments.simple_dpo_config import SimpleDPOConfig
from experiments.simple_sft_config import SimpleSFTConfig
from experiments.simple_train_config import SimpleTrainConfig
from levanter.utils.mesh import MeshConfig
from marin.datakit.download.huggingface import DownloadConfig, download_hf
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import (
    ExecutorStep,
    InputName,
    VersionedValue,
    ensure_versioned,
    this_output_path,
    unwrap_versioned_value,
    versioned,
)
from marin.processing.tokenize import (
    HfDatasetSpec,
    TokenizeConfig,
    TokenizerStep,
    add_validation_sets_to_mixture,
    lm_data_config,
    tokenize,
)
from marin.processing.tokenize.tokenize import HfTokenizeConfig, TokenizeConfigBase
from marin.training.training import (
    TrainDpoOnPodConfig,
    TrainLmOnPodConfig,
    run_levanter_train_dpo,
    run_levanter_train_lm,
)

logger = logging.getLogger(__name__)


HF_BUCKET_URI_PREFIX = "hf://buckets/"
HF_BUCKET_PATH_PREFIX = "buckets/"


def _is_hf_bucket_path(path: str) -> bool:
    return path.startswith(HF_BUCKET_URI_PREFIX) or path.startswith(HF_BUCKET_PATH_PREFIX)


def _normalize_hf_bucket_path(path: str) -> str:
    if path.startswith(HF_BUCKET_URI_PREFIX):
        return path.removeprefix("hf://")
    return path


DEFAULT_NEW_RUN_DATA_SHUFFLE = BlockShuffleConfig(
    io_block_size=256,
    window_blocks=512,
    perm_type="feistel",
)
"""Hierarchical block-shuffle default for newly constructed training runs."""


def _truncate_wandb_name(name: str) -> str:
    """Truncate a run name to fit WANDB's 64-character limit, preserving the trailing suffix."""
    if len(name) <= 64:
        return name
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
    return name


def _resolve_hf_export_steps(steps_per_hf_export: int | None, steps_per_export: int | None) -> int | None:
    """Resolve the HF export step interval: None means same as checkpoint, -1 means disabled."""
    if steps_per_hf_export is None:
        return steps_per_export
    if steps_per_hf_export == -1:
        return None
    return steps_per_hf_export


def _checkpoint_keep(steps_per_export: int | None) -> list[dict]:
    """Build the `keep` list for `CheckpointerConfig`.

    None means keep no permanent intermediate checkpoints (only the final checkpoint
    is saved at end-of-training, plus a rolling temporary checkpoint for resumption).
    """
    if steps_per_export is None:
        return []
    return [dict(every=steps_per_export)]


def _validate_train_length(train_seq_len: int | None, model_config: LmConfig) -> int:
    """Resolve and validate the training sequence length against the model's max."""
    actual = unwrap_versioned_value(model_config)
    train_length = train_seq_len or actual.max_seq_len
    if train_length > actual.max_seq_len:
        raise ValueError(f"train_length {train_length} exceeds model max_seq_len {actual.max_seq_len}.")
    return train_length


def default_download(
    name: str,
    hf_dataset_id: str,
    revision: str | None = None,
    override_output_path: str | None = None,
    **kwargs: Any,
) -> InputName:
    """
    Download a HuggingFace dataset and upload it to a specified path with default configuration.

    Args:
        name: The name of the Download step. It forms the basis of the output path
            unless override_output_path is explicitly specified.
        hf_dataset_id: Hugging Face source. Either `$ORG/$DATASET` on HF Hub or `hf://buckets/...`.
        revision: The revision of the dataset to download for Hub datasets.
            Optional for bucket paths.
        override_output_path: Optional. The output path for the dataset.
        **kwargs: Additional keyword arguments that are passed to the download config.

    The final output data will reside in '{output_path}/{revision}'.
    """

    download_kwargs = dict(kwargs)
    hf_repo_type_prefix = download_kwargs.pop("hf_repo_type_prefix", None)
    if _is_hf_bucket_path(hf_dataset_id):
        normalized_dataset_id = _normalize_hf_bucket_path(hf_dataset_id)
        description = f"Download {hf_dataset_id}"
        resolved_hf_repo_type_prefix = "" if hf_repo_type_prefix is None else hf_repo_type_prefix
        resolved_revision = "main" if revision is None else revision
    else:
        if revision is None:
            raise ValueError("revision is required for non-bucket Hugging Face dataset downloads.")
        normalized_dataset_id = hf_dataset_id
        description = f"Download {hf_dataset_id} revision {revision}"
        resolved_hf_repo_type_prefix = "datasets" if hf_repo_type_prefix is None else hf_repo_type_prefix
        resolved_revision = revision

    step = ExecutorStep(
        name=name,
        description=description,
        fn=download_hf,
        config=DownloadConfig(
            hf_dataset_id=normalized_dataset_id,
            revision=resolved_revision,
            gcs_output_path=this_output_path(),
            wait_for_completion=True,
            hf_repo_type_prefix=resolved_hf_repo_type_prefix,
            **download_kwargs,
        ),
        override_output_path=override_output_path,
    )

    return step.as_input_name()


def default_tokenize(
    name: str,
    dataset: InputName | ExecutorStep | str | HfDatasetSpec,
    tokenizer: str,
    format: LmDatasetFormatBase = TextLmDatasetFormat(),  # noqa
    *,
    sample_count: int | VersionedValue[int] | None = None,
    is_validation: bool = False,
    levanter_batch_size: int | None = None,
    resources: ResourceConfig | None = None,
    worker_resources: ResourceConfig | None = None,
) -> ExecutorStep:
    """
    Tokenizes a dataset using the specified tokenizer and Levanter's tokenization infrastructure.

    Args:
        name: The name of the tokenized dataset. This is used to form the output path for the executor step.
            `tokenized/` will be prepended to the name.
        dataset:  The dataset to tokenize. This can be an InputName, ExecutorStep, a string as a
            path to the dataset or a HuggingFace dataset ID, or ``HfDatasetSpec`` to specify a
            dataset with a particular subset name.
        tokenizer: string HuggingFace tokenizer name. Should be the same as you intend to use in the tokenizer
            spec for the training run.
        format: The format of the dataset. This is used to determine how to tokenize the data.

            See [Levanter's documentation](https://levanter.readthedocs.io/en/latest/reference/Data-Formats/)
            for more details.
        sample_count: Optional limit on the number of samples to tokenize per shard. If ``None``, tokenize everything.
        is_validation: Whether the dataset is a validation set. Doesn't do anything for HF datasets.
    Returns:
        An ExecutorStep that represents the tokenized dataset.
    """

    # Common kwargs for config constructors
    extra_kwargs: dict = {}
    if worker_resources is not None:
        extra_kwargs["worker_resources"] = worker_resources

    # sniff out if it's a HuggingFace dataset
    if isinstance(dataset, HfDatasetSpec):
        config = HfTokenizeConfig(
            id=dataset.id,
            name=dataset.name,
            cache_path=this_output_path(),
            tokenizer=ensure_versioned(tokenizer),
            format=format,
            sample_count=ensure_versioned(sample_count) if sample_count is not None else None,
            levanter_batch_size=levanter_batch_size,
            **extra_kwargs,
        )
    elif (
        isinstance(dataset, str)
        and not _is_hf_bucket_path(dataset)
        and dataset.count("/") == 1
        and not fsspec_utils.exists(dataset)
    ):
        config = HfTokenizeConfig(
            id=dataset,
            cache_path=this_output_path(),
            tokenizer=ensure_versioned(tokenizer),
            format=format,
            sample_count=ensure_versioned(sample_count) if sample_count is not None else None,
            levanter_batch_size=levanter_batch_size,
            **extra_kwargs,
        )
    else:
        config = TokenizeConfig(
            train_paths=[dataset] if not is_validation else [],
            validation_paths=[dataset] if is_validation else [],
            cache_path=this_output_path(),
            tokenizer=ensure_versioned(tokenizer),
            format=format,
            sample_count=ensure_versioned(sample_count) if sample_count is not None else None,
            levanter_batch_size=levanter_batch_size,
            **extra_kwargs,
        )

    return ExecutorStep(
        name=os.path.join("tokenized", name),
        description=f"Tokenize raw text using the {tokenizer} tokenizer.",
        fn=remote(
            tokenize,
            resources=resources or ResourceConfig.with_cpu(cpu=4, ram="16g", disk="10g"),
            pip_dependency_groups=["cpu"],
            env_vars={
                "TRANSFORMERS_NO_TORCH": "1",
                "TRANSFORMERS_NO_TORCHVISION": "1",
                "USE_TORCH": "0",
                "TORCH_DISABLE_GLOBAL_DEPS": "1",
            },
        ),
        config=config,
    )


@lru_cache  # LRU to make the executor happier
def default_validation_sets(tokenizer: str, base_path: str = "tokenized/") -> dict[str, TokenizerStep]:
    # Avoid circular dependencies
    # TODO: Will - break apart defaults a bit
    from experiments.evals.exp1600_uncheatable_evals import uncheatable_eval_tokenized

    validation_sets = dict(paloma_tokenized(base_path=base_path, tokenizer=tokenizer))
    validation_sets.update(uncheatable_eval_tokenized(base_path=base_path, tokenizer=tokenizer))
    return validation_sets


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

    train_length = _validate_train_length(train_config.train_seq_len, model_config)

    # Calculate the experiment token budget
    experiment_budget = train_config.train_batch_size * train_config.num_train_steps * train_length

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
    wandb_name: str | None = None,
    wandb_group: str | None = None,
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
        wandb_name: Optional W&B display name for this run. Defaults to W&B's auto-generated name.
        wandb_group: Optional W&B group to organize related runs (e.g., a sweep). If unset, defaults to $WANDB_GROUP.
    """

    pretraining_data = _prepare_data_config(tokenized, use_default_validation)

    tokenizer_name = unwrap_versioned_value(pretraining_data.tokenizer)
    steps_per_export = train_config.steps_per_export

    if wandb_group is None:
        wandb_group = os.environ.get("WANDB_GROUP")

    name = _truncate_wandb_name(name)

    if eval_harness_tasks:
        harness_config = LmEvalHarnessConfig(task_spec=convert_to_levanter_task_config(eval_harness_tasks))
    else:
        harness_config = None

    steps_per_export_hf = _resolve_hf_export_steps(train_config.steps_per_hf_export, steps_per_export)

    model_averaging = None
    if train_config.ema_beta is not None:
        from levanter.optim.model_averaging import EmaModelAveragingConfig

        model_averaging = EmaModelAveragingConfig(beta=train_config.ema_beta)

    if train_config.per_device_eval_parallelism is None:
        per_device_eval_parallelism = -1
    else:
        per_device_eval_parallelism = train_config.per_device_eval_parallelism

    schedule = BatchSchedule(unwrap_versioned_value(train_config.train_batch_size))
    total_examples = schedule.global_data_offset_by_step(unwrap_versioned_value(train_config.num_train_steps))

    checkpoint_path_to_load_from = train_config.initialize_from_checkpoint_path
    hf_checkpoint_path_to_load_from = train_config.initialize_from_hf

    if hf_checkpoint_path_to_load_from is not None and checkpoint_path_to_load_from is not None:
        raise ValueError("Cannot specify both initialize_from_checkpoint_path and initialize_from_hf")

    train_length = _validate_train_length(train_config.train_seq_len, model_config)

    inner_config = TrainLmConfig(
        data=pretraining_data,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                project=os.getenv("WANDB_PROJECT", "marin"),
                name=wandb_name,
                tags=[*tags],
                group=wandb_group,
                replicate_path=this_output_path(),
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=train_config.train_batch_size,
            per_device_parallelism=train_config.per_device_parallelism,
            num_train_steps=train_config.num_train_steps,
            steps_per_eval=train_config.steps_per_eval if train_config.steps_per_eval is not None else 1000,
            checkpointer=CheckpointerConfig(
                save_interval=timedelta(minutes=10),
                keep=_checkpoint_keep(steps_per_export),
            ),
            model_averaging=model_averaging,
            mesh=MeshConfig(
                axes={"replica": 1, "data": -1, "model": train_config.tensor_parallel_size},
                # Special axes for MoEs
                # TODO: this is actually bad and we should remove, but keeping for now
                compute_mapping={
                    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                },
            ),
            allow_partial_checkpoint=train_config.allow_partial_checkpoint,
            per_device_eval_parallelism=per_device_eval_parallelism,
            max_eval_batches=train_config.max_eval_batches,
            allow_nondivisible_batch_size=True,
            quantization=QuantizationConfig(int8=train_config.int8) if train_config.int8 else None,
            initialize_from=None if train_config.reset_data_loader_on_init else checkpoint_path_to_load_from,
            watch=train_config.watch,
            profiler=train_config.profiler,
            use_explicit_mesh_axes=train_config.explicit_mesh_axes,
        ),
        initialize_from_checkpoint_path=(
            checkpoint_path_to_load_from if train_config.reset_data_loader_on_init else None
        ),
        initialize_from_hf=hf_checkpoint_path_to_load_from or False,
        pad_tokenizer_to_match_model=train_config.pad_tokenizer_to_match_model,
        z_loss_weight=train_config.z_loss_weight,
        train_seq_len=train_length,
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
        hf_generation_eos_token_ids=train_config.hf_generation_eos_token_ids,
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
        env_vars=train_config.env_vars,
    )

    model_config = unwrap_versioned_value(model_config)

    return ExecutorStep(
        name=os.path.join("checkpoints", name),
        description=(
            f"Train a model (tokenizer={tokenizer_name}) for "
            f"{unwrap_versioned_value(train_config.num_train_steps)} (steps) * "
            f"{unwrap_versioned_value(train_config.train_batch_size)} (batch_size) * "
            f"{train_length} (train_seq_len) "
            f"= {total_examples * train_length} tokens."
        ),
        fn=run_levanter_train_lm,
        config=config,
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

    if sft_config.initialize_from_hf is not None and sft_config.initialize_from_checkpoint_path is not None:
        raise ValueError("Cannot specify both initialize_from_hf and initialize_from_checkpoint_path!")

    # now we just shell out to default_train
    normal_train_config = SimpleTrainConfig(
        resources=sft_config.resources,
        train_batch_size=sft_config.train_batch_size,
        num_train_steps=sft_config.num_train_steps,
        learning_rate=sft_config.learning_rate,
        lr_schedule=sft_config.lr_schedule,
        decay=sft_config.decay,
        weight_decay=sft_config.weight_decay,
        min_lr_ratio=sft_config.min_lr_ratio,
        max_grad_norm=sft_config.max_grad_norm,
        warmup=sft_config.warmup,
        steps_per_eval=sft_config.steps_per_eval,
        steps_per_export=sft_config.steps_per_checkpoint,
        int8=sft_config.int8,
        steps_per_hf_export=sft_config.steps_per_hf_export,
        initialize_from_hf=sft_config.initialize_from_hf,
        initialize_from_checkpoint_path=sft_config.initialize_from_checkpoint_path,
        train_seq_len=sft_config.max_seq_len,
        data_seed=sft_config.seed,
        z_loss_weight=sft_config.z_loss_weight,
        beta1=sft_config.beta1,
        beta2=sft_config.beta2,
        pad_tokenizer_to_match_model=sft_config.pad_tokenizer_to_match_model,
        per_device_parallelism=sft_config.per_device_parallelism,
        hf_generation_eos_token_ids=sft_config.hf_generation_eos_token_ids,
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


def default_dpo(
    name: str,
    tokenized: InputName | ExecutorStep | LMMixtureDatasetConfig,
    model_config: LlamaConfig,
    dpo_config: SimpleDPOConfig,
    tags: Sequence[str] = (),
    override_output_path: str | None = None,
) -> ExecutorStep:
    """
    Creates an ExecutorStep for DPO fine-tuning.

    Args:
        name: The name of the training run, forms the basis of the output path.
        tokenized: The tokenized preference data to train on.
        model_config: Levanter LlamaConfig for the model architecture to train.
        dpo_config: Configuration for the DPO training process.
        tags: Additional tags for WandB logging. Default: ().
        override_output_path: Optional override for executor output path.
    """
    if "dpo" not in tags:
        tags = [*tags, "dpo"]

    initialize_from_hf = dpo_config.initialize_from_hf

    if initialize_from_hf is None:
        initialize_from_hf = (
            dpo_config.model_name_or_path is not None and dpo_config.initialize_from_checkpoint_path is None
        )
    elif initialize_from_hf is True and dpo_config.model_name_or_path is None:
        raise ValueError("initialize_from_hf is True but model_name_or_path is not set")
    elif initialize_from_hf is False and dpo_config.initialize_from_checkpoint_path is None:
        raise ValueError("initialize_from_hf is False but initialize_from_checkpoint_path is not set")

    pretraining_data = _prepare_data_config(tokenized, use_default_validation=False)
    preference_data = PreferenceLmDataConfig.from_lm_data_config(pretraining_data)
    preference_data = dataclasses.replace(preference_data, permutation_type="feistel")
    dpo_tokenizer_name = unwrap_versioned_value(preference_data.tokenizer)

    name = _truncate_wandb_name(name)

    steps_per_export = dpo_config.steps_per_checkpoint
    steps_per_export_hf = _resolve_hf_export_steps(dpo_config.steps_per_hf_export, steps_per_export)

    train_length = _validate_train_length(dpo_config.train_seq_len, model_config)

    schedule = BatchSchedule(unwrap_versioned_value(dpo_config.train_batch_size))
    total_examples = schedule.global_data_offset_by_step(dpo_config.num_train_steps)

    reference_model_path = dpo_config.reference_model_path or dpo_config.model_name_or_path
    if reference_model_path is None:
        raise ValueError("reference_model_path must be set for DPO training.")

    inner_config = TrainDpoConfig(
        data=preference_data,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                project=dpo_config.wandb_project or "marin",
                tags=[*tags],
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=dpo_config.train_batch_size,
            num_train_steps=dpo_config.num_train_steps,
            steps_per_eval=dpo_config.steps_per_eval,
            checkpointer=CheckpointerConfig(
                save_interval=timedelta(minutes=10),
                keep=_checkpoint_keep(steps_per_export),
            ),
            model_averaging=None,
            mesh=MeshConfig(
                compute_mapping={
                    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                }
            ),
            allow_partial_checkpoint=dpo_config.allow_partial_checkpoint,
            allow_nondivisible_batch_size=True,
            quantization=QuantizationConfig(int8=dpo_config.int8) if dpo_config.int8 else None,
            initialize_from=None,
        ),
        initialize_from_checkpoint_path=dpo_config.initialize_from_checkpoint_path,
        initialize_from_hf=dpo_config.model_name_or_path if initialize_from_hf else False,
        train_seq_len=train_length,
        model=model_config,
        optimizer=AdamConfig(
            learning_rate=dpo_config.learning_rate,
            weight_decay=dpo_config.weight_decay,
            warmup=dpo_config.warmup,
            decay=dpo_config.cooldown,
            lr_schedule=dpo_config.lr_schedule,
            min_lr_ratio=dpo_config.min_lr_ratio,
            max_grad_norm=dpo_config.max_grad_norm,
        ),
        reference_model_path=reference_model_path,
        reference_is_hf=dpo_config.reference_is_hf,
        beta=dpo_config.beta,
        validation_split_fraction=dpo_config.validation_split_fraction,
        hf_save_steps=steps_per_export_hf,
        hf_save_dtype=dpo_config.hf_save_dtype,
        hf_generation_eos_token_ids=dpo_config.hf_generation_eos_token_ids,
        data_seed=dpo_config.seed,
    )

    config = TrainDpoOnPodConfig(
        train_config=inner_config,
        resources=dpo_config.resources,
        output_path=this_output_path(),
    )

    model_config = unwrap_versioned_value(model_config)

    return ExecutorStep(
        name=os.path.join("checkpoints", name),
        description=(
            f"Train a model (tokenizer={dpo_tokenizer_name}) for "
            f"{dpo_config.num_train_steps} (steps) * "
            f"{dpo_config.train_batch_size} (batch_size) * "
            f"{train_length} (train_seq_len) "
            f"= {total_examples * train_length} tokens."
        ),
        fn=run_levanter_train_dpo,
        config=config,
        override_output_path=override_output_path,
    )


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
        pretraining_data = lm_data_config(
            training_set=tokenized,
            validation_sets=validation_sets,
            shuffle=versioned(DEFAULT_NEW_RUN_DATA_SHUFFLE),
        )
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
