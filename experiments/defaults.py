# Copyright 2025 The Marin Authors
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
from haliax.partitioning import ResourceAxis
from haliax.quantization import QuantizationConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import LmDatasetFormatBase, LMMixtureDatasetConfig, TextLmDatasetFormat
from levanter.eval_harness import LmEvalHarnessConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.models.llama import LlamaConfig
from levanter.models.lm_model import LmConfig
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils import fsspec_utils
from levanter.utils.mesh import MeshConfig
from marin.download.huggingface.download_hf import DownloadConfig, download_hf
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.artifact import Artifact
from marin.execution.step_model import StepSpec
from marin.processing.tokenize.tokenize import (
    HfTokenizeConfig,
    TokenizeConfig,
    TokenizedMetadata,
    tokenize,
)
from marin.training.training import (
    TrainLmOnPodConfig,
    run_levanter_train_lm,
)

from experiments.evals.task_configs import (
    CORE_TASKS,
    convert_to_levanter_task_config,
)
from experiments.paloma import paloma_tokenized
from experiments.simple_sft_config import SimpleSFTConfig
from experiments.simple_train_config import SimpleTrainConfig
from marin.processing.tokenize import (
    HfDatasetSpec,
    add_validation_sets_to_mixture,
    lm_data_config,
    lm_mixture_data_config,
)

logger = logging.getLogger("ray")


def default_download(
    name: str,
    hf_dataset_id: str,
    revision: str,
    override_output_path: str | None = None,
    **kwargs: Any,
) -> StepSpec:
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
    return StepSpec(
        name=name,
        hash_attrs={"hf_dataset_id": hf_dataset_id, "revision": revision, **kwargs},
        override_output_path=override_output_path,
        fn=lambda output_path: download_hf(
            DownloadConfig(
                hf_dataset_id=hf_dataset_id,
                revision=revision,
                gcs_output_path=output_path,
                wait_for_completion=True,
                **kwargs,
            )
        ),
    )


def default_tokenize(
    name: str,
    dataset: StepSpec | str | HfDatasetSpec,
    tokenizer: str,
    format: LmDatasetFormatBase = TextLmDatasetFormat(),  # noqa
    *,
    sample_count: int | None = None,
    is_validation: bool = False,
) -> StepSpec:
    """
    Tokenizes a dataset using the specified tokenizer and Levanter's tokenization infrastructure.

    Args:
        name: The name of the tokenized dataset. This is used to form the output path for the executor step.
            `tokenized/` will be prepended to the name.
        dataset: The dataset to tokenize. This can be a StepSpec, a string path to the dataset
            or a HuggingFace dataset ID, or ``HfDatasetSpec`` to specify a dataset with a
            particular subset name.
        tokenizer: string HuggingFace tokenizer name. Should be the same as you intend to use in the tokenizer
            spec for the training run.
        format: The format of the dataset. This is used to determine how to tokenize the data.
        sample_count: Optional limit on the number of samples to tokenize per shard. If ``None``, tokenize everything.
        is_validation: Whether the dataset is a validation set. Doesn't do anything for HF datasets.
    Returns:
        A StepSpec that represents the tokenized dataset.
    """
    deps: list[StepSpec] = []
    hash_attrs: dict[str, Any] = {"tokenizer": tokenizer}
    if sample_count is not None:
        hash_attrs["sample_count"] = sample_count

    if isinstance(dataset, StepSpec):
        dataset_path = dataset.output_path
        deps = [dataset]
    elif isinstance(dataset, str):
        dataset_path = dataset
    elif isinstance(dataset, HfDatasetSpec):
        dataset_path = None  # HF dataset, not a file path
    else:
        raise ValueError(f"Unsupported dataset type: {type(dataset)}")

    # sniff out if it's a HuggingFace dataset
    if isinstance(dataset, HfDatasetSpec):
        hash_attrs["hf_id"] = dataset.id
        if dataset.name:
            hash_attrs["hf_name"] = dataset.name

        def fn(output_path: str) -> TokenizedMetadata:
            return tokenize(
                HfTokenizeConfig(
                    id=dataset.id,
                    name=dataset.name,
                    cache_path=output_path,
                    tokenizer=tokenizer,
                    format=format,
                    sample_count=sample_count,
                )
            )

    elif isinstance(dataset, str) and dataset.count("/") == 1 and not fsspec_utils.exists(dataset):
        hash_attrs["hf_id"] = dataset

        def fn(output_path: str) -> TokenizedMetadata:
            return tokenize(
                HfTokenizeConfig(
                    id=dataset,
                    cache_path=output_path,
                    tokenizer=tokenizer,
                    format=format,
                    sample_count=sample_count,
                )
            )

    else:

        def fn(output_path: str) -> TokenizedMetadata:
            return tokenize(
                TokenizeConfig(
                    train_paths=[dataset_path] if not is_validation else [],
                    validation_paths=[dataset_path] if is_validation else [],
                    cache_path=output_path,
                    tokenizer=tokenizer,
                    format=format,
                    sample_count=sample_count,
                )
            )

    return StepSpec(
        name=os.path.join("tokenized", name),
        hash_attrs=hash_attrs,
        deps=deps,
        fn=fn,
        resources=ResourceConfig.with_cpu(cpu=4, ram="16g", disk="10g"),
        pip_dependency_groups=["cpu"],
        env_vars={
            "TRANSFORMERS_NO_TORCH": "1",
            "TRANSFORMERS_NO_TORCHVISION": "1",
            "USE_TORCH": "0",
            "TORCH_DISABLE_GLOBAL_DEPS": "1",
        },
    )


@lru_cache  # LRU to make the executor happier
def default_validation_sets(tokenizer: str, base_path: str = "tokenized/") -> dict[str, StepSpec]:
    # Avoid circular dependencies
    # TODO: Will - break apart defaults a bit
    from experiments.evals.exp1600_uncheatable_evals import uncheatable_eval_tokenized

    validation_sets = dict(paloma_tokenized(base_path=base_path, tokenizer=tokenizer))
    validation_sets.update(uncheatable_eval_tokenized(base_path=base_path, tokenizer=tokenizer))
    return validation_sets


def simulated_epoching_train(
    name: str,
    tokenized: StepSpec | LMMixtureDatasetConfig,
    model_config: LmConfig,
    train_config: SimpleTrainConfig,
    target_budget: int,
    tags: Sequence[str] = (),
    use_default_validation: bool = True,
    eval_harness_tasks: Sequence[EvalTaskConfig] = CORE_TASKS,
    tokenizer: str | None = None,
) -> StepSpec:
    """
    Simulates the number of epochs seen in a full training run by sub-sampling individual datasets.
    Otherwise, operates the same as default_train.
    """
    # Use explicit training length rather than inferring from the model
    train_length = train_config.train_seq_len or model_config.max_seq_len
    if train_length > model_config.max_seq_len:
        raise ValueError(f"train_length {train_length} exceeds model max_seq_len {model_config.max_seq_len}.")

    # Calculate the experiment token budget
    experiment_budget = train_config.train_batch_size * train_config.num_train_steps * train_length

    logger.info(
        f"Simulating Epoching Behavior, Experiment Tokens {experiment_budget}, "
        + "Simulated Target Tokens {target_budget}"
    )

    return default_train(
        name,
        tokenized,
        model_config,
        train_config,
        tags,
        use_default_validation,
        eval_harness_tasks,
        tokenizer=tokenizer,
        _simulated_epoching_budgets=(target_budget, experiment_budget),
    )


def default_train(
    name: str,
    tokenized: StepSpec | LMMixtureDatasetConfig,
    model_config: LmConfig,
    train_config: SimpleTrainConfig,
    tags: Sequence[str] = (),
    use_default_validation: bool = True,
    eval_harness_tasks: Sequence[EvalTaskConfig] = CORE_TASKS,
    wandb_name: str | None = None,
    wandb_group: str | None = None,
    override_output_path: str | None = None,
    tokenizer: str | None = None,
    _simulated_epoching_budgets: tuple[int, int] | None = None,
) -> StepSpec:
    """
    Train a language model using the default configuration.

    Args:
        name:  The name of the training run. Will form the basis of the output path for the executor step.
        tokenized:  The tokenized data to train on. This can be a StepSpec or LMMixtureDatasetConfig.
        model_config: Levanter LmConfig for the model to train.
        train_config: SimpleTrainConfig for the training run.
        tags: Any additional tags to add to the Wandb tracker.
        use_default_validation: Whether to use the default validation sets (currently Paloma).
        eval_harness_tasks: List of evaluation harness tasks. Defaults to the CORE set of tasks. Use () or [] to disable
        wandb_name: Optional W&B display name for this run. Defaults to W&B's auto-generated name.
        wandb_group: Optional W&B group to organize related runs (e.g., a sweep). If unset, defaults to $WANDB_GROUP.
        tokenizer: The tokenizer name. Required when tokenized is a StepSpec.
    """
    # Resolve tokenizer from the input
    if tokenizer is None:
        if isinstance(tokenized, LMMixtureDatasetConfig):
            tokenizer = tokenized.tokenizer
        elif isinstance(tokenized, StepSpec) and "tokenizer" in tokenized.hash_attrs:
            tokenizer = tokenized.hash_attrs["tokenizer"]
        else:
            raise ValueError("tokenizer must be provided when tokenized is a StepSpec without a 'tokenizer' hash_attr")

    # Collect deps
    deps: list[StepSpec | str] = []
    validation_step_specs: dict[str, StepSpec] = {}
    if isinstance(tokenized, StepSpec):
        deps.append(tokenized)

    if use_default_validation:
        validation_step_specs = default_validation_sets(tokenizer=tokenizer)
        deps.extend(validation_step_specs.values())

    steps_per_export = train_config.steps_per_export

    if wandb_group is None:
        wandb_group = os.environ.get("WANDB_GROUP")

    # Max length of 64 characters for WANDB run is 64 characters
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

    checkpoint_path_to_load_from = train_config.initialize_from_checkpoint_path
    hf_checkpoint_path_to_load_from = train_config.initialize_from_hf

    if hf_checkpoint_path_to_load_from is not None and checkpoint_path_to_load_from is not None:
        raise ValueError("Cannot specify both initialize_from_checkpoint_path and initialize_from_hf")

    train_length = train_config.train_seq_len or model_config.max_seq_len
    if train_length > model_config.max_seq_len:
        raise ValueError(f"train_length {train_length} exceeds model max_seq_len {model_config.max_seq_len}.")

    # Build hash_attrs from key training parameters
    hash_attrs: dict[str, Any] = {
        "model": dataclasses.asdict(model_config),
        "train_batch_size": train_config.train_batch_size,
        "num_train_steps": train_config.num_train_steps,
        "learning_rate": train_config.learning_rate,
        "train_seq_len": train_length,
        "tokenizer": tokenizer,
    }

    # Add optional params that affect version
    for attr in [
        "weight_decay",
        "beta1",
        "beta2",
        "epsilon",
        "max_grad_norm",
        "warmup",
        "decay",
        "rewarmup",
        "lr_schedule",
        "min_lr_ratio",
        "cycle_length",
        "z_loss_weight",
        "ema_beta",
        "data_seed",
        "initialize_from_checkpoint_path",
        "initialize_from_hf",
        "int8",
        "steps_per_export",
    ]:
        val = getattr(train_config, attr, None)
        if val is not None:
            hash_attrs[attr] = val

    if eval_harness_tasks:
        hash_attrs["eval_tasks"] = [t.name for t in eval_harness_tasks]

    if _simulated_epoching_budgets is not None:
        hash_attrs["simulated_epoching_budgets"] = list(_simulated_epoching_budgets)

    # Capture values for closure
    _tokenized = tokenized
    _validation_step_specs = validation_step_specs
    _use_default_validation = use_default_validation
    _simulated_budgets = _simulated_epoching_budgets

    def train_fn(output_path: str):
        # Resolve data config at execution time (after tokenize steps have completed)
        if isinstance(_tokenized, StepSpec):
            metadata = Artifact.load(_tokenized, TokenizedMetadata)
            val_metadata = {
                vname: Artifact.load(vstep, TokenizedMetadata) for vname, vstep in _validation_step_specs.items()
            }
            pretraining_data = lm_data_config(
                training_set=(_tokenized.name, metadata),
                validation_sets=val_metadata if _use_default_validation else None,
            )
        else:
            pretraining_data = _tokenized
            if _use_default_validation and _validation_step_specs:
                val_metadata = {
                    vname: Artifact.load(vstep, TokenizedMetadata) for vname, vstep in _validation_step_specs.items()
                }
                pretraining_data = add_validation_sets_to_mixture(pretraining_data, val_metadata)

        if _simulated_budgets is not None:
            target_budget, experiment_budget = _simulated_budgets
            pretraining_data = dataclasses.replace(
                pretraining_data, target_budget=target_budget, experiment_budget=experiment_budget
            )

        inner_config = TrainLmConfig(
            data=pretraining_data,
            trainer=TrainerConfig(
                tracker=WandbConfig(
                    project="marin",
                    name=wandb_name,
                    tags=[*tags],
                    group=wandb_group,
                ),
                mp=jmp.get_policy("p=f32,c=bfloat16"),
                train_batch_size=train_config.train_batch_size,
                per_device_parallelism=train_config.per_device_parallelism,
                num_train_steps=train_config.num_train_steps,
                steps_per_eval=train_config.steps_per_eval if train_config.steps_per_eval is not None else 1000,
                checkpointer=CheckpointerConfig(
                    save_interval=timedelta(minutes=10),
                    keep=[dict(every=steps_per_export)],
                ),
                model_averaging=model_averaging,
                mesh=MeshConfig(
                    compute_mapping={
                        "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                        "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                    }
                ),
                allow_partial_checkpoint=train_config.allow_partial_checkpoint,
                per_device_eval_parallelism=per_device_eval_parallelism,
                max_eval_batches=train_config.max_eval_batches,
                allow_nondivisible_batch_size=True,
                quantization=QuantizationConfig(int8=train_config.int8) if train_config.int8 else None,
                initialize_from=None if train_config.reset_data_loader_on_init else checkpoint_path_to_load_from,
                watch=train_config.watch,
                profiler=train_config.profiler,
                profiler_start_step=train_config.profiler_start_step,
                profiler_num_steps=train_config.profiler_num_steps,
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
                        train_config.max_grad_norm
                        if train_config.max_grad_norm is not None
                        else AdamConfig().max_grad_norm
                    ),
                    warmup=(train_config.warmup if train_config.warmup is not None else AdamConfig().warmup),
                    rewarmup=(train_config.rewarmup if train_config.rewarmup is not None else AdamConfig().rewarmup),
                    decay=(train_config.decay if train_config.decay is not None else AdamConfig().decay),
                    lr_schedule=(
                        train_config.lr_schedule if train_config.lr_schedule is not None else AdamConfig().lr_schedule
                    ),
                    cycle_length=train_config.cycle_length,
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

        pod_config = train_config.resources

        config = TrainLmOnPodConfig(
            train_config=inner_config,
            resources=pod_config,
            output_path=output_path,
        )

        run_levanter_train_lm(config)

    return StepSpec(
        name=os.path.join("checkpoints", name),
        hash_attrs=hash_attrs,
        deps=deps,
        fn=train_fn,
        override_output_path=override_output_path,
    )


def default_sft(
    name: str,
    tokenized: StepSpec | LMMixtureDatasetConfig,
    model_config: LlamaConfig,
    sft_config: SimpleSFTConfig,
    tags: Sequence[str] = (),
    tokenizer: str | None = None,
) -> StepSpec:
    """
    Creates a StepSpec for supervised fine-tuning of a language model.

    Args:
        name: The name of the training run, forms the basis of the output path.
        tokenized: The tokenized data to train on.
        model_config: Levanter LlamaConfig for the model architecture to train.
        sft_config: Configuration for the SFT training process.
        tags: Additional tags for WandB logging. Default: ().
        tokenizer: The tokenizer name. Required when tokenized is a StepSpec.
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
    )

    if sft_config.reinit_tokens:
        raise NotImplementedError("reinit_tokens is not supported by default_train")

    # Create and return the StepSpec
    return default_train(
        name=name,
        tokenized=tokenized,
        model_config=model_config,
        train_config=normal_train_config,
        tags=tags,
        eval_harness_tasks=[],
        use_default_validation=False,
        tokenizer=tokenizer,
    )


def _prepare_data_config(
    tokenized: StepSpec | LMMixtureDatasetConfig,
    use_default_validation: bool = True,
    tokenizer: str | None = None,
) -> LMMixtureDatasetConfig:
    """Prepare a tokenized dataset for training by adding validation sets.

    Returns the data config with any validation sets added.
    """
    if tokenizer is None:
        if isinstance(tokenized, LMMixtureDatasetConfig):
            tokenizer = tokenized.tokenizer
        elif isinstance(tokenized, StepSpec) and "tokenizer" in tokenized.hash_attrs:
            tokenizer = tokenized.hash_attrs["tokenizer"]
        else:
            raise ValueError("tokenizer must be provided or inferrable from tokenized")

    if use_default_validation:
        validation_sets = default_validation_sets(tokenizer=tokenizer)
    else:
        validation_sets = {}

    if isinstance(tokenized, StepSpec):
        pretraining_data = lm_mixture_data_config(
            components={tokenized.name: tokenized},
            weights={tokenized.name: 1.0},
        )
        if validation_sets:
            pretraining_data = add_validation_sets_to_mixture(pretraining_data, validation_sets)
    else:
        pretraining_data = tokenized
        if validation_sets:
            pretraining_data = add_validation_sets_to_mixture(pretraining_data, validation_sets)
    return pretraining_data
