# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import gc
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, cast

import haliax as hax
import jax.numpy as jnp
import jax.random as jrandom
from haliax import Axis
from haliax.partitioning import named_jit, round_axis_for_partitioning

import levanter
import levanter.callbacks
from levanter import callbacks
from levanter.checkpoint import load_checkpoint
from levanter.compat.hf_checkpoints import HFCompatConfig
from levanter.data.mixture import MixtureDataset
from levanter.data.dataset import AsyncDataset
from levanter.data.text import (
    DatasetComponent,
    DpoExample,
    LmDataConfig,
    PreferenceChatLmDatasetFormat,
)
from levanter.models.llama import LlamaConfig
from levanter.models.lm_model import LmConfig, LmExample, LmHeadModel
from levanter.metrics import Metric, ReductionType
from levanter.optim import AdamConfig, OptimizerConfig
from levanter.schedule import BatchSchedule
from levanter.trainer import Trainer, TrainerConfig
from levanter.utils.jax_utils import parameter_count

logger = logging.getLogger(__name__)


def simpo_loss_from_logps(
    avg_logp_chosen: hax.NamedArray | jnp.ndarray,
    avg_logp_rejected: hax.NamedArray | jnp.ndarray,
    *,
    beta: float,
    gamma_beta_ratio: float,
) -> tuple[jnp.ndarray, dict[str, Metric]]:
    if isinstance(avg_logp_chosen, hax.NamedArray) or isinstance(avg_logp_rejected, hax.NamedArray):
        if not isinstance(avg_logp_chosen, hax.NamedArray) or not isinstance(avg_logp_rejected, hax.NamedArray):
            raise TypeError(
                "avg_logp_chosen and avg_logp_rejected must both be NamedArray when using named computations."
            )
        logits = (avg_logp_chosen - avg_logp_rejected) - gamma_beta_ratio
        loss = hax.mean(hax.nn.softplus(-beta * logits)).scalar()
        metrics = {
            "simpo_loss": Metric.from_value(loss, ReductionType.MEAN),
            "simpo_chosen_logp": Metric.from_value(hax.mean(avg_logp_chosen).scalar(), ReductionType.MEAN),
            "simpo_rejected_logp": Metric.from_value(hax.mean(avg_logp_rejected).scalar(), ReductionType.MEAN),
            "simpo_margin": Metric.from_value(
                hax.mean(avg_logp_chosen - avg_logp_rejected).scalar(), ReductionType.MEAN
            ),
            "simpo_accuracy": Metric.from_value(hax.mean(logits > 0).scalar(), ReductionType.MEAN),
        }
        return loss, metrics

    logits = (avg_logp_chosen - avg_logp_rejected) - gamma_beta_ratio
    loss = jnp.mean(hax.nn.softplus(-beta * logits))
    metrics = {
        "simpo_loss": Metric.from_value(loss, ReductionType.MEAN),
        "simpo_chosen_logp": Metric.from_value(jnp.mean(avg_logp_chosen), ReductionType.MEAN),
        "simpo_rejected_logp": Metric.from_value(jnp.mean(avg_logp_rejected), ReductionType.MEAN),
        "simpo_margin": Metric.from_value(jnp.mean(avg_logp_chosen - avg_logp_rejected), ReductionType.MEAN),
        "simpo_accuracy": Metric.from_value(jnp.mean(logits > 0), ReductionType.MEAN),
    }
    return loss, metrics


def _average_logp(model: LmHeadModel, example: LmExample, *, key=None) -> hax.NamedArray:
    nll = model.compute_next_token_loss(example, reduction=None, reduction_axis=(), key=key)
    Pos = example.tokens.resolve_axis("position")
    logp_sum = -hax.sum(nll, axis=Pos)
    denom = hax.sum(example.loss_weight, axis=Pos)
    zeros = hax.zeros_like(logp_sum)
    return hax.where(denom != 0, logp_sum / denom, zeros)


def _validate_preference_chat_formats(config: LmDataConfig) -> None:
    formats: dict[str, PreferenceChatLmDatasetFormat] = {}
    for name, component in config.components.items():
        if not isinstance(component, DatasetComponent):
            raise ValueError(f"SimPO training requires DatasetComponent, got {type(component)} for {name}")
        fmt = component.format
        if not isinstance(fmt, PreferenceChatLmDatasetFormat):
            raise ValueError(
                f"SimPO training requires preference_chat datasets. Component '{name}' has format {type(fmt).__name__}"
            )
        formats[name] = fmt

    packed = {name: fmt for name, fmt in formats.items() if fmt.pack}
    if packed:
        bad = ", ".join(sorted(packed.keys()))
        raise ValueError(f"Packed preference_chat datasets are not supported yet. Packed datasets: {bad}")

    non_raise = {name: fmt for name, fmt in formats.items() if fmt.slice_strategy != "raise"}
    if non_raise:
        bad = ", ".join(sorted(non_raise.keys()))
        raise ValueError(f"preference_chat slice_strategy must be 'raise' for now. Invalid datasets: {bad}")


def _num_validation_sequences(total_sequences: int, fraction: float) -> int:
    if total_sequences <= 1:
        return 0
    if fraction <= 0:
        return 0
    num_val = int(total_sequences * fraction)
    if num_val <= 0:
        num_val = 1
    if num_val >= total_sequences:
        num_val = total_sequences - 1
    return num_val


def _build_validation_split(
    config: LmDataConfig,
    Pos: Axis,
    *,
    batch_schedule: BatchSchedule,
    key: jrandom.PRNGKey,
    fraction: float,
) -> tuple[AsyncDataset[DpoExample], dict[str, AsyncDataset[DpoExample]]]:
    """Build train/validation split from LmDataConfig by holding out a fraction of each component."""
    train_caches = config.build_caches("train")
    token_datasets = config.build_token_datasets(train_caches, Pos, split="train")

    num_validation_sequences: dict[str, int] = {}
    for name, dataset in token_datasets.items():
        total_len = len(dataset.as_sync_dataset())
        num_val = _num_validation_sequences(total_len, fraction)
        if num_val > 0:
            num_validation_sequences[name] = num_val

    if not num_validation_sequences:
        train_dataset = cast(AsyncDataset[DpoExample], config.train_set(Pos, batch_schedule, key=key))
        return train_dataset, {}

    config_with_val = dataclasses.replace(config, num_validation_sequences=num_validation_sequences)
    train_dataset = cast(AsyncDataset[DpoExample], config_with_val.train_set(Pos, batch_schedule, key=key))
    validation_sets = cast(dict[str, AsyncDataset[DpoExample]], config_with_val.validation_sets(Pos))
    return train_dataset, validation_sets


@dataclass
class TrainSimpoConfig:
    data: LmDataConfig = field(default_factory=LmDataConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: LmConfig = field(default_factory=LlamaConfig)
    train_seq_len: int | None = None
    optimizer: OptimizerConfig = field(default_factory=AdamConfig)

    beta: float = 2.0
    gamma_beta_ratio: float = 0.5
    validation_split_fraction: float | None = 0.1

    initialize_from_hf: bool | str = False
    use_hf_model_config: bool = False

    hf_save_path: Optional[str] = None
    hf_upload: Optional[str] = None
    hf_save_steps: int = 10000
    hf_save_dtype: Optional[str] = None

    data_seed: Optional[int] = None
    initialize_from_checkpoint_path: Optional[str] = None


def main(config: TrainSimpoConfig):

    _validate_preference_chat_formats(config.data)

    tokenizer = config.data.the_tokenizer

    if config.initialize_from_hf:
        if config.trainer.initialize_from is not None:
            raise ValueError("Cannot specify both initialize_from_hf and initialize_from")

        assert isinstance(config.model, HFCompatConfig)
        converter = config.model.hf_checkpoint_converter()
        if hasattr(tokenizer, "vocab") and tokenizer.vocab != converter.tokenizer.vocab:
            logger.warning("The tokenizers appear to be different. You may want to check this.")

        if isinstance(config.initialize_from_hf, str):
            converter = converter.replaced(reference_checkpoint=config.initialize_from_hf, tokenizer=tokenizer)
        else:
            converter = converter.replaced(tokenizer=tokenizer)

        if config.use_hf_model_config:
            config.model = converter.config_from_hf_config(converter.default_hf_config)
    elif isinstance(config.model, HFCompatConfig):
        converter = config.model.hf_checkpoint_converter()
        converter = converter.replaced(tokenizer=tokenizer)
    else:
        converter = None

    levanter.initialize(config)
    optimizer = config.optimizer.build(config.trainer.num_train_steps)

    def loss_function(model: LmHeadModel, example: DpoExample, *, key=None):
        if key is not None:
            key_chosen, key_rejected = jrandom.split(key)
        else:
            key_chosen = None
            key_rejected = None

        avg_logp_chosen = _average_logp(model, example.chosen, key=key_chosen)
        avg_logp_rejected = _average_logp(model, example.rejected, key=key_rejected)

        return simpo_loss_from_logps(
            avg_logp_chosen,
            avg_logp_rejected,
            beta=config.beta,
            gamma_beta_ratio=config.gamma_beta_ratio,
        )

    with Trainer(config.trainer, optimizer, loss_function) as trainer:
        seed = config.trainer.seed
        data_key, loader_key, model_key, training_key = jrandom.split(jrandom.PRNGKey(seed), 4)
        del loader_key

        if config.data_seed is not None:
            logger.info(f"Overriding data seed with {config.data_seed}")
            data_key = jrandom.PRNGKey(config.data_seed)

        parameter_axis_mapping = trainer.parameter_axis_mapping

        model_max_seq_len = config.model.max_seq_len
        train_length = config.train_seq_len if config.train_seq_len is not None else model_max_seq_len

        if train_length <= 0:
            raise ValueError(f"train_length must be positive, got {train_length}")

        if train_length > model_max_seq_len:
            raise ValueError(f"train_length ({train_length}) cannot exceed model max_seq_len ({model_max_seq_len}).")

        if train_length != model_max_seq_len:
            logger.info(f"Training with sequence length {train_length} (model supports {model_max_seq_len}).")

        Pos = config.model.max_Pos.resize(train_length)

        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), parameter_axis_mapping)
        if vocab_size != Vocab.size:
            logger.info(f"Rounding vocab size from {vocab_size} to {Vocab.size} for partitioning")

        validation_sets: dict[str, AsyncDataset[DpoExample]] = {}
        if config.validation_split_fraction is not None:
            fraction = config.validation_split_fraction
            if fraction < 0 or fraction >= 1:
                raise ValueError(f"validation_split_fraction must be in [0, 1), got {fraction}")
            train_dataset, validation_sets = _build_validation_split(
                config.data,
                Pos,
                batch_schedule=config.trainer.batch_schedule,
                key=data_key,
                fraction=fraction,
            )
        else:
            train_dataset = cast(
                AsyncDataset[DpoExample],
                config.data.train_set(Pos, config.trainer.batch_schedule, key=data_key),
            )
            validation_sets = cast(dict[str, AsyncDataset[DpoExample]], config.data.validation_sets(Pos))

        state = trainer.initial_state(training_key, model_init=lambda: config.model.build(Vocab, key=model_key))

        if int(state.step) == 0:
            if config.initialize_from_hf:
                assert converter is not None
                logger.info(
                    "No training checkpoint found. Initializing model from HF checkpoint"
                    f" '{converter.reference_checkpoint}'"
                )
                state = dataclasses.replace(state, model=None)
                gc.collect()
                model = converter.load_pretrained(
                    config.model.model_type,
                    config=config.model if not config.use_hf_model_config else None,
                    axis_mapping=parameter_axis_mapping,
                    dtype=trainer.mp.compute_dtype,
                )
                model = named_jit(trainer.mp.cast_to_param, parameter_axis_mapping)(model)
                state = dataclasses.replace(state, model=model)
            elif config.initialize_from_checkpoint_path is not None:
                state = load_checkpoint(state, config.initialize_from_checkpoint_path)
                state = dataclasses.replace(state, step=jnp.array(0))
            else:
                logger.info("No checkpoint found. Starting from scratch.")

        levanter.tracker.log_summary({"parameter_count": parameter_count(state.model)})

        flops_per_token = config.model.flops_per_token(vocab_size, Pos.size)
        flops_per_example = 3 * flops_per_token * Pos.size if flops_per_token is not None else None
        trainer.add_hook(
            callbacks.log_performance_stats(Pos.size, trainer.config.batch_schedule, flops_per_example), every=1
        )

        if isinstance(train_dataset, MixtureDataset):
            last_stage = -1

            def log_mixture_weights(step_info):
                nonlocal last_stage
                seq_index = trainer.config.batch_schedule.global_data_offset_by_step(step_info.step)
                block_id = seq_index // train_dataset.block_size
                stage = train_dataset._get_stage_for_block(block_id)
                weights = train_dataset.weight_stages[stage][1]
                if stage != last_stage:
                    metrics = {f"mixture/weight/{name}": weight for name, weight in weights.items()}
                    metrics["mixture/stage"] = stage
                    levanter.tracker.log(metrics, step=step_info.step)
                    last_stage = stage

            trainer.add_hook(log_mixture_weights, every=1)

        if validation_sets:
            for name, dataset in validation_sets.items():
                trainer.add_eval_hook(dataset, name=name or None)
        else:
            logger.warning("No validation datasets provided.")

        if config.hf_save_path is not None and config.hf_save_steps is not None:
            assert converter is not None, "converter must be set when saving HF checkpoints"
            if config.trainer.checkpointer.append_run_id_to_base_path:
                full_save_path = os.path.join(config.hf_save_path, trainer.run_id)
            else:
                full_save_path = config.hf_save_path

            save_dtype: Optional[jnp.dtype] = None
            if config.hf_save_dtype is not None:
                try:
                    save_dtype = jnp.dtype(config.hf_save_dtype)
                except TypeError:
                    logger.warning(f"Invalid hf_save_dtype: {config.hf_save_dtype}. Defaulting to None.")

            def save_policy_hf_checkpoint(step):
                if step.step == 0:
                    return
                upload_to_hf = config.hf_upload or False
                hf_upload_kwargs = {}
                if upload_to_hf is not None:
                    hf_upload_kwargs["commit_message"] = f"Upload for step {step.step} from Levanter"

                converter.save_pretrained(
                    step.eval_model,
                    os.path.join(full_save_path, f"step-{step.step}"),
                    upload_to_hf=upload_to_hf,
                    dtype=save_dtype,
                    **hf_upload_kwargs,
                )

            trainer.add_hook(save_policy_hf_checkpoint, every=config.hf_save_steps)

        train_loader = trainer.data_loader(train_dataset)
        if state.step > 0:
            logger.info(f"Resuming training from step {state.step}")
            train_loader = train_loader.iter_from_step(state.step)
        else:
            train_loader = train_loader.iter_from_step(0)

        last_info = trainer.train(state, train_loader)

        if trainer.config.checkpointer is not None:
            trainer.run_hooks(last_info, force=True)
            checkpointer = trainer.config.checkpointer.create(trainer.run_id)
            checkpointer.wait_until_finished()

    trainer.tracker.finish()


if __name__ == "__main__":
    levanter.config.main(main)()
