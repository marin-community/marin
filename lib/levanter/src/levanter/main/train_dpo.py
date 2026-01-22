# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import gc
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Union, cast

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import jax.random as jrandom
from haliax import Axis
from haliax.partitioning import named_jit, round_axis_for_partitioning

import levanter
import levanter.callbacks
from levanter import callbacks
from levanter.checkpoint import load_checkpoint
from levanter.compat.hf_checkpoints import HFCompatConfig, save_hf_checkpoint_callback
from levanter.data.mixture import MixtureDataset
from levanter.data.dataset import AsyncDataset, EpochDataset
from levanter.data.text import (
    LMMixtureDatasetConfig,
    PreferenceChatLmDatasetFormat,
    SingleDatasetLMConfigBase,
    SingleDatasetLMConfig,
    UrlSingleDatasetLMConfig,
    dataset_for_format,
)
from levanter.models.llama import LlamaConfig
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.optim import AdamConfig, OptimizerConfig
from levanter.schedule import BatchSchedule
from levanter.trainer import Trainer, TrainerConfig
from levanter.utils.jax_utils import parameter_count, use_cpu_device
from levanter.utils.tree_utils import inference_mode

from levanter.data.text import DpoExample

logger = logging.getLogger(__name__)


def dpo_loss_from_logps(
    delta_pi: jnp.ndarray, delta_ref: jnp.ndarray, *, beta: float
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    logits = beta * (delta_pi - delta_ref)
    loss = jnp.mean(jax.nn.softplus(-logits))
    metrics = {
        "dpo_loss": loss,
        "dpo_margin_policy": jnp.mean(delta_pi),
        "dpo_margin_ref": jnp.mean(delta_ref),
        "dpo_accuracy": jnp.mean(logits > 0),
    }
    return loss, metrics


def _logp_sum(model: LmHeadModel, example, *, key=None) -> hax.NamedArray:
    nll = model.compute_next_token_loss(example, reduction=None, reduction_axis=())
    Pos = example.tokens.resolve_axis("position")
    return -hax.sum(nll, axis=Pos)


def _validate_preference_chat_formats(config: SingleDatasetLMConfig | LMMixtureDatasetConfig) -> None:
    formats = {name: source.format for name, source in config.sources.items()}

    non_preference = {name: fmt for name, fmt in formats.items() if not isinstance(fmt, PreferenceChatLmDatasetFormat)}
    if non_preference:
        bad = ", ".join(sorted(non_preference.keys()))
        raise ValueError(f"DPO training requires preference_chat datasets. Non-preference datasets: {bad}")

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


def _build_validation_split_single(
    config: SingleDatasetLMConfigBase,
    Pos: Axis,
    *,
    batch_schedule: BatchSchedule,
    key: jrandom.PRNGKey,
    epochs: int | None,
    fraction: float,
) -> tuple[AsyncDataset[DpoExample], dict[str, AsyncDataset[DpoExample]]]:
    cache = config.build_or_load_cache("train")
    if cache is None:
        raise ValueError("No training cache available for validation split.")

    base_dataset = dataset_for_format(
        config.format,
        Pos,
        cache,
        eos_id=config.the_tokenizer.eos_token_id,
        ignore_index=config.ignore_token_id,
        block_cross_document_attention=config.block_cross_document_attention,
    )

    total_len = len(base_dataset.as_sync_dataset())
    num_val = _num_validation_sequences(total_len, fraction)
    if num_val == 0:
        return config.train_set(Pos, batch_schedule, key=key, epochs=epochs), {}

    train_base = base_dataset.slice_dataset(end_index=total_len - num_val)
    val_base = base_dataset.slice_dataset(start_index=total_len - num_val, end_index=total_len)

    perm_type = config.permutation_type
    if perm_type is None:
        logger.warning("Defaulting to linear permutation for shuffling. This will change to Feistel in the future.")
        perm_type = "linear"

    train_dataset = train_base
    if config.shuffle is True:
        train_dataset = train_dataset.shuffle(key, perm_type=perm_type)
    elif isinstance(config.shuffle, int) and config.shuffle > 0:
        train_dataset = train_dataset.era_shuffle(config.shuffle, key=key, perm_type=perm_type)

    if epochs:
        train_dataset = EpochDataset(train_dataset, max_epochs=epochs)

    train_dataset = cast(AsyncDataset[DpoExample], train_dataset)
    val_base = cast(AsyncDataset[DpoExample], val_base)
    return train_dataset, {"": val_base}


def _build_validation_split_mixture(
    config: LMMixtureDatasetConfig,
    Pos: Axis,
    *,
    batch_schedule: BatchSchedule,
    key: jrandom.PRNGKey,
    epochs: int | None,
    fraction: float,
) -> tuple[AsyncDataset[DpoExample], dict[str, AsyncDataset[DpoExample]]]:
    if epochs:
        raise ValueError("Epochs are not supported for mixture datasets")

    train_caches = config.build_caches("train")
    token_datasets = config.build_token_datasets(train_caches, Pos)

    num_validation_sequences: dict[str, int] = {}
    for name, dataset in token_datasets.items():
        total_len = len(dataset.as_sync_dataset())
        num_val = _num_validation_sequences(total_len, fraction)
        if num_val > 0:
            num_validation_sequences[name] = num_val

    if not num_validation_sequences:
        train_dataset = cast(AsyncDataset[DpoExample], config.train_set(Pos, batch_schedule, key=key, epochs=epochs))
        return train_dataset, {}

    config_with_val = dataclasses.replace(config, num_validation_sequences=num_validation_sequences)
    train_dataset = cast(
        AsyncDataset[DpoExample], config_with_val.train_set(Pos, batch_schedule, key=key, epochs=epochs)
    )
    validation_sets = cast(dict[str, AsyncDataset[DpoExample]], config_with_val.validation_sets(Pos))
    return train_dataset, validation_sets


@dataclass
class TrainDpoConfig:
    data: Union[SingleDatasetLMConfig, LMMixtureDatasetConfig] = field(default_factory=UrlSingleDatasetLMConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: LmConfig = field(default_factory=LlamaConfig)
    train_seq_len: int | None = None
    optimizer: OptimizerConfig = field(default_factory=AdamConfig)

    reference_model_path: str = ""
    reference_is_hf: bool = True

    beta: float = 0.1

    initialize_from_hf: Union[bool, str] = False
    use_hf_model_config: bool = False

    validation_split_fraction: float | None = 0.1

    hf_save_path: Optional[str] = None
    hf_upload: Optional[str] = None
    hf_save_steps: int = 10000
    hf_save_dtype: Optional[str] = None

    data_seed: Optional[int] = None
    initialize_from_checkpoint_path: Optional[str] = None
    epoch: int = 0


def main(config: TrainDpoConfig):
    if not config.reference_model_path:
        raise ValueError("reference_model_path must be provided for DPO training.")

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

    reference_holder: dict[str, LmHeadModel] = {}

    def loss_function(model: LmHeadModel, example: DpoExample, *, key=None):
        if "model" not in reference_holder:
            raise ValueError("Reference model is not loaded.")

        reference_model = reference_holder["model"]

        if key is not None:
            key_chosen, key_rejected = jrandom.split(key)
        else:
            key_chosen = None
            key_rejected = None

        logp_pi_chosen = _logp_sum(model, example.chosen, key=key_chosen)
        logp_pi_rejected = _logp_sum(model, example.rejected, key=key_rejected)

        logp_ref_chosen = _logp_sum(reference_model, example.chosen, key=None)
        logp_ref_rejected = _logp_sum(reference_model, example.rejected, key=None)

        delta_pi = (logp_pi_chosen - logp_pi_rejected).array
        delta_ref = (logp_ref_chosen - logp_ref_rejected).array

        return dpo_loss_from_logps(delta_pi, delta_ref, beta=config.beta)

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
            if isinstance(config.data, SingleDatasetLMConfigBase):
                train_dataset, validation_sets = _build_validation_split_single(
                    config.data,
                    Pos,
                    batch_schedule=config.trainer.batch_schedule,
                    key=data_key,
                    epochs=config.epoch,
                    fraction=fraction,
                )
            elif isinstance(config.data, LMMixtureDatasetConfig):
                train_dataset, validation_sets = _build_validation_split_mixture(
                    config.data,
                    Pos,
                    batch_schedule=config.trainer.batch_schedule,
                    key=data_key,
                    epochs=config.epoch,
                    fraction=fraction,
                )
            else:
                raise TypeError(f"Unsupported data config type: {type(config.data)}")
        else:
            train_dataset = config.data.train_set(
                Pos,
                config.trainer.batch_schedule,
                key=data_key,
                epochs=config.epoch,
            )
            validation_sets = cast(dict[str, AsyncDataset[DpoExample]], config.data.validation_sets(Pos))

        train_dataset = cast(AsyncDataset[DpoExample], train_dataset)

        state = trainer.initial_state(training_key, model_init=lambda: config.model.build(Vocab, key=model_key))

        if int(state.step) == 0 and config.initialize_from_checkpoint_path is not None:
            state = load_checkpoint(state, config.initialize_from_checkpoint_path)
            state = dataclasses.replace(state, step=jnp.array(0))

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
            else:
                logger.info("No checkpoint found. Starting from scratch.")

        if config.reference_is_hf:
            if converter is None:
                raise ValueError("reference_is_hf requires a HFCompatConfig model.")
            reference_model = converter.load_pretrained(
                config.model.model_type,
                ref=config.reference_model_path,
                config=config.model if not config.use_hf_model_config else None,
                axis_mapping=parameter_axis_mapping,
                dtype=trainer.mp.compute_dtype,
            )
            reference_model = named_jit(trainer.mp.cast_to_param, parameter_axis_mapping)(reference_model)
        else:
            with use_cpu_device():
                reference_model = eqx.filter_eval_shape(config.model.build, Vocab, key=model_key)
                reference_model = load_checkpoint(reference_model, config.reference_model_path, subpath="model")
            reference_model = hax.shard(reference_model, parameter_axis_mapping)

        reference_model = cast(LmHeadModel, inference_mode(reference_model, True))
        reference_model = named_jit(trainer.mp.cast_to_compute, parameter_axis_mapping)(reference_model)
        reference_holder["model"] = reference_model

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

            trainer.add_hook(
                save_hf_checkpoint_callback(
                    full_save_path, converter, upload_to_hf=config.hf_upload or False, save_dtype=save_dtype
                ),
                every=config.hf_save_steps,
            )

        train_loader = trainer.data_loader(train_dataset)
        if state.step > 0:
            logger.info(f"Resuming training from step {state.step}")
            train_loader = train_loader.iter_from_step(state.step)
        else:
            train_loader = train_loader.iter_from_step(0)

        last_info = trainer.train(state, train_loader)

        if trainer.config.checkpointer is not None and config.epoch > 0:
            trainer.run_hooks(last_info, force=True)
            checkpointer = trainer.config.checkpointer.create(trainer.run_id)
            checkpointer.wait_until_finished()

    trainer.tracker.finish()


if __name__ == "__main__":
    levanter.config.main(main)()
