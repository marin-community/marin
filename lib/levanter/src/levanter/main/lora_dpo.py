# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""LoRA-DPO training script.

Combines LoRA fine-tuning (from lora_lm.py) with DPO preference optimization
(from train_dpo.py). The key insight: with LoRA, the base model (without adapters)
naturally serves as the reference model, and the model with LoRA adapters is the
policy. This eliminates the need for a second model copy, saving ~50% of model
parameter memory compared to standard DPO.
"""

import dataclasses
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, cast

import haliax as hax
import jax
import jax.random as jrandom
from haliax import Axis
from haliax.partitioning import named_jit, round_axis_for_partitioning

import levanter
import levanter.callbacks
from levanter import callbacks
from levanter.compat.hf_checkpoints import HFCheckpointConverter
from levanter.data.dataset import AsyncDataset
from levanter.data.mixture import MixtureDataset
from levanter.data.text import DpoExample, LmDataConfig, PreferenceChatLmDatasetFormat, dataset_for_preference_format
from levanter.lora import (
    LoraConfig,
    lora_trainable_params_filter,
    loraize,
    save_merged_hf_checkpoint_callback,
    save_peft_checkpoint_callback,
    unwrap_lora_modules,
)
from levanter.main.train_dpo import (
    _build_dpo_dataset,
    _build_validation_split,
    _validate_preference_chat_formats,
    dpo_loss_from_logps,
    _logp_sum,
)
from levanter.metrics import Metric, ReductionType
from levanter.models.lm_model import LmHeadModel
from levanter.optim import AdamConfig, OptimizerConfig
from levanter.trainer import Trainer, TrainerConfig
from levanter.utils.jax_utils import parameter_count
from levanter.utils.tree_utils import inference_mode


logger = logging.getLogger(__name__)


@dataclass
class LoraDpoConfig:
    data: LmDataConfig = field(default_factory=LmDataConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    optimizer: OptimizerConfig = field(default_factory=AdamConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)

    initialize_from_hf: str = ""
    """Required: HuggingFace model to load and apply LoRA to."""

    beta: float = 0.1
    """DPO temperature parameter."""

    train_seq_len: int | None = None
    use_hf_model_config: bool = False
    trust_remote_code: bool = False

    validation_split_fraction: float | None = 0.1

    peft_save_path: Optional[str] = None
    peft_hf_upload: Optional[str] = None
    merged_hf_save_path: Optional[str] = None
    merged_hf_upload: Optional[str] = None
    hf_save_steps: int = 10000

    data_seed: Optional[int] = None


def main(config: LoraDpoConfig):
    if not config.initialize_from_hf:
        raise ValueError("initialize_from_hf must be provided for LoRA-DPO training.")

    # DPO requires zero-initialized B matrix so the adapter starts as identity (policy = reference at step 0).
    # Without this, the policy immediately diverges from the reference, causing catastrophic training failure.
    if not config.lora.zero_init_b:
        logger.warning(
            "lora.zero_init_b is False — overriding to True for DPO. "
            "DPO requires zero-initialized LoRA B matrix so the policy matches the reference at initialization. "
            "Set lora.zero_init_b: true in your config to silence this warning."
        )
        config = dataclasses.replace(config, lora=dataclasses.replace(config.lora, zero_init_b=True))

    _validate_preference_chat_formats(config.data)

    tokenizer = config.data.the_tokenizer

    converter = HFCheckpointConverter.from_hf(config.initialize_from_hf, trust_remote_code=config.trust_remote_code)
    if hasattr(tokenizer, "vocab") and tokenizer.vocab != converter.tokenizer.vocab:
        logger.warning("The tokenizers appear to be different. You may want to check this.")
    converter = converter.replaced(tokenizer=tokenizer)

    model_config = converter.default_config
    if config.use_hf_model_config:
        model_config = converter.config_from_hf_config(converter.default_hf_config)

    levanter.initialize(config)
    optimizer = config.optimizer.build(config.trainer.num_train_steps)

    def loss_function(model: LmHeadModel, example: DpoExample, *, key=None):
        reference_model = unwrap_lora_modules(model)
        reference_model = inference_mode(reference_model, True)

        if key is not None:
            key_chosen, key_rejected = jrandom.split(key)
        else:
            key_chosen = None
            key_rejected = None

        # Policy log-probs (full model with LoRA adapters)
        logp_pi_chosen = _logp_sum(model, example.chosen, key=key_chosen)
        logp_pi_rejected = _logp_sum(model, example.rejected, key=key_rejected)

        # Reference log-probs (base model only, LoRA stripped)
        # stop_gradient prevents backprop through the reference forward pass,
        # avoiding useless base-weight gradient computation
        logp_ref_chosen = jax.lax.stop_gradient(_logp_sum(reference_model, example.chosen, key=key_chosen))
        logp_ref_rejected = jax.lax.stop_gradient(_logp_sum(reference_model, example.rejected, key=key_rejected))

        delta_pi = logp_pi_chosen - logp_pi_rejected
        delta_ref = logp_ref_chosen - logp_ref_rejected

        loss, metrics = dpo_loss_from_logps(delta_pi, delta_ref, beta=config.beta)
        chosen_reward = (logp_pi_chosen - logp_ref_chosen) * config.beta
        rejected_reward = (logp_pi_rejected - logp_ref_rejected) * config.beta
        metrics["dpo_chosen_reward"] = Metric.from_value(hax.mean(chosen_reward).scalar(), ReductionType.MEAN)
        metrics["dpo_rejected_reward"] = Metric.from_value(hax.mean(rejected_reward).scalar(), ReductionType.MEAN)
        return loss, metrics

    with Trainer(config.trainer, optimizer, loss_function) as trainer:
        seed = config.trainer.seed
        data_key, model_key, lora_key, training_key = jrandom.split(jrandom.PRNGKey(seed), 4)

        if config.data_seed is not None:
            logger.info(f"Overriding data seed with {config.data_seed}")
            data_key = jrandom.PRNGKey(config.data_seed)

        parameter_axis_mapping = trainer.parameter_axis_mapping

        model_max_seq_len = model_config.max_seq_len
        train_length = config.train_seq_len if config.train_seq_len is not None else model_max_seq_len

        if train_length <= 0:
            raise ValueError(f"train_length must be positive, got {train_length}")
        if train_length > model_max_seq_len:
            raise ValueError(f"train_length ({train_length}) cannot exceed model max_seq_len ({model_max_seq_len}).")
        if train_length != model_max_seq_len:
            logger.info(f"Training with sequence length {train_length} (model supports {model_max_seq_len}).")

        Pos = model_config.max_Pos.resize(train_length)

        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), parameter_axis_mapping)
        if vocab_size != Vocab.size:
            logger.info(f"Rounding vocab size from {vocab_size} to {Vocab.size} for partitioning")

        # Build datasets
        validation_sets: dict[str, AsyncDataset[DpoExample]] = {}
        if config.validation_split_fraction is not None:
            fraction = config.validation_split_fraction
            if fraction < 0 or fraction >= 1:
                raise ValueError(f"validation_split_fraction must be in [0, 1), got {fraction}")
            train_dataset, validation_sets = _build_validation_split(
                config.data,
                Pos,
                key=data_key,
                fraction=fraction,
            )
        else:
            train_dataset = _build_dpo_dataset(config.data, Pos, key=data_key)
            # Build validation sets from the validation cache
            val_caches = config.data.build_caches("validation")
            for name, component in config.data.components.items():
                cache = val_caches.get(name)
                if cache is None:
                    continue
                if not isinstance(component.format, PreferenceChatLmDatasetFormat):
                    continue
                validation_sets[name] = cast(
                    AsyncDataset[DpoExample], dataset_for_preference_format(component.format, Pos, cache)
                )

        # Load pretrained model and apply LoRA
        logger.info(f"Loading pretrained model from {converter.reference_checkpoint}")
        model = converter.load_pretrained(
            model_config.model_type,
            axis_mapping=parameter_axis_mapping,
            dtype=trainer.mp.compute_dtype,
        )
        model = named_jit(trainer.mp.cast_to_param, parameter_axis_mapping)(model)

        @named_jit(axis_resources=parameter_axis_mapping, donate_args=(True,))
        def loraize_hf_model(model):
            return loraize(model, config.lora, key=lora_key)

        model = loraize_hf_model(model)

        lora_param_filter = lora_trainable_params_filter(model)
        state = trainer.initial_state(training_key, model=model, is_trainable=lora_param_filter)

        all_param_count = parameter_count(state.model)
        just_lora_params = parameter_count(state.trainable_model)

        levanter.tracker.log_summary(
            {
                "parameter_count": all_param_count,
                "trainable_parameter_count": just_lora_params,
                "fraction_trainable": just_lora_params * 1.0 / all_param_count,
            }
        )

        logger.info(f"Total parameter count: {all_param_count}")
        logger.info(f"Trainable parameter count: {just_lora_params}")
        logger.info(f"Fraction of parameters that are trainable: {just_lora_params * 1.0 / all_param_count:.3e}")

        # Performance stats
        flops_per_token = model_config.flops_per_token(vocab_size, Pos.size)
        flops_per_example = 3 * flops_per_token * Pos.size if flops_per_token is not None else None
        trainer.add_hook(
            callbacks.log_performance_stats(Pos.size, trainer.config.batch_schedule, flops_per_example), every=1
        )

        # Mixture weight logging
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

        # Validation
        if validation_sets:
            for name, dataset in validation_sets.items():
                trainer.add_eval_hook(dataset, name=name or None)
        else:
            logger.warning("No validation datasets provided.")

        # PEFT checkpoint saving
        if config.peft_save_path is not None:
            if config.trainer.checkpointer.append_run_id_to_base_path:
                full_save_path = os.path.join(config.peft_save_path, trainer.run_id)
            else:
                full_save_path = config.peft_save_path
            trainer.add_hook(
                save_peft_checkpoint_callback(
                    full_save_path, config.lora, config.initialize_from_hf, tokenizer, config.peft_hf_upload
                ),
                every=config.hf_save_steps,
            )

        # Merged HF checkpoint saving
        if config.merged_hf_save_path is not None:
            if config.trainer.checkpointer.append_run_id_to_base_path:
                full_save_path = os.path.join(config.merged_hf_save_path, trainer.run_id)
            else:
                full_save_path = config.merged_hf_save_path
            trainer.add_hook(
                save_merged_hf_checkpoint_callback(full_save_path, converter, config.merged_hf_upload),
                every=config.hf_save_steps,
            )

        # Train
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
