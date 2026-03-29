# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional, cast

import draccus
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
from levanter.adaptation import (
    AdaptationConfig,
    AdaptationExportConfig,
    LoraAdaptationConfig,
    NoAdaptationConfig,
)
from levanter.data.dataset import AsyncDataset
from levanter.data.mixture import MixtureDataset
from levanter.data.text import (
    DpoExample,
    PreferenceChatLmDatasetFormat,
    PreferenceLmDataConfig,
    dataset_for_preference_format,
)
from levanter.main.model_init import load_model_from_source, prepare_model_init_context
from levanter.metrics import Metric, ReductionType
from levanter.models.llama import LlamaConfig
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.optim import AdamConfig, OptimizerConfig
from levanter.trainer import Trainer, TrainerConfig
from levanter.utils.jax_utils import parameter_count
from levanter.utils.tree_utils import inference_mode


logger = logging.getLogger(__name__)


class DpoModel(eqx.Module):
    policy: LmHeadModel
    reference: LmHeadModel


def _policy_model_for_hf_save(model: DpoModel | LmHeadModel) -> LmHeadModel:
    return model.policy if isinstance(model, DpoModel) else model


def dpo_loss_from_logps(
    delta_pi: hax.NamedArray,
    delta_ref: hax.NamedArray,
    *,
    beta: float,
) -> tuple[jnp.ndarray, dict[str, Metric]]:
    logits = (delta_pi - delta_ref) * beta
    loss = hax.mean(hax.nn.softplus(-logits)).scalar()
    metrics = {
        "dpo_loss": Metric.from_value(loss, ReductionType.MEAN),
        "dpo_margin_policy": Metric.from_value(hax.mean(delta_pi).scalar(), ReductionType.MEAN),
        "dpo_margin_ref": Metric.from_value(hax.mean(delta_ref).scalar(), ReductionType.MEAN),
        "dpo_accuracy": Metric.from_value(hax.mean(logits > 0).scalar(), ReductionType.MEAN),
    }
    return loss, metrics


def _logp_sum(model: LmHeadModel, example, *, key=None) -> hax.NamedArray:
    nll = model.compute_next_token_loss(example, reduction=hax.sum, reduction_axis="position", key=key)
    return -nll


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


def _get_training_components(config: PreferenceLmDataConfig) -> dict[str, Any]:
    """Get components with non-zero training weight."""
    weights = config.train_weights
    if weights is None:
        return dict(config.components)
    if isinstance(weights, dict):
        return {name: comp for name, comp in config.components.items() if weights.get(name, 0) > 0}

    has_weight = set()
    for _, stage_weights in weights:
        for name, weight in stage_weights.items():
            if weight > 0:
                has_weight.add(name)
    return {name: comp for name, comp in config.components.items() if name in has_weight}


def _build_dpo_dataset(
    config: PreferenceLmDataConfig,
    Pos: Axis,
    *,
    key: jrandom.PRNGKey,
) -> AsyncDataset[DpoExample]:
    """Build a DPO training dataset from a single component config."""
    training_components = _get_training_components(config)
    if len(training_components) != 1:
        raise ValueError(
            "DPO training only supports single-component configs for now. "
            f"Found {len(training_components)} training components: {list(training_components.keys())}"
        )

    name, component = next(iter(training_components.items()))
    caches = config.build_caches("train")
    cache = caches.get(name)
    if cache is None:
        raise ValueError(f"No training cache available for component {name}.")

    base_dataset = dataset_for_preference_format(cast(PreferenceChatLmDatasetFormat, component.format), Pos, cache)

    perm_type = config.permutation_type
    if perm_type == "linear":
        logger.warning("Using linear shuffling, not recommended. Please use Feistel permutation instead.")
    elif perm_type is None:
        perm_type = "feistel"

    train_dataset = base_dataset
    if config.shuffle is True:
        train_dataset = train_dataset.shuffle(key, perm_type=perm_type)
    elif isinstance(config.shuffle, int) and config.shuffle > 0:
        train_dataset = train_dataset.era_shuffle(config.shuffle, key=key, perm_type=perm_type)

    mix_key, _ = jrandom.split(key)
    mixture = MixtureDataset(
        datasets={name: train_dataset},
        weights={name: 1.0},
        block_size=config.mixture_block_size,
        key=mix_key,
        stop_strategy=config.stop_strategy,
    )

    return cast(AsyncDataset[DpoExample], mixture)


def _build_validation_split(
    config: PreferenceLmDataConfig,
    Pos: Axis,
    *,
    key: jrandom.PRNGKey,
    fraction: float,
) -> tuple[AsyncDataset[DpoExample], dict[str, AsyncDataset[DpoExample]]]:
    """Build train/validation split from a single component config."""
    training_components = _get_training_components(config)
    if len(training_components) != 1:
        raise ValueError(
            "DPO validation_split_fraction only supports single-component configs for now. "
            f"Found {len(training_components)} training components: {list(training_components.keys())}"
        )

    name, component = next(iter(training_components.items()))
    caches = config.build_caches("train")
    cache = caches.get(name)
    if cache is None:
        raise ValueError(f"No training cache available for component {name}.")

    base_dataset = dataset_for_preference_format(cast(PreferenceChatLmDatasetFormat, component.format), Pos, cache)

    total_len = len(base_dataset.as_sync_dataset())
    num_val = _num_validation_sequences(total_len, fraction)
    if num_val == 0:
        train_dataset = _build_dpo_dataset(config, Pos, key=key)
        return train_dataset, {}

    train_base = base_dataset.slice_dataset(end_index=total_len - num_val)
    val_base = base_dataset.slice_dataset(start_index=total_len - num_val, end_index=total_len)

    perm_type = config.permutation_type
    if perm_type == "linear":
        logger.warning("Using linear shuffling, not recommended. Please use Feistel permutation instead.")
    elif perm_type is None:
        perm_type = "feistel"

    train_dataset = train_base
    if config.shuffle is True:
        train_dataset = train_dataset.shuffle(key, perm_type=perm_type)
    elif isinstance(config.shuffle, int) and config.shuffle > 0:
        train_dataset = train_dataset.era_shuffle(config.shuffle, key=key, perm_type=perm_type)

    mix_key, _ = jrandom.split(key)
    mixture = MixtureDataset(
        datasets={name: train_dataset},
        weights={name: 1.0},
        block_size=config.mixture_block_size,
        key=mix_key,
        stop_strategy=config.stop_strategy,
    )

    train_dataset = cast(AsyncDataset[DpoExample], mixture)
    val_base = cast(AsyncDataset[DpoExample], val_base)
    return train_dataset, {name: val_base}


class DpoReferenceConfig(draccus.ChoiceRegistry):
    @classmethod
    def default_choice_name(cls) -> str | None:
        return "separate"


@DpoReferenceConfig.register_subclass("separate")
@dataclass(frozen=True)
class SeparateReferenceConfig(DpoReferenceConfig):
    model_path: str = ""
    is_hf: bool = True


@DpoReferenceConfig.register_subclass("adapter_base")
@dataclass(frozen=True)
class AdapterBaseReferenceConfig(DpoReferenceConfig):
    pass


@dataclass(frozen=True)
class AdapterBaseReferenceModelProvider:
    adapter: AdaptationConfig

    def model_for(self, policy_model: LmHeadModel) -> LmHeadModel:
        reference_model = self.adapter.base_model_view(policy_model)
        if reference_model is None:
            raise ValueError("reference.type=adapter_base requires an adapter that exposes a base_model_view.")
        return inference_mode(reference_model, True)


@dataclass
class TrainDpoConfig:
    data: PreferenceLmDataConfig = field(default_factory=PreferenceLmDataConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: LmConfig = field(default_factory=LlamaConfig)
    train_seq_len: int | None = None
    optimizer: OptimizerConfig = field(default_factory=AdamConfig)

    reference: DpoReferenceConfig = field(default_factory=SeparateReferenceConfig)
    adapter: AdaptationConfig = field(default_factory=NoAdaptationConfig)

    beta: float = 0.1

    initialize_from_hf: bool | str = False
    use_hf_model_config: bool = False

    validation_split_fraction: float | None = 0.1

    hf_save_path: Optional[str] = None
    hf_upload: bool | str = False
    hf_save_steps: int | None = 10000
    hf_save_dtype: Optional[str] = None

    peft_save_path: Optional[str] = None
    peft_hf_upload: bool | str = False
    merged_hf_save_path: Optional[str] = None
    merged_hf_upload: Optional[str] = None

    data_seed: Optional[int] = None
    initialize_from_checkpoint_path: Optional[str] = None


def _derive_training_keys(seed: int) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Derive training keys while preserving the legacy full-DPO model key path."""
    data_key, adapter_key, model_key, training_key = jrandom.split(jrandom.PRNGKey(seed), 4)
    policy_key, _ = jrandom.split(model_key)
    return data_key, model_key, policy_key, adapter_key, training_key


def _validate_dpo_config(config: TrainDpoConfig) -> None:
    if config.initialize_from_hf and config.trainer.initialize_from is not None:
        raise ValueError("Cannot specify both initialize_from_hf and initialize_from")

    if isinstance(config.reference, SeparateReferenceConfig):
        if not config.reference.model_path:
            raise ValueError("reference.model_path must be provided for reference.type=separate.")
        return

    if isinstance(config.reference, AdapterBaseReferenceConfig):
        if isinstance(config.adapter, NoAdaptationConfig):
            raise ValueError("reference.type=adapter_base requires a non-none adapter.")
        if isinstance(config.adapter, LoraAdaptationConfig) and not config.adapter.zero_init_b:
            raise ValueError("adapter.type=lora with reference.type=adapter_base requires zero_init_b=true.")
        return

    raise TypeError(f"Unsupported reference configuration: {type(config.reference).__name__}")


def _build_reference_provider(
    config: TrainDpoConfig,
):
    if isinstance(config.reference, AdapterBaseReferenceConfig):
        return AdapterBaseReferenceModelProvider(config.adapter)

    raise TypeError(f"Unsupported reference configuration: {type(config.reference).__name__}")


def _load_separate_reference_model(
    config: TrainDpoConfig,
    *,
    reference_model_key,
    model_context,
    Vocab,
    parameter_axis_mapping,
    trainer,
) -> LmHeadModel:
    if not isinstance(config.reference, SeparateReferenceConfig):
        raise TypeError(f"Unsupported reference configuration: {type(config.reference).__name__}")

    reference_model = load_model_from_source(
        context=model_context,
        Vocab=Vocab,
        model_key=reference_model_key,
        parameter_axis_mapping=parameter_axis_mapping,
        compute_dtype=trainer.mp.compute_dtype,
        cast_to_param=trainer.mp.cast_to_param,
        hf_ref=config.reference.model_path if config.reference.is_hf else False,
        checkpoint_path=None if config.reference.is_hf else config.reference.model_path,
    )
    reference_model = inference_mode(reference_model, True)
    return named_jit(trainer.mp.cast_to_compute, parameter_axis_mapping)(reference_model)


def _install_separate_reference_export_hooks(
    *,
    trainer,
    converter,
    export: AdaptationExportConfig,
) -> None:
    if export.peft_save_path is not None or export.merged_hf_save_path is not None:
        raise ValueError("peft_save_path and merged_hf_save_path require adapter.type: lora.")

    if export.hf_save_path is None or export.hf_save_steps is None:
        return

    if converter is None:
        raise ValueError("hf_save_path requires a HF-compatible model configuration.")

    full_save_path = export.hf_save_path
    if trainer.config.checkpointer is not None and trainer.config.checkpointer.append_run_id_to_base_path:
        full_save_path = os.path.join(full_save_path, trainer.run_id)

    save_dtype: jnp.dtype | None = None
    if export.hf_save_dtype is not None:
        try:
            save_dtype = jnp.dtype(export.hf_save_dtype)
        except TypeError:
            logger.warning(f"Invalid hf_save_dtype: {export.hf_save_dtype}. Defaulting to None.")

    def save_policy_hf_checkpoint(step):
        if step.step == 0:
            return

        upload_to_hf = export.hf_upload or False
        hf_upload_kwargs = {}
        if upload_to_hf is not None:
            hf_upload_kwargs["commit_message"] = f"Upload for step {step.step} from Levanter"

        converter.save_pretrained(
            _policy_model_for_hf_save(step.eval_model),
            os.path.join(full_save_path, f"step-{step.step}"),
            upload_to_hf=upload_to_hf,
            dtype=save_dtype,
            **hf_upload_kwargs,
        )

    trainer.add_hook(save_policy_hf_checkpoint, every=export.hf_save_steps)


def main(config: TrainDpoConfig):
    _validate_dpo_config(config)

    tokenizer = config.data.the_tokenizer
    model_context = prepare_model_init_context(
        config.model,
        tokenizer=tokenizer,
        initialize_from_hf=config.initialize_from_hf,
        use_hf_model_config=config.use_hf_model_config,
    )
    if model_context.model is not config.model:
        config = dataclasses.replace(config, model=model_context.model)

    levanter.initialize(config)
    optimizer = config.optimizer.build(config.trainer.num_train_steps)
    reference_provider: AdapterBaseReferenceModelProvider | None = None

    def loss_function(model: DpoModel | LmHeadModel, example: DpoExample, *, key=None):
        if isinstance(model, DpoModel):
            policy_model = model.policy
            reference_model = inference_mode(model.reference, True)
        else:
            if reference_provider is None:
                raise RuntimeError("Reference provider is not initialized.")
            policy_model = model
            reference_model = reference_provider.model_for(policy_model)

        if key is not None:
            key_chosen, key_rejected = jrandom.split(key)
        else:
            key_chosen = None
            key_rejected = None

        logp_pi_chosen = _logp_sum(policy_model, example.chosen, key=key_chosen)
        logp_pi_rejected = _logp_sum(policy_model, example.rejected, key=key_rejected)

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
        data_key, model_key, policy_key, adapter_key, training_key = _derive_training_keys(seed)

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
                key=data_key,
                fraction=fraction,
            )
        else:
            train_dataset = _build_dpo_dataset(config.data, Pos, key=data_key)
            val_caches = config.data.build_caches("validation")
            for name, component in config.data.components.items():
                cache = val_caches.get(name)
                if cache is None:
                    continue
                validation_sets[name] = cast(
                    AsyncDataset[DpoExample],
                    dataset_for_preference_format(cast(PreferenceChatLmDatasetFormat, component.format), Pos, cache),
                )

        initial_policy_model = config.model.build(Vocab, key=policy_key)
        initial_policy_model = config.adapter.apply(
            initial_policy_model,
            key=adapter_key,
            axis_mapping=parameter_axis_mapping,
        )
        if isinstance(config.reference, SeparateReferenceConfig):
            initial_reference_model = config.model.build(Vocab, key=model_key)
            initial_model = DpoModel(policy=initial_policy_model, reference=initial_reference_model)
            trainable_filter = DpoModel(policy=config.adapter.trainable_filter(initial_policy_model), reference=False)
        else:
            initial_model = initial_policy_model
            trainable_filter = config.adapter.trainable_filter(initial_policy_model)

        state = trainer.initial_state(training_key, model=initial_model, is_trainable=trainable_filter)

        if int(state.step) == 0:
            if config.initialize_from_hf:
                logger.info(
                    f"No training checkpoint found. Initializing model from HF checkpoint '{config.initialize_from_hf}'"
                )
                policy_model = load_model_from_source(
                    context=model_context,
                    Vocab=Vocab,
                    model_key=policy_key,
                    parameter_axis_mapping=parameter_axis_mapping,
                    compute_dtype=trainer.mp.compute_dtype,
                    cast_to_param=trainer.mp.cast_to_param,
                    hf_ref=config.initialize_from_hf,
                )
                policy_model = config.adapter.apply(
                    policy_model,
                    key=adapter_key,
                    axis_mapping=parameter_axis_mapping,
                )
            elif config.initialize_from_checkpoint_path is not None:
                policy_model = load_model_from_source(
                    context=model_context,
                    Vocab=Vocab,
                    model_key=policy_key,
                    parameter_axis_mapping=parameter_axis_mapping,
                    compute_dtype=trainer.mp.compute_dtype,
                    cast_to_param=trainer.mp.cast_to_param,
                    checkpoint_path=config.initialize_from_checkpoint_path,
                )
                policy_model = config.adapter.apply(
                    policy_model,
                    key=adapter_key,
                    axis_mapping=parameter_axis_mapping,
                )
            else:
                logger.info("No checkpoint found. Starting from scratch.")
                policy_model = state.model.policy if isinstance(state.model, DpoModel) else state.model
        else:
            logger.info(f"Resuming from step {state.step}, using checkpoint policy weights.")
            policy_model = state.model.policy if isinstance(state.model, DpoModel) else state.model

        if isinstance(config.reference, SeparateReferenceConfig):
            reference_model = _load_separate_reference_model(
                config,
                reference_model_key=model_key,
                model_context=model_context,
                Vocab=Vocab,
                parameter_axis_mapping=parameter_axis_mapping,
                trainer=trainer,
            )
            state = dataclasses.replace(state, model=DpoModel(policy=policy_model, reference=reference_model))
        else:
            reference_provider = _build_reference_provider(config)
            state = dataclasses.replace(state, model=policy_model)

        all_param_count = parameter_count(state.model)
        trainable_param_count = parameter_count(state.trainable_model)
        levanter.tracker.log_summary(
            {
                "parameter_count": all_param_count,
                "trainable_parameter_count": trainable_param_count,
                "fraction_trainable": trainable_param_count * 1.0 / all_param_count,
            }
        )

        flops_per_token = config.model.flops_per_token(vocab_size, Pos.size)
        flops_per_example = 3 * flops_per_token * Pos.size if flops_per_token is not None else None
        trainer.add_hook(
            callbacks.log_performance_stats(Pos.size, trainer.config.batch_schedule, flops_per_example),
            every=1,
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

        export_config = AdaptationExportConfig(
            hf_save_path=config.hf_save_path,
            hf_upload=config.hf_upload,
            hf_save_steps=config.hf_save_steps,
            hf_save_dtype=config.hf_save_dtype,
            peft_save_path=config.peft_save_path,
            peft_hf_upload=config.peft_hf_upload,
            merged_hf_save_path=config.merged_hf_save_path,
            merged_hf_upload=config.merged_hf_upload,
        )
        if isinstance(config.reference, SeparateReferenceConfig) and isinstance(config.adapter, NoAdaptationConfig):
            _install_separate_reference_export_hooks(
                trainer=trainer,
                converter=model_context.converter,
                export=export_config,
            )
        else:
            config.adapter.install_export_hooks(
                trainer=trainer,
                converter=model_context.converter,
                tokenizer=tokenizer,
                export=export_config,
            )

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
