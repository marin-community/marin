# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, cast

import draccus
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from haliax import Axis
from haliax.partitioning import named_jit, round_axis_for_partitioning

import levanter
import levanter.callbacks
import levanter.eval
from levanter import callbacks
from levanter.adaptation import (
    AdaptationConfig,
    AdaptationExportConfig,
    LoraAdaptationConfig,
    NoAdaptationConfig,
)
from levanter.compat.hf_checkpoints import build_generation_config
from levanter.data.dataset import AsyncDataset
from levanter.data.mixture import MixtureDataset
from levanter.data.text import (
    DpoExample,
    LmDataConfig,
    PreferenceChatLmDatasetFormat,
    PreferenceLmDataConfig,
    dataset_for_preference_format,
)
from levanter.dpo import (
    CachedDpoExample,
    CachedReferenceDataset,
    DpoModel,
    ReferenceEvalCacheConfig,
    ValidationDatasetSpec,
    build_or_load_reference_eval_cache,
    dpo_loss,
    reference_eval_cache_metadata,
    reference_eval_cache_path,
)
from levanter.main.model_init import load_model_from_source, prepare_model_init_context
from levanter.models.llama import LlamaConfig
from levanter.models.lm_model import LmConfig, LmExample, LmHeadModel
from levanter.optim import AdamConfig, OptimizerConfig
from levanter.trainer import Trainer, TrainerConfig
from levanter.trainer_state import trainables_only
from levanter.utils.jax_utils import parameter_count
from levanter.utils.tree_utils import inference_mode


logger = logging.getLogger(__name__)


def _policy_model_for_hf_save(model: DpoModel | LmHeadModel) -> LmHeadModel:
    return model.policy if isinstance(model, DpoModel) else model


def _restore_policy_model_from_partial_checkpoint(
    checkpointed_policy_model: LmHeadModel,
    source_policy_model: LmHeadModel,
    trainable_filter,
) -> LmHeadModel:
    checkpointed_trainables = trainables_only(checkpointed_policy_model, trainable_filter)
    return eqx.combine(checkpointed_trainables, source_policy_model)


def _initialize_policy_model_from_source(
    *,
    config: "TrainDpoConfig",
    model_context,
    Vocab: Axis,
    policy_key,
    adapter_key,
    parameter_axis_mapping,
    trainer: Trainer,
) -> LmHeadModel:
    if config.initialize_from_hf:
        policy_model = load_model_from_source(
            context=model_context,
            Vocab=Vocab,
            model_key=policy_key,
            parameter_axis_mapping=parameter_axis_mapping,
            compute_dtype=trainer.mp.compute_dtype,
            cast_to_param=trainer.mp.cast_to_param,
            hf_ref=config.initialize_from_hf,
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
    else:
        policy_model = config.model.build(Vocab, key=policy_key)

    return config.adapter.apply(
        policy_model,
        key=adapter_key,
        axis_mapping=parameter_axis_mapping,
    )


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
) -> tuple[AsyncDataset[DpoExample], dict[str, ValidationDatasetSpec]]:
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
    return train_dataset, {
        name: ValidationDatasetSpec(
            name=name,
            dataset=val_base,
            source_cache_path=cache.cache_dir,
            source_split="train",
            slice_start=total_len - num_val,
            slice_end=total_len,
        )
    }


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
    reference_eval_cache: ReferenceEvalCacheConfig = field(default_factory=ReferenceEvalCacheConfig)
    lm_validation_data: LmDataConfig | None = None
    lm_validation_prefix: str = "lm_eval"

    hf_save_path: Optional[str] = None
    hf_upload: bool | str = False
    hf_save_steps: int | None = 10000
    hf_save_dtype: Optional[str] = None
    hf_generation_eos_token_ids: Optional[list[int]] = None

    peft_save_path: Optional[str] = None
    peft_hf_upload: bool | str = False
    merged_hf_save_path: Optional[str] = None
    merged_hf_upload: Optional[str] = None

    data_seed: Optional[int] = None
    initialize_from_checkpoint_path: Optional[str] = None
    scheduled_eval_steps: Optional[list[int]] = None
    run_initial_eval: bool = False


def _derive_training_keys(seed: int) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Derive training keys while preserving the legacy full-DPO model key path."""
    data_key, adapter_key, model_key, training_key = jrandom.split(jrandom.PRNGKey(seed), 4)
    policy_key, _ = jrandom.split(model_key)
    return data_key, model_key, policy_key, adapter_key, training_key


def _periodic_eval_callback(callback: Callable[[Any], Any]) -> Callable[..., None]:
    last_eval_step: int | None = None

    def maybe_run_eval(step_info, *, force: bool = False):
        del force
        nonlocal last_eval_step
        if last_eval_step == step_info.step:
            return
        callback(step_info)
        last_eval_step = step_info.step

    return maybe_run_eval


def _scheduled_eval_callback(callback: Callable[[Any], Any], eval_steps: set[int]) -> Callable[..., None]:
    last_eval_step: int | None = None

    def maybe_run_eval(step_info, *, force: bool = False):
        nonlocal last_eval_step
        if last_eval_step == step_info.step:
            return
        if force or step_info.step in eval_steps:
            callback(step_info)
            last_eval_step = step_info.step

    return maybe_run_eval


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
        if isinstance(config.adapter, LoraAdaptationConfig):
            # Require B @ A = 0 so policy == reference at step 0: implicit reward is 0 and DPO loss is log 2.
            # Exactly one factor must be zero so gradients can still flow.
            has_zero_b = config.adapter.zero_init_b
            has_zero_a = config.adapter.a_init_mode == "zero"
            if has_zero_b == has_zero_a:
                raise ValueError(
                    "adapter.type=lora with reference.type=adapter_base requires zero adapter delta at init "
                    "with exactly one zero LoRA factor: set either zero_init_b=true or a_init_mode='zero', "
                    "but not both."
                )
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


def _reference_eval_cache_identity(config: TrainDpoConfig) -> dict[str, Any]:
    if isinstance(config.reference, SeparateReferenceConfig):
        return {
            "reference_type": "separate",
            "reference": config.reference,
        }

    if isinstance(config.reference, AdapterBaseReferenceConfig):
        base_source: dict[str, Any] = {
            "initialize_from_hf": config.initialize_from_hf,
            "initialize_from_checkpoint_path": config.initialize_from_checkpoint_path,
            "use_hf_model_config": config.use_hf_model_config,
        }
        if not config.initialize_from_hf and config.initialize_from_checkpoint_path is None:
            base_source["model"] = config.model
            base_source["seed"] = config.trainer.seed

        return {
            "reference_type": "adapter_base",
            "reference": config.reference,
            "adapter": config.adapter,
            "base_source": base_source,
        }

    raise TypeError(f"Unsupported reference configuration: {type(config.reference).__name__}")


def _build_validation_specs(config: PreferenceLmDataConfig, Pos: Axis) -> dict[str, ValidationDatasetSpec]:
    validation_specs: dict[str, ValidationDatasetSpec] = {}
    val_caches = config.build_caches("validation")
    for name, component in config.components.items():
        cache = val_caches.get(name)
        if cache is None:
            continue
        validation_specs[name] = ValidationDatasetSpec(
            name=name,
            dataset=cast(
                AsyncDataset[DpoExample],
                dataset_for_preference_format(cast(PreferenceChatLmDatasetFormat, component.format), Pos, cache),
            ),
            source_cache_path=cache.cache_dir,
            source_split="validation",
        )
    return validation_specs


def _maybe_prepare_cached_validation_specs(
    *,
    config: TrainDpoConfig,
    validation_specs: dict[str, ValidationDatasetSpec],
    trainer,
    reference_model: LmHeadModel,
    reference_identity: dict[str, Any],
    seq_len: int,
) -> dict[str, ValidationDatasetSpec]:
    if not validation_specs or config.reference_eval_cache.mode == "disabled":
        return validation_specs
    if config.reference_eval_cache.mode != "build_or_load":
        raise ValueError(f"Unsupported reference_eval_cache.mode: {config.reference_eval_cache.mode}")

    cached_specs: dict[str, ValidationDatasetSpec] = {}
    for name, spec in validation_specs.items():
        base_dataset = cast(AsyncDataset[DpoExample], spec.dataset)
        metadata = reference_eval_cache_metadata(spec, reference_identity=reference_identity, seq_len=seq_len)
        cache_dir = reference_eval_cache_path(
            spec,
            reference_identity=reference_identity,
            seq_len=seq_len,
            cache_dir=config.reference_eval_cache.cache_dir,
        )
        ref_chosen, ref_rejected = build_or_load_reference_eval_cache(
            reference_model=reference_model,
            dataset=base_dataset,
            eval_loader=trainer.data_loader(base_dataset, trainer.EvalBatch),
            compute_axis_mapping=trainer.compute_axis_mapping,
            mp=trainer.mp,
            cache_dir=cache_dir,
            metadata=metadata,
        )
        cached_specs[name] = dataclasses.replace(
            spec,
            dataset=CachedReferenceDataset(base_dataset, ref_chosen, ref_rejected),
        )

    return cached_specs


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
    generation_config = build_generation_config(tokenizer, config.hf_generation_eos_token_ids)
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

    def loss_function(model: DpoModel | LmHeadModel, example: DpoExample | CachedDpoExample, *, key=None):
        if isinstance(model, DpoModel):
            policy_model = model.policy
            reference_model = None if isinstance(example, CachedDpoExample) else inference_mode(model.reference, True)
        else:
            policy_model = model
            reference_model = None
            if not isinstance(example, CachedDpoExample):
                if reference_provider is None:
                    raise RuntimeError("Reference provider is not initialized.")
                reference_model = reference_provider.model_for(policy_model)

        if key is not None:
            key_chosen, key_rejected = jrandom.split(key)
        else:
            key_chosen = None
            key_rejected = None

        return dpo_loss(
            policy_model,
            reference_model,
            example,
            beta=config.beta,
            key_chosen=key_chosen,
            key_rejected=key_rejected,
        )

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

        validation_specs: dict[str, ValidationDatasetSpec] = {}
        if config.validation_split_fraction is not None:
            fraction = config.validation_split_fraction
            if fraction < 0 or fraction >= 1:
                raise ValueError(f"validation_split_fraction must be in [0, 1), got {fraction}")
            train_dataset, validation_specs = _build_validation_split(
                config.data,
                Pos,
                key=data_key,
                fraction=fraction,
            )
        else:
            train_dataset = _build_dpo_dataset(config.data, Pos, key=data_key)
            validation_specs = _build_validation_specs(config.data, Pos)

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
                policy_model = _initialize_policy_model_from_source(
                    config=config,
                    model_context=model_context,
                    Vocab=Vocab,
                    policy_key=policy_key,
                    adapter_key=adapter_key,
                    parameter_axis_mapping=parameter_axis_mapping,
                    trainer=trainer,
                )
            elif config.initialize_from_checkpoint_path is not None:
                policy_model = _initialize_policy_model_from_source(
                    config=config,
                    model_context=model_context,
                    Vocab=Vocab,
                    policy_key=policy_key,
                    adapter_key=adapter_key,
                    parameter_axis_mapping=parameter_axis_mapping,
                    trainer=trainer,
                )
            else:
                logger.info("No checkpoint found. Starting from scratch.")
                policy_model = state.model.policy if isinstance(state.model, DpoModel) else state.model
        else:
            logger.info(f"Resuming from step {state.step}, using checkpoint policy weights.")
            policy_model = state.model.policy if isinstance(state.model, DpoModel) else state.model
            if not isinstance(config.adapter, NoAdaptationConfig):
                logger.info(
                    "Adapter checkpoints only store trainable weights. Reconstructing the base policy model from the "
                    "configured source before overlaying resumed adapter parameters."
                )
                source_policy_model = _initialize_policy_model_from_source(
                    config=config,
                    model_context=model_context,
                    Vocab=Vocab,
                    policy_key=policy_key,
                    adapter_key=adapter_key,
                    parameter_axis_mapping=parameter_axis_mapping,
                    trainer=trainer,
                )
                policy_model = _restore_policy_model_from_partial_checkpoint(
                    policy_model,
                    source_policy_model,
                    config.adapter.trainable_filter(source_policy_model),
                )

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

        if validation_specs and config.reference_eval_cache.mode != "disabled":
            if isinstance(state.model, DpoModel):
                eval_reference_model = state.model.reference
            else:
                if reference_provider is None:
                    raise RuntimeError("Reference provider is not initialized.")
                eval_reference_model = reference_provider.model_for(state.model)

            validation_specs = _maybe_prepare_cached_validation_specs(
                config=config,
                validation_specs=validation_specs,
                trainer=trainer,
                reference_model=eval_reference_model,
                reference_identity=_reference_eval_cache_identity(config),
                seq_len=Pos.size,
            )

        all_param_count = parameter_count(state.model)
        trainable_param_count = parameter_count(state.trainable_model)
        levanter.tracker.log_summary(
            {
                "parameter_count": all_param_count,
                "trainable_parameter_count": trainable_param_count,
                "fraction_trainable": trainable_param_count * 1.0 / all_param_count,
            }
        )

        max_eval_examples_per_ds = config.trainer.max_eval_batches
        if max_eval_examples_per_ds is not None:
            max_eval_examples_per_ds *= config.trainer.eval_batch_size

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

        lm_eval_callback: Callable[[Any], Any] | None = None
        if config.lm_validation_data is not None:
            tagged_lm_eval_datasets = config.lm_validation_data.tagged_eval_sets(Pos)
            if tagged_lm_eval_datasets:
                checkpoint_path = None
                if config.trainer.checkpointer is not None:
                    checkpoint_path = config.trainer.checkpointer.expanded_path(trainer.run_id)

                def lm_eval_loss(model: DpoModel | LmHeadModel, batch: LmExample) -> levanter.eval.LossFnOutput:
                    policy_model = inference_mode(_policy_model_for_hf_save(model), True)
                    policy_model = trainer.mp.cast_to_compute(policy_model)
                    per_pos_loss = policy_model.compute_next_token_loss(batch, reduction=None, reduction_axis=()).array
                    per_pos_weight = batch.loss_weight.array
                    per_pos_token_id = jnp.roll(batch.tokens.array, -1, axis=-1)
                    return per_pos_loss, per_pos_weight, per_pos_token_id

                lm_eval_callback = levanter.eval.cb_tagged_lm_evaluate(
                    trainer.EvalBatch,
                    tagged_lm_eval_datasets,
                    tokenizer=tokenizer,
                    device_mesh=trainer.device_mesh,
                    axis_mapping=trainer.compute_axis_mapping,
                    max_examples_per_dataset=max_eval_examples_per_ds,
                    prefix=config.lm_validation_prefix,
                    mp=trainer.mp,
                    checkpoint_path=checkpoint_path,
                    loss_fn=lm_eval_loss,
                    eval_ema=False,
                )
            else:
                logger.warning("No LM evaluation datasets provided for DPO.")

        initial_eval_callbacks: list[Any] = []
        if validation_specs or lm_eval_callback is not None:
            eval_steps = set(config.scheduled_eval_steps) if config.scheduled_eval_steps is not None else None
            hook_every = 1 if eval_steps is not None else config.trainer.steps_per_eval

            for name, spec in validation_specs.items():
                eval_loader = trainer.data_loader(spec.dataset, trainer.EvalBatch)
                if not eval_loader or (
                    trainer.config.max_eval_batches is not None and trainer.config.max_eval_batches <= 0
                ):
                    continue

                @eqx.filter_jit
                def eval_loss(model, *batch, **batch_kwargs):
                    model = trainer.mp.cast_to_compute(model)
                    return trainer.loss_fn(model, *batch, **batch_kwargs, key=None)

                compute_loss = callbacks.compute_validation_loss(
                    eval_loss,
                    eval_loader,
                    max_batches=trainer.config.max_eval_batches,
                    name=name or None,
                )
                if eval_steps is None:
                    wrapped_callback = _periodic_eval_callback(compute_loss)
                else:
                    wrapped_callback = _scheduled_eval_callback(compute_loss, eval_steps)
                initial_eval_callbacks.append(wrapped_callback)
                trainer.add_hook(wrapped_callback, every=hook_every)

            if lm_eval_callback is not None:
                if eval_steps is None:
                    wrapped_callback = _periodic_eval_callback(lm_eval_callback)
                else:
                    wrapped_callback = _scheduled_eval_callback(lm_eval_callback, eval_steps)
                initial_eval_callbacks.append(wrapped_callback)
                trainer.add_hook(wrapped_callback, every=hook_every)
        else:
            logger.warning("No validation datasets provided.")

        export_config = AdaptationExportConfig(
            hf_save_path=config.hf_save_path,
            hf_upload=config.hf_upload,
            hf_save_steps=config.hf_save_steps,
            hf_save_dtype=config.hf_save_dtype,
            generation_config=generation_config,
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

        if config.run_initial_eval and initial_eval_callbacks and state.step == 0:
            logger.info("Running initial validation before training.")
            initial_eval_state = dataclasses.replace(state, step=1)
            initial_eval_info = callbacks.StepInfo(initial_eval_state, 0.0, 0.0)
            for callback in initial_eval_callbacks:
                callback(initial_eval_info, force=True)

        trainer.train(state, train_loader)

        if trainer.config.checkpointer is not None:
            checkpointer = trainer.config.checkpointer.create(trainer.run_id)
            checkpointer.wait_until_finished()

    trainer.tracker.finish()


if __name__ == "__main__":
    levanter.config.main(main)()
