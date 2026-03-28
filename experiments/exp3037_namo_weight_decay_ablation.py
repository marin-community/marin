# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exp3037: NAMO/NAMO-D weight-decay ablation for issue #3037.

This experiment directly tests whether performance deltas come from optimizer
updates or from matrix weight-decay semantics.

Architecture families:
- GPT-2 style (small proxy)
- Llama style (small proxy)

Variants per family:
- AdamW standard decay
- AdamW reduced decay (0.2x)
- Muon standard decay
- Muon reduced decay (0.2x)
- NAMO adaptive matrix decay (paper style)
- NAMO AdamW-style matrix decay
- NAMO-D adaptive matrix decay (paper style)
- NAMO-D AdamW-style matrix decay

Issue: https://github.com/marin-community/marin/issues/3037
"""

from __future__ import annotations

import dataclasses
import os
import os.path
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Literal, Protocol, TypeAlias

import jmp
import jax
import optax
from fray.cluster import ResourceConfig
from haliax.nn import Linear
from haliax.partitioning import ResourceAxis
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import LmDataConfig
from levanter.main import train_lm
from levanter.models.gpt2 import Gpt2Config
from levanter.models.llama import LlamaConfig
from levanter.optim import AdamConfig, MuonConfig, NamoConfig, NamoDConfig
from levanter.optim.config import OptimizerConfig
from levanter.optim.namo import scale_with_namo, scale_with_namod
from levanter.optim.util import CoefficientType, flatten_linear_layers, unflatten_linear_layers
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig

from experiments.marin_models import marin_tokenizer
from experiments.paloma import paloma_tokenized
from experiments.simple_train_config import SimpleTrainConfig
from experiments.speedrun.prebuilt_caches import fineweb_edu_subcache_10B
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.processing.tokenize import TokenizeConfig, lm_mixture_data_config
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm

STEPS: int = 50_000
SEQ_LEN: int = 1024
BATCH_SIZE: int = 6

USE_LOCAL_CACHES_ENV: str = "EXP3037_USE_LOCAL_CACHES"
USE_LOCAL_C4_ENV: str = "EXP3037_USE_LOCAL_C4"
LOCAL_TRAIN_CACHE_PATH_ENV: str = "EXP3037_LOCAL_TRAIN_CACHE_PATH"
LOCAL_C4_CACHE_PATH_ENV: str = "EXP3037_LOCAL_C4_CACHE_PATH"
VARIANTS_ENV: str = "EXP3037_VARIANTS"
MODEL_FAMILIES_ENV: str = "EXP3037_MODEL_FAMILIES"

Scalar: TypeAlias = float | jax.Array
ModelFamily: TypeAlias = Literal["gpt2", "llama"]


class _TrainOptimizer(Protocol):
    learning_rate: float
    weight_decay: float

    def build(self, num_train_steps: int) -> optax.GradientTransformation: ...


@dataclass(frozen=True)
class Variant:
    name: str
    optimizer: _TrainOptimizer
    tags: tuple[str, ...]


def _is_linear_or_none(x: Any) -> bool:
    return isinstance(x, Linear) or x is None


def _create_namo_mask(params: Any) -> Any:
    from levanter.utils.jax_utils import leaf_key_paths

    paths = leaf_key_paths(params)

    def mask_fn(param: Any, path: Any) -> Any:
        path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
        if "Embedding" in path_str or "lm_head" in path_str:
            return "adamw"
        if isinstance(param, Linear):
            return dataclasses.replace(param, weight="namo", bias="adamw" if param.bias is not None else None)
        return "adamw"

    return jax.tree_util.tree_map(mask_fn, params, paths, is_leaf=lambda x: isinstance(x, Linear))


def _adamw_fallback_transform(
    *,
    max_grad_norm: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    adam_lr: Scalar,
    weight_decay: float,
    build_weight_decay_mask: Any,
) -> optax.GradientTransformation:
    components = []
    if max_grad_norm:
        components.append(optax.clip_by_global_norm(max_grad_norm))
    components.append(optax.scale_by_adam(beta1, beta2, epsilon))
    if weight_decay > 0:
        components.append(optax.add_decayed_weights(weight_decay, build_weight_decay_mask()))
    components.append(optax.scale(-adam_lr))
    return optax.chain(*components)


def _add_matrix_adamw_decay(
    *,
    learning_rate: Scalar,
    weight_decay: float,
) -> optax.GradientTransformation:
    def init_fn(params: optax.Params) -> optax.EmptyState:
        del params
        return optax.EmptyState()

    def update_fn(
        updates: optax.Updates,
        state: optax.EmptyState,
        params: optax.Params | None = None,
    ) -> tuple[optax.Updates, optax.EmptyState]:
        if params is None:
            raise ValueError("Matrix-AdamW-decay variants require params to apply decoupled decay")

        if weight_decay <= 0:
            return updates, state

        flat_updates = flatten_linear_layers(updates)
        flat_params = flatten_linear_layers(params)

        def apply_decay(update_node: Any, param_node: Any) -> Any:
            if not isinstance(update_node, Linear) or not isinstance(param_node, Linear):
                return update_node

            decay_delta = -learning_rate * weight_decay * param_node.weight.array
            new_weight = dataclasses.replace(update_node.weight, array=update_node.weight.array + decay_delta)
            return dataclasses.replace(update_node, weight=new_weight)

        decayed_flat_updates = jax.tree_util.tree_map(
            apply_decay,
            flat_updates,
            flat_params,
            is_leaf=_is_linear_or_none,
        )
        return unflatten_linear_layers(updates, decayed_flat_updates), state

    return optax.GradientTransformation(init_fn, update_fn)


@dataclass(frozen=True)
class NamoAdamWMatrixDecayConfig(OptimizerConfig):
    """Experiment-local NAMO variant with AdamW-style matrix decay."""

    learning_rate: float = 1e-2
    adam_lr: float = 6e-4
    momentum: float = 0.95
    mu2: float = 0.99
    adamnorm_eps: float = 1e-8
    nesterov: bool = True
    backend_steps: int = 5
    muon_epsilon: float = 1e-8
    scale_coeff: float = 0.2
    adam_weight_decay: float | None = None
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    coefficient_type: CoefficientType = "simple"

    def build(self, num_train_steps: int) -> optax.GradientTransformation:
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)

        def optimizer(learning_rate: Scalar, adam_lr: Scalar) -> optax.GradientTransformation:
            adam_weight_decay = self.adam_weight_decay if self.adam_weight_decay is not None else self.weight_decay
            transformations = {
                "namo": optax.chain(
                    scale_with_namo(
                        momentum=self.momentum,
                        mu2=self.mu2,
                        nesterov=self.nesterov,
                        steps=self.backend_steps,
                        muon_eps=self.muon_epsilon,
                        learning_rate=learning_rate,
                        weight_decay=0.0,
                        adamnorm_eps=self.adamnorm_eps,
                        scale_coeff=self.scale_coeff,
                        coefficient_type=self.coefficient_type,
                    ),
                    _add_matrix_adamw_decay(
                        learning_rate=learning_rate,
                        weight_decay=self.weight_decay,
                    ),
                ),
                "adamw": _adamw_fallback_transform(
                    max_grad_norm=self.max_grad_norm,
                    beta1=self.beta1,
                    beta2=self.beta2,
                    epsilon=self.epsilon,
                    adam_lr=adam_lr,
                    weight_decay=adam_weight_decay,
                    build_weight_decay_mask=self.build_weight_decay_mask,
                ),
            }
            return optax.multi_transform(transformations, _create_namo_mask)

        return optax.inject_hyperparams(optimizer)(learning_rate=learning_rate_schedule, adam_lr=adam_lr_schedule)


@dataclass(frozen=True)
class NamoDAdamWMatrixDecayConfig(OptimizerConfig):
    """Experiment-local NAMO-D variant with AdamW-style matrix decay."""

    learning_rate: float = 1e-2
    adam_lr: float = 6e-4
    momentum: float = 0.95
    mu2: float = 0.99
    adamnorm_eps: float = 1e-8
    nesterov: bool = True
    backend_steps: int = 5
    muon_epsilon: float = 1e-8
    scale_coeff: float = 0.2
    clamp_c: float = 0.75
    adam_weight_decay: float | None = None
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    coefficient_type: CoefficientType = "simple"

    def build(self, num_train_steps: int) -> optax.GradientTransformation:
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)

        def optimizer(learning_rate: Scalar, adam_lr: Scalar) -> optax.GradientTransformation:
            adam_weight_decay = self.adam_weight_decay if self.adam_weight_decay is not None else self.weight_decay
            transformations = {
                "namo": optax.chain(
                    scale_with_namod(
                        momentum=self.momentum,
                        mu2=self.mu2,
                        nesterov=self.nesterov,
                        steps=self.backend_steps,
                        muon_eps=self.muon_epsilon,
                        learning_rate=learning_rate,
                        weight_decay=0.0,
                        adamnorm_eps=self.adamnorm_eps,
                        scale_coeff=self.scale_coeff,
                        clamp_c=self.clamp_c,
                        coefficient_type=self.coefficient_type,
                    ),
                    _add_matrix_adamw_decay(
                        learning_rate=learning_rate,
                        weight_decay=self.weight_decay,
                    ),
                ),
                "adamw": _adamw_fallback_transform(
                    max_grad_norm=self.max_grad_norm,
                    beta1=self.beta1,
                    beta2=self.beta2,
                    epsilon=self.epsilon,
                    adam_lr=adam_lr,
                    weight_decay=adam_weight_decay,
                    build_weight_decay_mask=self.build_weight_decay_mask,
                ),
            }
            return optax.multi_transform(transformations, _create_namo_mask)

        return optax.inject_hyperparams(optimizer)(learning_rate=learning_rate_schedule, adam_lr=adam_lr_schedule)


def _build_data_config() -> LmDataConfig:
    use_local_caches = os.environ.get(USE_LOCAL_CACHES_ENV, "0") == "1"
    local_train_cache_path = os.environ.get(LOCAL_TRAIN_CACHE_PATH_ENV)
    local_c4_cache_path = os.environ.get(LOCAL_C4_CACHE_PATH_ENV)
    local_c4_available = local_c4_cache_path is not None and os.path.exists(
        f"{local_c4_cache_path}/validation/shard_ledger.json"
    )
    use_local_c4 = os.environ.get(USE_LOCAL_C4_ENV, "1") == "1" and local_c4_available

    c4_component: TokenizeConfig | ExecutorStep
    if use_local_c4 and local_c4_cache_path is not None:
        c4_component = TokenizeConfig(
            train_paths=[],
            validation_paths=[local_c4_cache_path],
            cache_path=local_c4_cache_path,
            tokenizer=marin_tokenizer,
        )
    else:
        c4_component = paloma_tokenized(tokenizer=marin_tokenizer)["paloma/c4_en"]

    if use_local_caches:
        if local_train_cache_path is None:
            raise ValueError(
                f"{USE_LOCAL_CACHES_ENV}=1 requires {LOCAL_TRAIN_CACHE_PATH_ENV} to point to a local cache path"
            )
        train_cfg = TokenizeConfig(
            train_paths=[local_train_cache_path],
            validation_paths=[],
            cache_path=local_train_cache_path,
            tokenizer=marin_tokenizer,
        )
        components = {
            "train/fineweb_edu_10B_local": train_cfg,
            "paloma/c4_en": c4_component,
        }
        weights = {"train/fineweb_edu_10B_local": 1.0}
    else:
        components = {
            "train/fineweb_edu_10B": fineweb_edu_subcache_10B,
            "paloma/c4_en": c4_component,
        }
        weights = {"train/fineweb_edu_10B": 1.0}

    return lm_mixture_data_config(
        components=components,
        weights=weights,
        include_raw_paths=False,
    )


def _common_train_config(optimizer_config: _TrainOptimizer) -> SimpleTrainConfig:
    return SimpleTrainConfig(
        resources=ResourceConfig.with_gpu("A100-80G", count=1),
        train_batch_size=BATCH_SIZE,
        num_train_steps=STEPS,
        train_seq_len=SEQ_LEN,
        optimizer_config=optimizer_config,
        learning_rate=float(optimizer_config.learning_rate),
        weight_decay=float(optimizer_config.weight_decay),
        steps_per_eval=5_000,
        steps_per_export=1_000_000,
        steps_per_task_eval=1_000_000,
        steps_per_hf_export=-1,
    )


def _build_train_step(
    *,
    name: str,
    data_config: LmDataConfig,
    model_config: Any,
    train_config: SimpleTrainConfig,
    tags: list[str],
) -> ExecutorStep:
    inner_config = train_lm.TrainLmConfig(
        data=data_config,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                project="marin",
                tags=tags,
                replicate_path=this_output_path(),
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=train_config.train_batch_size,
            per_device_parallelism=train_config.per_device_parallelism,
            num_train_steps=train_config.num_train_steps,
            steps_per_eval=train_config.steps_per_eval if train_config.steps_per_eval is not None else 1000,
            checkpointer=CheckpointerConfig(
                save_interval=timedelta(minutes=10),
                keep=[dict(every=train_config.steps_per_export)],
            ),
            mesh=MeshConfig(
                compute_mapping={
                    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                }
            ),
            allow_partial_checkpoint=train_config.allow_partial_checkpoint,
            per_device_eval_parallelism=(
                train_config.per_device_eval_parallelism if train_config.per_device_eval_parallelism is not None else -1
            ),
            max_eval_batches=train_config.max_eval_batches,
            allow_nondivisible_batch_size=True,
            watch=train_config.watch,
            profiler=train_config.profiler,
            use_explicit_mesh_axes=train_config.explicit_mesh_axes,
        ),
        train_seq_len=train_config.train_seq_len or model_config.max_seq_len,
        model=model_config,
        optimizer=train_config.optimizer_config,
        data_seed=train_config.data_seed,
        eval_harness_steps=train_config.steps_per_task_eval or 1_000_000,
        eval_harness=None,
    )

    pod_config = TrainLmOnPodConfig(
        train_config=inner_config,
        resources=train_config.resources,
        output_path=this_output_path(),
    )

    return ExecutorStep(
        name=os.path.join("checkpoints", name),
        fn=run_levanter_train_lm,
        config=pod_config,
    )


def _build_model_config(model_family: ModelFamily) -> Any:
    if model_family == "gpt2":
        return Gpt2Config(max_seq_len=SEQ_LEN)

    return LlamaConfig(
        max_seq_len=SEQ_LEN,
        hidden_dim=512,
        intermediate_dim=1792,
        num_heads=8,
        num_kv_heads=8,
        num_layers=6,
    )


def _variants(model_family: ModelFamily) -> list[Variant]:
    namo_common = dict(
        learning_rate=1.2e-2,
        adam_lr=1.3e-3,
        weight_decay=0.01,
        warmup=0.0,
        min_lr_ratio=0.0,
        lr_schedule="linear",
        decay=1.0,
        momentum=0.95,
        mu2=0.99,
        adamnorm_eps=1e-8,
        muon_epsilon=1e-8,
        max_grad_norm=1.0,
        coefficient_type="simple",
    )
    muon_common = dict(
        learning_rate=1.3e-3,
        adam_lr=1.3e-3,
        warmup=0.0,
        min_lr_ratio=0.0,
        lr_schedule="linear",
        decay=1.0,
        momentum=0.95,
        beta1=0.9,
        beta2=0.95,
        epsilon=1e-8,
        muon_epsilon=1e-8,
        max_grad_norm=1.0,
        use_kimi_scaling=(model_family == "gpt2"),
    )
    adamw_common = dict(
        learning_rate=1.3e-3,
        warmup=0.0,
        min_lr_ratio=0.0,
        lr_schedule="linear",
        decay=1.0,
        beta1=0.9,
        beta2=0.95,
        epsilon=1e-8,
        max_grad_norm=1.0,
    )

    return [
        Variant(
            name="adamw_standard_decay",
            optimizer=AdamConfig(**adamw_common, weight_decay=0.01),
            tags=("adamw", "standard-decay", "weight-decay-ablation"),
        ),
        Variant(
            name="adamw_reduced_decay",
            optimizer=AdamConfig(**adamw_common, weight_decay=0.002),
            tags=("adamw", "reduced-decay-0.2x", "weight-decay-ablation"),
        ),
        Variant(
            name="muon_standard_decay",
            optimizer=MuonConfig(**muon_common, weight_decay=0.01),
            tags=("muon", "standard-decay", "weight-decay-ablation"),
        ),
        Variant(
            name="muon_reduced_decay",
            optimizer=MuonConfig(**muon_common, weight_decay=0.002),
            tags=("muon", "reduced-decay-0.2x", "weight-decay-ablation"),
        ),
        Variant(
            name="namo_adaptive_decay",
            optimizer=NamoConfig(**namo_common),
            tags=("namo", "adaptive-decay", "weight-decay-ablation"),
        ),
        Variant(
            name="namo_adamw_matrix_decay",
            optimizer=NamoAdamWMatrixDecayConfig(**namo_common),
            tags=("namo", "adamw-matrix-decay", "weight-decay-ablation"),
        ),
        Variant(
            name="namod_adaptive_decay",
            optimizer=NamoDConfig(**namo_common),
            tags=("namod", "adaptive-decay", "weight-decay-ablation"),
        ),
        Variant(
            name="namod_adamw_matrix_decay",
            optimizer=NamoDAdamWMatrixDecayConfig(**namo_common),
            tags=("namod", "adamw-matrix-decay", "weight-decay-ablation"),
        ),
    ]


def _selected_model_families() -> list[ModelFamily]:
    raw = os.environ.get(MODEL_FAMILIES_ENV, os.environ.get("EXP3037_MODEL_FAMILY", "gpt2,llama"))
    requested = [part.strip().lower() for part in raw.split(",") if part.strip()]
    valid = {"gpt2", "llama"}

    invalid = [name for name in requested if name not in valid]
    if invalid:
        raise ValueError(f"Unsupported {MODEL_FAMILIES_ENV} entries: {invalid}; valid={sorted(valid)}")

    if not requested:
        return ["gpt2", "llama"]

    # preserve order while de-duplicating
    return list(dict.fromkeys(requested))  # type: ignore[return-value]


def build_experiment_steps() -> list[ExecutorStep]:
    data_config = _build_data_config()
    selected = {name.strip() for name in os.environ.get(VARIANTS_ENV, "").split(",") if name.strip()}

    steps: list[ExecutorStep] = []
    for model_family in _selected_model_families():
        model_config = _build_model_config(model_family)
        family_tag = f"model-{model_family}"
        for variant in _variants(model_family):
            if selected and variant.name not in selected:
                continue
            train_cfg = _common_train_config(variant.optimizer)
            step = _build_train_step(
                name=f"exp3037/namo_decay_ablation/{model_family}/{variant.name}",
                data_config=data_config,
                model_config=model_config,
                train_config=train_cfg,
                tags=["experiment", "namo-weight-decay", "issue-3037", family_tag, *variant.tags],
            )
            steps.append(step)

    return steps


if __name__ == "__main__":
    executor_main(
        steps=build_experiment_steps(),
        description="Exp3037: NAMO/NAMO-D weight-decay ablation for issue #3037",
    )
