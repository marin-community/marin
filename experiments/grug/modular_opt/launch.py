# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Template: grug-modular-opt trial run.

This variant shows how to use optax.multi_transform to configure multiple
optimizers for different modules. See #3075 for more context.
"""

import dataclasses
import os
from dataclasses import dataclass, field
from datetime import timedelta

import jax
import jmp
import optax
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import LmDataConfig
from levanter.optim import OptimizerConfig
from levanter.tracker import TrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import leaf_key_paths
from marin.execution.dag import ExecutorStep, this_output_path, versioned
from marin.execution.executor import executor_main
from marin.processing.tokenize import add_validation_sets_to_mixture
from marin.training.training import temporary_checkpoint_base_path

from experiments.defaults import default_validation_sets
from experiments.grug.modular_opt.model import GrugModelConfig
from experiments.grug.modular_opt.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug
from experiments.pretraining_datasets import nemotron_mix_block_shuffle


@OptimizerConfig.register_subclass("grug_param_group_adam")
@dataclass(frozen=True)
class GrugParamGroupAdamConfig(OptimizerConfig):
    """AdamW with path-based parameter groups.

    Group routing is string-pattern based, so one rule can apply to many layers
    without listing each layer explicitly.
    """

    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    max_grad_norm: float | None = 1.0

    embed_head_lr_multiplier: float = 0.5
    embed_head_weight_decay_multiplier: float = 0.0
    embed_head_beta1: float = 0.5
    embed_head_beta2: float = 0.95

    special_lr_multiplier: float = 5.0
    special_weight_decay_multiplier: float = 0.0
    special_beta1: float = 0.9
    special_beta2: float = 0.99

    embed_head_patterns: tuple[str, ...] = ("embed", "lm_head")
    special_patterns: tuple[str, ...] = ("scalar", "gate", "lambda")
    no_decay_patterns: tuple[str, ...] = ("norm", "bias")

    def build(self, num_train_steps):
        lr_schedule = self.lr_scheduler(num_train_steps)
        embed_patterns = tuple(pattern.lower() for pattern in self.embed_head_patterns)
        special_patterns = tuple(pattern.lower() for pattern in self.special_patterns)
        no_decay_patterns = tuple(pattern.lower() for pattern in self.no_decay_patterns)

        def _group_transform(
            *,
            learning_rate: float,
            weight_decay_multiplier: float,
            beta1: float,
            beta2: float,
            lr_multiplier: float,
        ) -> optax.GradientTransformation:
            components: list[optax.GradientTransformation] = [
                optax.scale_by_adam(b1=beta1, b2=beta2, eps=self.epsilon),
            ]
            decayed_weight = self.weight_decay * weight_decay_multiplier
            if decayed_weight > 0:
                components.append(optax.add_decayed_weights(decayed_weight))
            components.append(optax.scale(-learning_rate * lr_multiplier))
            return optax.chain(*components)

        def _create_mask(params):
            paths = leaf_key_paths(params)

            def _label_for_path(_, path):
                path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
                path_lower = path_str.lower()
                if any(pattern in path_lower for pattern in no_decay_patterns):
                    return "no_decay"
                if any(pattern in path_lower for pattern in special_patterns):
                    return "special"
                if any(pattern in path_lower for pattern in embed_patterns):
                    return "embed_head"
                return "default"

            return jax.tree.map(_label_for_path, params, paths)

        def _optimizer(base_learning_rate):
            transforms = {
                "default": _group_transform(
                    learning_rate=base_learning_rate,
                    weight_decay_multiplier=1.0,
                    beta1=self.beta1,
                    beta2=self.beta2,
                    lr_multiplier=1.0,
                ),
                "no_decay": _group_transform(
                    learning_rate=base_learning_rate,
                    weight_decay_multiplier=0.0,
                    beta1=self.beta1,
                    beta2=self.beta2,
                    lr_multiplier=1.0,
                ),
                "embed_head": _group_transform(
                    learning_rate=base_learning_rate,
                    weight_decay_multiplier=self.embed_head_weight_decay_multiplier,
                    beta1=self.embed_head_beta1,
                    beta2=self.embed_head_beta2,
                    lr_multiplier=self.embed_head_lr_multiplier,
                ),
                "special": _group_transform(
                    learning_rate=base_learning_rate,
                    weight_decay_multiplier=self.special_weight_decay_multiplier,
                    beta1=self.special_beta1,
                    beta2=self.special_beta2,
                    lr_multiplier=self.special_lr_multiplier,
                ),
            }
            grouped = optax.multi_transform(transforms, _create_mask)
            if self.max_grad_norm is None:
                return grouped
            return optax.chain(optax.clip_by_global_norm(self.max_grad_norm), grouped)

        return optax.inject_hyperparams(_optimizer)(base_learning_rate=lr_schedule)


@dataclass(frozen=True)
class GrugModularOptLaunchConfig:
    """Last-mile run config for the modular-opt grug variant."""

    model: GrugModelConfig
    data: LmDataConfig
    output_path: str
    run_id: str
    resources: ResourceConfig
    steps: int
    batch_size: int
    seed: int
    mp: str  # jmp policy string, e.g. "params=float32,compute=bfloat16,output=bfloat16".
    tracker: TrackerConfig
    optimizer: OptimizerConfig
    grug_trainer: GrugTrainerConfig = field(default_factory=GrugTrainerConfig)
    eval: GrugEvalConfig | None = field(default_factory=GrugEvalConfig)


GRUG_130M_MODEL = GrugModelConfig(
    vocab_size=128_256,
    hidden_dim=512,
    intermediate_dim=1792,
    num_layers=6,
    num_heads=8,
    num_kv_heads=8,
    max_seq_len=4096,
    head_dim=None,
)

NEMOTRON_MIX_WITH_DEFAULT_VALIDATION = add_validation_sets_to_mixture(
    nemotron_mix_block_shuffle,
    default_validation_sets(tokenizer=nemotron_mix_block_shuffle.tokenizer),
)


def _resolve_run_id(default_run_id: str) -> str:
    """Resolve run id and append `FERRY_DATE` when launching from ferry workflows."""
    run_id = os.environ.get("GRUG_RUN_ID", default_run_id)
    ferry_date = os.environ.get("FERRY_DATE")
    if ferry_date:
        run_id = f"{run_id}-{ferry_date}"
    return run_id


def _resolve_tracker(tracker: TrackerConfig, run_id: str) -> TrackerConfig:
    if isinstance(tracker, WandbConfig):
        return dataclasses.replace(tracker, name=run_id)
    return tracker


def run_grug_modular_opt_trial(config: GrugModularOptLaunchConfig) -> None:
    # Map template launch knobs onto full Levanter TrainerConfig.
    trainer = TrainerConfig(
        id=config.run_id,
        seed=config.seed,
        train_batch_size=config.batch_size,
        num_train_steps=config.steps,
        profiler=ProfilerConfig(enabled=False, start_step=5, num_steps=100, perfetto_link=False),
        mp=jmp.get_policy(config.mp),
        tracker=_resolve_tracker(config.tracker, config.run_id),
        use_explicit_mesh_axes=True,
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        checkpointer=CheckpointerConfig(
            base_path=os.path.join(config.output_path, "checkpoints"),
            temporary_base_path=temporary_checkpoint_base_path(config.output_path),
            append_run_id_to_base_path=False,
            save_interval=timedelta(minutes=10),
            keep=[{"every": 1000}],
        ),
    )

    grug_trainer = dataclasses.replace(config.grug_trainer, trainer=trainer)

    run_config = GrugRunConfig(
        model=config.model,
        data=config.data,
        resources=config.resources,
        optimizer=config.optimizer,
        trainer=grug_trainer,
        eval=config.eval,
    )
    run_grug(run_config)


RESOLVED_RUN_ID = _resolve_run_id("grug-modular-opt-trial")


grug_modular_opt_trial = ExecutorStep(
    name="grug/modular-opt-trial",
    fn=run_grug_modular_opt_trial,
    config=GrugModularOptLaunchConfig(
        model=versioned(GRUG_130M_MODEL),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        # this_output_path() resolves to this step's output root (e.g. gs://.../grug/modular-opt-trial-<version>).
        output_path=this_output_path(),
        # Keep run id out of versioning so changing job metadata doesn't create a new output path.
        run_id=RESOLVED_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu("v4-8")),
        steps=versioned(2_000),
        batch_size=versioned(512),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin",
            tags=["grug", "template", "modular_opt", "issue-3075"],
            group="grug-modular-opt-trial",
            name=None,  # filled from run_id in _resolve_tracker
            replicate_path=this_output_path(),
        ),
        optimizer=versioned(
            GrugParamGroupAdamConfig(
                learning_rate=3e-3,
                weight_decay=0.1,
                lr_schedule="cosine",
                decay=0.2,
                min_lr_ratio=0.1,
                warmup=1000,
                embed_head_lr_multiplier=0.5,
                embed_head_weight_decay_multiplier=0.0,
                embed_head_beta1=0.5,
                embed_head_beta2=0.95,
                special_lr_multiplier=5.0,
                special_weight_decay_multiplier=0.0,
                special_beta1=0.9,
                special_beta2=0.99,
                special_patterns=("scalar", "gate", "lambda"),
            )
        ),
        grug_trainer=versioned(
            GrugTrainerConfig(
                z_loss_weight=1e-4,
                ema_beta=None,
                log_every=1,
            )
        ),
        eval=versioned(
            GrugEvalConfig(
                eval_batch_size=512,
                steps_per_eval=1000,
                max_eval_batches=8,
                eval_current=True,
                eval_ema=False,
            )
        ),
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[grug_modular_opt_trial],
        description="Template grug modular-opt 130M trial run (~2000 steps) with parameter-group Adam.",
    )
