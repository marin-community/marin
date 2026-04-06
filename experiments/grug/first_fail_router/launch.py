# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Template: grug-moe trial run.

This keeps model, train loop, and launch wiring in `experiments/grug/moe_max` so
the MoE variant can be iterated independently from the dense base template.
"""

import dataclasses
import os
from dataclasses import dataclass, field
from datetime import timedelta
from functools import partial

import jax
import jax.numpy as jnp
import jmp
import optax
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import LmDataConfig
from levanter.optim import OptimizerConfig
from levanter.optim import GrugMuonConfig
from levanter.optim.grugmuon import _grug_scale_with_muon
from levanter.tracker import TrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import leaf_key_paths
from levanter.utils.mesh import MeshConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize import add_validation_sets_to_mixture

from experiments.defaults import default_validation_sets
from experiments.grug.no_aux.model import GrugModelConfig
from experiments.grug.no_aux.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug
from experiments.tootsie.exp1295_32b import nemotron_mix


def _cautious_weight_decay(decay: float, cooldown_start: int, total_steps: int) -> optax.GradientTransformation:
    """Cautious weight decay that linearly phases out to zero during cooldown."""

    def init_fn(params):
        del params
        return (jnp.zeros([], jnp.int32),)

    def update_fn(updates, state, params):
        (count,) = state
        # Full decay before cooldown, linearly ramp to zero during cooldown
        scale = jnp.where(
            count < cooldown_start,
            1.0,
            1.0 - jnp.minimum((count - cooldown_start) / (total_steps - cooldown_start), 1.0),
        )

        def apply(u, p):
            mask = (jnp.sign(u) == jnp.sign(p)).astype(u.dtype)
            return u + decay * scale * mask * p
        updates = jax.tree.map(apply, updates, params)
        return updates, (count + 1,)

    return optax.GradientTransformation(init_fn, update_fn)


@OptimizerConfig.register_subclass("no_aux_muon")
@dataclass(frozen=True)
class MoeMaxMuonConfig(GrugMuonConfig):
    """Muon + per-group Adam for moe_max model."""

    expert_lr_mul: float = 0.5

    def build(self, num_train_steps):
        lr_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)
        cfg = self

        def optimizer(learning_rate, adam_lr):
            def _muon_components():
                return [
                    _grug_scale_with_muon(
                        cfg.momentum, cfg.nesterov, cfg.backend_steps,
                        cfg.muon_epsilon, cfg.use_kimi_scaling, cfg.coefficient_type,
                    ),
                ]

            def muon_tx(lr_mul=1.0):
                components = _muon_components()
                if cfg.weight_decay > 0:
                    cooldown_start = int(num_train_steps * (1.0 - cfg.decay))
                    components.append(_cautious_weight_decay(cfg.weight_decay, cooldown_start, num_train_steps))
                components.append(optax.scale(-learning_rate * lr_mul))
                return optax.chain(*components)

            adam = optax.chain(
                optax.scale_by_adam(b1=cfg.beta1, b2=cfg.beta2, eps=cfg.epsilon),
                optax.scale(-adam_lr),
            )

            transforms = {
                "muon": muon_tx(),
                "expert_muon": muon_tx(cfg.expert_lr_mul),
                "adam": adam,
            }

            grouped = optax.multi_transform(transforms, partial(self._create_mask))
            if cfg.max_grad_norm:
                return optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), grouped)
            return grouped

        return optax.inject_hyperparams(optimizer)(learning_rate=lr_schedule, adam_lr=adam_lr_schedule)

    def _create_mask(self, params):
        paths = leaf_key_paths(params)

        def classify(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            p = path_str.lower()
            if "token_embed" in p or "output_proj" in p or "gate" in p or "router" in p:
                return "adam"
            if hasattr(param, "ndim") and param.ndim == 3:
                return "expert_muon"
            if hasattr(param, "ndim") and param.ndim >= 2:
                return "muon"
            return "adam"

        return jax.tree.map(classify, params, paths)


@dataclass(frozen=True)
class GrugMoeLaunchConfig:
    """Last-mile run config for the MoE grug template.

    Keep this as the main entry point for day-to-day edits (model/data/optimizer/trainer/eval knobs).
    """

    model: GrugModelConfig
    data: LmDataConfig
    output_path: str
    run_id: str
    resources: ResourceConfig
    steps: int
    batch_size: int | list
    seed: int
    mp: str  # jmp policy string, e.g. "params=float32,compute=bfloat16,output=bfloat16".
    tracker: TrackerConfig
    optimizer: OptimizerConfig
    grug_trainer: GrugTrainerConfig = field(default_factory=GrugTrainerConfig)
    eval: GrugEvalConfig | None = field(default_factory=GrugEvalConfig)


GRUG_MOE_TRIAL_MODEL = GrugModelConfig(
    vocab_size=128_256,
    hidden_dim=512,
    intermediate_dim=512,
    shared_expert_intermediate_dim=512,
    num_experts=16,
    num_experts_per_token=2,
    num_layers=8,
    num_heads=4,
    num_kv_heads=4,
    max_seq_len=2048,
    head_dim=None,
    initializer_std=0.02,
    load_balancing_loss_coef=0.0,
    router_z_loss_coef=0.001,
    num_ve_layers=2,
    sliding_window=None,
    rope_theta=1024,
    bias_update_rate=0.01,
    num_dense_layers=2,
    dense_intermediate_dim=1536,
)

NEMOTRON_MIX_WITH_DEFAULT_VALIDATION = add_validation_sets_to_mixture(
    nemotron_mix,
    default_validation_sets(tokenizer=nemotron_mix.tokenizer),
)


def _resolve_run_id(default_run_id: str) -> str:
    """Resolve run id and append `FERRY_DATE` when launching from ferry workflows."""
    run_id = os.environ.get("GRUG_RUN_ID", default_run_id)
    ferry_date = os.environ.get("FERRY_DATE")
    if ferry_date:
        run_id = f"{run_id}-{ferry_date}"
    return run_id


def _resolve_tracker(tracker: TrackerConfig, run_id: str, output_path: str) -> TrackerConfig:
    if isinstance(tracker, WandbConfig):
        return dataclasses.replace(tracker, name=run_id, replicate_path=output_path)
    return tracker


def run_grug_moe_trial(config: GrugMoeLaunchConfig) -> None:
    # Map template launch knobs onto full Levanter TrainerConfig.
    trainer = TrainerConfig(
        id=config.run_id,
        seed=config.seed,
        train_batch_size=config.batch_size,
        num_train_steps=config.steps,
        profiler=ProfilerConfig(enabled=False, start_step=5, num_steps=100, perfetto_link=False),
        mp=jmp.get_policy(config.mp),
        tracker=_resolve_tracker(config.tracker, config.run_id, config.output_path),
        use_explicit_mesh_axes=True,
        mesh=MeshConfig(axes={"data": -1, "expert": 1, "model": 1}),
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        checkpointer=CheckpointerConfig(
            base_path=os.path.join(config.output_path, "checkpoints"),
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


RESOLVED_RUN_ID = _resolve_run_id("mar7_no_aux_dense_1536")


grug_moe_trial = ExecutorStep(
    name="grug/mar7_no_aux_dense_1536",
    fn=run_grug_moe_trial,
    config=GrugMoeLaunchConfig(
        model=versioned(GRUG_MOE_TRIAL_MODEL),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        output_path=this_output_path(),
        run_id=RESOLVED_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu("v4-16")),
        steps=versioned(1000),
        batch_size=versioned(128),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="dial_moe",
            tags=["grug", "template", "moe"],
            group="no_aux",
            name=None,
        ),
        optimizer=versioned(
            MoeMaxMuonConfig(
                learning_rate=0.02,
                adam_lr=0.0064,
                weight_decay=0.03,
                min_lr_ratio=0.1,
                warmup=0,
                momentum=0.95,
                beta1=0.8,
                beta2=0.95,
                epsilon=1e-15,
                muon_epsilon=1e-8,
                max_grad_norm=1,
                lr_schedule="linear",
                decay=0.5,
            )
        ),
        grug_trainer=versioned(
            GrugTrainerConfig(
                z_loss_weight=0.0,
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
        steps=[grug_moe_trial],
        description="Aux-loss-free MoE with sigmoid gating.",
    )
