# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""EKN scaling experiment: expert-count sweep at Nano scale on v4-8.

Nano config (from scaling law):
    hidden_dim=512, intermediate_dim=512, shared_expert_intermediate_dim=512,
    num_experts_per_token=2, num_layers=8, num_heads=8, num_kv_heads=2,
    max_seq_len=4096, vocab_size=128256, initializer_std=0.006,
    load_balancing_loss_coef=0.01, router_z_loss_coef=0.001,
    AdamW (lr=1.68e-3, b1=0.9, b2=0.95, wd=0.1), linear LR with 10% decay,
    batch_size=96, ~16,810 steps, compute C=3.61e18, TPU v4-8.
"""

import dataclasses
import os
from dataclasses import dataclass, field
from datetime import timedelta

import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import LmDataConfig
from levanter.optim import AdamConfig, OptimizerConfig
from levanter.tracker import TrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize import add_validation_sets_to_mixture

from experiments.defaults import default_validation_sets
from experiments.grug.ekn_scaling.model import GrugModelConfig
from experiments.grug.ekn_scaling.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug
from experiments.tootsie.exp1295_32b import nemotron_mix

# Base active width = K * intermediate_dim = 2 * 512 = 1024.
# When scaling K, we reduce intermediate_dim to keep active width constant.
BASE_K = 2
BASE_INTERMEDIATE_DIM = 512
SWEEP_K = [2, 4, 8]
SWEEP_E = [8, 32, 128]
SWEEP_LBL = [0.0025, 0.01, 0.04]


@dataclass(frozen=True)
class EknScalingLaunchConfig:
    """Launch config for the EKN scaling experiment."""

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
    description: str = ""
    grug_trainer: GrugTrainerConfig = field(default_factory=GrugTrainerConfig)
    eval: GrugEvalConfig | None = field(default_factory=GrugEvalConfig)


EKN_NANO_MODEL = GrugModelConfig(
    vocab_size=128_256,
    hidden_dim=512,
    intermediate_dim=512,
    shared_expert_intermediate_dim=512,
    num_experts=8,
    num_experts_per_token=2,
    num_layers=8,
    num_heads=8,
    num_kv_heads=2,
    max_seq_len=4096,
    initializer_std=0.006,
    load_balancing_loss_coef=0.01,
    router_z_loss_coef=0.001,
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


def run_ekn_scaling_trial(config: EknScalingLaunchConfig) -> None:
    """Run a single EKN scaling training trial."""
    trainer = TrainerConfig(
        mesh=MeshConfig(axes={"data": -1, "expert": 1, "model": 1}),
        id=config.run_id,
        seed=config.seed,
        train_batch_size=config.batch_size,
        num_train_steps=config.steps,
        profiler=ProfilerConfig(enabled=False, start_step=5, num_steps=100, perfetto_link=False),
        mp=jmp.get_policy(config.mp),
        tracker=_resolve_tracker(config.tracker, config.run_id, config.output_path),
        use_explicit_mesh_axes=True,
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
        description=config.description,
        optimizer=config.optimizer,
        trainer=grug_trainer,
        eval=config.eval,
    )
    run_grug(run_config)


# ---------------------------------------------------------------------------
# Build one training step per (K, E, lbl) combination
# ---------------------------------------------------------------------------

RESOLVED_RUN_ID = _resolve_run_id("ekn-scaling-nano-run1-sweep")

training_steps: list[ExecutorStep] = []
for _k in SWEEP_K:
    _intermediate = BASE_INTERMEDIATE_DIM * BASE_K // _k
    for _e in SWEEP_E:
        for _lbl in SWEEP_LBL:
            _model = dataclasses.replace(
                EKN_NANO_MODEL,
                num_experts=_e,
                num_experts_per_token=_k,
                intermediate_dim=_intermediate,
                shared_expert_intermediate_dim=_intermediate,
                load_balancing_loss_coef=_lbl,
            )
            _lbl_tag = f"lbl{_lbl}".replace(".", "p")
            _run_id = f"{RESOLVED_RUN_ID}-k{_k}-e{_e}-{_lbl_tag}"

            _step = ExecutorStep(
                name=f"grug/ekn-scaling-nano-run1-k{_k}-e{_e}-{_lbl_tag}",
                fn=run_ekn_scaling_trial,
                config=EknScalingLaunchConfig(
                    model=versioned(_model),
                    data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
                    output_path=this_output_path(),
                    run_id=_run_id,
                    resources=versioned(ResourceConfig.with_tpu("v4-8")),
                    steps=versioned(16_810),
                    batch_size=versioned(96),
                    seed=versioned(0),
                    mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
                    description=(
                        f"EKN scaling Nano: K={_k}, E={_e}, d_expert={_intermediate},"
                        f" lbl={_lbl}, aux loss averaged over layers."
                    ),
                    tracker=WandbConfig(
                        project="dial_moe",
                        tags=["grug", "ekn_scaling", "nano", "moe", f"k{_k}", f"e{_e}", _lbl_tag],
                        group="ekn-scaling-nano-sweep",
                        name=None,
                    ),
                    optimizer=versioned(
                        AdamConfig(
                            learning_rate=1.68e-3,
                            weight_decay=0.1,
                            beta1=0.9,
                            beta2=0.95,
                            lr_schedule="linear",
                            decay=0.1,
                            min_lr_ratio=0.1,
                            warmup=0.01,
                        )
                    ),
                    grug_trainer=versioned(
                        GrugTrainerConfig(
                            z_loss_weight=0,
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
            training_steps.append(_step)


if __name__ == "__main__":
    executor_main(
        steps=training_steps,
        description="EKN scaling Nano: expert-count sweep (aux loss averaged over layers).",
    )
