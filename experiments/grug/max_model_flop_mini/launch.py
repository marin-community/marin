# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Max-features MoE trial run.

Results: eval/paloma/c4_en/bpb: 1.1136 @ 5000 steps, ~9.14e17 model FLOPs.

Enhancements on top of experiments/grug/moe:

Architecture:
- QK-norm: non-parametric RMS norm on Q/K before RoPE.
- Partial RoPE: only rotates first 50% of head_dim (partial_rotary_factor=0.5).
- Parameter-free RMSNorm: no learnable scale weight.
- Embed norm: RMSNorm applied immediately after token embedding lookup.
- Per-head attention gate: learned sigmoid gate on each attention head output.
- Value embeddings (VE): auxiliary vocabulary embedding mixed into V on the
  last num_ve_layers layers via learnable lambda/gate parameters.
- Residual stream mixing (x0): per-layer learnable interpolation between the
  current hidden state and the original post-embed-norm state.
- Sliding window attention: alternating short/long causal windows across layers
  (long every 4th layer, short = sliding_window // 2).
- Zero-init output projections: lm_head, attn w_o, dense MLP w_down, and MoE
  w_down all initialized to zeros.
- Load-balancing loss and router z-loss (configurable, None disables each).

Training / config:
- 16 experts (vs 8 in base), top-2 routing.
- Configurable ep_capacity_factor for expert parallelism.
- GrugMuonConfig (Muon) replaces AdamConfig, with 3D expert-weight support
  (Newton-Schulz vmapped over the expert dim).
- Full Levanter TrainerConfig with checkpointing, profiler, mixed-precision
  policy, and WandB tracking.
- Runs on Nemotron data mix with default validation sets.
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
from levanter.optim import OptimizerConfig
from levanter.optim import GrugMuonConfig
from levanter.tracker import TrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize import add_validation_sets_to_mixture

from experiments.defaults import default_validation_sets
from experiments.grug.moe_test.model import GrugModelConfig
from experiments.grug.moe_test.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug
from experiments.tootsie.exp1295_32b import nemotron_mix


@dataclass(frozen=True)
class GrugMoeLaunchConfig:
    """Last-mile run config for the MoE grug template.

    Keep this as the main entry point for day-to-day edits (model/data/optimizer/trainer/eval knobs).
    """

    model: GrugModelConfig
    data: LmDataConfig
    output_path: str
    run_id: str
    steps: int
    batch_size: int
    seed: int
    mp: str  # jmp policy string, e.g. "params=float32,compute=bfloat16,output=bfloat16".
    tracker: TrackerConfig
    optimizer: OptimizerConfig
    grug_trainer: GrugTrainerConfig = field(default_factory=GrugTrainerConfig)
    eval: GrugEvalConfig | None = field(default_factory=GrugEvalConfig)


GRUG_MOE_TRIAL_MODEL = GrugModelConfig(
    vocab_size=128_256,
    hidden_dim=512,
    intermediate_dim=512 * 2,
    shared_expert_intermediate_dim=0,
    num_experts=16,
    num_experts_per_token=2,
    num_layers=8,
    num_heads=4,
    num_kv_heads=4,
    max_seq_len=2048,
    head_dim=None,
    initializer_std=0.02,
    lbl_coef=0.01,
    rzl_coef=0.001,
    num_ve_layers=2,
    sliding_window=2048,
    rope_theta=1024,
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


def _resolve_tracker(tracker: TrackerConfig, run_id: str) -> TrackerConfig:
    if isinstance(tracker, WandbConfig):
        return dataclasses.replace(tracker, name=run_id)
    return tracker


def run_grug_moe_trial(config: GrugMoeLaunchConfig) -> None:
    # Map template launch knobs onto full Levanter TrainerConfig.
    trainer = TrainerConfig(
        mesh=MeshConfig(axes={"data": -1, "expert": 1, "model": 1}),
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
            append_run_id_to_base_path=False,
            save_interval=timedelta(minutes=10),
            keep=[{"every": 1000}],
        ),
    )

    grug_trainer = dataclasses.replace(config.grug_trainer, trainer=trainer)

    run_config = GrugRunConfig(
        model=config.model,
        data=config.data,
        optimizer=config.optimizer,
        trainer=grug_trainer,
        eval=config.eval,
    )
    run_grug(run_config)


RESOLVED_RUN_ID = _resolve_run_id("moe_feat_max_mini")


grug_moe_trial = ExecutorStep(
    name="grug/moe_feat_max_mini",
    fn=run_grug_moe_trial,
    config=GrugMoeLaunchConfig(
        model=versioned(GRUG_MOE_TRIAL_MODEL),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        # this_output_path() resolves to this step's output root (e.g. gs://.../grug/moe-trial-<version>).
        output_path=this_output_path(),
        # Keep run id out of versioning so changing job metadata doesn't create a new output path.
        run_id=RESOLVED_RUN_ID,
        steps=versioned(5000),
        batch_size=versioned(128),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="dial_moe",
            tags=["grug", "template", "moe"],
            group="moe_feat_max",
            name=None,  # filled from run_id in _resolve_tracker
        ),
        optimizer=versioned(
            GrugMuonConfig(
                learning_rate=0.02,
                adam_lr=0.0064,
                weight_decay=0,
                min_lr_ratio=0.1,
                warmup=0,
                momentum=0.95,
                beta1=0.8,
                beta2=0.95,
                epsilon=1e-15,
                muon_epsilon=1e-5,
                max_grad_norm=1,
                lr_schedule="linear",
                decay=0.5,
            )
        ),
        grug_trainer=versioned(
            GrugTrainerConfig(
                z_loss_weight=0,
                ema_beta=None,
                log_every=1,
            )
        ),
    ),
    resources=ResourceConfig.with_tpu("v4-8"),
)


if __name__ == "__main__":
    executor_main(
        steps=[grug_moe_trial],
        description="Applying max features to small scale MoE on Nemotron mix.",
    )
