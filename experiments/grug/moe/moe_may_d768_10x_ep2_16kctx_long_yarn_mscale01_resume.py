# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resume d=768 10x EP=2 from step 39000 with 4x context, long-only YaRN (coef=0.1).

10x source-of-truth (``marin-big-run-moe_may_compute_opt_d768_10x``): bs=256,
seq=4096, num_train_steps=42188. Resume from step-39000 (nearest available
checkpoint to 40k; 10x kept every-3000 steps), seq -> 16384, bs -> 64 (tpb
preserved at 1_048_576, so MuonH yields identical peak LR / beta / epsilon
and the cosine schedule continues exactly), sliding_window stays at the
trained 2048.

YaRN NTK-by-parts inv_freq rescaling (old=4k -> new=16k) is applied on the
long-window layers only (every-4th + last). qk_mult on long layers gets the
paper m-scale ``1.3 * (0.1*ln(4) + 1) = 1.4802``. Short layers stay vanilla.

Submit (us-central2, v4-32, production priority)::

    .venv/bin/iris --cluster=marin job run --no-wait --region us-central2 \\
        --priority production \\
        -e WANDB_API_KEY "$WANDB_API_KEY" \\
        -- python -m experiments.grug.moe.moe_may_d768_10x_ep2_16kctx_long_yarn_mscale01_resume
"""

import dataclasses
import math
import os
from dataclasses import dataclass, field
from datetime import timedelta

import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import LmDataConfig
from levanter.optim import OptimizerConfig
from levanter.tracker import TrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned
from marin.training.training import temporary_checkpoint_base_path

from experiments.grug.moe.heuristic_v2 import MoeMuonHHeuristic
from experiments.grug.moe.launch import NEMOTRON_MIX_WITH_DEFAULT_VALIDATION
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug

_SOURCE_CKPT_PATH: str = (
    "gs://marin-us-central2/grug/moe_may_compute_opt/"
    "marin-big-run-moe_may_compute_opt_d768_10x-f809e7/checkpoints/step-39000"
)
_DIM: int = 768
_ORIG_BS: int = 256
_ORIG_SEQ: int = 4096
_NEW_BS: int = 64
_NEW_SEQ: int = 16_384
_YARN_OLD_SEQ_LEN: int = 4096
_YARN_ALPHA: int = 1
_YARN_BETA: int = 32
_YARN_MSCALE_COEF: float = 0.1
_TOTAL_STEPS: int = 42_188
_EXPERT_PARALLEL: int = 2


@dataclass(frozen=True)
class GrugMoeResumeLaunchConfig:
    model: GrugModelConfig
    data: LmDataConfig
    output_path: str
    run_id: str
    resources: ResourceConfig
    steps: int
    batch_size: int
    seed: int
    mp: str
    tracker: TrackerConfig
    optimizer: OptimizerConfig
    load_checkpoint_path: str
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    grug_trainer: GrugTrainerConfig = field(default_factory=GrugTrainerConfig)
    eval: GrugEvalConfig | None = field(default_factory=GrugEvalConfig)
    expert_parallel: int = 1


def _resolve_tracker(tracker: TrackerConfig, run_id: str) -> TrackerConfig:
    if isinstance(tracker, WandbConfig):
        return dataclasses.replace(tracker, name=run_id)
    return tracker


def run_grug_moe_resume(config: GrugMoeResumeLaunchConfig) -> None:
    trainer = TrainerConfig(
        id=config.run_id,
        seed=config.seed,
        train_batch_size=config.batch_size,
        num_train_steps=config.steps,
        profiler=config.profiler,
        mp=jmp.get_policy(config.mp),
        tracker=_resolve_tracker(config.tracker, config.run_id),
        use_explicit_mesh_axes=True,
        mesh=MeshConfig(axes={"expert": config.expert_parallel}),
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        checkpointer=CheckpointerConfig(
            base_path=os.path.join(config.output_path, "checkpoints"),
            temporary_base_path=temporary_checkpoint_base_path(config.output_path),
            append_run_id_to_base_path=False,
            save_interval=timedelta(minutes=10),
            keep=[{"every": 1000}],
        ),
        load_checkpoint=True,
        load_checkpoint_path=config.load_checkpoint_path,
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


_heuristic = MoeMuonHHeuristic()
_base_model_cfg = _heuristic.build_model_config(_DIM, seq_len=_NEW_SEQ)
# Heuristic gives sw=seq//2=8192 at seq=16k; the 10x effective sliding_window
# was 2048 (the heuristic's d=768 baseline value at seq=4096).
_base_model_cfg = dataclasses.replace(_base_model_cfg, sliding_window=2048)

_long_mscale = _YARN_MSCALE_COEF * math.log(_NEW_SEQ / _ORIG_SEQ) + 1.0
_model_cfg = dataclasses.replace(
    _base_model_cfg,
    long_yarn_old_seq_len=_YARN_OLD_SEQ_LEN,
    long_qk_mult=_base_model_cfg.qk_mult * _long_mscale,
    yarn_alpha=_YARN_ALPHA,
    yarn_beta=_YARN_BETA,
)

_orig_tokens = float(_TOTAL_STEPS * _ORIG_BS * _ORIG_SEQ)
_optimizer = _heuristic.build_muonh_config(_NEW_BS, _orig_tokens, _DIM, seq_len=_NEW_SEQ)

_run_id = "moe_may_compute_opt_d768_10x_ep2_16kctx_long_yarn_mscale01_from39k"
resume_step = ExecutorStep(
    name=f"grug/{_run_id}",
    fn=run_grug_moe_resume,
    config=GrugMoeResumeLaunchConfig(
        model=versioned(_model_cfg),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        output_path=this_output_path(),
        run_id=_run_id,
        resources=versioned(ResourceConfig.with_tpu("v4-32")),
        steps=versioned(_TOTAL_STEPS),
        batch_size=versioned(_NEW_BS),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin_moe",
            tags=["moe", "moe_may_compute_opt", f"d{_DIM}", "10x", "16kctx_long_yarn_mscale01_resume"],
            group="moe-may-compute-opt",
            name=None,
        ),
        optimizer=versioned(_optimizer),
        load_checkpoint_path=versioned(_SOURCE_CKPT_PATH),
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
        expert_parallel=_EXPERT_PARALLEL,
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[resume_step],
        description=(
            f"Resume d={_DIM} 10x EP={_EXPERT_PARALLEL} from step 39000 with "
            f"seq={_NEW_SEQ}, sliding_window=2048, bs={_NEW_BS}; long-only YaRN "
            f"(old=4k, alpha=1, beta=32, mscale_coef={_YARN_MSCALE_COEF}, "
            f"long_qk_mult={_model_cfg.long_qk_mult:.4f}). Trains through step "
            f"{_TOTAL_STEPS} (3188 remaining) on v4-32 us-central2."
        ),
    )
