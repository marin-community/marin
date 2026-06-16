# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resume d=768 EP=1 from step 15000 with 4x context length, rope changes on long layers only.

3-arm ablation. Sliding window stays at the trained 2048 (so short layers are
identical to training). Only the long-window layers (every-4th + last,
``use_pko=True``) get the context-extension rope adjustment:

  1. ``_long_theta40k``       long-only RoPE theta 10k -> 40k (uniform NTK)
  2. ``_long_yarn``            long-only YaRN NTK-by-parts + m-scale (coef=0.2)
  3. ``_long_yarn_mscale01``   long-only YaRN NTK-by-parts + paper m-scale (coef=0.1)

All arms: bs 64 -> 16, tokens-per-step preserved at 64*4096 = 16*16384 = 262144,
identical peak LR / beta2 / epsilon, cosine schedule continues from step 15000.

Submit (us-central2, v4-32, production priority)::

    .venv/bin/iris --cluster=marin job run --no-wait --region us-central2 \\
        --priority production \\
        -e WANDB_API_KEY "$WANDB_API_KEY" \\
        -- python -m experiments.grug.moe.moe_may_d768_ep1_16kctx_long_only_ablation
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
from levanter.grug.attention import RotaryConfig
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

_SOURCE_CKPT_PATH: str = "gs://marin-us-central2/grug/moe_may_compute_opt_d768_ep1-579754/checkpoints/step-15000"
_DIM: int = 768
_ORIG_BS: int = 64
_ORIG_SEQ: int = 4096
_NEW_BS: int = 16
_NEW_SEQ: int = 16_384
_BASE_SW: int = 2048  # trained sliding_window; left unchanged on short layers
_YARN_OLD_SEQ_LEN: int = 4096
_YARN_ALPHA: int = 1
_YARN_BETA: int = 32
_TOTAL_STEPS: int = 16_875


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
_base_model_cfg = dataclasses.replace(
    _heuristic.build_model_config(_DIM, seq_len=_NEW_SEQ),
    sliding_window=_BASE_SW,
)
_orig_tokens = float(_TOTAL_STEPS * _ORIG_BS * _ORIG_SEQ)
_optimizer = _heuristic.build_muonh_config(_NEW_BS, _orig_tokens, _DIM, seq_len=_NEW_SEQ)


@dataclass(frozen=True)
class _Arm:
    tag: str
    long_rope_theta: float | None  # None = keep base 10k
    long_yarn_old_seq_len: int | None
    long_mscale_coef: float


_ARMS: tuple[_Arm, ...] = (
    _Arm("long_theta40k", 40_000.0, None, 0.0),
    _Arm("long_yarn", None, _YARN_OLD_SEQ_LEN, 0.2),
    _Arm("long_yarn_mscale01", None, _YARN_OLD_SEQ_LEN, 0.1),
)


def _build_arm_step(arm: _Arm) -> ExecutorStep:
    long_mscale = arm.long_mscale_coef * math.log(_NEW_SEQ / _ORIG_SEQ) + 1.0
    long_rope = RotaryConfig(theta=arm.long_rope_theta) if arm.long_rope_theta is not None else None
    long_qk_mult = _base_model_cfg.qk_mult * long_mscale if long_mscale != 1.0 else None
    model_cfg = dataclasses.replace(
        _base_model_cfg,
        long_rope=long_rope,
        long_yarn_old_seq_len=arm.long_yarn_old_seq_len,
        long_qk_mult=long_qk_mult,
        yarn_alpha=_YARN_ALPHA,
        yarn_beta=_YARN_BETA,
    )
    run_id = f"moe_may_compute_opt_d{_DIM}_ep1_16kctx_{arm.tag}_from15k"
    return ExecutorStep(
        name=f"grug/{run_id}",
        fn=run_grug_moe_resume,
        config=GrugMoeResumeLaunchConfig(
            model=versioned(model_cfg),
            data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
            output_path=this_output_path(),
            run_id=run_id,
            resources=versioned(ResourceConfig.with_tpu("v4-32")),
            steps=versioned(_TOTAL_STEPS),
            batch_size=versioned(_NEW_BS),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                project="marin_moe",
                tags=["moe", "moe_may_compute_opt", f"d{_DIM}", f"16kctx_{arm.tag}_resume"],
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
        ),
    )


resume_steps: list[ExecutorStep] = [_build_arm_step(arm) for arm in _ARMS]


if __name__ == "__main__":
    executor_main(
        steps=resume_steps,
        description=(
            f"Resume d={_DIM} EP=1 from step 15000 with seq={_NEW_SEQ}, bs={_NEW_BS}, "
            f"sliding_window={_BASE_SW}; long-layers-only rope ablation "
            f"(theta40k / yarn / yarn_mscale01). v4-32 us-central2."
        ),
    )
