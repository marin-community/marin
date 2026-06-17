# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resume d=768 EP=1 from step 15000 with 4x context length (4096 -> 16384).

Six-arm context-extension ablation mirroring the d=512 sweep:

  1. ``_sw4k``           naive 4x seq, sliding_window doubled 2k -> 4k
  2. ``_sw2k``           naive 4x seq, sliding_window unchanged at 2k
  3. ``_theta40k``       naive 4x seq, RoPE theta 10k -> 40k (uniform NTK)
  4. ``_yarn``           YaRN NTK-by-parts + m-scale (coef=0.2)
  5. ``_yarn_mscale01``  YaRN NTK-by-parts + paper m-scale (coef=0.1)
  6. ``_yarn_nomscale``  YaRN NTK-by-parts only, qk_mult stays at 1.3

All arms: bs 64 -> 16, tokens-per-step preserved at 64*4096 = 16*16384 = 262144,
so MuonH yields identical peak LR / beta2 / epsilon and the cosine schedule
(built for ``num_train_steps=16875``) is picked up where step 15000 left off
(1875 remaining steps).

Submit (us-central2, v4-32)::

    .venv/bin/iris --cluster=marin job run --no-wait --region us-central2 \\
        -e WANDB_API_KEY "$WANDB_API_KEY" \\
        -- python -m experiments.grug.moe.moe_may_d768_ep1_16kctx_ablation
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

from experiments.grug.moe.heuristic_v2 import MoeHeuristicV2
from experiments.grug.moe.launch import NEMOTRON_MIX_WITH_DEFAULT_VALIDATION
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug

_SOURCE_CKPT_PATH: str = "gs://marin-us-central2/grug/moe_may_compute_opt_d768_ep1-579754/checkpoints/step-15000"
_DIM: int = 768
_ORIG_BS: int = 64
_ORIG_SEQ: int = 4096
_NEW_BS: int = 16
_NEW_SEQ: int = 16_384
_BASE_SW: int = 2048  # d=768 baseline sliding_window from MoeHeuristicV2
_EXTENDED_SW: int = 4096
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


_heuristic = MoeHeuristicV2()
_base_model_cfg = _heuristic.build_model_config(_DIM, seq_len=_NEW_SEQ)
_orig_tokens = float(_TOTAL_STEPS * _ORIG_BS * _ORIG_SEQ)
_optimizer = _heuristic.build_muonh_config(_NEW_BS, _orig_tokens, _DIM, seq_len=_NEW_SEQ)


@dataclass(frozen=True)
class _Arm:
    tag: str
    sliding_window: int
    rope_theta: float
    yarn_old_seq_len: int | None
    mscale_coef: float


_ARMS: tuple[_Arm, ...] = (
    _Arm("sw4k", _EXTENDED_SW, 10_000.0, None, 0.0),
    _Arm("sw2k", _BASE_SW, 10_000.0, None, 0.0),
    _Arm("theta40k", _EXTENDED_SW, 40_000.0, None, 0.0),
    _Arm("yarn", _EXTENDED_SW, 10_000.0, _YARN_OLD_SEQ_LEN, 0.2),
    _Arm("yarn_mscale01", _EXTENDED_SW, 10_000.0, _YARN_OLD_SEQ_LEN, 0.1),
    _Arm("yarn_nomscale", _EXTENDED_SW, 10_000.0, _YARN_OLD_SEQ_LEN, 0.0),
)


def _build_arm_step(arm: _Arm) -> ExecutorStep:
    mscale = arm.mscale_coef * math.log(_NEW_SEQ / _ORIG_SEQ) + 1.0
    model_cfg = dataclasses.replace(
        _base_model_cfg,
        sliding_window=arm.sliding_window,
        qk_mult=_base_model_cfg.qk_mult * mscale,
        rope=RotaryConfig(theta=arm.rope_theta),
        yarn_old_seq_len=arm.yarn_old_seq_len,
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
            f"Resume d={_DIM} EP=1 from step 15000 with seq={_NEW_SEQ}, bs={_NEW_BS}; "
            f"6-arm context-extension ablation (sw4k / sw2k / theta40k / yarn / "
            f"yarn_mscale01 / yarn_nomscale). v4-32 us-central2."
        ),
    )
