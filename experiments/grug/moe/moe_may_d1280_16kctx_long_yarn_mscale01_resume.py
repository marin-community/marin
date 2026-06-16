# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resume d=1280 EP=1 from step 13000 with long-only YaRN (coef=0.1) at 4x context.

Two arms, sharing the same source checkpoint and YaRN/qk_mult settings, differing
only in expert parallelism:

  1. ``_ep1`` -- same EP=1 as training
  2. ``_ep2`` -- swap to EP=2 at load time (params reshard across the expert axis)

Both: seq 4096 -> 16384, sliding_window stays at the trained 2048, bs 256 -> 64
(tokens-per-step preserved at 1_048_576, so MuonH yields identical peak LR /
beta / epsilon and the cosine schedule continues exactly). Long-window-only YaRN
(NTK-by-parts on inv_freq) is applied on the every-4th + last layers, with the
paper m-scale coefficient 0.1 giving ``long_qk_mult = 1.3 * (0.1*ln(4) + 1) = 1.4802``.
Short layers stay vanilla.

Submit (us-central2, v4-32, production priority)::

    .venv/bin/iris --cluster=marin job run --no-wait --region us-central2 \\
        --priority production \\
        -e WANDB_API_KEY "$WANDB_API_KEY" \\
        -- python -m experiments.grug.moe.moe_may_d1280_16kctx_long_yarn_mscale01_resume
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

from experiments.grug.moe.heuristic_muonh import MoeMuonHHeuristic
from experiments.grug.moe.launch import NEMOTRON_MIX_WITH_DEFAULT_VALIDATION
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug

_SOURCE_CKPT_PATH: str = "gs://marin-us-central2/grug/moe_may_compute_opt_d1280_ep1-b9a7ad/checkpoints/step-13000"
_DIM: int = 1280
_ORIG_BS: int = 256
_ORIG_SEQ: int = 4096
_NEW_BS: int = 64
_NEW_SEQ: int = 16_384
_YARN_OLD_SEQ_LEN: int = 4096
_YARN_ALPHA: int = 1
_YARN_BETA: int = 32
_YARN_MSCALE_COEF: float = 0.1
_TOTAL_STEPS: int = 14_325


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
# build_model_config gives sw=seq//2=8192 at seq=16k; the d=1280 baseline trained
# at sw=2048, which is what we keep on the short layers.
_base_model_cfg = dataclasses.replace(
    _heuristic.build_model_config(_DIM, seq_len=_NEW_SEQ),
    sliding_window=2048,
)
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


def _build_arm(expert_parallel: int) -> ExecutorStep:
    run_id = f"moe_may_compute_opt_d{_DIM}_ep{expert_parallel}_16kctx_long_yarn_mscale01_from13k"
    return ExecutorStep(
        name=f"grug/{run_id}",
        fn=run_grug_moe_resume,
        config=GrugMoeResumeLaunchConfig(
            model=versioned(_model_cfg),
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
                tags=[
                    "moe",
                    "moe_may_compute_opt",
                    f"d{_DIM}",
                    f"ep{expert_parallel}",
                    "16kctx_long_yarn_mscale01_resume",
                ],
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
                    eval_batch_size=128,
                    steps_per_eval=1000,
                    max_eval_batches=8,
                    eval_current=True,
                    eval_ema=False,
                )
            ),
            expert_parallel=expert_parallel,
        ),
    )


resume_steps: list[ExecutorStep] = [_build_arm(1), _build_arm(2)]


if __name__ == "__main__":
    executor_main(
        steps=resume_steps,
        description=(
            f"Resume d={_DIM} from step 13000 (EP=1 trained checkpoint) with "
            f"seq={_NEW_SEQ}, sliding_window=2048, bs={_NEW_BS}; long-only YaRN "
            f"(old={_YARN_OLD_SEQ_LEN}, alpha={_YARN_ALPHA}, beta={_YARN_BETA}, "
            f"mscale_coef={_YARN_MSCALE_COEF}, long_qk_mult={_model_cfg.long_qk_mult:.4f}). "
            f"Two arms: EP=1 and EP=2. v4-32 us-central2."
        ),
    )
