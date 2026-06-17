# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resume d=512 EP=1 from step 7000 with 32k context length.

seq 4096 -> 32768, sliding_window stays at 4096, batch_size 32 -> 4. Total
tokens-per-step (bs * seq) is preserved at 131072, so the MuonH heuristic
yields the same peak LR / beta2 / epsilon as the original run, and the
cosine schedule (built for ``num_train_steps=10980``) is picked up where
the checkpoint left off. Only the no-window (every-4th + last) layers see
context past 4k.

Submit (us-central2, v4-32)::

    .venv/bin/iris --cluster=marin job run --no-wait --region us-central2 \\
        -e WANDB_API_KEY "$WANDB_API_KEY" \\
        -- python -m experiments.grug.moe.moe_may_d512_ep1_32kctx_resume
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

_SOURCE_CKPT_PATH: str = "gs://marin-us-central2/grug/moe_may_compute_opt_d512_ep1-05c39b/checkpoints/step-7000"
_DIM: int = 512
_ORIG_BS: int = 32
_ORIG_SEQ: int = 4096
_NEW_BS: int = 4
_NEW_SEQ: int = 32_768
_SLIDING_WINDOW: int = 4096
_TOTAL_STEPS: int = 10_980


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
# build_model_config would set sliding_window = seq_len // 2 = 16384; override to 4096.
_model_cfg = dataclasses.replace(
    _heuristic.build_model_config(_DIM, seq_len=_NEW_SEQ),
    sliding_window=_SLIDING_WINDOW,
)
_orig_tokens = float(_TOTAL_STEPS * _ORIG_BS * _ORIG_SEQ)
_optimizer = _heuristic.build_muonh_config(_NEW_BS, _orig_tokens, _DIM, seq_len=_NEW_SEQ)

_run_id = "moe_may_compute_opt_d512_ep1_32kctx_from7k"
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
            tags=["moe", "moe_may_compute_opt", f"d{_DIM}", "32kctx_resume"],
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


if __name__ == "__main__":
    executor_main(
        steps=[resume_step],
        description=(
            f"Resume d={_DIM} EP=1 from step 7000 with seq={_NEW_SEQ}, "
            f"sliding_window={_SLIDING_WINDOW}, bs={_NEW_BS}. Trains through "
            f"step {_TOTAL_STEPS} (3980 remaining) on v4-32 us-central2."
        ),
    )
