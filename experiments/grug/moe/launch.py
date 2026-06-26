# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Template: grug-moe trial run.

This keeps model, train loop, and launch wiring in `experiments/grug/moe` so
the MoE variant can be iterated independently from the dense base template.
"""

import dataclasses
import os
from dataclasses import dataclass, field
from datetime import timedelta

import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import BlockShuffleConfig, LmDataConfig, TextLmDatasetFormat
from levanter.optim import OptimizerConfig
from levanter.tracker import TrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned
from marin.processing.tokenize import add_validation_sets_to_mixture
from marin.processing.tokenize.data_configs import lm_data_config
from marin.training.training import temporary_checkpoint_base_path

from experiments.defaults import default_validation_sets
from experiments.grug.moe.heuristic_muonh import MoeMuonHHeuristic
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug
from experiments.llama import llama3_tokenizer
from experiments.pretraining_datasets import nemotron_mix
from experiments.tokenization import default_tokenize


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
    batch_size: int
    seed: int
    mp: str  # jmp policy string, e.g. "params=float32,compute=bfloat16,output=bfloat16".
    tracker: TrackerConfig
    optimizer: OptimizerConfig
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    grug_trainer: GrugTrainerConfig = field(default_factory=GrugTrainerConfig)
    eval: GrugEvalConfig | None = field(default_factory=GrugEvalConfig)
    # Mesh size along the "expert" axis (expert-parallelism). 1 = no EP.
    expert_parallel: int = 1
    checkpointer: CheckpointerConfig | None = None
    """Override the checkpointer. None builds the default (periodic + final saves
    under output_path). Throughput experiments point this at node-local disk so a
    slow object-store commit can't wedge the end-of-run barrier."""
    checkpoint_keep: list[dict] | None = None
    """When ``checkpointer`` is None, splice this into the default ``CheckpointerConfig.keep``.
    Each entry is a CheckpointInterval dict, e.g. ``[{"every": 10_000}]`` for periodic
    permanent checkpoints, or ``[{"every": phase_1_start_step, "until": phase_1_start_step}]``
    to snapshot a specific transition step. Ignored when ``checkpointer`` is set."""
    save_interval_minutes: int = 10
    """Temp checkpoint cadence (minutes) when ``checkpointer`` is None. Long-running
    production jobs on reserved (non-preemptible) hardware can safely raise this to
    cut GCS write churn; 10 min is appropriate for short jobs that may hit preemption."""
    load_checkpoint_path: str | None = None
    """Resume training from this checkpoint path. The trainer restores model
    weights, optimizer state, and step counter — so the run continues the
    original LR schedule from that step. ``None`` starts fresh from step 0."""


NEMOTRON_MIX_WITH_DEFAULT_VALIDATION = add_validation_sets_to_mixture(
    nemotron_mix,
    default_validation_sets(tokenizer=nemotron_mix.tokenizer),
)


def env_int(key: str, default: int) -> int:
    """Read an int from ``os.environ[key]``, falling back to ``default`` when unset/empty."""
    raw = os.environ.get(key, "")
    return int(raw) if raw else default


def slimpajama_6b_data() -> LmDataConfig:
    """SlimPajama-6B, llama3-tokenized with block-shuffle, re-tokenized on first run.

    A small, R2-local corpus for GPU smoke/scale runs; returns a ready-to-train
    ``LmDataConfig``. A production pretraining mixture would instead need its
    tokenized cache already materialized to avoid a cross-region tokenize.
    """
    tokenize_step = default_tokenize(
        name="slimpajama-6b-cw",
        dataset="DKYoon/SlimPajama-6B",
        tokenizer=llama3_tokenizer,
        format=TextLmDatasetFormat(),
    )
    tokenize_step = dataclasses.replace(
        tokenize_step,
        config=dataclasses.replace(
            tokenize_step.config,
            # SlimPajama-6B tokenization OOMs at the default 10g worker_resources.
            worker_resources=ResourceConfig(ram="64g", disk="64g"),
        ),
    )
    return lm_data_config(
        training_set=tokenize_step,
        shuffle=BlockShuffleConfig(io_block_size=256, window_blocks=256, perm_type="feistel"),
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
        checkpointer=config.checkpointer
        or CheckpointerConfig(
            base_path=os.path.join(config.output_path, "checkpoints"),
            temporary_base_path=temporary_checkpoint_base_path(config.output_path),
            append_run_id_to_base_path=False,
            save_interval=timedelta(minutes=config.save_interval_minutes),
            keep=config.checkpoint_keep,
        ),
        # load_checkpoint=None lets levanter auto-detect existing checkpoints
        # under the configured output_path on restart, so iris preemption
        # restarts resume from the latest temp/permanent checkpoint instead of
        # silently starting over at step 0. An explicit load_checkpoint_path
        # still forces a load from that path.
        load_checkpoint=None,
        load_checkpoint_path=config.load_checkpoint_path,
    )

    grug_trainer = dataclasses.replace(
        config.grug_trainer,
        trainer=trainer,
        expert_axis_size=config.expert_parallel,
    )

    run_config = GrugRunConfig(
        model=config.model,
        data=config.data,
        resources=config.resources,
        optimizer=config.optimizer,
        trainer=grug_trainer,
        eval=config.eval,
    )
    run_grug(run_config)


# May Recipe compute-optimal cells from the drop-1e18 isoflop fit
# (issue #6074). ``MoeMuonHHeuristic`` (heuristic_muonh) supplies LR / beta2 /
# epsilon; (bs, steps) hardcoded so callers don't depend on
# ``compute_tokens_and_batch`` heuristics for cell selection.
#
#   dim   budget     bs    steps   tokens     muonh_lr   tpu
#   512   3.82e17    32    10_980  1.44e9     0.00980    v4-32 (EP=1)
#   768   2.81e18    64    16_875  4.42e9     0.00837    v4-32 (EP=1)
#   1024  1.16e19    128   16_080  8.43e9     0.00879    v4-32 (EP=1)
#   1280  3.46e19    256   14_325  1.50e10    0.00957    v4-32 (EP=1)
_SEQ_LEN: int = 4096
_COMPUTE_OPT_CELLS: tuple[tuple[int, int, int], ...] = (
    (512, 32, 10_980),
    # (768, 64, 16_875),
    # (1024, 128, 16_080),
    # (1280, 256, 14_325),
)

_heuristic = MoeMuonHHeuristic()

# Public alias for the d=512 baseline GrugModelConfig. Kept because
# consumers (e.g. experiments/ferries/canary_ferry.py) import it by name.
GRUG_MOE_TRIAL_MODEL: GrugModelConfig = _heuristic.build_model_config(512, seq_len=_SEQ_LEN)

compute_opt_steps: list[ExecutorStep] = []
for _dim, _bs, _steps in _COMPUTE_OPT_CELLS:
    _model = _heuristic.build_model_config(_dim, seq_len=_SEQ_LEN)
    _tokens = float(_steps * _bs * _SEQ_LEN)
    _optimizer = _heuristic.build_muonh_config(_bs, _tokens, _dim, seq_len=_SEQ_LEN)
    _run_id = f"moe_may_compute_opt_d{_dim}_demo"
    compute_opt_steps.append(
        ExecutorStep(
            name=f"grug/{_run_id}",
            fn=run_grug_moe_trial,
            config=GrugMoeLaunchConfig(
                model=versioned(_model),
                data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
                output_path=this_output_path(),
                run_id=_run_id,
                resources=versioned(ResourceConfig.with_tpu("v4-32")),
                steps=versioned(_steps),
                batch_size=versioned(_bs),
                seed=versioned(0),
                mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
                tracker=WandbConfig(
                    project="marin_moe",
                    tags=["moe", "moe_may_compute_opt", f"d{_dim}"],
                    group="moe-may-compute-opt",
                    name=None,
                ),
                optimizer=versioned(_optimizer),
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
    )


if __name__ == "__main__":
    executor_main(
        steps=compute_opt_steps,
        description="May Recipe compute-optimal cells at d ∈ {512, 768, 1024, 1280} on v4-32 (EP=1).",
    )
