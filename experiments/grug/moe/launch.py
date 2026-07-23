# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""grug-moe trial run, authored as a lazy artifact.

The run is a function that returns a typed :class:`Checkpoint` handle addressed by an
explicit ``name@version``. The model, optimizer, data mixture, token budget, z-loss,
and evals are all stated inline; the output path is ``ctx.output_path`` and the TPU is a
run-arg, so neither bears on the artifact's identity.

The grug-moe training mechanism (``GrugMoeLaunchConfig`` + ``run_grug_moe_trial`` ->
``run_grug``) is grug-specific compute and is kept as-is: the recipe's ``fn`` is
``run_grug_moe_trial``, which builds the Levanter trainer and dispatches the training
job to Fray. Only the data/validation wiring around it is assembled lazily from
dataset handles.
"""

import dataclasses
import os
from dataclasses import dataclass, field
from datetime import timedelta

import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.checkpoint import CheckpointerConfig, latest_checkpoint_path
from levanter.data.text.datasets import LmDataConfig
from levanter.optim.config import OptimizerConfig
from levanter.tracker import TrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import experiment_main
from marin.experiment.data import mixture, tokenized
from marin.experiment.namespacing import user_namespaced_name
from marin.processing.tokenize.tokenize import TokenizedCache
from marin.training.training import LevanterCheckpoint, resolve_checkpointer_output_path

from experiments.datasets.nemotron import nemotron_datasets
from experiments.datasets.paloma import paloma_datasets
from experiments.datasets.proofpile import proofpile_dataset
from experiments.datasets.starcoder import starcoder_dataset
from experiments.datasets.uncheatable import uncheatable_datasets
from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug
from experiments.llama import llama3_tokenizer

# SlimPajama-6B tokenization OOMs at the default 10g worker resources.
_SLIMPAJAMA_TOKENIZE_RESOURCES = ResourceConfig(ram="64g", disk="64g")

# The TPU the training job is dispatched onto. A run-arg, not part of the config's
# identity: re-running on a different TPU is the same checkpoint. The launcher step
# runs inline (run_grug dispatches its own Fray job).
_TRAIN_RESOURCES = ResourceConfig.with_tpu("v5p-8")

# Nemotron CC mixture weights: the corpus's TiB proportions, plus starcoder and
# proof-pile at their published weights. Policy lives here, in the experiment.
_NEMOTRON_WEIGHTS = {
    "hq_actual": 0.91351,
    "hq_synth": 2.72,
    "medium_high": 0.82471,
    "medium": 3.38,
    "medium_low": 1.54,
    "low_actual": 0.70123,
    "low_synth": 0.62771,
}
_STARCODER_WEIGHT = 0.25
_PROOFPILE_WEIGHT = 0.055


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
    processes_per_task: int = 1
    """GPU processes per task. > 1 fans each node into one JAX process per GPU
    (multi-controller) via the iris.hooks.multigpu_main supervisor; 1 keeps the
    single-process-per-node model."""
    checkpointer: CheckpointerConfig | None = None
    """Override the checkpointer. None builds the default (periodic + final saves
    under output_path). Throughput experiments point this at node-local disk so a
    slow object-store commit can't wedge the end-of-run barrier."""
    init_from: str | None = None
    """Checkpoint base directory to initialize weights from (the latest checkpoint
    under it is loaded). None trains from scratch. Used to chain training phases —
    a midtrain/SFT/RL run points this at the prior phase's ``checkpoints`` directory."""


def env_int(key: str, default: int) -> int:
    """Read an int from ``os.environ[key]``, falling back to ``default`` when unset/empty."""
    raw = os.environ.get(key, "")
    return int(raw) if raw else default


def slimpajama_6b_dataset() -> ArtifactStep[TokenizedCache]:
    """SlimPajama-6B, llama3-tokenized — a small corpus for GPU smoke/scale runs.

    Returns the tokenized :class:`TokenizedCache` handle; the launcher assembles it into an
    ``LmDataConfig`` with :func:`~marin.experiment.data.mixture`. Tokenization runs as
    its own Fray job (a production pretraining mixture would instead pin an already
    materialized cache to avoid a cross-region tokenize).
    """
    return tokenized(
        "slimpajama-6b",
        source="DKYoon/SlimPajama-6B",
        tokenizer=llama3_tokenizer,
        resources=_SLIMPAJAMA_TOKENIZE_RESOURCES,
        version="2026.06.28",
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
    """Map template launch knobs onto a full Levanter trainer and dispatch the run.

    Runs inline on the launcher; ``run_grug`` submits the training job to Fray and
    blocks until it completes.
    """
    initialize_from = latest_checkpoint_path(config.init_from) if config.init_from is not None else None
    trainer = TrainerConfig(
        id=config.run_id,
        seed=config.seed,
        train_batch_size=config.batch_size,
        num_train_steps=config.steps,
        profiler=config.profiler,
        mp=jmp.get_policy(config.mp),
        tracker=_resolve_tracker(config.tracker, config.run_id),
        use_explicit_mesh_axes=True,
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        initialize_from=initialize_from,
        checkpointer=config.checkpointer
        or resolve_checkpointer_output_path(
            CheckpointerConfig(save_interval=timedelta(minutes=10), keep=None),
            config.output_path,
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
        processes_per_task=config.processes_per_task,
    )
    run_grug(run_config)


RESOLVED_RUN_ID = _resolve_run_id("4_10_test_moe")


# Baseline: 1e18 compute budget, d1024. Model + optimizer + batch + steps are
# all derived from `MoeAdamHHeuristic`. To override any of these, swap in
# an explicit `GrugModelConfig` / `GrugMoeAdamHConfig` below.
_BASELINE_BUDGET: float = 1e18
_BASELINE_HIDDEN_DIM: int = 1024
_BASELINE_TARGET_STEPS: int = 2**14
_baseline_model, _baseline_optimizer, _baseline_batch, _baseline_steps = build_from_heuristic(
    budget=_BASELINE_BUDGET,
    hidden_dim=_BASELINE_HIDDEN_DIM,
    target_steps=_BASELINE_TARGET_STEPS,
)

# Public alias for the heuristic-derived baseline GrugModelConfig. Kept
# because consumers (e.g. experiments/ferries/canary_ferry.py) import it by
# name.
GRUG_MOE_TRIAL_MODEL: GrugModelConfig = _baseline_model


def grug_moe_baseline(*, version: str | None = None) -> ArtifactStep[LevanterCheckpoint]:
    """The baseline grug MoE (QB+GN+XSA+zloss) on the Nemotron mix as a lazy checkpoint.

    Every component is a :class:`Dataset` handle, so the whole graph lowers via
    :func:`~marin.execution.lazy.lower`. Pinned components never re-tokenize; the
    paloma/uncheatable suites are validation (weight 0).
    """
    name = "grug/4_10_baseline_moe"
    version = resolve_version(name, version)
    nem = nemotron_datasets(tokenizer=llama3_tokenizer)
    train = {nem[split]: weight for split, weight in _NEMOTRON_WEIGHTS.items()}
    train[starcoder_dataset(tokenizer=llama3_tokenizer)] = _STARCODER_WEIGHT
    train[proofpile_dataset(tokenizer=llama3_tokenizer)] = _PROOFPILE_WEIGHT
    validation = [
        *paloma_datasets(tokenizer=llama3_tokenizer).values(),
        *uncheatable_datasets(tokenizer=llama3_tokenizer).values(),
    ]

    def build_config(ctx: StepContext) -> GrugMoeLaunchConfig:
        return GrugMoeLaunchConfig(
            model=_baseline_model,
            data=mixture(ctx, train, validation=validation),
            output_path=ctx.output_path,
            run_id=RESOLVED_RUN_ID,
            resources=ctx.runtime_arg("train_resources"),
            steps=_baseline_steps,
            batch_size=_baseline_batch,
            seed=0,
            mp="params=float32,compute=bfloat16,output=bfloat16",
            tracker=WandbConfig(project="marin_moe", tags=["moe"], group="moe-iter04", name=None),
            optimizer=_baseline_optimizer,
            grug_trainer=GrugTrainerConfig(z_loss_weight=1e-4, ema_beta=None, log_every=1),
            eval=GrugEvalConfig(
                eval_batch_size=512,
                steps_per_eval=1000,
                max_eval_batches=8,
                eval_current=True,
                eval_ema=False,
            ),
        )

    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=LevanterCheckpoint,
        run=run_grug_moe_trial,
        build_config=build_config,
        deps=(*train, *validation),
        runtime_args={"train_resources": _TRAIN_RESOURCES},
    )


if __name__ == "__main__":
    experiment_main(grug_moe_baseline)()
