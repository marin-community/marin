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
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointerConfig, latest_checkpoint_path
from levanter.data.text import LmDataConfig
from levanter.optim import OptimizerConfig
from levanter.tracker import TrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.step_runner import StepRunner
from marin.experiment.data import mixture, tokenized
from marin.experiment.namespacing import user_namespaced_name
from marin.processing.tokenize.tokenize import TokenizedCache
from marin.training.training import LevanterCheckpoint, temporary_checkpoint_base_path

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
    watch: WatchConfig = field(default_factory=WatchConfig)
    grug_trainer: GrugTrainerConfig = field(default_factory=GrugTrainerConfig)
    eval: GrugEvalConfig | None = field(default_factory=GrugEvalConfig)
    processes_per_task: int = 1
    """GPU processes per task. > 1 fans each node into one JAX process per GPU
    (multi-controller) via the iris.runtime.multigpu supervisor; 1 keeps the
    single-process-per-node model."""
    checkpointer: CheckpointerConfig | None = None
    """Override the checkpointer. None builds the default (periodic + final saves
    under output_path). Throughput experiments point this at node-local disk so a
    slow object-store commit can't wedge the end-of-run barrier."""
    checkpointing_enabled: bool = True
    """False disables checkpoint restore, periodic saves, and final forced saves.
    Use this for disposable throughput probes where checkpoint I/O would dominate
    the measured run tail."""
    init_from: str | None = None
    """Checkpoint base directory to initialize weights from (the latest checkpoint
    under it is loaded). None trains from scratch. Used to chain training phases —
    a midtrain/SFT/RL run points this at the prior phase's ``checkpoints`` directory."""
    log_jaxprs: bool = True
    log_xla_hlo: bool = True


def env_int(key: str, default: int) -> int:
    """Read an int from ``os.environ[key]``, falling back to ``default`` when unset/empty."""
    raw = os.environ.get(key, "")
    return int(raw) if raw else default


def env_bool(key: str, default: bool) -> bool:
    """Read a boolean from ``os.environ[key]``, falling back to ``default`` when unset/empty."""
    raw = os.environ.get(key, "")
    if not raw:
        return default
    normalized = raw.lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{key}={raw!r} must be a boolean")


def validate_local_expert_model_axes(
    *,
    expert_axis: int,
    model_axis: int,
    local_device_count: int,
    env_prefix: str,
) -> None:
    """Validate that expert/model axes fit cleanly inside one worker."""
    if expert_axis <= 0:
        raise ValueError(f"{env_prefix}_EXPERT_AXIS must be positive, got {expert_axis}")
    if model_axis <= 0:
        raise ValueError(f"{env_prefix}_MODEL_AXIS must be positive, got {model_axis}")
    if local_device_count <= 0:
        raise ValueError(f"local_device_count must be positive, got {local_device_count}")

    local_axis_product = expert_axis * model_axis
    if local_device_count % local_axis_product != 0:
        raise ValueError(
            f"{env_prefix}_EXPERT_AXIS * {env_prefix}_MODEL_AXIS must divide the "
            f"{local_device_count} GPUs on each worker so expert/model groups stay local; "
            f"got {expert_axis} * {model_axis} = {local_axis_product}."
        )


def validate_ring_expert_model_axes(
    *,
    expert_axis: int,
    model_axis: int,
    moe_implementation: str | None,
    env_prefix: str,
) -> None:
    """Reject ring EP/model-axis combinations that fail on CW H100s."""
    implementation = moe_implementation or "ring"
    if implementation == "ring" and expert_axis > 1 and model_axis > 1:
        raise ValueError(
            f"{env_prefix}_MOE_IMPLEMENTATION=ring currently requires either "
            f"{env_prefix}_EXPERT_AXIS=1 or {env_prefix}_MODEL_AXIS=1 on CoreWeave H100s; "
            f"got {env_prefix}_EXPERT_AXIS={expert_axis}, {env_prefix}_MODEL_AXIS={model_axis}. "
            "Use model_axis>1 only for attention/model-axis diagnostics with expert_axis=1, "
            "or keep model_axis=1 for ring expert-parallel runs."
        )


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


def _trainer_mesh_for_grug(grug_trainer: GrugTrainerConfig) -> MeshConfig:
    """Build the validation mesh that matches Grug's compact batch sharding."""
    return MeshConfig(
        axes={
            "data": -1,
            "expert": grug_trainer.expert_axis_size,
            "model": grug_trainer.model_axis_size,
        },
        dcn_axes={"replica_dcn": -1},
        compute_mapping={"batch": ["replica_dcn", "data", "expert"]},
    )


class DisabledCheckpointerConfig(CheckpointerConfig):
    """TrainerConfig-compatible checkpoint config that never creates a checkpointer."""

    def create(self, run_id):
        del run_id
        return None


def run_grug_moe_trial(config: GrugMoeLaunchConfig) -> None:
    """Map template launch knobs onto a full Levanter trainer and dispatch the run.

    Runs inline on the launcher; ``run_grug`` submits the training job to Fray and
    blocks until it completes.
    """
    initialize_from = latest_checkpoint_path(config.init_from) if config.init_from is not None else None
    checkpointer = config.checkpointer or CheckpointerConfig(
        base_path=os.path.join(config.output_path, "checkpoints"),
        temporary_base_path=temporary_checkpoint_base_path(config.output_path),
        append_run_id_to_base_path=False,
        save_interval=timedelta(minutes=10),
        keep=None,
    )
    load_checkpoint = None
    if not config.checkpointing_enabled:
        checkpointer = DisabledCheckpointerConfig(base_path="/tmp/grug-disabled-checkpoints")
        load_checkpoint = False

    trainer = TrainerConfig(
        id=config.run_id,
        seed=config.seed,
        train_batch_size=config.batch_size,
        num_train_steps=config.steps,
        profiler=config.profiler,
        watch=config.watch,
        mp=jmp.get_policy(config.mp),
        tracker=_resolve_tracker(config.tracker, config.run_id),
        mesh=_trainer_mesh_for_grug(config.grug_trainer),
        use_explicit_mesh_axes=True,
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        initialize_from=initialize_from,
        load_checkpoint=load_checkpoint,
        checkpointer=checkpointer,
        log_jaxprs=config.log_jaxprs,
        log_xla_hlo=config.log_xla_hlo,
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


def grug_moe_baseline(*, version: str = "dev") -> ArtifactStep[LevanterCheckpoint]:
    """The baseline grug MoE (QB+GN+XSA+zloss) on the Nemotron mix as a lazy checkpoint.

    Every component is a :class:`Dataset` handle, so the whole graph lowers via
    :func:`~marin.execution.lazy.lower`. Pinned components never re-tokenize; the
    paloma/uncheatable suites are validation (weight 0).
    """
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
        name=user_namespaced_name("grug/4_10_baseline_moe", version),
        version=version,
        artifact_type=LevanterCheckpoint,
        run=run_grug_moe_trial,
        build_config=build_config,
        deps=(*train, *validation),
        runtime_args={"train_resources": _TRAIN_RESOURCES},
    )


if __name__ == "__main__":
    StepRunner().run([grug_moe_baseline().lower()])
