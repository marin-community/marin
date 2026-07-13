# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""grug-base trial run, authored as a lazy artifact.

The run is a function that returns a typed :class:`Checkpoint` handle addressed by an
explicit ``name@version``. The model, optimizer, data mixture, token budget, and evals
are stated inline; the output path is ``ctx.output_path`` and the TPU is a run-arg, so neither
bears on the artifact's identity.

The grug training mechanism (``GrugBaseLaunchConfig`` + ``build_grug_run_config`` ->
``run_grug``) is grug-specific compute and is kept as-is: the recipe's ``fn`` builds the
Levanter trainer (its checkpointer paths derive from ``output_path``) and dispatches the
training job to Fray. Only the data/validation wiring around it is assembled lazily.
"""

import dataclasses
import os
from dataclasses import dataclass
from datetime import timedelta

import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import LmDataConfig
from levanter.optim.config import AdamConfig, OptimizerConfig
from levanter.tracker import TrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.step_runner import StepRunner
from marin.experiment.data import mixture
from marin.experiment.namespacing import user_namespaced_name
from marin.training.training import LevanterCheckpoint, resolve_checkpointer_output_path

from experiments.datasets.nemotron import nemotron_datasets
from experiments.datasets.paloma import paloma_datasets
from experiments.datasets.proofpile import proofpile_dataset
from experiments.datasets.starcoder import starcoder_dataset
from experiments.datasets.uncheatable import uncheatable_datasets
from experiments.grug.base.model import GrugModelConfig
from experiments.grug.base.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug
from experiments.llama import llama3_tokenizer

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


GRUG_130M_MODEL = GrugModelConfig(
    vocab_size=128_256,
    hidden_dim=512,
    intermediate_dim=1792,
    num_layers=6,
    num_heads=8,
    num_kv_heads=8,
    max_seq_len=4096,
    head_dim=None,
)


@dataclass(frozen=True)
class GrugBaseLaunchConfig:
    """Last-mile run config for the base grug template.

    Keep this as the main entry point for day-to-day edits (model/data/optimizer/trainer/eval knobs).

    The trainer and eval knobs are flat scalars rather than nested
    ``GrugTrainerConfig`` / ``GrugEvalConfig`` objects: those carry
    ``jax.sharding.PartitionSpec`` batch-sharding defaults, which the artifact
    fingerprint cannot serialize. ``build_grug_run_config`` reconstitutes them (with
    their default pspecs) at run time, keeping the sharding plumbing out of the
    artifact's identity.
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
    z_loss_weight: float = 1e-4
    ema_beta: float | None = None  # EMA coefficient for eval/checkpoint model; None disables EMA.
    log_every: int = 1
    loss_implementation: str | tuple[str, ...] | None = None  # cross-entropy kernel; None uses the trainer default.
    eval_batch_size: int | None = 512  # None disables perplexity eval.
    steps_per_eval: int = 1000
    max_eval_batches: int = 8
    eval_current: bool = True
    eval_ema: bool = False


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


def build_grug_run_config(launch: GrugBaseLaunchConfig) -> GrugRunConfig:
    """Map launch knobs onto the trainer's full ``GrugRunConfig``.

    The checkpointer's ``base_path`` and ``temporary_base_path`` derive from
    ``launch.output_path``, so a run resolves its checkpoint locations under the
    region its output path lives in.
    """
    output_path = launch.output_path
    trainer = TrainerConfig(
        id=launch.run_id,
        seed=launch.seed,
        train_batch_size=launch.batch_size,
        num_train_steps=launch.steps,
        profiler=ProfilerConfig(enabled=False, start_step=5, num_steps=100, perfetto_link=False),
        mp=jmp.get_policy(launch.mp),
        tracker=_resolve_tracker(launch.tracker, launch.run_id),
        use_explicit_mesh_axes=True,
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        checkpointer=resolve_checkpointer_output_path(
            CheckpointerConfig(save_interval=timedelta(minutes=10), keep=None),
            output_path,
        ),
    )

    grug_trainer = GrugTrainerConfig(
        trainer=trainer,
        z_loss_weight=launch.z_loss_weight,
        ema_beta=launch.ema_beta,
        log_every=launch.log_every,
        loss_implementation=launch.loss_implementation,
    )

    eval_config = (
        GrugEvalConfig(
            eval_batch_size=launch.eval_batch_size,
            steps_per_eval=launch.steps_per_eval,
            max_eval_batches=launch.max_eval_batches,
            eval_current=launch.eval_current,
            eval_ema=launch.eval_ema,
        )
        if launch.eval_batch_size is not None
        else None
    )

    return GrugRunConfig(
        model=launch.model,
        data=launch.data,
        resources=launch.resources,
        output_path=output_path,
        optimizer=launch.optimizer,
        trainer=grug_trainer,
        eval=eval_config,
    )


def run_grug_base_trial(config: GrugBaseLaunchConfig) -> None:
    """Build the full grug run config and dispatch the training job to Fray.

    Runs inline on the launcher; ``run_grug`` submits the job and blocks until it
    completes.
    """
    run_grug(build_grug_run_config(config))


def grug_base_trial(*, version: str = "dev") -> ArtifactStep[LevanterCheckpoint]:
    """The base grug trial on the Nemotron mix as a lazy checkpoint.

    Every component is a :class:`Dataset` handle, so the whole graph lowers via
    :func:`~marin.execution.lazy.lower`. The paloma/uncheatable suites are validation
    (weight 0).
    """
    nem = nemotron_datasets(tokenizer=llama3_tokenizer)
    train = {nem[split]: weight for split, weight in _NEMOTRON_WEIGHTS.items()}
    train[starcoder_dataset(tokenizer=llama3_tokenizer)] = _STARCODER_WEIGHT
    train[proofpile_dataset(tokenizer=llama3_tokenizer)] = _PROOFPILE_WEIGHT
    validation = [
        *paloma_datasets(tokenizer=llama3_tokenizer).values(),
        *uncheatable_datasets(tokenizer=llama3_tokenizer).values(),
    ]

    run_id = _resolve_run_id("grug-base-trial")

    def build_config(ctx: StepContext) -> GrugBaseLaunchConfig:
        return GrugBaseLaunchConfig(
            model=GRUG_130M_MODEL,
            data=mixture(ctx, train, validation=validation),
            output_path=ctx.output_path,
            run_id=run_id,
            resources=ctx.runtime_arg("train_resources"),
            steps=2_000,
            batch_size=512,
            seed=0,
            mp="params=float32,compute=bfloat16,output=bfloat16",
            tracker=WandbConfig(
                project="marin",
                tags=["grug", "template"],
                group="grug-base-trial",
                name=None,  # filled from run_id in _resolve_tracker
                replicate_path=ctx.output_path,
            ),
            optimizer=AdamConfig(
                learning_rate=3e-3,
                weight_decay=0.1,
                lr_schedule="cosine",
                decay=0.2,
                min_lr_ratio=0.1,
                warmup=1000,
            ),
            z_loss_weight=1e-4,
            ema_beta=None,
            log_every=1,
            eval_batch_size=512,
            steps_per_eval=1000,
            max_eval_batches=8,
            eval_current=True,
            eval_ema=False,
        )

    return ArtifactStep(
        name=user_namespaced_name("grug/base-trial", version),
        version=version,
        artifact_type=LevanterCheckpoint,
        run=run_grug_base_trial,
        build_config=build_config,
        deps=(*train, *validation),
        runtime_args={"train_resources": _TRAIN_RESOURCES},
    )


if __name__ == "__main__":
    StepRunner().run([grug_base_trial().lower()])
