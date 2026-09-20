# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Base-size replication arms for arXiv 2609.19107 (paper Table 6).

Three arms at the paper's base size (d8, width 1024, 1B tokens, batch 524,288,
GPT-2 tokenizer, FineWeb):

- ``vanilla``: the plain transformer under its own Table 5 recipe
  (paper target: 3.3279).
- ``op1``: Operator-1 (K=1 boundary operator, 2/3/3 split) under its own
  recipe (paper target: 3.3057).
- ``op1-vanilla-recipe``: Operator-1 under the Vanilla recipe — the paper's
  transfer probe (paper target: 3.3135).

Comparing these against Table 6 tests whether the paper's central
base-size claim (Operator-1 beats Vanilla even under the transferred recipe)
replicates outside the paper's stack.

Submit (v4-8, one arm at a time)::

    .venv/bin/iris --cluster=marin job run --no-wait \\
        -e WANDB_API_KEY "$WANDB_API_KEY" \\
        -- python -m experiments.grug.paper_rep.launch \\
            --version dev --run --arm vanilla
"""

import dataclasses
import os
from dataclasses import dataclass, field
from datetime import timedelta

import click
import jmp
from fray.cluster import ResourceConfig
from levanter.checkpoint import CheckpointerConfig, latest_checkpoint_path
from levanter.data.text.datasets import LmDataConfig
from levanter.optim.config import OptimizerConfig
from levanter.tracker import TrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import build_options
from marin.experiment.data import mixture
from marin.experiment.namespacing import user_namespaced_name
from marin.processing.tokenize.tokenize import TokenizedCache
from marin.training.training import LevanterCheckpoint, resolve_checkpointer_output_path

from experiments.grug.paper_rep.data import fineweb_10bt_train, fineweb_validation
from experiments.grug.paper_rep.model import GrugModelConfig
from experiments.grug.paper_rep.recipes import (
    OPERATOR1_RECIPE,
    VANILLA_RECIPE,
    model_config,
    optimizer_config,
)
from experiments.grug.paper_rep.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug

# The TPU the training job is dispatched onto. A run-arg, not part of the
# config's identity: re-running on a different TPU is the same checkpoint.
# v4-16 rather than v4-8: the v4-8 pools were degraded/booting at launch
# time (2026-09-20) while preemptible v4-16 slices were ready. Batch 256
# sequences divides over 16 devices; the global batch and data order are
# unchanged, so the arms stay comparable to each other and to the paper.
_TRAIN_RESOURCES = ResourceConfig.with_tpu("v4-16")

# Paper base-size run geometry: d8 (width 1024), 1B tokens, batch 524,288.
_NUM_LAYERS = 8
_BATCH_SEQS = 256  # 256 sequences x 2048 tokens = 524,288 tokens/step.
_NUM_STEPS = 1907  # 1907 x 524,288 ~= 1.0B tokens.
_MAX_SEQ_LEN = 2048


@dataclass(frozen=True)
class PaperRepLaunchConfig:
    """Last-mile run config for the paper-replication arms."""

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
    grug_trainer: GrugTrainerConfig = field(default_factory=GrugTrainerConfig)
    eval: GrugEvalConfig | None = field(default_factory=GrugEvalConfig)
    checkpointer: CheckpointerConfig | None = None
    """Override the checkpointer. None builds the default (periodic + final saves
    under output_path)."""
    init_from: str | None = None
    """Checkpoint base directory to initialize weights from. None trains from scratch."""


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


def run_paper_rep_trial(config: PaperRepLaunchConfig) -> None:
    """Map launch knobs onto a full Levanter trainer and dispatch the run.

    Runs inline on the launcher; ``run_grug`` submits the training job to Fray
    and blocks until it completes.
    """
    initialize_from = latest_checkpoint_path(config.init_from) if config.init_from is not None else None
    trainer = TrainerConfig(
        id=config.run_id,
        seed=config.seed,
        train_batch_size=config.batch_size,
        num_train_steps=config.steps,
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

    run_config = GrugRunConfig(
        model=config.model,
        data=config.data,
        resources=config.resources,
        optimizer=config.optimizer,
        trainer=dataclasses.replace(config.grug_trainer, trainer=trainer),
        eval=config.eval,
    )
    run_grug(run_config)


def paper_rep_arm(
    *,
    name_suffix: str,
    recipe,
    boundary_operator: bool,
    version: str | None = None,
) -> ArtifactStep[LevanterCheckpoint]:
    """One base-size replication arm as a lazy checkpoint.

    Model + optimizer derive from the recipe (paper Table 5); the boundary
    operator follows the paper Table 2 split (d8: 2/3/3) with alpha = 1.
    """
    name = f"grug/paper_rep_d8_{name_suffix}"
    version = resolve_version(name, version)
    model = model_config(
        recipe,
        num_layers=_NUM_LAYERS,
        vocab_size=50_304,
        max_seq_len=_MAX_SEQ_LEN,
        boundary_operator=boundary_operator,
    )
    optimizer = optimizer_config(recipe)
    train_cache = fineweb_10bt_train()
    validation_cache = fineweb_validation()

    def build_config(ctx: StepContext) -> PaperRepLaunchConfig:
        return PaperRepLaunchConfig(
            model=model,
            data=mixture(ctx, {train_cache: 1.0}, validation=[validation_cache]),
            output_path=ctx.output_path,
            run_id=_resolve_run_id(f"paper_rep_d8_{name_suffix}"),
            resources=ctx.runtime_arg("train_resources"),
            steps=_NUM_STEPS,
            batch_size=_BATCH_SEQS,
            seed=0,
            mp="params=float32,compute=bfloat16,output=bfloat16",
            tracker=WandbConfig(
                project="marin",
                tags=["grug", "paper-rep", "boundary-operator"],
                group="paper-rep-d8",
                name=None,
            ),
            optimizer=optimizer,
            grug_trainer=GrugTrainerConfig(z_loss_weight=0.0, ema_beta=None, log_every=1),
            eval=GrugEvalConfig(
                eval_batch_size=256,
                steps_per_eval=500,
                max_eval_batches=200,
                eval_current=True,
                eval_ema=False,
            ),
        )

    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=LevanterCheckpoint,
        run=run_paper_rep_trial,
        build_config=build_config,
        deps=(train_cache, validation_cache),
        runtime_args={"train_resources": _TRAIN_RESOURCES},
    )


def vanilla_d8(*, version: str | None = None) -> ArtifactStep[LevanterCheckpoint]:
    """Vanilla transformer, own recipe (paper Table 6 target: 3.3279)."""
    return paper_rep_arm(
        name_suffix="vanilla",
        recipe=VANILLA_RECIPE,
        boundary_operator=False,
        version=version,
    )


def operator1_d8(*, version: str | None = None) -> ArtifactStep[LevanterCheckpoint]:
    """Operator-1, own recipe (paper Table 6 target: 3.3057)."""
    return paper_rep_arm(
        name_suffix="op1",
        recipe=OPERATOR1_RECIPE,
        boundary_operator=True,
        version=version,
    )


def operator1_vanilla_recipe_d8(*, version: str | None = None) -> ArtifactStep[LevanterCheckpoint]:
    """Operator-1 under the Vanilla recipe — the transfer probe (target: 3.3135)."""
    return paper_rep_arm(
        name_suffix="op1_vanilla_recipe",
        recipe=VANILLA_RECIPE,
        boundary_operator=True,
        version=version,
    )


_ARMS = {
    "vanilla": vanilla_d8,
    "op1": operator1_d8,
    "op1-vanilla-recipe": operator1_vanilla_recipe_d8,
}


def materialize_data(*, version: str | None = None) -> list[ArtifactStep[TokenizedCache]]:
    """Materialize the FineWeb caches once, before any training arm.

    Returns the two tokenized-cache handles; running this step downloads and
    tokenizes FineWeb (sample/10BT + the held-out validation file). The
    training arms depend on the same handles, so they reuse the caches and
    hold their reserved TPU only for training.
    """
    return [fineweb_10bt_train(), fineweb_validation()]


@click.command()
@click.option(
    "--arm",
    type=click.Choice([*sorted(_ARMS), "data"]),
    required=True,
    help="Which Table 6 arm to run, or 'data' to materialize the FineWeb caches only.",
)
@build_options
def build(arm: str):
    """Build one paper-replication base-size arm (or materialize the data)."""
    if arm == "data":
        return materialize_data()
    return _ARMS[arm]()


if __name__ == "__main__":
    build()
