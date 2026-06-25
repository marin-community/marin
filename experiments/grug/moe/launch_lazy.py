# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""grug-moe baseline, authored as a lazy artifact.

This is the same run as ``baseline_moe`` in ``launch.py``, written in the
artifact model: a zero-arg function returns a typed :class:`Checkpoint` handle
addressed by an explicit ``name@version``. The experiment file contains no
``ExecutorStep``, ``executor_main``, ``versioned()``, or ``this_output_path()`` —
the decisions are stated inline, and the output path is ``ctx.out``.

Two variants show the migration arc:

- ``grug_moe_baseline`` keeps the legacy ``nemotron_mix`` data catalog, so it
  materializes through the bridge (``marin.execution.lazy.to_executor_step`` →
  the existing ``Executor``). A golden test
  (``tests/experiment/test_grug_moe_lazy_parity.py``) pins it to produce the same
  materialized config as ``baseline_moe``.
- ``grug_moe_slimpajama`` builds its data from a :class:`Dataset` handle, so the
  whole graph lowers via ``lower()`` with the ``Executor`` out of the path.
"""

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.lazy import BuildContext, Checkpoint, Recipe, lower
from marin.execution.step_runner import StepRunner
from marin.experiment.data import mixture, tokenize

from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    RESOLVED_RUN_ID,
    GrugMoeLaunchConfig,
    run_grug_moe_trial,
)
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig
from experiments.llama import llama3_tokenizer
from experiments.pretraining_datasets.nemotron_lazy import nemotron_components

# Tokenization runs as its own Fray job (keeps the launcher pod light).
_TOKENIZE_RESOURCES = ResourceConfig(ram="64g", disk="64g")

# 1e18 compute budget, d1024 — model/optimizer/batch/steps derived from the heuristic.
_BUDGET = 1e18
_HIDDEN_DIM = 1024
_TARGET_STEPS = 2**14


def grug_moe_baseline(*, version: str = "v1") -> Checkpoint:
    model, optimizer, batch_size, steps = build_from_heuristic(
        budget=_BUDGET, hidden_dim=_HIDDEN_DIM, target_steps=_TARGET_STEPS
    )

    def build_config(ctx: BuildContext) -> GrugMoeLaunchConfig:
        return _grug_launch_config(ctx, model, optimizer, batch_size, steps, data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION)

    return Checkpoint(
        name="grug/4_10_baseline_moe",
        version=version,
        # resources live in the config (run_grug dispatches its own Fray TPU job),
        # so the launcher step itself runs inline — matching baseline_moe.
        recipe=Recipe(fn=run_grug_moe_trial, build_config=build_config, resources=None),
    )


def _grug_launch_config(ctx, model, optimizer, batch_size, steps, *, data):
    """Shared GrugMoeLaunchConfig assembly for the lazy grug-moe runs."""
    return GrugMoeLaunchConfig(
        model=model,
        data=data,
        output_path=ctx.out,
        run_id=RESOLVED_RUN_ID,
        resources=ResourceConfig.with_tpu("v5p-8"),
        steps=steps,
        batch_size=batch_size,
        seed=0,
        mp="params=float32,compute=bfloat16,output=bfloat16",
        tracker=WandbConfig(project="marin_moe", tags=["moe"], group="moe-iter04", name=None),
        optimizer=optimizer,
        grug_trainer=GrugTrainerConfig(z_loss_weight=1e-4, ema_beta=None, log_every=1),
        eval=GrugEvalConfig(
            eval_batch_size=512,
            steps_per_eval=1000,
            max_eval_batches=8,
            eval_current=True,
            eval_ema=False,
        ),
    )


def grug_moe_slimpajama(*, version: str = "v1") -> Checkpoint:
    """Same grug-moe run on a single tokenized dataset, authored fully in the
    artifact model: the data is a :class:`Dataset` handle, so the whole graph
    lowers via ``lower()`` with the Executor out of the path."""
    model, optimizer, batch_size, steps = build_from_heuristic(
        budget=_BUDGET, hidden_dim=_HIDDEN_DIM, target_steps=_TARGET_STEPS
    )
    slim = tokenize(
        "tokenized/slimpajama-6b",
        "v1",
        source="DKYoon/SlimPajama-6B",
        tokenizer=llama3_tokenizer,
        resources=_TOKENIZE_RESOURCES,
    )

    def build_config(ctx: BuildContext) -> GrugMoeLaunchConfig:
        return _grug_launch_config(ctx, model, optimizer, batch_size, steps, data=mixture(ctx, {slim: 1.0}))

    return Checkpoint(
        name="grug/4_10_baseline_moe_slim",
        version=version,
        recipe=Recipe(fn=run_grug_moe_trial, build_config=build_config, deps=(slim,), resources=None),
    )


def grug_moe_baseline_nemotron(*, version: str = "v1") -> Checkpoint:
    """The grug-moe baseline on the Nemotron CC splits, authored fully in the
    artifact model. The splits are pinned ``Dataset`` handles (existing tokenized
    data), so the whole graph lowers via ``lower()`` with the Executor out of the
    path and nothing re-tokenizes.

    (The full ``baseline_moe`` mixture also blends starcoder/proofpile and the
    default validation sets; those are unpinned content-addressed caches and need
    their existing paths discovered before they can join this pure graph.)
    """
    model, optimizer, batch_size, steps = build_from_heuristic(
        budget=_BUDGET, hidden_dim=_HIDDEN_DIM, target_steps=_TARGET_STEPS
    )
    components = nemotron_components(tokenizer=llama3_tokenizer)

    def build_config(ctx: BuildContext) -> GrugMoeLaunchConfig:
        return _grug_launch_config(ctx, model, optimizer, batch_size, steps, data=mixture(ctx, components))

    return Checkpoint(
        name="grug/4_10_baseline_moe_nemotron",
        version=version,
        recipe=Recipe(fn=run_grug_moe_trial, build_config=build_config, deps=tuple(components), resources=None),
    )


if __name__ == "__main__":
    # Pure lazy graph (nemotron splits -> train), run by StepRunner with no Executor.
    StepRunner().run([lower(grug_moe_baseline_nemotron())])
