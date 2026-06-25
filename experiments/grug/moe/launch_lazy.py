# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""grug-moe baseline, authored as a lazy artifact.

Each run is a function that returns a typed :class:`Checkpoint` handle addressed by
an explicit ``name@version``. The experiment file contains no ``ExecutorStep``,
``executor_main``, ``versioned()``, or ``this_output_path()`` — the decisions are
stated inline, and the output path is ``ctx.out``.

The data mixture reads as a flat table: the catalog modules provide dataset
*handles* (mechanism); this experiment states the *weights* (policy). The Nemotron
weights are the corpus's TiB proportions.

Three variants:

- ``grug_moe_slimpajama`` builds its data from a single :class:`Dataset` handle,
  the minimal fully-lazy demo.
- ``grug_moe_baseline_pure`` assembles the full baseline mixture — Nemotron CC +
  starcoder + proofpile, plus the paloma/uncheatable validation suites at weight 0
  — from handles, so the whole graph lowers via ``lower()`` with the ``Executor``
  out of the path.
- ``grug_moe_fineweb_mini`` is a tiny, fast smoke on the prebuilt fineweb-edu 10M
  cache: it exercises the same lazy path (a download dependency, then one short TPU
  training job) cheaply, for validating the model end to end on a real cluster.
"""

import sys

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.lazy import Checkpoint, Recipe, RunContext, lower
from marin.execution.step_runner import StepRunner
from marin.experiment.data import mixture, pretokenized, tokenized

from experiments.evals.uncheatable_lazy import uncheatable_validation
from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.launch import RESOLVED_RUN_ID, GrugMoeLaunchConfig, run_grug_moe_trial
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig
from experiments.llama import llama3_tokenizer
from experiments.marin_tokenizer import marin_tokenizer
from experiments.paloma_lazy import paloma_validation
from experiments.pretraining_datasets.nemotron_lazy import nemotron_datasets
from experiments.pretraining_datasets.simple_lazy import proofpile_dataset, starcoder_dataset

# Tokenization runs as its own Fray job (keeps the launcher pod light).
_TOKENIZE_RESOURCES = ResourceConfig(ram="64g", disk="64g")

# Downloading a prebuilt cache is light I/O — a small CPU job, not a tokenize.
_DOWNLOAD_RESOURCES = ResourceConfig(ram="16g", disk="32g")

# The TPU the training job is dispatched onto. A run-arg, not part of the config's
# identity: re-running on a different TPU is the same checkpoint, so it must not fork
# the artifact. The launcher step itself runs inline (run_grug dispatches its own job).
_TRAIN_RESOURCES = ResourceConfig.with_tpu("v5p-8")

# 1e18 compute budget, d1024 — model/optimizer/batch/steps derived from the heuristic.
_BUDGET = 1e18
_HIDDEN_DIM = 1024
_TARGET_STEPS = 2**14

_STANDARD_EVAL = GrugEvalConfig(
    eval_batch_size=512,
    steps_per_eval=1000,
    max_eval_batches=8,
    eval_current=True,
    eval_ema=False,
)


def _grug_launch_config(
    ctx,
    model,
    optimizer,
    batch_size,
    steps,
    *,
    data,
    run_id=RESOLVED_RUN_ID,
    eval_config=_STANDARD_EVAL,
    wandb_group="moe-iter04",
):
    """Shared GrugMoeLaunchConfig assembly for the lazy grug-moe runs."""
    return GrugMoeLaunchConfig(
        model=model,
        data=data,
        output_path=ctx.out,
        run_id=run_id,
        resources=ctx.run_arg("train_resources"),
        steps=steps,
        batch_size=batch_size,
        seed=0,
        mp="params=float32,compute=bfloat16,output=bfloat16",
        tracker=WandbConfig(project="marin_moe", tags=["moe"], group=wandb_group, name=None),
        optimizer=optimizer,
        grug_trainer=GrugTrainerConfig(z_loss_weight=1e-4, ema_beta=None, log_every=1),
        eval=eval_config,
    )


def grug_moe_slimpajama(*, version: str = "v1") -> Checkpoint:
    """Same grug-moe run on a single tokenized dataset, authored fully in the
    artifact model: the data is a :class:`Dataset` handle, so the whole graph
    lowers via ``lower()`` with the Executor out of the path."""
    model, optimizer, batch_size, steps = build_from_heuristic(
        budget=_BUDGET, hidden_dim=_HIDDEN_DIM, target_steps=_TARGET_STEPS
    )
    slim = tokenized(
        "slimpajama-6b",
        source="DKYoon/SlimPajama-6B",
        tokenizer=llama3_tokenizer,
        resources=_TOKENIZE_RESOURCES,
    )

    def build_config(ctx: RunContext) -> GrugMoeLaunchConfig:
        return _grug_launch_config(ctx, model, optimizer, batch_size, steps, data=mixture(ctx, {slim: 1.0}))

    return Checkpoint(
        name="grug/4_10_baseline_moe_slim",
        version=version,
        recipe=Recipe(
            fn=run_grug_moe_trial,
            build_config=build_config,
            deps=(slim,),
            run_args={"train_resources": _TRAIN_RESOURCES},
        ),
    )


def grug_moe_baseline_pure(*, version: str = "v1") -> Checkpoint:
    """The full grug-moe baseline, authored entirely in the artifact model. Every
    component is a :class:`Dataset` handle, so the whole graph lowers via ``lower()``
    with the Executor, InputName, this_output_path, and content-addressing all out of
    the path. Pinned components (nemotron/starcoder/proofpile) never re-tokenize.

    The mixture is a flat table: catalog modules supply the dataset handles, and the
    weights (policy) are stated here. Nemotron weights are the corpus's TiB proportions;
    the paloma/uncheatable suites are validation (weight 0).
    """
    model, optimizer, batch_size, steps = build_from_heuristic(
        budget=_BUDGET, hidden_dim=_HIDDEN_DIM, target_steps=_TARGET_STEPS
    )

    nem = nemotron_datasets(tokenizer=llama3_tokenizer)
    train = {
        nem["hq_actual"]: 0.91351,
        nem["hq_synth"]: 2.72,
        nem["medium_high"]: 0.82471,
        nem["medium"]: 3.38,
        nem["medium_low"]: 1.54,
        nem["low_actual"]: 0.70123,
        nem["low_synth"]: 0.62771,
        starcoder_dataset(tokenizer=llama3_tokenizer): 0.25,
        proofpile_dataset(tokenizer=llama3_tokenizer): 0.055,
    }
    validation = [*paloma_validation(tokenizer=llama3_tokenizer), *uncheatable_validation(tokenizer=llama3_tokenizer)]

    def build_config(ctx: RunContext) -> GrugMoeLaunchConfig:
        data = mixture(ctx, train, validation=validation)
        return _grug_launch_config(ctx, model, optimizer, batch_size, steps, data=data)

    return Checkpoint(
        name="grug/4_10_baseline_moe_pure",
        version=version,
        recipe=Recipe(
            fn=run_grug_moe_trial,
            build_config=build_config,
            deps=(*train, *validation),
            run_args={"train_resources": _TRAIN_RESOURCES},
        ),
    )


# Mini smoke: tiny model, the prebuilt fineweb-edu 10M cache, a short run.
_FINEWEB_EDU_10M_REPO = "marin-community/fineweb-edu-pretokenized-10M"
_MINI_HIDDEN_DIM = 256
_MINI_BUDGET = 1e15
_MINI_BATCH = 32
_MINI_STEPS = 20

# v5p lives in us-east5 / us-central1; the smoke's data cache is in us-east5. Pin the
# region so the dispatched TPU job runs region-local with the data instead of
# inheriting the launcher pod's (arbitrary) region, which may have no v5p at all.
_MINI_TRAIN_RESOURCES = ResourceConfig.with_tpu("v5p-8", regions=["us-east5"])


def grug_moe_fineweb_mini(*, version: str = "v1") -> Checkpoint:
    """A tiny, fast grug-moe smoke on the prebuilt fineweb-edu 10M cache.

    Exercises the full lazy path end to end — a download dependency materializes as
    its own Fray job, then the launcher dispatches one short TPU training job —
    without the cost of a real pretraining run. The model is sized down (``d256``,
    ``batch=32``, 20 steps) so the whole graph runs in minutes.
    """
    model, optimizer, _, _ = build_from_heuristic(
        budget=_MINI_BUDGET, hidden_dim=_MINI_HIDDEN_DIM, target_steps=_MINI_STEPS
    )
    fineweb = pretokenized(
        "fineweb-edu-10M",
        repo_id=_FINEWEB_EDU_10M_REPO,
        tokenizer=marin_tokenizer,
        resources=_DOWNLOAD_RESOURCES,
    )

    def build_config(ctx: RunContext) -> GrugMoeLaunchConfig:
        return _grug_launch_config(
            ctx,
            model,
            optimizer,
            _MINI_BATCH,
            _MINI_STEPS,
            data=mixture(ctx, {fineweb: 1.0}),
            run_id="grug-moe-fineweb-mini",
            eval_config=None,
            wandb_group="mini-smoke",
        )

    return Checkpoint(
        name="grug/moe_fineweb_mini",
        version=version,
        recipe=Recipe(
            fn=run_grug_moe_trial,
            build_config=build_config,
            deps=(fineweb,),
            run_args={"train_resources": _MINI_TRAIN_RESOURCES},
        ),
    )


_VARIANTS = {
    "slimpajama": grug_moe_slimpajama,
    "pure": grug_moe_baseline_pure,
    "mini": grug_moe_fineweb_mini,
}


if __name__ == "__main__":
    # Lower the chosen variant to a pure StepSpec graph and run it (no Executor).
    # `python -m experiments.grug.moe.launch_lazy [slimpajama|pure|mini]`, default pure.
    selected = sys.argv[1] if len(sys.argv) > 1 else "pure"
    StepRunner().run([lower(_VARIANTS[selected]())])
