# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sketch: a grug-moe checkpoint chain across training phases.

A model's lifecycle — pretrain -> midtrain (anneal / continued pretraining) -> SFT
-> RL — is a chain of checkpoints, each initialized from the previous. In the lazy
artifact model each phase is a :class:`~marin.execution.lazy.Checkpoint` whose recipe
depends on the prior phase and initializes from it: ``init_from`` points at the
parent's ``checkpoints`` directory, resolved via ``ctx.path(parent)``. The dependency
edge *is* the lineage — recorded in each artifact's provenance — and the build-once
guard serves a cached parent when only a downstream phase is re-run. Branching is just
two children off one parent (e.g. two SFT mixes from the same pretrain).

``pretrain`` and ``midtrain`` run the real grug pretraining loop. SFT and RL need their
own training objectives (a supervised loss over prompt/response data; a rollout/reward
loop) which the grug template does not yet provide — they are sketched here as the seam
where those loops slot in, so the *shape* of the chain runs end to end while the
objective-specific code is named but not yet built.

Run it (pretrain + midtrain train; SFT/RL raise until implemented)::

    python -m experiments.grug.moe.phases_lazy
"""

from collections.abc import Callable

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.lazy import Checkpoint, Recipe, RunContext, lower
from marin.execution.step_runner import StepRunner
from marin.experiment.data import mixture, pretokenized

from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.launch import GrugMoeLaunchConfig, run_grug_moe_trial
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig
from experiments.marin_tokenizer import marin_tokenizer

# Tiny, fast sizing so the whole chain can smoke on one slice; real phases scale up.
# v5p lives in us-east5 / us-central1; pin the region so each phase's TPU job runs
# region-local with the prebuilt cache (us-east5) rather than inheriting the launcher's.
_TRAIN_RESOURCES = ResourceConfig.with_tpu("v5p-8", regions=["us-east5"])
_DOWNLOAD_RESOURCES = ResourceConfig(ram="16g", disk="32g")

# The architecture is fixed across phases (you cannot resize a model mid-chain), so
# model + optimizer + batch are derived once; phases differ in data, steps, and parent.
_MODEL, _OPTIMIZER, _BATCH, _ = build_from_heuristic(budget=1e15, hidden_dim=256, target_steps=2000)

_EVAL = GrugEvalConfig(eval_batch_size=256, steps_per_eval=500, max_eval_batches=8, eval_current=True, eval_ema=False)

# The prebuilt fineweb-edu 10M cache, shared by every phase here. Real SFT/RL phases
# would swap in their own supervised / preference datasets.
_fineweb = pretokenized(
    "fineweb-edu-10M",
    repo_id="marin-community/fineweb-edu-pretokenized-10M",
    tokenizer=marin_tokenizer,
    resources=_DOWNLOAD_RESOURCES,
)


def run_grug_sft_phase(config: GrugMoeLaunchConfig) -> None:
    """Supervised fine-tuning phase (sketch).

    Initializes weights from ``config.init_from`` and would train a supervised loss over
    a prompt/response dataset. The grug template has no SFT loop yet; this is the seam
    where it slots in.
    """
    raise NotImplementedError("grug SFT phase: supervised fine-tuning loop not yet implemented")


def run_grug_rl_phase(config: GrugMoeLaunchConfig) -> None:
    """RL phase (sketch).

    Initializes weights from ``config.init_from`` and would run a rollout/reward
    optimization loop. Not yet implemented in the grug template.
    """
    raise NotImplementedError("grug RL phase: rollout/reward loop not yet implemented")


def _phase(
    name: str,
    *,
    fn: Callable[[GrugMoeLaunchConfig], None],
    steps: int,
    parent: Checkpoint | None,
    run_id: str,
    wandb_group: str,
    version: str = "v1",
) -> Checkpoint:
    """One phase as a :class:`Checkpoint`, initialized from ``parent`` (None = from scratch).

    The data dependency is the shared fineweb cache; ``parent`` (when set) is both a
    dependency — so it builds first — and the checkpoint this phase initializes from.
    """
    deps = (_fineweb,) if parent is None else (_fineweb, parent)

    def build_config(ctx: RunContext) -> GrugMoeLaunchConfig:
        return GrugMoeLaunchConfig(
            model=_MODEL,
            data=mixture(ctx, {_fineweb: 1.0}),
            output_path=ctx.out,
            run_id=run_id,
            resources=ctx.run_arg("train_resources"),
            steps=steps,
            batch_size=_BATCH,
            seed=0,
            mp="params=float32,compute=bfloat16,output=bfloat16",
            tracker=WandbConfig(project="marin_moe", tags=["moe", "phases"], group=wandb_group, name=None),
            optimizer=_OPTIMIZER,
            grug_trainer=GrugTrainerConfig(z_loss_weight=1e-4, ema_beta=None, log_every=1),
            eval=_EVAL,
            # The latest checkpoint under the parent phase's output is loaded as the
            # starting weights; the parent's name@version travels into this phase's
            # fingerprint, so re-pointing at a different parent is a different artifact.
            init_from=f"{ctx.path(parent)}/checkpoints" if parent is not None else None,
        )

    return Checkpoint(
        name=name,
        version=version,
        recipe=Recipe(fn=fn, build_config=build_config, deps=deps, run_args={"train_resources": _TRAIN_RESOURCES}),
    )


def phase_chain(*, version: str = "v1") -> Checkpoint:
    """The full pretrain -> midtrain -> SFT -> RL checkpoint chain.

    Returns the final RL checkpoint; lowering it pulls the whole lineage. pretrain and
    midtrain run; SFT and RL raise until their objectives are implemented.
    """
    pretrain = _phase(
        "grug/phases/pretrain",
        fn=run_grug_moe_trial,
        steps=2000,
        parent=None,
        run_id="grug-phase-pretrain",
        wandb_group="phases-pretrain",
        version=version,
    )
    midtrain = _phase(
        "grug/phases/midtrain",
        fn=run_grug_moe_trial,
        steps=500,
        parent=pretrain,
        run_id="grug-phase-midtrain",
        wandb_group="phases-midtrain",
        version=version,
    )
    sft = _phase(
        "grug/phases/sft",
        fn=run_grug_sft_phase,
        steps=300,
        parent=midtrain,
        run_id="grug-phase-sft",
        wandb_group="phases-sft",
        version=version,
    )
    return _phase(
        "grug/phases/rl",
        fn=run_grug_rl_phase,
        steps=200,
        parent=sft,
        run_id="grug-phase-rl",
        wandb_group="phases-rl",
        version=version,
    )


if __name__ == "__main__":
    # Lower the chain to a StepSpec graph and run it: pretrain + midtrain train, each
    # initialized from its parent; SFT/RL raise NotImplementedError until built.
    StepRunner().run([lower(phase_chain())])
