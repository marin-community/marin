# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Boundary-operator compute-optimal cells (arXiv 2609.19107).

Runs the prelude-core-coda boundary operator (no looping, no growth) at the
May Recipe compute-optimal budgets from ``experiments/grug/moe/README.md``,
matched to the baseline cells in parameters and FLOPs by construction - the
operator adds no weights and one vector-add per boundary.

The README baselines were measured under the previous template defaults
(``seq_len=4096``, PKO on for long layers, RoPE on long layers, final-logit
z-loss off), so each cell pins those flags and the batch/steps of the README
table rather than re-deriving them from the rounded budgets. Cell budgets:
d512 / 3.82e17, d768 / 2.81e18, d1024 / 1.16e19, d1280 / 3.46e19.

Phase 0 of the replication plan runs one arm at a time: ``--dim 512
``--alpha 1.0`` first, then the alpha-insurance twin ``--alpha 0.707``; the
winner proceeds to d768/d1024/d1280 (gate 1 / gate 2).

If gate 1 fails at d512, the debug ladder sweeps the learning rate around the
heuristic value: ``--lr-scale 0.5`` and ``--lr-scale 2.0`` at the winning
alpha. The paper's transfer-regret table (7.8e-3 for Operator-1) predicts a
transferred recipe underperforms a re-centered one, and GLR is its most
sensitive knob.

Submit (v4-32, EP=1)::

    .venv/bin/iris --cluster=marin job run --no-wait \\
        -e WANDB_API_KEY "$WANDB_API_KEY" \\
        -- python -m experiments.grug.moe_boundary.launch_compute_opt \\
            --version dev --run --dim 512 --alpha 1.0
"""

import dataclasses

import click
from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import build_options
from marin.experiment.data import mixture
from marin.experiment.namespacing import user_namespaced_name
from marin.training.training import LevanterCheckpoint

from experiments.grug.moe_boundary.heuristic import build_from_heuristic
from experiments.grug.moe_boundary.launch import (
    GrugMoeLaunchConfig,
    grug_moe_boundary_mix,
    run_grug_moe_trial,
)
from experiments.grug.moe_boundary.model import split_prelude_core_coda
from experiments.grug.moe_boundary.train import GrugEvalConfig, GrugTrainerConfig

_SEQ: int = 4096  # README baseline measurement condition
_EP: int = 1

# May Recipe compute-optimal cells (README table): dim -> (budget, batch, steps)
_BASELINE_CELLS: dict[int, tuple[float, int, int]] = {
    512: (3.82e17, 32, 10_980),
    768: (2.81e18, 64, 16_875),
    1024: (1.16e19, 128, 16_080),
    1280: (3.46e19, 256, 14_325),
}

_TRAIN_RESOURCES = ResourceConfig.with_tpu("v4-32")


def boundary_cell(
    *, hidden_dim: int, alpha: float, lr_scale: float = 1.0, version: str | None = None
) -> ArtifactStep[LevanterCheckpoint]:
    """Compute-optimal boundary-operator cell at a May Recipe baseline point.

    ``hidden_dim`` is one of the four README scales; ``alpha`` is the boundary
    operator's injection weight (paper arms: 0.707 and 1.0). ``lr_scale``
    multiplies the heuristic's optimizer learning rates (debug ladder only).
    """
    budget, batch_size, steps = _BASELINE_CELLS[hidden_dim]
    # Model + optimizer from the heuristic at this budget, then pin the legacy
    # measurement conditions and add the boundary operator with the paper's
    # even P/C/D split.
    model, optimizer, _, _ = build_from_heuristic(budget=budget, hidden_dim=hidden_dim, seq_len=_SEQ)
    split = split_prelude_core_coda(model.num_layers)
    prelude_len, coda_len = split.prelude, split.coda
    if lr_scale != 1.0:
        optimizer = dataclasses.replace(
            optimizer, learning_rate=optimizer.learning_rate * lr_scale, adam_lr=optimizer.adam_lr * lr_scale
        )
    boundary_model = dataclasses.replace(
        model,
        max_seq_len=_SEQ,
        disable_pko=False,
        disable_long_rope=False,
        boundary_operator=True,
        prelude_len=prelude_len,
        coda_len=coda_len,
        injection_scale=alpha,
    )
    name = f"grug/moe_boundary_compute_opt_d{hidden_dim}_ep{_EP}_alpha{alpha:g}"
    if lr_scale != 1.0:
        name += f"_lr{lr_scale:g}"
    version = resolve_version(name, version)
    train, validation = grug_moe_boundary_mix()
    run_id = f"moe_boundary_compute_opt_d{hidden_dim}_ep{_EP}_alpha{alpha:g}"
    if lr_scale != 1.0:
        run_id += f"_lr{lr_scale:g}"

    def build_config(ctx: StepContext) -> GrugMoeLaunchConfig:
        return GrugMoeLaunchConfig(
            model=boundary_model,
            data=mixture(ctx, train, validation=validation),
            output_path=ctx.output_path,
            run_id=run_id,
            resources=ctx.runtime_arg("train_resources"),
            steps=steps,
            batch_size=batch_size,
            seed=0,
            mp="params=float32,compute=bfloat16,output=bfloat16",
            tracker=WandbConfig(
                project="marin_moe",
                tags=["moe", "boundary-operator", f"d{hidden_dim}", f"alpha{alpha:g}"],
                group="moe-boundary",
                name=None,
            ),
            optimizer=optimizer,
            grug_trainer=GrugTrainerConfig(z_loss_weight=0.0, ema_beta=None, log_every=1),
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


@click.command()
@click.option("--dim", type=click.Choice(["512", "768", "1024", "1280"]), required=True)
@click.option("--alpha", type=float, default=1.0, help="Boundary-operator injection scale (paper: 0.707 or 1.0).")
@click.option("--lr-scale", type=float, default=1.0, help="Multiplier on the heuristic LR (debug ladder: 0.5/2.0).")
@build_options
def build(dim: str, alpha: float, lr_scale: float):
    """Build one boundary-operator compute-optimal cell."""
    return boundary_cell(hidden_dim=int(dim), alpha=alpha, lr_scale=lr_scale)


if __name__ == "__main__":
    build()
