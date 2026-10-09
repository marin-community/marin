# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""PivotRL data: pass rates of the frozen policy on prepared candidates, then the selected pivots.

The pass-rate artifact is unfiltered and reusable; each difficulty threshold is its own small
selection artifact. Training consumes ``train.parquet`` from the selection, for example as
``ArtifactDataSource(pivots, relative_path=TRAIN_FILENAME)`` in a ``skyrl_step``.

Print the plan with ``uv run python -m experiments.post_training.pivotrl.pipeline --version dev``,
choose runs with ``--experiment``, and add ``--run`` to build them.
"""

import click
from fray.types import ResourceConfig
from marin.evaluation.eval_env import EVAL_RUNTIME_ENV_KEYS, env_vars_from_keys
from marin.execution.artifact import Artifact
from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.execution.remote import remote
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_owned_name
from marin.rl.pass_rates import measure_pass_rates

from experiments.datasets.nemotron_pivot import nemotron_pivot_datasets
from experiments.post_training.pivotrl.runs import (
    RUNS,
    NemotronPivotCandidates,
    PivotRLRun,
)
from experiments.post_training.pivotrl.selection import head_rows, select_pivots


def candidate_step(
    candidates: NemotronPivotCandidates,
) -> ArtifactStep[Artifact]:
    """The full candidate set, or its first ``limit`` rows."""
    full = nemotron_pivot_datasets()[candidates.dataset]
    if candidates.limit is None:
        return full
    return apply(
        user_owned_name(f"pivotrl/{candidates.label}/candidates"),
        remote(head_rows, resources=ResourceConfig.with_cpu(cpu=2, ram="16g")),
        rows_path=full,
        rows_filename=candidates.filename,
        output_path=OUT,
        limit=candidates.limit,
    )


def pivot_steps(run: PivotRLRun) -> dict[str, ArtifactStep[Artifact]]:
    """The pass-rate and selection artifacts for one run."""
    dataset = run.candidates.label
    # Seeded runs of the same candidates and policy get their own artifacts.
    policy = run.model.name if run.sampling.seed is None else f"{run.model.name}/seed-{run.sampling.seed}"
    candidates = candidate_step(run.candidates)
    grading_env = env_vars_from_keys(run.secret_env_keys)
    pass_rates = apply(
        user_owned_name(f"pivotrl/{dataset}/pass-rates/{policy}"),
        # The orchestrator is a CPU job; remote inference launches the GPU serving job beside it.
        remote(
            measure_pass_rates,
            resources=ResourceConfig.with_cpu(cpu=8, ram="64g"),
            env_vars={**env_vars_from_keys(EVAL_RUNTIME_ENV_KEYS), **grading_env},
        ),
        rows_path=candidates,
        rows_filename=run.candidates.filename,
        output_path=OUT,
        task=run.candidates.task,
        model=run.model,
        accelerator=run.accelerator,
        sampling=run.sampling,
    )
    # Each criterion and lambda is its own small artifact over the same pass rates.
    criterion = "" if run.criterion == "passed" else f"/{run.criterion}"
    pivots = apply(
        user_owned_name(f"pivotrl/{dataset}/pivots/{policy}{criterion}/lambda-{run.difficulty_threshold}"),
        remote(select_pivots, resources=ResourceConfig.with_cpu(cpu=2, ram="16g")),
        pass_rates_path=pass_rates,
        output_path=OUT,
        difficulty_threshold=run.difficulty_threshold,
        criterion=run.criterion,
    )
    return {"pass-rates": pass_rates, "pivots": pivots}


@click.command(help=__doc__)
@click.option(
    "--experiment",
    "names",
    type=click.Choice(sorted(RUNS)),
    multiple=True,
    help="Run to build; repeatable. Default: every run.",
)
@build_options
def main(names: tuple[str, ...]) -> dict[str, ArtifactStep[Artifact]]:
    return {f"{name}/{stage}": step for name in names or RUNS for stage, step in pivot_steps(RUNS[name]).items()}


if __name__ == "__main__":
    main()
