# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""PivotRL data: pass rates of the frozen policy on prepared candidates, then the selected pivots.

The pass-rate artifact is unfiltered and reusable; each difficulty threshold is its own small
selection artifact. Training consumes ``train.parquet`` from the selection, for example as
``ArtifactDataSource(pivots, relative_path=TRAIN_FILENAME)`` in a ``skyrl_step``.

Print the plan with ``uv run python -m experiments.post_training.pivotrl.pipeline --version dev``,
choose runs with ``--experiment``, and add ``--run`` to build them.

For mixed-domain training, pass one ``--experiment`` and ``--curated-path``, the curated
task-curation source of its candidate set (``data/rl/pivotrl-<set>-<hash>``), to add a ``tasks``
stage: the selected pivots as lowered tasks in ``tasks.parquet`` (:mod:`.compendium`).
"""

import click
from fray.types import ResourceConfig
from marin.evaluation.eval_env import EVAL_RUNTIME_ENV_KEYS, env_vars_from_keys
from marin.execution.artifact import Artifact
from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.execution.remote import remote
from marin.experiment.cli import build_options
from marin.experiment.data import hf_download
from marin.experiment.namespacing import user_owned_name
from marin.rl.openhands_pivot import TOOLS_FILENAME, TRAJECTORIES_FILENAME, prepare_openhands_candidates
from marin.rl.pass_rates import measure_pass_rates

from experiments.datasets.nemotron_pivot import nemotron_pivot_datasets
from experiments.post_training.pivotrl.compendium import PivotSession, pivot_tasks
from experiments.post_training.pivotrl.runs import (
    RUNS,
    NemotronPivotCandidates,
    OpenHandsCandidates,
    PivotRLRun,
)
from experiments.post_training.pivotrl.selection import head_rows, select_pivots

# MarinSkyRL's task-session defaults: no model-turn deadline and a 30-second cleanup.
PIVOT_SESSION = PivotSession(model_turn_timeout=None, cleanup_timeout=30.0)


def full_candidate_step(
    candidates: NemotronPivotCandidates | OpenHandsCandidates,
) -> ArtifactStep[Artifact]:
    """The candidate set prepared from a pinned trajectory release."""
    match candidates:
        case NemotronPivotCandidates():
            return nemotron_pivot_datasets()[candidates.dataset]
        case OpenHandsCandidates():
            raw = hf_download(
                candidates.raw_name,
                hf_id=candidates.hf_id,
                revision=candidates.revision,
                version=candidates.raw_version,
                urls_glob=(TRAJECTORIES_FILENAME, TOOLS_FILENAME),
            )
            return apply(
                f"processed/{candidates.dataset}-pivot-candidates",
                remote(prepare_openhands_candidates, resources=ResourceConfig.with_cpu(cpu=8, ram="64g")),
                version=candidates.version,
                dataset_path=raw,
                output_path=OUT,
                trajectories=candidates.trajectories,
                validation_trajectories=candidates.validation_trajectories,
                seed=candidates.seed,
                max_prompt_chars=candidates.max_prompt_chars,
            )


def candidate_step(
    candidates: NemotronPivotCandidates | OpenHandsCandidates,
) -> ArtifactStep[Artifact]:
    """The full candidate set, or its first ``limit`` rows."""
    full = full_candidate_step(candidates)
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


def pivot_steps(run: PivotRLRun, curated_path: str | None = None) -> dict[str, ArtifactStep[Artifact]]:
    """The pass-rate and selection artifacts for one run, and its lowered tasks given its curated source."""
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
    selection = f"pivotrl/{dataset}/pivots/{policy}{criterion}/lambda-{run.difficulty_threshold}"
    pivots = apply(
        user_owned_name(selection),
        remote(select_pivots, resources=ResourceConfig.with_cpu(cpu=2, ram="16g")),
        pass_rates_path=pass_rates,
        output_path=OUT,
        difficulty_threshold=run.difficulty_threshold,
        criterion=run.criterion,
    )
    steps = {"pass-rates": pass_rates, "pivots": pivots}
    if curated_path is not None:
        steps["tasks"] = apply(
            user_owned_name(f"{selection}/tasks"),
            remote(pivot_tasks, resources=ResourceConfig.with_cpu(cpu=2, ram="16g")),
            curated_path=curated_path,
            pivots_path=pivots,
            output_path=OUT,
            session=PIVOT_SESSION,
        )
    return steps


@click.command(help=__doc__)
@click.option(
    "--experiment",
    "names",
    type=click.Choice(sorted(RUNS)),
    multiple=True,
    help="Run to build; repeatable. Default: every run.",
)
@click.option(
    "--curated-path",
    default=None,
    help="Curated PivotRL source of the one --experiment's candidate set; adds its lowered-task stage.",
)
@build_options
def main(names: tuple[str, ...], curated_path: str | None) -> dict[str, ArtifactStep[Artifact]]:
    if curated_path is not None and len(names) != 1:
        raise click.UsageError("--curated-path belongs to exactly one --experiment")
    return {
        f"{name}/{stage}": step
        for name in names or RUNS
        for stage, step in pivot_steps(RUNS[name], curated_path).items()
    }


if __name__ == "__main__":
    main()
