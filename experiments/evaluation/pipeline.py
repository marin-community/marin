# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Eval runs as pipeline steps.

:func:`eval_step` wraps one launcher run (model x eval selection) as an :class:`ArtifactStep`, so
evals compose into ``StepRunner`` pipelines and can be triggered programmatically -- e.g. right
after a training pipeline exports a checkpoint, or fanned out over a model sweep. The step runs the
same orchestration as the CLI (serve the model once, run evalchemy against the served URL, write
``record.json`` + results + per-question parquet), with the records rooted at the step's own
artifact path: an identical (model, evals, limit, version) config is a cache hit, and downstream
steps can depend on the records like any other artifact.

The step process acts as the orchestrator, so the pipeline must run as an Iris job::

    uv run iris --cluster marin job run -- python -m experiments.evaluation.pipeline

The demo pipeline below runs the smoke suite for one small model.
"""

from __future__ import annotations

from dataclasses import dataclass

from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.step_runner import StepRunner

from experiments.evaluation.evals import SUITES
from experiments.evaluation.hardware import default_platform
from experiments.evaluation.launch import LaunchSpec, run_inline
from experiments.evaluation.models import MODELS


@dataclass(frozen=True)
class EvalStepConfig:
    """One eval run's identity (model + eval selection + instance cap) and its output root."""

    model: str
    evals: str
    limit: int | None
    records_prefix: str
    accelerator: str | None
    version: str


def run_eval_pipeline_step(config: EvalStepConfig) -> None:
    keys = SUITES.get(config.evals) or (config.evals,)
    spec = LaunchSpec(
        model=config.model,
        evals=keys,
        platform=default_platform(MODELS[config.model]),
        accelerator=config.accelerator,
        limit=config.limit,
        records_prefix=config.records_prefix,
        cluster="ambient",
        version=config.version,
    )
    run_inline(spec)


def eval_step(
    model: str,
    evals: str,
    *,
    version: str,
    limit: int | None = None,
    accelerator: str | None = None,
) -> ArtifactStep[Artifact]:
    """A lazy handle for one eval run of ``model`` on ``evals`` (a suite name or eval key).

    ``model``/``evals``/``limit`` and ``version`` bear the artifact's identity; ``accelerator`` is a
    runtime arg (an execution choice), so overriding the slice never forks the artifact.
    """

    def build_config(ctx: StepContext) -> EvalStepConfig:
        return EvalStepConfig(
            model=model,
            evals=evals,
            limit=limit,
            records_prefix=ctx.output_path,
            accelerator=ctx.runtime_arg("accelerator"),
            version=version,
        )

    return ArtifactStep(
        name=f"evals/{model}/{evals}",
        version=version,
        artifact_type=Artifact,
        run=run_eval_pipeline_step,
        build_config=build_config,
        runtime_args={"accelerator": accelerator},
    )


def main() -> None:
    step = eval_step("qwen3-1.7b", "smoke", version="2026.07.19")
    StepRunner().run([step.lower()])


if __name__ == "__main__":
    main()
