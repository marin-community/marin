# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the Snowball plumbing SFT geometry (32k/32 GPU), not target, → GS8 RL → Micro210 loop."""

from dataclasses import replace

import click
from fray.types import GpuConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import graph_handles
from marin.rl.cli import _skyrl_executions, rl_build_options
from marin.rl.skyrl import ArtifactHfModel
from rigging.secrets import require_secret

from experiments.post_training.long_loop.endpoint_probe import endpoint_step
from experiments.post_training.long_loop.evaluation import micro_inputs, smoke_evaluation
from experiments.post_training.long_loop.rl import DEFAULT_SELECTION_URI, SELECTION_PREFIX, smoke_rl_step
from experiments.post_training.long_loop.runtime_timing import report_step, timed
from experiments.post_training.long_loop.sft import TOKENIZER_REPO, TOKENIZER_REVISION, sft_steps


def plumbing_loop(version: str, source_revision: str) -> ArtifactStep:
    """Sequence real artifact bindings, including the live SFT endpoint check."""
    inputs = timed(micro_inputs(version))
    trained, exported = sft_steps(version, source_revision, prerequisites=(inputs,))
    endpoint = timed(endpoint_step(version, exported))
    selection = ArtifactStep.adopt(f"{SELECTION_PREFIX}/snowball-100k", "2026.11.04", DEFAULT_SELECTION_URI)
    rl = smoke_rl_step(
        version=version,
        model=ArtifactHfModel(
            step=exported,
            tokenizer_uri=TOKENIZER_REPO,
            tokenizer_revision=TOKENIZER_REVISION,
            relative_path="hf",
        ),
        selection=selection,
    )
    rl = timed(replace(rl, deps=(*rl.deps, endpoint)))
    evaluated = timed(smoke_evaluation(version, rl, inputs))
    return report_step(version, [inputs, trained.deps[0], trained, exported, endpoint, rl, evaluated])


@click.command(help=__doc__)
@click.option("--stage", type=click.Choice(["inputs", "loop"]), default="loop", show_default=True)
@click.option("--source-revision", required=True, help="Immutable 40-character Marin worktree commit.")
@rl_build_options
def main(source_revision: str, stage: str) -> ArtifactStep:
    if len(source_revision) != 40 or any(c not in "0123456789abcdef" for c in source_revision):
        raise click.BadParameter("Expected a full Git commit SHA", param_hint="--source-revision")
    require_secret("HF_TOKEN")
    version = resolve_version("long-loop/micro-smoke", None)
    loop = plumbing_loop(version, source_revision)
    if stage == "loop":
        return loop
    inputs = next(step for step in graph_handles([loop]) if step.name == "long-loop/micro-inputs")
    execution = replace(
        _skyrl_executions([loop])[0],
        coordinator_name=f"long-loop-inputs-{version}",
        coordinator_gpu=GpuConfig(variant="H100", count=1),
    )
    return replace(inputs, runtime_args={**inputs.runtime_args, "skyrl_execution": execution})


if __name__ == "__main__":
    main()
