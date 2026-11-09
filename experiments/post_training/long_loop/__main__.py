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
from experiments.post_training.long_loop.sft import (
    TOKENIZER_REPO,
    TOKENIZER_REVISION,
    read_json,
    sft_export_step,
    sft_steps,
    validate_handoff,
)


def plumbing_loop(
    version: str, source_revision: str, reuse_sft_version: str | None = None, handoff: str = "reuse"
) -> ArtifactStep:
    """Sequence real artifact bindings, including the live SFT endpoint check."""
    sft_version = reuse_sft_version or version
    inputs = timed(micro_inputs(sft_version))
    trained, exported = sft_steps(sft_version, source_revision, prerequisites=(inputs,))
    base = trained.deps[0]
    if reuse_sft_version:
        if handoff == "reuse":
            validate_handoff(exported.path())
        pinned = []
        for step in ((inputs, base, trained, exported) if handoff == "reuse" else (inputs, base, trained)):
            record = read_json(step.path() + "/.artifact.json")
            if record["name"] != step.name or record["version"] != reuse_sft_version:
                raise ValueError(f"Committed artifact identity differs: {step.path()}")
            pinned.append(
                ArtifactStep.adopt(
                    step.name.replace("long-loop/", "long-loop/reused-"),
                    version,
                    step.path(),
                    kind=step.artifact_type,
                    config={"original_record": record},
                )
            )
        inputs, base, trained = pinned[:3]
        if handoff == "reuse":
            exported = pinned[3]
        else:
            model = read_json(trained.path() + "/.artifact.json")["config"]["recipe"]["model"]
            exported = sft_export_step(version, source_revision, trained, model)
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
    return report_step(version, [inputs, base, trained, exported, endpoint, rl, evaluated])


@click.command(help=__doc__)
@click.option("--stage", type=click.Choice(["inputs", "loop"]), default="loop", show_default=True)
@click.option("--source-revision", required=True, help="Immutable 40-character Marin worktree commit.")
@click.option("--reuse-sft-version", help="Reuse committed input, base, SFT and export artifacts from this version.")
@click.option("--handoff", type=click.Choice(["reuse", "reexport"]), default="reuse", show_default=True)
@rl_build_options
def main(source_revision: str, stage: str, reuse_sft_version: str | None, handoff: str) -> ArtifactStep:
    if len(source_revision) != 40 or any(c not in "0123456789abcdef" for c in source_revision):
        raise click.BadParameter("Expected a full Git commit SHA", param_hint="--source-revision")
    if stage == "inputs" and reuse_sft_version:
        raise click.UsageError("--reuse-sft-version applies only to --stage loop")
    require_secret("HF_TOKEN")
    version = resolve_version("long-loop/micro-smoke", None)
    loop = plumbing_loop(version, source_revision, reuse_sft_version, handoff)
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
