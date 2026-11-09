# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the Snowball plumbing SFT geometry (32k/32 GPU), not target, → GS8 RL → Micro210 loop."""

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import click
from fray.types import GpuConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, materialized_config
from marin.experiment.cli import graph_handles
from marin.rl.cli import _skyrl_executions, rl_build_options
from marin.rl.skyrl import ArtifactHfModel, SkyRLRun, SkyRLRunConfig
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.secrets import require_secret

from experiments.post_training.long_loop.endpoint_probe import endpoint_step
from experiments.post_training.long_loop.eval_runtime.judge import JudgedEvaluation
from experiments.post_training.long_loop.evaluation import micro_inputs, smoke_evaluation
from experiments.post_training.long_loop.launch_validation import validate_launch
from experiments.post_training.long_loop.rl import DEFAULT_SELECTION_URI, SELECTION_PREFIX, smoke_rl_step
from experiments.post_training.long_loop.runtime_timing import report_step, timed
from experiments.post_training.long_loop.serving import checked_serving_release
from experiments.post_training.long_loop.sft import (
    TOKENIZER_REPO,
    TOKENIZER_REVISION,
    read_json,
    sft_export_step,
    sft_steps,
    validate_handoff,
)


def plumbing_loop(
    version: str,
    source_revision: str,
    reuse_sft_version: str | None = None,
    handoff: str = "reuse",
    reuse_export_version: str | None = None,
) -> ArtifactStep:
    """Sequence real artifact bindings, including the live SFT endpoint check."""
    checked_serving_release()
    sft_version = reuse_sft_version or version
    inputs = timed(micro_inputs(sft_version))
    trained, exported = sft_steps(sft_version, source_revision, prerequisites=(inputs,))
    base = trained.deps[0]
    if reuse_export_version:
        exported = sft_export_step(reuse_export_version, source_revision, trained, {})
    if reuse_sft_version:
        if handoff == "reuse":
            validate_handoff(exported.path())
        pinned = []
        for step in ((inputs, base, trained, exported) if handoff == "reuse" else (inputs, base, trained)):
            record = read_json(step.path() + "/.artifact.json")
            if record["name"] != step.name or record["version"] != step.version:
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


def eval_continuation(version: str, rl_version: str, inputs_version: str) -> ArtifactStep:
    checked_serving_release()
    prefix = "s3://marin-us-east-02a/marin/"
    sources = (
        ("long-loop/micro-inputs", inputs_version, None),
        ("users/benfeuer/checkpoints/long-loop-plumbing-rl", rl_version, SkyRLRun),
    )
    adopted = []
    for name, original_version, kind in sources:
        source = prefix + name + "/" + original_version
        record = read_json(source + "/.artifact.json")
        if record["name"] != name or record["version"] != original_version:
            raise ValueError(f"Committed artifact identity differs: {source}")
        if kind is SkyRLRun:
            terminal = read_json(source + "/terminal.json")["result"]
            if terminal["state"] != "succeeded" or terminal["model"]["global_step"] != 3:
                raise ValueError("Eval-only continuation requires committed successful RL step 3")
        options = {"kind": kind} if kind else {}
        adopted.append(
            ArtifactStep.adopt(
                "long-loop/reused-eval-" + name.rsplit("/", 1)[-1],
                version,
                source,
                config={"original_record": record},
                **options,
            )
        )
    inputs, trained = adopted
    evaluated = timed(smoke_evaluation(version, trained, inputs))
    return report_step(version, [inputs, trained, evaluated])


def validate_graph(terminal: ArtifactStep, directory: Path) -> None:
    configure_coreweave_s3()
    rows = []
    prefix = os.environ["MARIN_PREFIX"]
    for step in graph_handles([terminal]):
        if step.adopt_source:
            rows.append({"stage": step.name, "status": "adopted", "source": step.adopt_source})
            continue
        config = materialized_config(step, prefix)
        launch = config.skyrl if isinstance(config, JudgedEvaluation) else config
        row = {"stage": step.name, "status": "constructed; not a SkyRL launch"}
        if isinstance(launch, SkyRLRunConfig):
            digest = hashlib.sha256(launch.launch_config_yaml.encode()).hexdigest()
            row = {
                "stage": step.name,
                "status": "strictly composed",
                **validate_launch(launch.launch_config_yaml, directory / digest),
            }
        rows.append(row)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "graph.json").write_text(json.dumps(rows, indent=2))


@click.command(help=__doc__)
@click.option("--stage", type=click.Choice(["inputs", "loop", "eval"]), default="loop", show_default=True)
@click.option("--validation-output", type=click.Path(path_type=Path), default=Path("/tmp/long-loop-validation"))
@click.option("--reuse-rl-version", help="Committed step-3 RL version for eval-only continuation.")
@click.option("--source-revision", required=True, help="Immutable 40-character Marin worktree commit.")
@click.option("--reuse-sft-version", help="Reuse committed input, base, SFT and export artifacts from this version.")
@click.option(
    "--reuse-export-version", help="Reuse a separately committed export while retaining the original SFT lineage."
)
@click.option("--handoff", type=click.Choice(["reuse", "reexport"]), default="reuse", show_default=True)
@rl_build_options
def main(
    source_revision: str,
    stage: str,
    validation_output: Path,
    reuse_sft_version: str | None,
    handoff: str,
    reuse_export_version: str | None,
    reuse_rl_version: str | None,
) -> ArtifactStep:
    if len(source_revision) != 40 or any(c not in "0123456789abcdef" for c in source_revision):
        raise click.BadParameter("Expected a full Git commit SHA", param_hint="--source-revision")
    if stage == "inputs" and reuse_sft_version:
        raise click.UsageError("--reuse-sft-version applies only to --stage loop")
    if reuse_export_version and (not reuse_sft_version or handoff != "reuse"):
        raise click.UsageError("--reuse-export-version requires --reuse-sft-version and --handoff reuse")
    require_secret("HF_TOKEN")
    version = resolve_version("long-loop/micro-smoke", None)
    if stage == "eval":
        if not reuse_rl_version or not reuse_sft_version:
            raise click.UsageError("Eval requires --reuse-rl-version and --reuse-sft-version for inputs")
        terminal = eval_continuation(version, reuse_rl_version, reuse_sft_version)
        validate_graph(terminal, validation_output / version)
        return terminal
    loop = plumbing_loop(version, source_revision, reuse_sft_version, handoff, reuse_export_version)
    if stage == "loop":
        validate_graph(loop, validation_output / version)
        return loop
    inputs = next(step for step in graph_handles([loop]) if step.name == "long-loop/micro-inputs")
    execution = replace(
        _skyrl_executions([loop])[0],
        coordinator_name=f"long-loop-inputs-{version}",
        coordinator_gpu=GpuConfig(variant="H100", count=1),
    )
    terminal = replace(inputs, runtime_args={**inputs.runtime_args, "skyrl_execution": execution})
    validate_graph(terminal, validation_output / version)
    return terminal


if __name__ == "__main__":
    main()
