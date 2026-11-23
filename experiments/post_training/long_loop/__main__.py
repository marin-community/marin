# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve and validate a pinned long-loop run spec before any allocation."""

import hashlib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

import click
import draccus
import yaml
from huggingface_hub import hf_hub_download
from iris.cluster.client.job_info import get_job_info
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import graph_handles
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import SkyRLRunConfig
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.secrets import require_secret

from experiments.evaluation.micro210 import Micro210Run
from experiments.post_training.long_loop.eval_runtime.judge import JudgedEvaluation
from experiments.post_training.long_loop.launch_validation import validate_launch
from experiments.post_training.long_loop.nvl72 import PROFILES, NVL72RunSpec, run_from_spec, validate_model_geometry
from experiments.post_training.long_loop.nvl72_evaluation import NVL72EvaluationSpec, evaluation_from_spec
from experiments.post_training.long_loop.recipe_bundle import require_clean_revision, runtime_source_manifest
from experiments.post_training.long_loop.run_spec import resolve_spec, stage_blockers
from experiments.post_training.long_loop.spec_pipeline import build_spec_graph


def validate_graph(
    terminal: ArtifactStep,
    directory: Path,
    artifact_prefix: str,
    *,
    model_config: bytes | None = None,
    check_vocabulary: bool = False,
) -> None:
    rows = []
    for step in graph_handles([terminal]):
        if step.adopt_source:
            rows.append({"stage": step.name, "status": "adopted", "source": step.adopt_source})
            continue
        context = StepContext.for_run(
            step.path(artifact_prefix), artifact_prefix, runtime_args=step.runtime_args, deps=step.deps
        )
        # Future checkpoint selection is unavailable before training. Use the
        # builder's identity branch, retaining actual runtime config and paths.
        config = step.build_config(replace(context, is_fingerprint=True))
        launch = config.skyrl if isinstance(config, (JudgedEvaluation, Micro210Run)) else config
        row = {"stage": step.name, "status": "constructed; not a SkyRL launch"}
        if isinstance(launch, SkyRLRunConfig):
            digest = hashlib.sha256(launch.launch_config_yaml.encode()).hexdigest()
            row = {
                "stage": step.name,
                "status": "strictly composed; future checkpoint identity unresolved",
                **validate_launch(
                    launch.launch_config_yaml,
                    directory / digest,
                    model_config=model_config,
                    check_vocabulary=check_vocabulary,
                ),
            }
        rows.append(row)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "graph.json").write_text(json.dumps(rows, indent=2))


@click.command(help=__doc__)
@click.option("--spec", required=True, help="Pinned local run spec or regional S3 NVL72 spec URI.")
@click.option("--smoke", is_flag=True, help="Apply only the spec's explicit scale overrides.")
@click.option("--through", type=click.Choice(["sft", "rl", "eval", "archive"]), default="archive", show_default=True)
@click.option("--validation-output", type=click.Path(path_type=Path), required=True)
@click.option("--i-know-i-am-using-a-deprecated-config-format", is_flag=True)
@click.option("--i-know-i-am-breaking-experimental-controls", is_flag=True)
@click.option(
    "--experimental-controls-reason", help="Decision recorded in the archive for nonstandard allocations/deadlines."
)
@rl_build_options
def main(
    spec: str,
    smoke: bool,
    through: str,
    validation_output: Path,
    i_know_i_am_using_a_deprecated_config_format: bool,
    i_know_i_am_breaking_experimental_controls: bool,
    experimental_controls_reason: str | None,
) -> ArtifactStep:
    configure_coreweave_s3()
    document = yaml.safe_load(StoragePath(spec).read_bytes())
    if document.get("format") == "nvl72_eval":
        if through != "eval" or smoke:
            raise click.UsageError("An NVL72 evaluation requires --through eval without --smoke")
        evaluation = draccus.decode(NVL72EvaluationSpec, document)
        if not evaluation.output_prefix.startswith("s3://marin-us-east-08a/"):
            raise ValueError("NVL72 outputs must use regional S3 in us-east-08a")
        if not evaluation.model_uri.startswith("s3://marin-us-east-08a/"):
            raise ValueError("NVL72 evaluation requires a regional model")
        os.environ["MARIN_PREFIX"] = evaluation.output_prefix
        if get_job_info() is None:
            require_clean_revision(Path(__file__).resolve().parents[3], evaluation.marin_revision)
        validation_output.mkdir(parents=True, exist_ok=True)
        terminal = evaluation_from_spec(evaluation, validation_output / "launch.yaml")
        model_config = (StoragePath(evaluation.model_uri) / "config.json").read_bytes()
        validate_graph(
            terminal, validation_output / terminal.version, evaluation.output_prefix, model_config=model_config
        )
        (validation_output / "resolved-plan.json").write_text(json.dumps(asdict(evaluation), indent=2))
        return terminal
    if document.get("format") == "nvl72_rl":
        if through != "rl":
            raise click.UsageError("An adopted NVL72 RL input requires --through rl")
        nvl72 = draccus.decode(NVL72RunSpec, document)
        if not nvl72.output_prefix.startswith("s3://marin-us-east-08a/"):
            raise ValueError("NVL72 outputs must use regional S3 in us-east-08a")
        os.environ["MARIN_PREFIX"] = nvl72.output_prefix
        if get_job_info() is None:
            require_clean_revision(Path(__file__).resolve().parents[3], nvl72.marin_revision)
        configure_coreweave_s3()
        validation_output.mkdir(parents=True, exist_ok=True)
        fs, model_path = filesystem_for(prefix_join(nvl72.model_uri, "config.json"))
        model_config = fs.cat_file(model_path)
        validate_model_geometry(json.loads(model_config), PROFILES[nvl72.profile])
        terminal = run_from_spec(nvl72, smoke=smoke, config_output=validation_output / "launch.yaml")
        (validation_output / "resolved-plan.json").write_text(
            json.dumps({**asdict(nvl72), "smoke": smoke, "through": through}, indent=2)
        )
        validate_graph(terminal, validation_output / terminal.version, nvl72.output_prefix, model_config=model_config)
        return terminal
    resolved = resolve_spec(
        Path(spec),
        smoke=smoke,
        acknowledge_deprecated_config_format=i_know_i_am_using_a_deprecated_config_format,
        acknowledge_experimental_controls=i_know_i_am_breaking_experimental_controls,
        experimental_controls_reason=experimental_controls_reason,
    )
    os.environ["MARIN_PREFIX"] = resolved["storage"]["output_prefix"]
    os.environ["MARIN_TEMP_PREFIX"] = resolved["storage"]["temporary_prefix"]
    resolved["through"] = through
    resolved["active_stage_blockers"] = stage_blockers(resolved, through)
    # Iris ships the validated workspace bundle without Git metadata.
    if get_job_info() is None:
        require_clean_revision(Path(__file__).resolve().parents[3], resolved["marin_revision"])
    resolved["runtime_source"] = runtime_source_manifest(Path(__file__).resolve().parents[3], resolved["marin_revision"])
    validation_output.mkdir(parents=True, exist_ok=True)
    plan = validation_output / "resolved-plan.json"
    plan.write_text(json.dumps(resolved, indent=2))
    click.echo(json.dumps(resolved, indent=2))
    if resolved["active_stage_blockers"]:
        raise click.ClickException("Run is not admitted: " + "; ".join(resolved["active_stage_blockers"]))
    configure_coreweave_s3()
    require_secret("HF_TOKEN")
    version = resolve_version(resolved["name"], None)
    terminal = build_spec_graph(resolved, version, through=through)
    model_config = None
    if through != "sft" and resolved["adopted_handoff"] is None:
        # SFT preserves the initializer architecture, but its export does not exist at submission.
        initializer = resolved["initializer"]
        model_config = Path(
            hf_hub_download(initializer["repo"], "config.json", revision=initializer["revision"])
        ).read_bytes()
    validate_graph(terminal, validation_output / version, os.environ["MARIN_PREFIX"], model_config=model_config)
    return terminal


if __name__ == "__main__":
    main()
