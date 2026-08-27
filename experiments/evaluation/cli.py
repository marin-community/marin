# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Command-line entry point for the eval launcher.

``uv run python -m experiments.evaluation.cli launch --model qwen3-8b --evals smoke``. Two commands:
``launch`` submits runs and optionally waits for their object-store records; ``backfill-samples``
rebuilds every run's finestore sample archive from its kept ``samples_*.jsonl`` sources.
"""

from __future__ import annotations

from pathlib import Path

import click
from finestore.eval import export_lm_eval_samples
from iris.cli.connect import open_iris_client
from iris.rpc import job_pb2
from iris.rpc.proto_display import PRIORITY_BAND_NAMES, priority_band_name, priority_band_value
from marin.evaluation.harbor.runner import canonical_served_name
from marin.evaluation.hardware import Platform, default_platform
from marin.evaluation.model_config import ModelConfig, load_model_config
from marin.evaluation.records import DEFAULT_SCAN_PREFIXES, list_records
from marin.evaluation.runner import EvaluationBatch, wait_and_report
from rigging.config_discovery import find_project_root
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.evaluation.evals import EVALS, SUITES, EvalchemyDefinition, HarborDefinition
from experiments.evaluation.launch import (
    EVALUATION_CONTROLLER_CLUSTER,
    LaunchSpec,
    launch_group,
    prepare_evaluation_batch,
)
from experiments.evaluation.models import models


def _resolve_eval_keys(evals_arg: str) -> tuple[str, ...]:
    keys: tuple[str, ...] = SUITES.get(evals_arg) or tuple(part.strip() for part in evals_arg.split(",") if part.strip())
    if not keys:
        raise click.BadParameter("no evals selected")
    unknown = [key for key in keys if key not in EVALS]
    if unknown:
        raise click.BadParameter(f"unknown eval(s) {unknown}; known: {sorted(EVALS)} or suites {sorted(SUITES)}")
    return keys


def resolve_model_config(model_key: str | None, config_path: Path | None) -> ModelConfig:
    """Resolve exactly one model registry key or catalog-schema file."""
    if (model_key is None) == (config_path is None):
        raise click.BadParameter(
            "specify exactly one of --model or --model-config",
            param_hint="--model/--model-config",
        )
    if config_path is not None:
        try:
            return load_model_config(config_path)
        except Exception as exc:
            raise click.BadParameter(str(exc), param_hint="--model-config") from exc

    assert model_key is not None
    catalog = models()
    if model_key not in catalog:
        raise click.BadParameter(f"unknown model {model_key!r}; known: {sorted(catalog)}", param_hint="--model")
    return catalog[model_key]


def _print_plan(spec: LaunchSpec, batch: EvaluationBatch) -> None:
    click.echo(
        f"model: {spec.model.name}  platform: {spec.platform.value}  "
        f"controller_cluster={EVALUATION_CONTROLLER_CLUSTER}  "
        f"target_cluster={batch.accelerator.target_cluster or 'none'}  "
        f"priority={priority_band_name(batch.priority_band)}"
    )
    for evaluation in batch.evaluations:
        tasks = [task.name for task in evaluation.identity.eval_ref.tasks]
        click.echo(
            f"  eval={evaluation.identity.eval_ref.name}  location={batch.model.location}  "
            f"backend={batch.model.serve.backend.value}  accel={batch.accelerator.label}  "
            f"region_or_cluster={batch.accelerator.target_cluster or batch.accelerator.region}  "
            f"tasks={tasks}  "
            f"records={batch.records_prefix}"
        )


@click.group()
def cli() -> None:
    """Launch and track model evaluations."""


@cli.command()
@click.option("--model", default=None, help="Model registry key. Mutually exclusive with --model-config.")
@click.option(
    "--model-config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Model catalog YAML or JSON. Mutually exclusive with --model.",
)
@click.option(
    "--evals",
    "evals_arg",
    default=None,
    help="Suite name (e.g. 'smoke') or comma-separated eval keys; defaults to smoke.",
)
@click.option(
    "--evalchemy-config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    multiple=True,
    help="Portable Evalchemy YAML or JSON. Repeatable and additive with --evals and --harbor-config.",
)
@click.option(
    "--harbor-config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    multiple=True,
    help="Harbor JobConfig YAML or JSON. Repeatable and additive with --evals.",
)
@click.option(
    "--platform",
    type=click.Choice([p.value for p in Platform]),
    default=None,
    help="Force tpu or gpu; defaults from the model.",
)
@click.option("--accelerator", default=None, help="Slice override, e.g. 'v6e-8' or 'H100x8'.")
@click.option("--limit", type=int, default=None, help="Override max eval instances per task.")
@click.option(
    "--version",
    "version",
    default=None,
    help="Human version label for this launch, e.g. '2026.07.20' or 'rl-fix-sweep'.",
)
@click.option("--description", default=None, help="Free-text note on why this launch was run.")
@click.option("--no-wait", is_flag=True, help="Submit and return without waiting for results.")
@click.option("--dry-run", is_flag=True, help="Print the resolved plan without submitting.")
@click.option(
    "--records-prefix",
    default=None,
    help="Object-store prefix for run records; defaults to GCS, or CW S3 for CoreWeave-routed runs.",
)
@click.option(
    "--federated_cluster",
    default=None,
    help="Override the federated cluster for GPU serving; defaults to the fleet profile.",
)
@click.option(
    "--priority",
    type=click.Choice(PRIORITY_BAND_NAMES, case_sensitive=False),
    default=None,
    help="Iris priority band for the orchestrator and serve jobs; defaults to inherit.",
)
def launch(
    model: str | None,
    model_config: Path | None,
    evals_arg: str | None,
    evalchemy_config: tuple[Path, ...],
    harbor_config: tuple[Path, ...],
    platform: str | None,
    accelerator: str | None,
    limit: int | None,
    version: str | None,
    description: str | None,
    no_wait: bool,
    dry_run: bool,
    records_prefix: str | None,
    federated_cluster: str | None,
    priority: str | None,
) -> None:
    """Submit one serve group for MODEL: serve once, run every selected eval, record each one."""
    selected_model = resolve_model_config(model, model_config)
    resolved_platform = Platform(platform) if platform else default_platform(selected_model)
    evalchemy_definitions = [
        EvalchemyDefinition(
            name=canonical_served_name(path.stem),
            config_path=path,
        )
        for path in evalchemy_config
    ]
    harbor_definitions = [
        HarborDefinition(
            name=canonical_served_name(path.stem),
            config_path=path,
        )
        for path in harbor_config
    ]
    evals = (
        _resolve_eval_keys(evals_arg)
        if evals_arg is not None
        else (() if evalchemy_definitions or harbor_definitions else _resolve_eval_keys("smoke"))
    )
    spec = LaunchSpec(
        model=selected_model,
        evals=evals,
        evalchemy_definitions=tuple(evalchemy_definitions),
        harbor_definitions=tuple(harbor_definitions),
        platform=resolved_platform,
        accelerator=accelerator,
        limit=limit,
        records_prefix=records_prefix,
        federated_cluster=federated_cluster,
        priority_band=(job_pb2.PRIORITY_BAND_INHERIT if priority is None else priority_band_value(priority)),
        version=version,
        description=description,
    )
    try:
        batch = prepare_evaluation_batch(spec)
    except ValueError as exc:
        param_hint = "--evalchemy-config/--harbor-config" if evalchemy_config or harbor_config else "--evals"
        raise click.BadParameter(str(exc), param_hint=param_hint) from exc
    if dry_run:
        _print_plan(spec, batch)
        return
    with open_iris_client(cluster_name=EVALUATION_CONTROLLER_CLUSTER, workspace=find_project_root()) as client:
        group = launch_group(batch, client)
        click.echo(
            f"submitted group {group.group_id} ({len(group.evaluations)} evals, one serve) "
            f"through cluster {EVALUATION_CONTROLLER_CLUSTER!r}"
        )
        for evaluation in group.evaluations:
            click.echo(f"  {evaluation.run_id}  ({group.model_name} / {evaluation.eval_name})")
        if no_wait:
            return
        wait_and_report([group])


@cli.command("backfill-samples")
@click.option(
    "--prefix",
    "prefixes",
    multiple=True,
    default=DEFAULT_SCAN_PREFIXES,
    show_default=True,
    help="Object-store prefix(es) to scan for records; repeatable.",
)
def backfill_samples(prefixes: tuple[str, ...]) -> None:
    """Rebuild every run's finestore sample archive from its kept ``samples_*.jsonl`` sources."""
    configure_coreweave_s3()
    for prefix in prefixes:
        for record in list_records(prefix):
            written = export_lm_eval_samples(record.results_path)
            click.echo(f"{record.run_id}  {written} sample(s)  {record.results_path}")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
