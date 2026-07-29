# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Command-line entry point for the eval launcher.

``uv run python -m experiments.evaluation.cli launch --model qwen3-8b --evals smoke``. Two commands:
``launch`` submits runs and optionally waits for their object-store records; ``backfill-samples``
rewrites every run's per-sample parquet exports from its kept ``samples_*.jsonl`` sources.
"""

from __future__ import annotations

from pathlib import Path

import click
from iris.cli.connect import open_iris_client
from marin.evaluation.harbor.driver_config import load_harbor_job_config
from marin.evaluation.harbor.runner import canonical_served_name
from marin.evaluation.hardware import Platform, default_platform
from marin.evaluation.records import CW_RECORDS_PREFIX, DEFAULT_RECORDS_PREFIX, list_records
from marin.evaluation.runner import LaunchProvenance, wait_and_report
from marin.evaluation.samples import export_lm_eval_samples
from rigging.config_discovery import find_project_root
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.evaluation.evals import EVALS, SUITES
from experiments.evaluation.launch import (
    HarborConfigSelection,
    LaunchSpec,
    build_evaluation_batch,
    launch_group,
)
from experiments.evaluation.models import models

_DRY_RUN_IDENTITY = "dry-run"


def _resolve_eval_keys(evals_arg: str) -> tuple[str, ...]:
    keys: tuple[str, ...] = SUITES.get(evals_arg) or tuple(part.strip() for part in evals_arg.split(",") if part.strip())
    if not keys:
        raise click.BadParameter("no evals selected")
    unknown = [key for key in keys if key not in EVALS]
    if unknown:
        raise click.BadParameter(f"unknown eval(s) {unknown}; known: {sorted(EVALS)} or suites {sorted(SUITES)}")
    return keys


def _print_plan(spec: LaunchSpec) -> None:
    batch = build_evaluation_batch(
        spec,
        LaunchProvenance(
            git_sha=_DRY_RUN_IDENTITY,
            launch_host=_DRY_RUN_IDENTITY,
        ),
        _DRY_RUN_IDENTITY,
    )
    click.echo(f"model: {spec.model}  platform: {spec.platform.value}  cluster: {spec.cluster}")
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
@click.option("--model", required=True, help="Model registry key.")
@click.option(
    "--evals",
    "evals_arg",
    default=None,
    help="Suite name (e.g. 'smoke') or comma-separated eval keys; defaults to smoke.",
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
@click.option("--cluster", default="marin", envvar="IRIS_CLUSTER", help="Named iris cluster to submit to.")
def launch(
    model: str,
    evals_arg: str | None,
    harbor_config: tuple[Path, ...],
    platform: str | None,
    accelerator: str | None,
    limit: int | None,
    version: str | None,
    description: str | None,
    no_wait: bool,
    dry_run: bool,
    records_prefix: str | None,
    cluster: str,
) -> None:
    """Submit one serve group for MODEL: serve once, run every selected eval, record each one."""
    catalog = models()
    if model not in catalog:
        raise click.BadParameter(f"unknown model {model!r}; known: {sorted(catalog)}")
    model_config = catalog[model]
    resolved_platform = Platform(platform) if platform else default_platform(model_config)
    harbor_configs: list[HarborConfigSelection] = []
    for path in harbor_config:
        try:
            config = load_harbor_job_config(path)
        except ValueError as exc:
            raise click.BadParameter(str(exc), param_hint="--harbor-config") from exc
        harbor_configs.append(
            HarborConfigSelection(
                name=canonical_served_name(path.stem),
                config=config,
            )
        )
    evals = (
        _resolve_eval_keys(evals_arg)
        if evals_arg is not None
        else (() if harbor_configs else _resolve_eval_keys("smoke"))
    )
    spec = LaunchSpec(
        model=model,
        evals=evals,
        harbor_configs=tuple(harbor_configs),
        platform=resolved_platform,
        accelerator=accelerator,
        limit=limit,
        records_prefix=records_prefix,
        cluster=cluster,
        version=version,
        description=description,
    )
    if dry_run:
        _print_plan(spec)
        return
    with open_iris_client(cluster_name=cluster, workspace=find_project_root()) as client:
        group = launch_group(spec, client)
        click.echo(
            f"submitted group {group.group_id} ({len(group.evaluations)} evals, one serve) to cluster {cluster!r}"
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
    default=(DEFAULT_RECORDS_PREFIX, CW_RECORDS_PREFIX),
    show_default=True,
    help="Object-store prefix(es) to scan for records; repeatable.",
)
def backfill_samples(prefixes: tuple[str, ...]) -> None:
    """Rewrite every run's per-sample parquets from its kept ``samples_*.jsonl`` sources."""
    configure_coreweave_s3()
    for prefix in prefixes:
        for record in list_records(prefix):
            written = export_lm_eval_samples(record.results_path)
            click.echo(f"{record.run_id}  {len(written)} parquet(s)  {record.results_path}")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
