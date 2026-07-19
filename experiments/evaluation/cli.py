# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Command-line entry point for the eval launcher.

``uv run python -m experiments.evaluation.cli launch --model qwen3-8b --evals smoke``. Three commands:
``launch`` submits runs (and, by default, waits and mirrors results to Postgres); ``ingest`` rebuilds
the Postgres index from object-store records; ``runs`` queries it.
"""

from __future__ import annotations

import click
from iris.cli.connect import open_iris_client
from marin.evaluation.records import CW_RECORDS_PREFIX, DEFAULT_RECORDS_PREFIX, list_records
from marin.evaluation.results_db import (
    connect_engine,
    ensure_schema,
    fetch_runs,
    resolve_db_config,
    upsert_record,
)
from rigging.config_discovery import find_project_root
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.evaluation.evals import EVALS, SUITES
from experiments.evaluation.hardware import Platform, default_platform
from experiments.evaluation.launch import LaunchSpec, launch_group, plan_runs, records_prefix_for, wait_and_report
from experiments.evaluation.models import MODELS


def _resolve_eval_keys(evals_arg: str) -> tuple[str, ...]:
    keys: tuple[str, ...] = SUITES.get(evals_arg) or tuple(part.strip() for part in evals_arg.split(",") if part.strip())
    if not keys:
        raise click.BadParameter("no evals selected")
    unknown = [key for key in keys if key not in EVALS]
    if unknown:
        raise click.BadParameter(f"unknown eval(s) {unknown}; known: {sorted(EVALS)} or suites {sorted(SUITES)}")
    return keys


def _connect_db_or_fail():
    config = resolve_db_config()
    if config is None:
        raise click.ClickException("eval DB unavailable: set EVAL_DB_PASSWORD or grant Secret Manager access")
    engine = connect_engine(config.instance, config.db, config.user, config.password)
    ensure_schema(engine)
    return engine


def _maybe_engine():
    config = resolve_db_config()
    if config is None:
        click.echo("eval DB unavailable (no password); results will not be mirrored to Postgres")
        return None
    engine = connect_engine(config.instance, config.db, config.user, config.password)
    ensure_schema(engine)
    return engine


def _print_plan(spec: LaunchSpec) -> None:
    click.echo(f"model: {spec.model}  platform: {spec.platform.value}  cluster: {spec.cluster}")
    for plan in plan_runs(spec):
        target = plan.accel.target_cluster or plan.accel.region or spec.cluster
        click.echo(
            f"  eval={plan.eval_key}  location={plan.model.location}  backend={plan.model.backend.value}  "
            f"accel={plan.accel.label}  region_or_cluster={target}  limit={plan.limit}  "
            f"chat_template={plan.model.apply_chat_template}  tasks={[t.name for t in plan.suite.tasks]}  "
            f"records={records_prefix_for(plan.accel, spec)}"
        )


@click.group()
def cli() -> None:
    """Launch and track model evaluations."""


@cli.command()
@click.option("--model", required=True, help="Model registry key (see experiments.evaluation.models.MODELS).")
@click.option("--evals", "evals_arg", default="smoke", help="Suite name (e.g. 'smoke') or comma-separated eval keys.")
@click.option(
    "--platform",
    type=click.Choice([p.value for p in Platform]),
    default=None,
    help="Force tpu or gpu; defaults from the model.",
)
@click.option("--accelerator", default=None, help="Slice override, e.g. 'v6e-8' or 'H100x8'.")
@click.option("--limit", type=int, default=None, help="Override max eval instances per task.")
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
    evals_arg: str,
    platform: str | None,
    accelerator: str | None,
    limit: int | None,
    no_wait: bool,
    dry_run: bool,
    records_prefix: str | None,
    cluster: str,
) -> None:
    """Submit one serve group for MODEL: serve once, run every selected eval, record each one."""
    if model not in MODELS:
        raise click.BadParameter(f"unknown model {model!r}; known: {sorted(MODELS)}")
    model_config = MODELS[model]
    resolved_platform = Platform(platform) if platform else default_platform(model_config)
    spec = LaunchSpec(
        model=model,
        evals=_resolve_eval_keys(evals_arg),
        platform=resolved_platform,
        accelerator=accelerator,
        limit=limit,
        records_prefix=records_prefix,
        cluster=cluster,
    )
    if dry_run:
        _print_plan(spec)
        return
    with open_iris_client(cluster_name=cluster, workspace=find_project_root()) as client:
        group = launch_group(spec, client)
        click.echo(f"submitted group {group.group_id} ({len(group.runs)} evals, one serve) to cluster {cluster!r}")
        for ref in group.runs:
            click.echo(f"  {ref.run_id}  ({group.model_key} / {ref.eval_key})")
        if no_wait:
            return
        wait_and_report([group], _maybe_engine())


@cli.command()
@click.option(
    "--prefix",
    "prefixes",
    multiple=True,
    default=(DEFAULT_RECORDS_PREFIX, CW_RECORDS_PREFIX),
    show_default=True,
    help="Object-store prefix(es) to scan for records; repeatable.",
)
def ingest(prefixes: tuple[str, ...]) -> None:
    """Rebuild the Postgres index from every record under the given prefixes."""
    configure_coreweave_s3()
    engine = _connect_db_or_fail()
    for prefix in prefixes:
        records = list_records(prefix)
        for record in records:
            upsert_record(engine, record)
        click.echo(f"ingested {len(records)} record(s) from {prefix}")


@cli.command()
@click.option("--model", default=None, help="Filter by model name.")
@click.option("--eval", "eval_name", default=None, help="Filter by eval name.")
@click.option("--user", default=None, help="Filter by user.")
@click.option("--status", default=None, help="Filter by status (succeeded/failed/infra_failed).")
@click.option("--limit", type=int, default=200, help="Maximum rows to return.")
def runs(model: str | None, eval_name: str | None, user: str | None, status: str | None, limit: int) -> None:
    """List recorded eval runs, newest first."""
    engine = _connect_db_or_fail()
    rows = fetch_runs(engine, model=model, eval_name=eval_name, user=user, status=status, limit=limit)
    if not rows:
        click.echo("no runs")
        return
    for row in rows:
        click.echo(
            f"{str(row['created_at'])[:19]}  {row['run_id']:<46}  {row['user_name']:<12}  "
            f"{row['model_name']:<20}  {row['eval_name']:<14}  {row['accelerator']:<10}  {row['status']}"
        )


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
