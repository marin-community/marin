# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Click CLI for the monitoring dispatcher."""

import logging
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import click

from marin.dispatch.agent_adapter import ClaudeCodeAdapter, CodexAdapter
from marin.dispatch.schema import (
    IrisRunConfig,
    MonitoringCollection,
    RayRunConfig,
    RunPointer,
    RunState,
    RunTrack,
    TickEventKind,
)
from marin.dispatch.storage import (
    delete_collection,
    list_collections,
    load_collection,
    load_state,
    save_collection,
    save_state,
)
from marin.dispatch.tick import process_tick

logger = logging.getLogger(__name__)


def _resolve_repo_root() -> Path:
    p = Path.cwd()
    while p != p.parent:
        if (p / ".git").exists():
            return p
        p = p.parent
    raise click.UsageError("Could not find git repository root from current directory")


@click.group("dispatch")
@click.option("--repo-root", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--verbose", "-v", is_flag=True)
@click.pass_context
def cli(ctx: click.Context, repo_root: Path | None, verbose: bool) -> None:
    """Monitoring dispatcher for long-running research threads."""
    ctx.ensure_object(dict)
    ctx.obj["repo_root"] = repo_root or _resolve_repo_root()
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


@cli.command()
@click.option("--name", required=True, help="Unique collection name")
@click.option("--prompt", required=True, help="Operator instructions for the agent")
@click.option("--logbook", required=True, help="Logbook path relative to repo root")
@click.option("--branch", required=True, help="Long-lived branch for this thread")
@click.option("--issue", required=True, type=int, help="GitHub issue number")
@click.pass_context
def register(ctx: click.Context, name: str, prompt: str, logbook: str, branch: str, issue: int) -> None:
    """Register a new monitoring collection."""
    repo_root = ctx.obj["repo_root"]
    now = datetime.now(timezone.utc).isoformat()
    collection = MonitoringCollection(
        name=name,
        prompt=prompt,
        logbook=logbook,
        branch=branch,
        issue=issue,
        created_at=now,
    )
    save_collection(repo_root, collection)
    click.echo(f"Registered collection: {name}")


@cli.command("list")
@click.pass_context
def list_cmd(ctx: click.Context) -> None:
    """List all monitoring collections."""
    repo_root = ctx.obj["repo_root"]
    names = list_collections(repo_root)
    if not names:
        click.echo("No collections registered.")
        return
    for name in names:
        collection = load_collection(repo_root, name)
        states = load_state(repo_root, name)
        status_summary = ", ".join(s.last_status for s in states) if states else "no state"
        paused = " [PAUSED]" if collection.paused else ""
        n_runs = len(collection.runs)
        click.echo(
            f"  {name}: branch={collection.branch} issue=#{collection.issue} "
            f"runs={n_runs} status=[{status_summary}]{paused}"
        )


@cli.command()
@click.argument("name")
@click.pass_context
def show(ctx: click.Context, name: str) -> None:
    """Show details of a monitoring collection."""
    repo_root = ctx.obj["repo_root"]
    collection = load_collection(repo_root, name)
    states = load_state(repo_root, name)

    click.echo(f"Name:      {collection.name}")
    click.echo(f"Branch:    {collection.branch}")
    click.echo(f"Issue:     #{collection.issue}")
    click.echo(f"Logbook:   {collection.logbook}")
    click.echo(f"Paused:    {collection.paused}")
    click.echo(f"Created:   {collection.created_at}")
    click.echo(f"Prompt:    {collection.prompt[:100]}{'...' if len(collection.prompt) > 100 else ''}")
    click.echo(f"Runs ({len(collection.runs)}):")
    for i, rp in enumerate(collection.runs):
        state = states[i] if i < len(states) else None
        job_id = rp.ray.job_id if rp.ray else (rp.iris.job_id if rp.iris else "?")
        status = state.last_status if state else "no state"
        failures = state.consecutive_failures if state else 0
        click.echo(f"  [{i}] {rp.track}: {job_id} — status={status} failures={failures}")


@cli.command()
@click.argument("name")
@click.option("--prompt", default=None, help="Update operator prompt")
@click.option("--paused", type=bool, default=None, help="Pause or unpause")
@click.pass_context
def update(ctx: click.Context, name: str, prompt: str | None, paused: bool | None) -> None:
    """Update a monitoring collection's mutable fields."""
    repo_root = ctx.obj["repo_root"]
    collection = load_collection(repo_root, name)
    changes: dict = {}
    if prompt is not None:
        changes["prompt"] = prompt
    if paused is not None:
        changes["paused"] = paused
    if not changes:
        click.echo("Nothing to update.")
        return
    updated = replace(collection, **changes)
    save_collection(repo_root, updated)
    click.echo(f"Updated collection: {name}")


@cli.command("add-run")
@click.argument("name")
@click.option("--track", required=True, type=click.Choice(["ray", "iris"]))
@click.option("--job-id", required=True)
@click.option("--cluster", default=None, help="Ray cluster name")
@click.option("--experiment", default=None, help="Ray experiment script path")
@click.option("--config", "iris_config", default=None, help="Iris config path")
@click.option("--resubmit-command", default=None, help="Iris resubmit command")
@click.pass_context
def add_run(
    ctx: click.Context,
    name: str,
    track: str,
    job_id: str,
    cluster: str | None,
    experiment: str | None,
    iris_config: str | None,
    resubmit_command: str | None,
) -> None:
    """Add a run pointer to an existing collection."""
    repo_root = ctx.obj["repo_root"]
    collection = load_collection(repo_root, name)

    run_track = RunTrack(track)
    if run_track == RunTrack.RAY:
        if not cluster or not experiment:
            raise click.UsageError("--cluster and --experiment are required for ray runs")
        rp = RunPointer(track=run_track, ray=RayRunConfig(job_id=job_id, cluster=cluster, experiment=experiment))
    else:
        if not iris_config or not resubmit_command:
            raise click.UsageError("--config and --resubmit-command are required for iris runs")
        rp = RunPointer(
            track=run_track, iris=IrisRunConfig(job_id=job_id, config=iris_config, resubmit_command=resubmit_command)
        )

    updated = replace(collection, runs=(*collection.runs, rp))
    save_collection(repo_root, updated)
    click.echo(f"Added {track} run {job_id} to {name} (index {len(collection.runs)})")


@cli.command("remove-run")
@click.argument("name")
@click.option("--job-id", required=True, help="Job ID of the run to remove")
@click.pass_context
def remove_run(ctx: click.Context, name: str, job_id: str) -> None:
    """Remove a run pointer by job ID."""
    repo_root = ctx.obj["repo_root"]
    collection = load_collection(repo_root, name)

    index: int | None = None
    for i, rp in enumerate(collection.runs):
        rp_job_id = rp.ray.job_id if rp.ray else (rp.iris.job_id if rp.iris else None)
        if rp_job_id == job_id:
            index = i
            break

    if index is None:
        raise click.UsageError(f"No run with job-id '{job_id}' in collection {name}")

    runs = list(collection.runs)
    removed = runs.pop(index)
    updated = replace(collection, runs=tuple(runs))
    save_collection(repo_root, updated)

    # Remove the corresponding state entry so indices stay aligned.
    states = load_state(repo_root, name)
    if index < len(states):
        states.pop(index)
        save_state(repo_root, name, states)

    click.echo(f"Removed run ({removed.track}: {job_id}) from {name}")


@cli.command()
@click.option("--collection", default=None, help="Process only this collection")
@click.option("--event-kind", type=click.Choice([e.value for e in TickEventKind]), default="scheduled_poll")
@click.option("--agent", "agent_type", type=click.Choice(["claude-code", "codex"]), default="claude-code")
@click.pass_context
def tick(ctx: click.Context, collection: str | None, event_kind: str, agent_type: str) -> None:
    """Process one monitoring tick."""
    repo_root = ctx.obj["repo_root"]
    kind = TickEventKind(event_kind)

    if agent_type == "claude-code":
        agent = ClaudeCodeAdapter()
    else:
        agent = CodexAdapter()

    names = [collection] if collection else list_collections(repo_root)
    if not names:
        click.echo("No collections to process.")
        return

    for name in names:
        click.echo(f"Processing {name}...")
        outcome = process_tick(name, kind, agent, repo_root)
        click.echo(
            f"  dispatched={outcome.dispatched} succeeded={outcome.succeeded} "
            f"failed={outcome.failed} escalated={outcome.escalated}"
        )


@cli.command("reset-failures")
@click.argument("name")
@click.option("--job-id", default=None, help="Reset only this run; omit to reset all")
@click.pass_context
def reset_failures(ctx: click.Context, name: str, job_id: str | None) -> None:
    """Reset consecutive failure count so dispatching resumes after manual intervention."""
    repo_root = ctx.obj["repo_root"]
    collection = load_collection(repo_root, name)
    states = load_state(repo_root, name)
    while len(states) < len(collection.runs):
        states.append(RunState())

    if job_id is not None:
        index: int | None = None
        for i, rp in enumerate(collection.runs):
            rp_job_id = rp.ray.job_id if rp.ray else (rp.iris.job_id if rp.iris else None)
            if rp_job_id == job_id:
                index = i
                break
        if index is None:
            raise click.UsageError(f"No run with job-id '{job_id}' in collection {name}")
        states[index].consecutive_failures = 0
        states[index].last_error = ""
        click.echo(f"Reset failures for run {job_id} in {name}")
    else:
        for s in states:
            s.consecutive_failures = 0
            s.last_error = ""
        click.echo(f"Reset failures for all runs in {name}")

    save_state(repo_root, name, states)


@cli.command()
@click.argument("name")
@click.pass_context
def delete(ctx: click.Context, name: str) -> None:
    """Delete a monitoring collection."""
    repo_root = ctx.obj["repo_root"]
    delete_collection(repo_root, name)
    click.echo(f"Deleted collection: {name}")
