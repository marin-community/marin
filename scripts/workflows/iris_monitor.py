#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Iris job monitoring CLI: status, wait, and failure-diagnostics collection.

Replaces copy-pasted shell wait loops and diagnostics blocks across Marin GitHub
Actions workflows. Invoke as:

    uv run python scripts/workflows/iris_monitor.py <command> [options]

Commands:
  status   Print the current state of a single job.
  wait     Poll until the job reaches a terminal state.
  collect  Collect failure diagnostics into an output directory.
"""

import json
import logging
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import click

# Scripts in scripts/workflows/ import sibling modules via importlib so the
# package is not installed. We add the directory to sys.path here so that the
# leading-underscore sibling modules are importable without installing anything.
sys.path.insert(0, str(Path(__file__).parent))

from _iris_cli import (
    DiagnosticsRequest,
    IrisJobState,
    IrisJobStatus,
    _iris_flags,
    iris_command,
    job_status,
    wait_for_job,
)
from _iris_diagnostics_coreweave import collect_coreweave_diagnostics
from _iris_diagnostics_gcp import collect_gcp_diagnostics

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parents[2]


# ---------------------------------------------------------------------------
# collect_diagnostics
# ---------------------------------------------------------------------------


def collect_diagnostics(
    request: DiagnosticsRequest,
    *,
    repo_root: Path,
    run: Callable[[list[str]], subprocess.CompletedProcess] | None = None,
) -> Path:
    """Collect Iris controller, job tree, and provider-specific diagnostics.

    Always writes controller-process.log, job-tree.json, and summary.json.
    Provider-specific artifacts (controller-*.log for GCP, kubernetes-pods.json
    for CoreWeave) are best-effort; failures are recorded in summary.json but do
    not abort collection unless no required provider artifact could be written.

    Args:
        request: DiagnosticsRequest describing what to collect and where.
        repo_root: Repository root used to locate the iris binary.
        run: Injectable subprocess callable for testing. Defaults to
            subprocess.run with capture_output=True, text=True, check=False.

    Returns:
        request.output_dir after writing all available artifacts.

    Raises:
        RuntimeError: When no required provider artifact could be collected.
    """
    if run is None:

        def run(cmd: list[str]) -> subprocess.CompletedProcess:
            return subprocess.run(cmd, capture_output=True, text=True, check=False)

    output_dir = request.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_files: list[str] = []
    errors: list[str] = []

    # --- controller-process.log ---
    iris_cmd = [*iris_command(repo_root), *_iris_flags(request.iris_config, request.controller_url)]
    process_log_cmd = [*iris_cmd, "process", "logs", "--max-lines=500"]
    result = run(process_log_cmd)
    process_log_path = output_dir / "controller-process.log"
    process_log_path.write_text(result.stdout + result.stderr)
    all_files.append("controller-process.log")
    if result.returncode != 0:
        errors.append(f"iris process logs failed (exit {result.returncode}): {result.stderr.strip()}")

    # --- job-tree.json ---
    job_tree_cmd = [*iris_cmd, "job", "list", "--json", "--prefix", request.job_id]
    result = run(job_tree_cmd)
    job_tree_path = output_dir / "job-tree.json"
    job_tree_path.write_text(result.stdout or result.stderr or "")
    if result.returncode != 0:
        errors.append(f"iris job list failed (exit {result.returncode}): {result.stderr.strip()}")
    else:
        all_files.append("job-tree.json")

    # --- provider-specific ---
    provider_files: list[str] = []
    required_files: list[str] = ["job-tree.json", "summary.json"]
    provider_errors: list[str] = []

    if request.provider == "gcp":
        if not request.project or not request.controller_label:
            provider_errors.append("GCP diagnostics require --project and --controller-label")
        else:
            gcp_files, gcp_required, gcp_errors = collect_gcp_diagnostics(
                request.job_id,
                output_dir,
                request.project,
                request.controller_label,
                service_account=request.service_account,
                ssh_key=request.ssh_key,
                run=run,
            )
            provider_files.extend(gcp_files)
            required_files.extend(gcp_required)
            provider_errors.extend(gcp_errors)
            all_files.extend(gcp_files)

    elif request.provider == "coreweave":
        if not request.namespace:
            provider_errors.append("CoreWeave diagnostics require --namespace")
        else:
            cw_files, cw_required, cw_errors = collect_coreweave_diagnostics(
                request.job_id,
                output_dir,
                request.namespace,
                kubeconfig=request.kubeconfig,
                run=run,
            )
            provider_files.extend(cw_files)
            required_files.extend(cw_required)
            provider_errors.extend(cw_errors)
            all_files.extend(cw_files)

    errors.extend(provider_errors)

    # --- summary.json ---
    missing_required = [f for f in required_files if f not in [*all_files, "summary.json"]]
    summary = {
        "job_id": request.job_id,
        "provider": request.provider,
        "files": all_files,
        "required_files": required_files,
        "missing_required_files": missing_required,
        "errors": errors,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Fail if no required provider artifact was collected.
    if request.provider == "gcp" and not provider_files:
        raise RuntimeError(f"No GCP controller logs could be collected. Errors: {'; '.join(errors)}")
    if request.provider == "coreweave" and "kubernetes-pods.json" not in all_files:
        raise RuntimeError(f"CoreWeave kubernetes-pods.json could not be collected. Errors: {'; '.join(errors)}")

    return output_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write_github_output(status: IrisJobStatus) -> None:
    """Write job_id, state, and succeeded to $GITHUB_OUTPUT when the variable is set."""
    github_output = os.environ.get("GITHUB_OUTPUT")
    if not github_output:
        return
    succeeded = "true" if status.state == IrisJobState.SUCCEEDED else "false"
    with open(github_output, "a") as fh:
        fh.write(f"job_id={status.job_id}\n")
        fh.write(f"state={status.state}\n")
        fh.write(f"succeeded={succeeded}\n")


@click.group()
@click.option("-v", "--verbose", is_flag=True, default=False, help="Enable verbose logging.")
def cli(verbose: bool) -> None:
    """Iris job monitor: status, wait, and collect commands."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s", stream=sys.stderr)


@cli.command()
@click.option("--job-id", required=True, help="Iris job ID to inspect.")
@click.option(
    "--iris-config",
    default=None,
    type=click.Path(path_type=Path),
    help="Path to iris config file.",
)
@click.option(
    "--controller-url",
    default=None,
    help="Iris controller URL (e.g. http://localhost:PORT). Overrides --iris-config.",
)
@click.option("--prefix", default=None, help="Prefix for iris job list query (defaults to job-id).")
def status(job_id: str, iris_config: Path | None, controller_url: str | None, prefix: str | None) -> None:
    """Print the current state of an Iris job."""
    try:
        s = job_status(
            job_id, iris_config=iris_config, controller_url=controller_url, prefix=prefix, repo_root=_REPO_ROOT
        )
    except LookupError as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)
    except RuntimeError as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)

    click.echo(f"job_id: {s.job_id}")
    click.echo(f"state:  {s.state}")
    if s.error:
        click.echo(f"error:  {s.error}")


@cli.command()
@click.option("--job-id", required=True, help="Iris job ID to wait on.")
@click.option(
    "--iris-config",
    default=None,
    type=click.Path(path_type=Path),
    help="Path to iris config file.",
)
@click.option(
    "--controller-url",
    default=None,
    help="Iris controller URL (e.g. http://localhost:PORT). Overrides --iris-config.",
)
@click.option("--prefix", default=None, help="Prefix for iris job list query (defaults to job-id).")
@click.option("--poll-interval", default=30.0, type=float, help="Seconds between polls.", show_default=True)
@click.option(
    "--timeout",
    default=None,
    type=float,
    help="Maximum seconds to wait. No limit if omitted.",
)
@click.option(
    "--github-output",
    is_flag=True,
    default=False,
    help="Write job_id, state, and succeeded to $GITHUB_OUTPUT on terminal exit.",
)
def wait(
    job_id: str,
    iris_config: Path | None,
    controller_url: str | None,
    prefix: str | None,
    poll_interval: float,
    timeout: float | None,
    github_output: bool,
) -> None:
    """Poll until an Iris job reaches a terminal state.

    Exits 0 on SUCCEEDED. Exits non-zero on failure, cancellation, or timeout.
    SIGINT/SIGTERM stop polling without cancelling the remote job.
    """
    interrupted = False

    def _handle_signal(signum, frame):
        nonlocal interrupted
        interrupted = True
        click.echo(f"\nReceived signal {signum}; stopping poll. Iris job continues running.", err=True)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    click.echo(f"Polling job {job_id!r} every {poll_interval}s ...", err=True)

    try:
        s = wait_for_job(
            job_id,
            iris_config=iris_config,
            controller_url=controller_url,
            prefix=prefix,
            poll_interval=poll_interval,
            timeout=timeout,
            repo_root=_REPO_ROOT,
            sleep=lambda secs: _interruptible_sleep(secs, interrupted_flag=lambda: interrupted),
        )
    except TimeoutError as exc:
        click.echo(str(exc), err=True)
        sys.exit(2)
    except (LookupError, RuntimeError) as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)

    if interrupted:
        click.echo("Polling interrupted by signal.", err=True)
        sys.exit(1)

    if github_output:
        _write_github_output(s)

    click.echo(f"Job {job_id!r} finished with state: {s.state}", err=True)
    if s.error:
        click.echo(f"Error: {s.error}", err=True)

    if s.state != IrisJobState.SUCCEEDED:
        sys.exit(1)


def _interruptible_sleep(seconds: float, *, interrupted_flag: callable) -> None:
    """Sleep for `seconds` but wake up early if interrupted_flag() returns True."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if interrupted_flag():
            return
        time.sleep(min(1.0, deadline - time.monotonic()))


@cli.command()
@click.option("--job-id", required=True, help="Iris job ID to collect diagnostics for.")
@click.option(
    "--iris-config",
    default=None,
    type=click.Path(path_type=Path),
    help="Path to iris config file.",
)
@click.option(
    "--controller-url",
    default=None,
    help="Iris controller URL (e.g. http://localhost:PORT). Overrides --iris-config.",
)
@click.option(
    "--provider",
    required=True,
    type=click.Choice(["gcp", "coreweave"]),
    help="Cloud provider for provider-specific diagnostics.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(path_type=Path),
    help="Directory to write diagnostic files into.",
)
@click.option("--project", default=None, help="GCP project ID (GCP only).")
@click.option(
    "--controller-label",
    default=None,
    help="GCE instance label key identifying controller VMs (GCP only).",
)
@click.option(
    "--service-account",
    default=None,
    help="Service account to impersonate for gcloud SSH (GCP only).",
)
@click.option(
    "--ssh-key",
    default=None,
    type=click.Path(path_type=Path),
    help="Path to SSH key file for gcloud SSH (GCP only).",
)
@click.option("--namespace", default=None, help="Kubernetes namespace (CoreWeave only).")
@click.option(
    "--kubeconfig",
    default=None,
    type=click.Path(path_type=Path),
    help="Path to kubeconfig file (CoreWeave only).",
)
def collect(
    job_id: str,
    iris_config: Path | None,
    controller_url: str | None,
    provider: Literal["gcp", "coreweave"],
    output_dir: Path,
    project: str | None,
    controller_label: str | None,
    service_account: str | None,
    ssh_key: Path | None,
    namespace: str | None,
    kubeconfig: Path | None,
) -> None:
    """Collect failure diagnostics for an Iris job into an output directory.

    Always writes controller-process.log, job-tree.json, and summary.json.
    GCP: additionally SSHes into controller VMs to fetch docker logs.
    CoreWeave: additionally runs kubectl get pods for job task pods.

    Exit non-zero only when no required provider artifact could be collected.
    Optional artifacts missing → still exit 0.
    """
    request = DiagnosticsRequest(
        job_id=job_id,
        output_dir=output_dir,
        iris_config=iris_config,
        controller_url=controller_url,
        provider=provider,
        project=project,
        controller_label=controller_label,
        namespace=namespace,
        service_account=service_account,
        ssh_key=ssh_key,
        kubeconfig=kubeconfig,
    )

    click.echo(f"Collecting diagnostics for job {job_id!r} into {output_dir} ...", err=True)
    try:
        out = collect_diagnostics(request, repo_root=_REPO_ROOT)
        click.echo(f"Diagnostics written to {out}", err=True)
    except RuntimeError as exc:
        click.echo(f"Diagnostics collection failed: {exc}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
