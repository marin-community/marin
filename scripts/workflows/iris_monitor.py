#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Iris job monitoring CLI: status, wait, and failure-diagnostics collection."""

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

import click

_REPO_ROOT = Path(__file__).parents[2]

# SSH command run on each controller VM. Fetches docker state and full iris-controller logs.
_CONTROLLER_SSH_COMMAND = """\
set +e
echo '=== docker ps -a ==='
sudo docker ps -a
echo '=== docker logs iris-controller (last 5000 lines) ==='
sudo docker logs --timestamps --tail 5000 iris-controller 2>&1
"""


class IrisJobState(StrEnum):
    PENDING = "JOB_STATE_PENDING"
    BUILDING = "JOB_STATE_BUILDING"
    RUNNING = "JOB_STATE_RUNNING"
    SUCCEEDED = "JOB_STATE_SUCCEEDED"
    FAILED = "JOB_STATE_FAILED"
    CANCELLED = "JOB_STATE_CANCELLED"


_ACTIVE = {IrisJobState.PENDING, IrisJobState.BUILDING, IrisJobState.RUNNING}


@dataclass(frozen=True)
class IrisJobStatus:
    job_id: str
    state: IrisJobState
    error: str | None


def iris_command(repo_root: Path) -> list[str]:
    venv_iris = repo_root / ".venv" / "bin" / "iris"
    if venv_iris.exists():
        return [str(venv_iris)]
    return ["uv", "run", "--package", "iris", "iris"]


def _iris_flags(iris_config: Path | None, controller_url: str | None) -> list[str]:
    if controller_url is not None:
        return [f"--controller-url={controller_url}"]
    if iris_config is not None:
        return [f"--config={iris_config}"]
    return []


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def job_status(
    job_id: str,
    *,
    iris_config: Path | None,
    repo_root: Path,
    controller_url: str | None = None,
) -> IrisJobStatus:
    """Look up a job by exact job_id via `iris job list --json --prefix <job_id>`."""
    cmd = [
        *iris_command(repo_root),
        *_iris_flags(iris_config, controller_url),
        "job",
        "list",
        "--json",
        "--prefix",
        job_id,
    ]
    result = _run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"iris job list failed (exit {result.returncode}): {result.stderr.strip()}")

    for row in json.loads(result.stdout):
        if row.get("job_id") == job_id:
            return IrisJobStatus(job_id=job_id, state=IrisJobState(row["state"]), error=row.get("error") or None)

    raise LookupError(f"Job not found in iris job list output: {job_id!r}")


def wait_for_job(
    job_id: str,
    *,
    iris_config: Path | None,
    poll_interval: float,
    timeout: float | None,
    repo_root: Path,
    controller_url: str | None = None,
) -> IrisJobStatus:
    """Poll until the job reaches a terminal state. Raises TimeoutError on timeout."""
    start = time.monotonic()
    while True:
        status = job_status(job_id, iris_config=iris_config, repo_root=repo_root, controller_url=controller_url)
        if status.state not in _ACTIVE:
            return status
        if timeout is not None and (time.monotonic() - start) >= timeout:
            raise TimeoutError(f"Timed out waiting for job {job_id!r} after {timeout}s")
        time.sleep(poll_interval)


def _list_controller_instances(project: str, controller_label: str) -> list[tuple[str, str]]:
    result = _run(
        [
            "gcloud",
            "compute",
            "instances",
            "list",
            f"--project={project}",
            f"--filter=labels.{controller_label}=true",
            "--format=csv[no-heading](name,zone)",
        ]
    )
    if result.returncode != 0:
        return []
    instances = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",", 1)]
        if len(parts) == 2 and parts[0]:
            instances.append((parts[0], parts[1]))
    return instances


def _fetch_controller_log(
    name: str,
    zone: str,
    project: str,
    output_path: Path,
    *,
    service_account: str | None,
    ssh_key: Path | None,
) -> str | None:
    cmd = [
        "gcloud",
        "compute",
        "ssh",
        name,
        f"--project={project}",
        f"--zone={zone}",
        "--quiet",
        f"--command={_CONTROLLER_SSH_COMMAND}",
    ]
    if service_account:
        cmd.append(f"--impersonate-service-account={service_account}")
    if ssh_key:
        cmd.append(f"--ssh-key-file={ssh_key}")
    result = _run(cmd)
    output_path.write_text(result.stdout + result.stderr)
    if result.returncode != 0:
        return f"SSH to {name} ({zone}) failed with exit {result.returncode}"
    return None


def _collect_gcp(
    output_dir: Path,
    project: str,
    controller_label: str,
    *,
    service_account: str | None,
    ssh_key: Path | None,
) -> tuple[list[str], list[str]]:
    """Return (files_written, errors). Each controller log is also a required artifact."""
    instances = _list_controller_instances(project, controller_label)
    if not instances:
        return [], [f"gcloud found no instances with label {controller_label}=true in project {project}"]

    written: list[str] = []
    errors: list[str] = []
    for name, zone in instances:
        filename = f"controller-{name}.log"
        error = _fetch_controller_log(
            name, zone, project, output_dir / filename, service_account=service_account, ssh_key=ssh_key
        )
        if error:
            errors.append(error)
        else:
            written.append(filename)
    return written, errors


def _collect_coreweave(output_dir: Path, job_id: str, namespace: str, kubeconfig: Path | None) -> tuple[bool, list[str]]:
    """Return (wrote_pods_json, errors). Kubernetes label values cap at 63 chars and disallow underscores."""
    label = job_id.lstrip("/").replace("_", "-")[:63]
    cmd = ["kubectl"]
    if kubeconfig:
        cmd += [f"--kubeconfig={kubeconfig}"]
    cmd += ["-n", namespace, "get", "pods", f"-l=iris.job_id={label}", "-o", "json"]

    result = _run(cmd)
    pods_path = output_dir / "kubernetes-pods.json"
    if result.returncode != 0:
        # Write whatever we got so the artifact upload has something.
        pods_path.write_text(result.stdout or result.stderr or "")
        return False, [f"kubectl get pods failed (exit {result.returncode}): {result.stderr.strip()}"]
    pods_path.write_text(result.stdout)
    return True, []


def collect_diagnostics(
    job_id: str,
    output_dir: Path,
    provider: Literal["gcp", "coreweave"],
    *,
    iris_config: Path | None,
    controller_url: str | None,
    project: str | None,
    controller_label: str | None,
    service_account: str | None,
    ssh_key: Path | None,
    namespace: str | None,
    kubeconfig: Path | None,
    repo_root: Path,
) -> Path:
    """Collect Iris controller, job tree, and provider-specific diagnostics into output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    iris_cmd = [*iris_command(repo_root), *_iris_flags(iris_config, controller_url)]
    files: list[str] = []
    errors: list[str] = []

    process_log = _run([*iris_cmd, "process", "logs", "--max-lines=500"])
    (output_dir / "controller-process.log").write_text(process_log.stdout + process_log.stderr)
    files.append("controller-process.log")
    if process_log.returncode != 0:
        errors.append(f"iris process logs failed (exit {process_log.returncode}): {process_log.stderr.strip()}")

    job_tree = _run([*iris_cmd, "job", "list", "--json", "--prefix", job_id])
    (output_dir / "job-tree.json").write_text(job_tree.stdout or job_tree.stderr or "")
    if job_tree.returncode != 0:
        errors.append(f"iris job list failed (exit {job_tree.returncode}): {job_tree.stderr.strip()}")
    else:
        files.append("job-tree.json")

    if provider == "gcp":
        if not project or not controller_label:
            raise click.UsageError("GCP diagnostics require --project and --controller-label")
        gcp_files, gcp_errors = _collect_gcp(
            output_dir, project, controller_label, service_account=service_account, ssh_key=ssh_key
        )
        files.extend(gcp_files)
        errors.extend(gcp_errors)
        if not gcp_files:
            _write_summary(output_dir, job_id, provider, files, errors)
            raise RuntimeError(f"No GCP controller logs could be collected. Errors: {'; '.join(errors)}")
    else:
        if not namespace:
            raise click.UsageError("CoreWeave diagnostics require --namespace")
        wrote, cw_errors = _collect_coreweave(output_dir, job_id, namespace, kubeconfig)
        if wrote:
            files.append("kubernetes-pods.json")
        errors.extend(cw_errors)
        if not wrote:
            _write_summary(output_dir, job_id, provider, files, errors)
            raise RuntimeError(f"CoreWeave kubernetes-pods.json could not be collected. Errors: {'; '.join(errors)}")

    _write_summary(output_dir, job_id, provider, files, errors)
    return output_dir


def _write_summary(output_dir: Path, job_id: str, provider: str, files: list[str], errors: list[str]) -> None:
    summary = {"job_id": job_id, "provider": provider, "files": files, "errors": errors}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))


@click.group()
def cli() -> None:
    """Iris job monitor: status, wait, and collect commands."""


@cli.command()
@click.option("--job-id", required=True, help="Iris job ID to inspect.")
@click.option("--iris-config", default=None, type=click.Path(path_type=Path), help="Path to iris config file.")
@click.option(
    "--controller-url", default=None, help="Iris controller URL (e.g. http://localhost:PORT). Overrides --iris-config."
)
def status(job_id: str, iris_config: Path | None, controller_url: str | None) -> None:
    """Print the current state of an Iris job."""
    s = job_status(job_id, iris_config=iris_config, controller_url=controller_url, repo_root=_REPO_ROOT)
    click.echo(f"job_id: {s.job_id}")
    click.echo(f"state:  {s.state}")
    if s.error:
        click.echo(f"error:  {s.error}")


@cli.command()
@click.option("--job-id", required=True, help="Iris job ID to wait on.")
@click.option("--iris-config", default=None, type=click.Path(path_type=Path), help="Path to iris config file.")
@click.option(
    "--controller-url", default=None, help="Iris controller URL (e.g. http://localhost:PORT). Overrides --iris-config."
)
@click.option("--poll-interval", default=30.0, type=float, help="Seconds between polls.", show_default=True)
@click.option("--timeout", default=None, type=float, help="Maximum seconds to wait. No limit if omitted.")
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
    poll_interval: float,
    timeout: float | None,
    github_output: bool,
) -> None:
    """Poll until an Iris job reaches a terminal state. Exit non-zero unless SUCCEEDED."""
    click.echo(f"Polling job {job_id!r} every {poll_interval}s ...", err=True)
    s = wait_for_job(
        job_id,
        iris_config=iris_config,
        controller_url=controller_url,
        poll_interval=poll_interval,
        timeout=timeout,
        repo_root=_REPO_ROOT,
    )

    if github_output and (path := os.environ.get("GITHUB_OUTPUT")):
        succeeded = "true" if s.state == IrisJobState.SUCCEEDED else "false"
        with open(path, "a") as fh:
            fh.write(f"job_id={s.job_id}\nstate={s.state}\nsucceeded={succeeded}\n")

    click.echo(f"Job {job_id!r} finished with state: {s.state}", err=True)
    if s.error:
        click.echo(f"Error: {s.error}", err=True)
    if s.state != IrisJobState.SUCCEEDED:
        sys.exit(1)


@cli.command()
@click.option("--job-id", required=True, help="Iris job ID to collect diagnostics for.")
@click.option("--iris-config", default=None, type=click.Path(path_type=Path), help="Path to iris config file.")
@click.option(
    "--controller-url", default=None, help="Iris controller URL (e.g. http://localhost:PORT). Overrides --iris-config."
)
@click.option(
    "--provider",
    required=True,
    type=click.Choice(["gcp", "coreweave"]),
    help="Cloud provider for provider-specific diagnostics.",
)
@click.option(
    "--output-dir", required=True, type=click.Path(path_type=Path), help="Directory to write diagnostic files into."
)
@click.option("--project", default=None, help="GCP project ID (GCP only).")
@click.option("--controller-label", default=None, help="GCE instance label key identifying controller VMs (GCP only).")
@click.option("--service-account", default=None, help="Service account to impersonate for gcloud SSH (GCP only).")
@click.option(
    "--ssh-key", default=None, type=click.Path(path_type=Path), help="Path to SSH key file for gcloud SSH (GCP only)."
)
@click.option("--namespace", default=None, help="Kubernetes namespace (CoreWeave only).")
@click.option(
    "--kubeconfig", default=None, type=click.Path(path_type=Path), help="Path to kubeconfig file (CoreWeave only)."
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
    """Collect failure diagnostics for an Iris job into an output directory."""
    click.echo(f"Collecting diagnostics for job {job_id!r} into {output_dir} ...", err=True)
    out = collect_diagnostics(
        job_id,
        output_dir,
        provider,
        iris_config=iris_config,
        controller_url=controller_url,
        project=project,
        controller_label=controller_label,
        service_account=service_account,
        ssh_key=ssh_key,
        namespace=namespace,
        kubeconfig=kubeconfig,
        repo_root=_REPO_ROOT,
    )
    click.echo(f"Diagnostics written to {out}", err=True)


if __name__ == "__main__":
    cli()
