#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Iris job monitoring CLI: status, wait, failure-diagnostics, and CoreWeave port-forward."""

import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import click

# Cross-module imports from iris: the script lives downstream of iris and must
# stay in lockstep with how the controller labels managed pods (otherwise our
# kubectl selectors silently miss everything) and what set of states iris
# considers terminal (otherwise `wait` falls through to a ValueError on real
# terminal states like JOB_STATE_KILLED). Importing rather than mirroring
# means future iris changes propagate automatically.
from iris.cluster.providers.k8s.tasks import _sanitize_label_value
from iris.cluster.types import is_job_finished
from iris.rpc import job_pb2

_REPO_ROOT = Path(__file__).parents[2]

JOB_STATE_SUCCEEDED = job_pb2.JobState.Name(job_pb2.JOB_STATE_SUCCEEDED)

# SSH command run on every Iris-managed VM. Dumps every container's logs (not
# just iris-controller) plus the GCE startup-script and cloud-init journals,
# so we capture worker-side failures and "controller never booted" cases.
_HOST_SSH_COMMAND = """\
set +e
echo '=== docker ps -a ==='
sudo docker ps -a
for cid in $(sudo docker ps -aq); do
  echo "=== docker logs $cid ==="
  sudo docker logs --timestamps --tail 5000 "$cid" 2>&1
done
echo '=== startup script journal ==='
sudo journalctl -u google-startup-scripts.service --no-pager 2>&1 | tail -n 2000
echo '=== cloud-final journal ==='
sudo journalctl -u cloud-final.service --no-pager 2>&1 | tail -n 500
"""


@dataclass(frozen=True)
class IrisJobStatus:
    job_id: str
    state: str  # iris proto state name, e.g. "JOB_STATE_RUNNING"
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
            return IrisJobStatus(job_id=job_id, state=row["state"], error=row.get("error") or None)

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
        if is_job_finished(job_pb2.JobState.Value(status.state)):
            return status
        if timeout is not None and (time.monotonic() - start) >= timeout:
            raise TimeoutError(f"Timed out waiting for job {job_id!r} after {timeout}s")
        time.sleep(poll_interval)


def _list_managed_instances(
    project: str,
    controller_label: str,
    managed_label: str | None,
) -> list[tuple[str, str, str]]:
    """Return (name, zone, role) for every controller and managed worker VM."""
    if managed_label:
        filter_ = f"labels.{managed_label}=true OR labels.{controller_label}=true"
    else:
        filter_ = f"labels.{controller_label}=true"

    result = _run(
        [
            "gcloud",
            "compute",
            "instances",
            "list",
            f"--project={project}",
            f"--filter={filter_}",
            "--format=csv[no-heading](name,zone,labels.list())",
        ]
    )
    if result.returncode != 0:
        return []
    out: list[tuple[str, str, str]] = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",", 2)]
        if len(parts) < 2 or not parts[0]:
            continue
        name, zone = parts[0], parts[1]
        labels = parts[2] if len(parts) == 3 else ""
        role = "controller" if controller_label in labels else "worker"
        out.append((name, zone, role))
    return out


def _fetch_host_log(
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
        f"--command={_HOST_SSH_COMMAND}",
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
    managed_label: str | None,
    *,
    service_account: str | None,
    ssh_key: Path | None,
) -> tuple[list[str], list[str]]:
    """Return (files_written, errors). Each controller log is a required artifact."""
    instances = _list_managed_instances(project, controller_label, managed_label)
    if not instances:
        labels = f"{controller_label}=true"
        if managed_label:
            labels = f"{managed_label}=true OR {controller_label}=true"
        return [], [f"gcloud found no instances with label {labels} in project {project}"]

    written: list[str] = []
    errors: list[str] = []
    for name, zone, role in instances:
        filename = f"{role}-{name}.log"
        error = _fetch_host_log(
            name, zone, project, output_dir / filename, service_account=service_account, ssh_key=ssh_key
        )
        if error:
            errors.append(error)
        else:
            written.append(filename)
    return written, errors


def _kubectl(kubeconfig: Path | None) -> list[str]:
    cmd = ["kubectl"]
    if kubeconfig:
        cmd += [f"--kubeconfig={kubeconfig}"]
    return cmd


def _kubectl_dump(
    cmd: list[str],
    output_path: Path,
    description: str,
) -> str | None:
    """Run kubectl, write combined stdout+stderr to output_path, return error string or None."""
    result = _run(cmd)
    output_path.write_text(result.stdout or result.stderr or "")
    if result.returncode != 0:
        return f"{description} failed (exit {result.returncode}): {result.stderr.strip()}"
    return None


def _collect_coreweave(
    output_dir: Path,
    job_id: str,
    namespace: str,
    kubeconfig: Path | None,
    *,
    managed_label: str | None,
    include_cluster_context: bool,
    iris_cmd: list[str],
) -> tuple[list[str], list[str]]:
    """Collect controller pod logs, worker pod logs, and (optionally) cluster-wide context.

    Returns (files_written, errors). `kubernetes-pods.json` is required.
    """
    written: list[str] = []
    errors: list[str] = []
    kctl = _kubectl(kubeconfig)

    # Required: pod listing for the job. Must use the same sanitization the
    # iris controller uses when writing iris.job_id labels (interior '/' →
    # '.'), or the selector matches nothing.
    label = _sanitize_label_value(job_id.lstrip("/"))[:63]
    pods_path = output_dir / "kubernetes-pods.json"
    err = _kubectl_dump(
        [*kctl, "-n", namespace, "get", "pods", f"-l=iris.job_id={label}", "-o", "json"],
        pods_path,
        "kubectl get pods (job)",
    )
    if err:
        errors.append(err)
    else:
        written.append("kubernetes-pods.json")

    # Controller pod logs / describe (current + previous container).
    for fname, args, desc in [
        (
            "controller.log",
            ["-n", namespace, "logs", "-l", "app=iris-controller", "--tail=-1", "--all-containers"],
            "kubectl logs controller",
        ),
        (
            "controller-previous.log",
            ["-n", namespace, "logs", "-l", "app=iris-controller", "--tail=-1", "--all-containers", "--previous"],
            "kubectl logs controller --previous",
        ),
        (
            "controller-describe.txt",
            ["-n", namespace, "describe", "pod", "-l", "app=iris-controller"],
            "kubectl describe controller",
        ),
    ]:
        err = _kubectl_dump([*kctl, *args], output_dir / fname, desc)
        if err is None:
            written.append(fname)
        # Failures here are best-effort; we always at least attempted to write the file.

    # Per-managed-pod logs and describes. Without a managed label we can't
    # enumerate workers cheaply; skip silently in that case.
    if managed_label:
        list_result = _run([*kctl, "-n", namespace, "get", "pods", "-l", f"{managed_label}=true", "-o", "name"])
        if list_result.returncode == 0:
            for line in list_result.stdout.strip().splitlines():
                if not line:
                    continue
                safe = line.replace("/", "-")
                _kubectl_dump(
                    [*kctl, "-n", namespace, "logs", line, "--tail=-1", "--all-containers"],
                    output_dir / f"{safe}.log",
                    f"kubectl logs {line}",
                )
                _kubectl_dump(
                    [*kctl, "-n", namespace, "describe", line],
                    output_dir / f"{safe}-describe.txt",
                    f"kubectl describe {line}",
                )
                written.append(f"{safe}.log")
                written.append(f"{safe}-describe.txt")
        else:
            errors.append(f"kubectl get pods -l {managed_label}=true failed: {list_result.stderr.strip()}")

    # Recent events scoped to namespace.
    err = _kubectl_dump(
        [*kctl, "-n", namespace, "get", "events", "--sort-by=.lastTimestamp"],
        output_dir / "events.txt",
        "kubectl get events",
    )
    if err is None:
        written.append("events.txt")

    if include_cluster_context:
        for fname, args, desc in [
            (
                "nodepools.txt",
                ["get", "nodepools.compute.coreweave.com", "-A", "-o", "wide"],
                "kubectl get nodepools",
            ),
            (
                "nodepools.yaml",
                ["get", "nodepools.compute.coreweave.com", "-A", "-o", "yaml"],
                "kubectl get nodepools -o yaml",
            ),
            ("nodes.txt", ["get", "nodes", "-o", "wide"], "kubectl get nodes"),
        ]:
            err = _kubectl_dump([*kctl, *args], output_dir / fname, desc)
            if err is None:
                written.append(fname)

        for fname, rpc in [
            ("autoscaler-status.txt", "get-autoscaler-status"),
            ("scheduler-state.txt", "get-scheduler-state"),
        ]:
            result = _run([*iris_cmd, "rpc", "controller", rpc])
            (output_dir / fname).write_text(result.stdout + result.stderr)
            if result.returncode == 0:
                written.append(fname)
            else:
                errors.append(f"iris rpc controller {rpc} failed (exit {result.returncode})")

    return written, errors


# Required artifacts per provider — used by summary.json's missing_required_files
# field so the workflow run can tell at a glance whether collection succeeded.
_REQUIRED_GCP = ("controller-*.log",)  # at least one match
_REQUIRED_COREWEAVE = ("kubernetes-pods.json",)


def _missing_required(provider: str, files: list[str]) -> list[str]:
    if provider == "gcp":
        # At least one SSH-fetched controller log must be present.
        # `controller-process.log` (iris RPC dump) does not satisfy this — it
        # comes from RPC, not from a host SSH, and is present even when every
        # GCE VM was unreachable, which is exactly the case we want to flag.
        if any(f.startswith("controller-") and f.endswith(".log") and f != "controller-process.log" for f in files):
            return []
        return ["controller-*.log"]
    return [f for f in _REQUIRED_COREWEAVE if f not in files]


def collect_diagnostics(
    job_id: str,
    output_dir: Path,
    provider: Literal["gcp", "coreweave"],
    *,
    iris_config: Path | None,
    controller_url: str | None,
    project: str | None,
    controller_label: str | None,
    managed_label: str | None,
    service_account: str | None,
    ssh_key: Path | None,
    namespace: str | None,
    kubeconfig: Path | None,
    include_cluster_context: bool,
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
            output_dir,
            project,
            controller_label,
            managed_label,
            service_account=service_account,
            ssh_key=ssh_key,
        )
        files.extend(gcp_files)
        errors.extend(gcp_errors)
    else:
        if not namespace:
            raise click.UsageError("CoreWeave diagnostics require --namespace")
        cw_files, cw_errors = _collect_coreweave(
            output_dir,
            job_id,
            namespace,
            kubeconfig,
            managed_label=managed_label,
            include_cluster_context=include_cluster_context,
            iris_cmd=iris_cmd,
        )
        files.extend(cw_files)
        errors.extend(cw_errors)

    missing_required = _missing_required(provider, files)
    _write_summary(output_dir, job_id, provider, files, missing_required, errors)
    if missing_required:
        raise RuntimeError(
            f"Required {provider} diagnostics missing: {missing_required}. Errors: {'; '.join(errors) or '(none)'}"
        )
    return output_dir


def _write_summary(
    output_dir: Path,
    job_id: str,
    provider: str,
    files: list[str],
    missing_required: list[str],
    errors: list[str],
) -> None:
    summary = {
        "job_id": job_id,
        "provider": provider,
        "files": files,
        "required_files": list(_REQUIRED_GCP if provider == "gcp" else _REQUIRED_COREWEAVE),
        "missing_required_files": missing_required,
        "errors": errors,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))


def _free_local_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _start_port_forward(
    namespace: str,
    service: str,
    local_port: int,
    target_port: int,
    kubeconfig: Path | None,
) -> subprocess.Popen:
    cmd = [
        *_kubectl(kubeconfig),
        "-n",
        namespace,
        "port-forward",
        f"svc/{service}",
        f"{local_port}:{target_port}",
    ]
    # start_new_session detaches from the parent so the forwarder survives this
    # python process exiting; the workflow step records PF_PID and kills it later.
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def port_forward_until_healthy(
    namespace: str,
    service: str,
    *,
    target_port: int,
    timeout: float,
    poll_interval: float,
    kubeconfig: Path | None,
    rollout_deployment: str | None,
    rollout_timeout: float,
    health_path: str,
) -> tuple[int, int]:
    """Wait for the controller deployment to roll out, then port-forward and probe /health.

    Returns (local_port, pf_pid). Raises TimeoutError if the controller never becomes healthy.
    """
    import urllib.error
    import urllib.request

    kctl = _kubectl(kubeconfig)
    if rollout_deployment:
        rollout = _run(
            [
                *kctl,
                "-n",
                namespace,
                "rollout",
                "status",
                f"deployment/{rollout_deployment}",
                f"--timeout={int(rollout_timeout)}s",
            ]
        )
        if rollout.returncode != 0:
            raise RuntimeError(
                f"deployment/{rollout_deployment} rollout failed: {(rollout.stderr or rollout.stdout).strip()}"
            )

    local_port = _free_local_port()
    proc = _start_port_forward(namespace, service, local_port, target_port, kubeconfig)
    health_url = f"http://localhost:{local_port}{health_path}"

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        # Restart the forwarder if it died (konnectivity flakes after rollout).
        if proc.poll() is not None:
            proc = _start_port_forward(namespace, service, local_port, target_port, kubeconfig)
            time.sleep(poll_interval)
            continue
        try:
            with urllib.request.urlopen(health_url, timeout=poll_interval) as resp:
                if 200 <= resp.status < 300:
                    return local_port, proc.pid
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            pass
        time.sleep(poll_interval)

    proc.terminate()
    raise TimeoutError(f"controller {health_url} never became healthy within {timeout}s")


@click.group()
def cli() -> None:
    """Iris job monitor: status, wait, collect, and port-forward."""


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
        succeeded = "true" if s.state == JOB_STATE_SUCCEEDED else "false"
        with open(path, "a") as fh:
            fh.write(f"job_id={s.job_id}\nstate={s.state}\nsucceeded={succeeded}\n")

    click.echo(f"Job {job_id!r} finished with state: {s.state}", err=True)
    if s.error:
        click.echo(f"Error: {s.error}", err=True)
    if s.state != JOB_STATE_SUCCEEDED:
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
@click.option(
    "--managed-label",
    default=None,
    help=(
        "Label key identifying every managed VM/pod (controllers + workers). When set, GCP diagnostics also "
        "SSH worker VMs and CoreWeave diagnostics also dump per-pod logs."
    ),
)
@click.option("--service-account", default=None, help="Service account to impersonate for gcloud SSH (GCP only).")
@click.option(
    "--ssh-key", default=None, type=click.Path(path_type=Path), help="Path to SSH key file for gcloud SSH (GCP only)."
)
@click.option("--namespace", default=None, help="Kubernetes namespace (CoreWeave only).")
@click.option(
    "--kubeconfig", default=None, type=click.Path(path_type=Path), help="Path to kubeconfig file (CoreWeave only)."
)
@click.option(
    "--include-cluster-context",
    is_flag=True,
    default=False,
    help="CoreWeave: also dump nodepools, nodes, autoscaler-status, and scheduler-state.",
)
def collect(
    job_id: str,
    iris_config: Path | None,
    controller_url: str | None,
    provider: Literal["gcp", "coreweave"],
    output_dir: Path,
    project: str | None,
    controller_label: str | None,
    managed_label: str | None,
    service_account: str | None,
    ssh_key: Path | None,
    namespace: str | None,
    kubeconfig: Path | None,
    include_cluster_context: bool,
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
        managed_label=managed_label,
        service_account=service_account,
        ssh_key=ssh_key,
        namespace=namespace,
        kubeconfig=kubeconfig,
        include_cluster_context=include_cluster_context,
        repo_root=_REPO_ROOT,
    )
    click.echo(f"Diagnostics written to {out}", err=True)


@cli.command(name="port-forward")
@click.option("--namespace", required=True, help="Kubernetes namespace.")
@click.option("--service", required=True, help="Service name to port-forward (without svc/ prefix).")
@click.option("--target-port", default=10000, show_default=True, type=int, help="Service port to forward to.")
@click.option("--kubeconfig", default=None, type=click.Path(path_type=Path), help="Path to kubeconfig file.")
@click.option(
    "--rollout-deployment",
    default="iris-controller",
    show_default=True,
    help="Deployment to wait for before port-forwarding. Pass empty string to skip.",
)
@click.option(
    "--rollout-timeout", default=120.0, show_default=True, type=float, help="Seconds to wait for rollout to settle."
)
@click.option(
    "--timeout", default=300.0, show_default=True, type=float, help="Seconds to wait for /health to return 2xx."
)
@click.option("--poll-interval", default=5.0, show_default=True, type=float, help="Seconds between probes.")
@click.option(
    "--health-path",
    default="/health",
    show_default=True,
    help="HTTP path to probe for readiness on the forwarded service.",
)
@click.option(
    "--url-var",
    default="IRIS_CONTROLLER_URL",
    show_default=True,
    help="$GITHUB_ENV variable name to write the controller URL under.",
)
def port_forward(
    namespace: str,
    service: str,
    target_port: int,
    kubeconfig: Path | None,
    rollout_deployment: str,
    rollout_timeout: float,
    timeout: float,
    poll_interval: float,
    health_path: str,
    url_var: str,
) -> None:
    """Port-forward to an Iris controller service and wait for it to be healthy.

    Spawns a detached `kubectl port-forward` and writes IRIS_CONTROLLER_URL,
    LOCAL_PORT, and PF_PID to $GITHUB_ENV. The forwarder survives this process
    exiting so subsequent workflow steps can reach the controller.
    """
    click.echo(f"Port-forwarding to {service} in {namespace} ...", err=True)
    local_port, pf_pid = port_forward_until_healthy(
        namespace,
        service,
        target_port=target_port,
        timeout=timeout,
        poll_interval=poll_interval,
        kubeconfig=kubeconfig,
        rollout_deployment=rollout_deployment or None,
        rollout_timeout=rollout_timeout,
        health_path=health_path,
    )
    url = f"http://localhost:{local_port}"
    click.echo(f"Controller healthy on {url} (pid={pf_pid})", err=True)

    if path := os.environ.get("GITHUB_ENV"):
        with open(path, "a") as fh:
            fh.write(f"{url_var}={url}\nLOCAL_PORT={local_port}\nPF_PID={pf_pid}\n")


if __name__ == "__main__":
    cli()
