#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Iris job monitoring CLI used by GitHub workflows."""

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import click
from iris.cluster.providers.k8s.tasks import _sanitize_label_value
from iris.cluster.types import is_job_finished
from iris.rpc import job_pb2
from rigging.redaction import redact_value

_REPO_ROOT = Path(__file__).parents[2]

JOB_STATE_SUCCEEDED = job_pb2.JobState.Name(job_pb2.JOB_STATE_SUCCEEDED)

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


@dataclass(frozen=True)
class K8sPodStatus:
    name: str
    phase: str
    ready: bool
    deleting: bool


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
    """Poll until the job reaches a terminal state. Raises TimeoutError if `timeout` elapses."""
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


def _redact_pod_doc(value: object) -> object:
    """Redact a parsed Kubernetes pod document via the shared rigging redactor.

    Kubernetes encodes container env as ``[{"name": ..., "value": ...}, ...]``,
    which hides the env-var name from a generic dict redactor. We lift each
    entry's value into a single-key ``{name: value}`` dict so name-based
    sensitivity matching applies, then write the redacted value back.
    """
    if isinstance(value, list):
        return [_redact_pod_doc(item) for item in value]
    if not isinstance(value, dict):
        return redact_value(value)
    return {
        key: _redact_env_array(val) if key == "env" and isinstance(val, list) else _redact_pod_doc(val)
        for key, val in value.items()
    }


def _redact_env_array(env_list: list) -> list:
    """Redact a Kubernetes container env array.

    Each ``{"name": X, "value": Y}`` entry is lifted into ``{X: Y}`` so the
    shared redactor matches by env-var name. JSON-encoded values (e.g.
    ``IRIS_JOB_ENV``) are parsed first so nested secrets are caught, then
    re-serialized. ``valueFrom``-only entries and malformed shapes flow
    through the general dict walker unchanged.
    """
    out = []
    for entry in env_list:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str) or "value" not in entry:
            out.append(_redact_pod_doc(entry))
            continue
        name = entry["name"]
        raw = entry["value"]
        parsed = raw
        if isinstance(raw, str) and raw.lstrip().startswith(("{", "[")):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                pass
        redacted = redact_value({name: parsed})[name]
        if isinstance(redacted, (dict, list)):
            redacted = json.dumps(redacted, separators=(",", ":"))
        new_entry = {key: _redact_pod_doc(val) for key, val in entry.items() if key != "value"}
        new_entry["value"] = redacted
        out.append(new_entry)
    return out


def _kubectl_dump(
    cmd: list[str],
    output_path: Path,
    description: str,
) -> str | None:
    result = _run(cmd)
    output_path.write_text(result.stdout or result.stderr or "")
    if result.returncode != 0:
        return f"{description} failed (exit {result.returncode}): {result.stderr.strip()}"
    return None


def _kubectl_pod_json_dump(
    cmd: list[str],
    output_path: Path,
    description: str,
) -> str | None:
    result = _run(cmd)
    if result.returncode != 0:
        output_path.write_text(result.stderr)
        return f"{description} failed (exit {result.returncode}): {result.stderr.strip()}"

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        output_path.write_text("")
        return f"{description} returned invalid JSON: {exc}"

    output_path.write_text(json.dumps(_redact_pod_doc(data), indent=2) + "\n")
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
    written: list[str] = []
    errors: list[str] = []
    kctl = _kubectl(kubeconfig)

    # Match the controller's iris.job_id label sanitization, or the selector misses everything.
    label = _sanitize_label_value(job_id.lstrip("/"))[:63]
    pods_path = output_dir / "kubernetes-pods.json"
    err = _kubectl_pod_json_dump(
        [*kctl, "-n", namespace, "get", "pods", f"-l=iris.job_id={label}", "-o", "json"],
        pods_path,
        "kubectl get pods (job)",
    )
    if err:
        errors.append(err)
    else:
        written.append("kubernetes-pods.json")

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
    ]:
        err = _kubectl_dump([*kctl, *args], output_dir / fname, desc)
        if err is None:
            written.append(fname)

    err = _kubectl_pod_json_dump(
        [*kctl, "-n", namespace, "get", "pods", "-l", "app=iris-controller", "-o", "json"],
        output_dir / "controller-pods.json",
        "kubectl get controller pods",
    )
    if err is None:
        written.append("controller-pods.json")

    if managed_label:
        list_result = _run([*kctl, "-n", namespace, "get", "pods", "-l", f"{managed_label}=true", "-o", "name"])
        if list_result.returncode == 0:
            for line in list_result.stdout.strip().splitlines():
                if not line:
                    continue
                safe = line.replace("/", "-")
                log_err = _kubectl_dump(
                    [*kctl, "-n", namespace, "logs", line, "--tail=-1", "--all-containers"],
                    output_dir / f"{safe}.log",
                    f"kubectl logs {line}",
                )
                pod_err = _kubectl_pod_json_dump(
                    [*kctl, "-n", namespace, "get", line, "-o", "json"],
                    output_dir / f"{safe}.json",
                    f"kubectl get {line}",
                )
                if log_err is None:
                    written.append(f"{safe}.log")
                else:
                    errors.append(log_err)
                if pod_err is None:
                    written.append(f"{safe}.json")
                else:
                    errors.append(pod_err)
        else:
            errors.append(f"kubectl get pods -l {managed_label}=true failed: {list_result.stderr.strip()}")

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


_REQUIRED_GCP = ("controller-*.log",)
_REQUIRED_COREWEAVE = ("kubernetes-pods.json",)


def _missing_required(provider: str, files: list[str]) -> list[str]:
    if provider == "gcp":
        # `controller-process.log` is from iris RPC, not host SSH — it's present even when every VM was unreachable,
        # which is exactly the case we want to flag.
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


@click.group()
def cli() -> None:
    pass


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


def _pod_ready(pod: dict) -> bool:
    return any(
        condition.get("type") == "Ready" and condition.get("status") == "True"
        for condition in pod.get("status", {}).get("conditions", [])
    )


def _controller_pods_from_json(payload: str) -> list[K8sPodStatus]:
    data = json.loads(payload)
    pods: list[K8sPodStatus] = []
    for pod in data.get("items", []):
        metadata = pod.get("metadata", {})
        status = pod.get("status", {})
        name = metadata.get("name", "")
        if not name:
            continue
        pods.append(
            K8sPodStatus(
                name=name,
                phase=status.get("phase", "Unknown"),
                ready=_pod_ready(pod),
                deleting=metadata.get("deletionTimestamp") is not None,
            )
        )
    return pods


def _settled_controller_pod_name(pods: list[K8sPodStatus]) -> str | None:
    if len(pods) != 1:
        return None
    pod = pods[0]
    if pod.phase != "Running" or not pod.ready or pod.deleting:
        return None
    return pod.name


def _format_controller_pods(pods: list[K8sPodStatus]) -> str:
    if not pods:
        return "(none)"
    return ", ".join(f"{pod.name}:phase={pod.phase},ready={pod.ready},deleting={pod.deleting}" for pod in pods)


def _kubectl_rollout_status(
    namespace: str,
    kubeconfig: Path | None,
    *,
    timeout: float,
) -> None:
    result = _run(
        [
            *_kubectl(kubeconfig),
            "-n",
            namespace,
            "rollout",
            "status",
            "deployment/iris-controller",
            f"--timeout={max(1, int(timeout))}s",
        ]
    )
    if result.stdout:
        click.echo(result.stdout.rstrip(), err=True)
    if result.stderr:
        click.echo(result.stderr.rstrip(), err=True)
    if result.returncode != 0:
        raise RuntimeError(f"kubectl rollout status failed (exit {result.returncode})")


def _controller_pods(
    namespace: str,
    kubeconfig: Path | None,
    *,
    controller_selector: str,
) -> list[K8sPodStatus]:
    result = _run(
        [
            *_kubectl(kubeconfig),
            "-n",
            namespace,
            "get",
            "pods",
            "-l",
            controller_selector,
            "-o",
            "json",
        ]
    )
    if result.returncode != 0:
        raise RuntimeError(f"kubectl get controller pods failed (exit {result.returncode}): {result.stderr.strip()}")
    return _controller_pods_from_json(result.stdout)


def wait_for_settled_coreweave_controller(
    namespace: str,
    kubeconfig: Path | None,
    *,
    controller_selector: str,
    timeout: float,
    poll_interval: float,
) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pods = _controller_pods(namespace, kubeconfig, controller_selector=controller_selector)
        pod_name = _settled_controller_pod_name(pods)
        if pod_name is not None:
            click.echo(f"Controller rollout settled on pod/{pod_name}", err=True)
            return pod_name

        click.echo(f"waiting for one ready controller pod: {_format_controller_pods(pods)}", err=True)
        time.sleep(poll_interval)

    pods = _controller_pods(namespace, kubeconfig, controller_selector=controller_selector)
    raise TimeoutError(f"controller rollout did not settle to one ready pod: {_format_controller_pods(pods)}")


def _terminate_process(proc: subprocess.Popen, *, timeout: float = 5.0) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _health_url(controller_url: str, health_path: str) -> str:
    return controller_url.rstrip("/") + "/" + health_path.lstrip("/")


def _wait_for_controller_health(
    controller_url: str,
    *,
    health_path: str,
    timeout: float,
    poll_interval: float,
    port_forward: subprocess.Popen,
) -> None:
    url = _health_url(controller_url, health_path)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if port_forward.poll() is not None:
            raise RuntimeError(f"kubectl port-forward exited with code {port_forward.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=min(poll_interval, 5.0)) as resp:
                if 200 <= resp.status < 300:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            pass
        time.sleep(poll_interval)
    raise TimeoutError(f"controller {url} never became healthy within {timeout}s")


def _start_coreweave_port_forward(
    namespace: str,
    kubeconfig: Path | None,
    *,
    pod_name: str,
    local_port: int,
    remote_port: int,
    log_path: Path,
) -> subprocess.Popen:
    """Spawn ``kubectl port-forward pod/X local:remote`` detached.

    Uses ``start_new_session=True`` so the kubectl process survives this
    Python script exiting — the workflow's later ``Stop port-forward`` step
    kills it via ``$PF_PID``. The log is opened line-buffered so failure
    diagnostics that ``cp`` it before SIGTERM see complete output.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        *_kubectl(kubeconfig),
        "-n",
        namespace,
        "port-forward",
        f"pod/{pod_name}",
        f"{local_port}:{remote_port}",
    ]
    log_file = log_path.open("a", buffering=1)
    try:
        return subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
    finally:
        # Parent's copy of the fd can be closed once Popen has dup'd it into the child.
        log_file.close()


def open_coreweave_controller_tunnel(
    namespace: str,
    kubeconfig: Path | None,
    *,
    controller_selector: str,
    local_port: int,
    remote_port: int,
    timeout: float,
    poll_interval: float,
    health_path: str,
    log_path: Path,
) -> tuple[str, int]:
    """Wait for the controller rollout to settle, start ``kubectl port-forward``, probe ``/health``.

    Returns ``(controller_url, kubectl_pid)``. The kubectl child runs in a new
    session so it outlives this Python process — workflows hold on to the PID
    and tear it down later via ``kill $PF_PID``.
    """
    _kubectl_rollout_status(namespace, kubeconfig, timeout=timeout)
    pod_name = wait_for_settled_coreweave_controller(
        namespace,
        kubeconfig,
        controller_selector=controller_selector,
        timeout=timeout,
        poll_interval=poll_interval,
    )

    port_forward = _start_coreweave_port_forward(
        namespace,
        kubeconfig,
        pod_name=pod_name,
        local_port=local_port,
        remote_port=remote_port,
        log_path=log_path,
    )
    controller_url = f"http://127.0.0.1:{local_port}"
    try:
        _wait_for_controller_health(
            controller_url,
            health_path=health_path,
            timeout=timeout,
            poll_interval=poll_interval,
            port_forward=port_forward,
        )
    except (RuntimeError, TimeoutError):
        _terminate_process(port_forward)
        raise
    return controller_url, port_forward.pid


@cli.command(name="coreweave-controller")
@click.option("--namespace", required=True, help="Kubernetes namespace containing the Iris controller.")
@click.option(
    "--kubeconfig",
    default=None,
    type=click.Path(path_type=Path),
    help="Path to the CoreWeave kubeconfig. Uses kubectl's default config when omitted.",
)
@click.option("--controller-selector", default="app=iris-controller", show_default=True)
@click.option("--local-port", default=10000, show_default=True, type=int)
@click.option("--remote-port", default=10000, show_default=True, type=int)
@click.option(
    "--timeout",
    default=600.0,
    show_default=True,
    type=float,
    help="Seconds to wait for each controller readiness phase.",
)
@click.option("--poll-interval", default=5.0, show_default=True, type=float)
@click.option("--health-path", default="/health", show_default=True)
@click.option(
    "--log-path",
    required=True,
    type=click.Path(path_type=Path),
    help="Path where kubectl port-forward writes its stdout/stderr.",
)
@click.option("--url-var", default="IRIS_CONTROLLER_URL", show_default=True)
@click.option("--pid-var", default="PF_PID", show_default=True)
@click.option("--log-var", default="PF_LOG", show_default=True)
def coreweave_controller(
    namespace: str,
    kubeconfig: Path | None,
    controller_selector: str,
    local_port: int,
    remote_port: int,
    timeout: float,
    poll_interval: float,
    health_path: str,
    log_path: Path,
    url_var: str,
    pid_var: str,
    log_var: str,
) -> None:
    """Wait for the CoreWeave controller rollout and export a stable local controller URL.

    Kept for the iris-smoke-coreweave workflow, which still tunnels into the
    shared controller from the runner. Once that workflow migrates to running
    its tests as in-cluster iris jobs (see ``--config`` path used by the canary
    in marin-canary-ferry-coreweave.yaml), this subcommand can be removed.
    """
    url, pf_pid = open_coreweave_controller_tunnel(
        namespace,
        kubeconfig,
        controller_selector=controller_selector,
        local_port=local_port,
        remote_port=remote_port,
        timeout=timeout,
        poll_interval=poll_interval,
        health_path=health_path,
        log_path=log_path,
    )
    click.echo(f"Controller healthy on {url} (kubectl pid={pf_pid}, log={log_path})", err=True)

    if path := os.environ.get("GITHUB_ENV"):
        with open(path, "a") as fh:
            fh.write(f"{url_var}={url}\n{pid_var}={pf_pid}\n{log_var}={log_path}\n")


if __name__ == "__main__":
    cli()
