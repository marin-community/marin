# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared CoreWeave dev-pod plumbing behind the ``dev_gpu`` and ``dev_cpu`` tools.

Everything here is tool-neutral: neither the holder command, the kubectl
builders nor the error messages may mention a device kind, so a change to one
tool cannot silently change the other tool's behavior or messages.
"""

import json
import logging
import os
import shlex
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType

import click
from iris.client.client import IrisClient, Job
from iris.cluster.backends.k8s.tasks import _LABEL_TASK_ID, _sanitize_label_value
from iris.cluster.composer import provider_bundle
from iris.cluster.config import IrisClusterConfig, load_config
from iris.cluster.types import JobName
from iris.resources.state import TERMINAL_JOB_STATES, JobState, TaskState
from iris.rpc import job_pb2

logger = logging.getLogger(__name__)

# The holder only sleeps. Exiting 0 on SIGTERM/SIGINT makes a release or pod shutdown end it at once
# with a clean exit status; Python's default SIGTERM handling would kill it by signal and record a failure.
HOLDER_COMMAND = (
    "import signal, sys, time; "
    "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0)); "
    "signal.signal(signal.SIGINT, lambda *_: sys.exit(0)); "
    "print('iris dev holder ready', flush=True); "
    "time.sleep(365 * 24 * 60 * 60)"
)

TASK_CONTAINER = "task"

FAILED_JOB_STATES = TERMINAL_JOB_STATES - {JobState.SUCCEEDED}


class Priority(StrEnum):
    SYSTEM = "system"
    PRODUCTION = "production"
    INTERACTIVE = "interactive"
    BATCH = "batch"


PRIORITY_BANDS = MappingProxyType(
    {
        Priority.SYSTEM: job_pb2.PRIORITY_BAND_SYSTEM,
        Priority.PRODUCTION: job_pb2.PRIORITY_BAND_PRODUCTION,
        Priority.INTERACTIVE: job_pb2.PRIORITY_BAND_INTERACTIVE,
        Priority.BATCH: job_pb2.PRIORITY_BAND_BATCH,
    }
)


@dataclass(frozen=True)
class PodRef:
    """The k8s pod backing a dev session."""

    namespace: str
    pod_name: str
    container: str = TASK_CONTAINER


@dataclass(frozen=True)
class CoreweaveTarget:
    """Cluster-level kubectl target. Empty kubeconfig_path => kubectl default resolution;
    empty kube_context => the file's current-context."""

    namespace: str
    kubeconfig_path: str
    kube_context: str = ""


def require_coreweave_platform(config: IrisClusterConfig) -> CoreweaveTarget:
    """Resolve the kubectl target for a CoreWeave cluster, or fail fast.

    Inverts dev_tpu.py's GCP gate: these tools only work against
    CoreWeave/Kubernetes-backed clusters.
    """
    if config.platform.platform_kind() != "coreweave":
        raise click.ClickException(
            "This tool requires a CoreWeave/Kubernetes-backed cluster. For GCP TPU clusters use scripts/iris/dev_tpu.py."
        )
    # Namespace must come from kubernetes_provider: that is where Iris actually
    # creates and lists task pods (defaulting to "iris" when unset).
    # Kubeconfig comes from platform.coreweave: that is the operator's laptop
    # kubeconfig (kubernetes_provider.kubeconfig is the in-cluster controller's,
    # empty => in-cluster auth, which is wrong for a CLI running off-cluster).
    kp = config.kubernetes_provider
    namespace = (kp.namespace if kp is not None else "") or "iris"
    cw_kubeconfig = config.platform.coreweave.kubeconfig_path
    kubeconfig_path = os.path.expanduser(cw_kubeconfig) if cw_kubeconfig else ""
    return CoreweaveTarget(
        namespace=namespace,
        kubeconfig_path=kubeconfig_path,
        kube_context=config.platform.coreweave.kube_context,
    )


def pod_label_selector(task_id: str) -> str:
    """k8s label selector matching the task pod Iris created for ``task_id``."""
    return f"{_LABEL_TASK_ID}={_sanitize_label_value(task_id)}"


def kubectl_base(target: CoreweaveTarget) -> list[str]:
    cmd = ["kubectl"]
    if target.kubeconfig_path:
        cmd.append(f"--kubeconfig={target.kubeconfig_path}")
    if target.kube_context:
        cmd.append(f"--context={target.kube_context}")
    cmd.append(f"--namespace={target.namespace}")
    return cmd


def kubectl_get_pods_cmd(target: CoreweaveTarget, selector: str) -> list[str]:
    return [*kubectl_base(target), "get", "pods", "-l", selector, "-o", "json"]


def kubectl_connect_cmd(target: CoreweaveTarget, pod: PodRef) -> list[str]:
    return [*kubectl_base(target), "exec", "-it", pod.pod_name, "-c", pod.container, "--", "bash", "-l"]


def kubectl_exec_cmd(target: CoreweaveTarget, pod: PodRef, command: tuple[str, ...]) -> list[str]:
    return [*kubectl_base(target), "exec", pod.pod_name, "-c", pod.container, "--", *command]


def kubectl_copy_to_cmd(target: CoreweaveTarget, pod: PodRef, local: str, remote: str) -> list[str]:
    return [*kubectl_base(target), "cp", "-c", pod.container, local, f"{pod.pod_name}:{remote}"]


def kubectl_copy_from_cmd(target: CoreweaveTarget, pod: PodRef, remote: str, local: str) -> list[str]:
    return [*kubectl_base(target), "cp", "-c", pod.container, f"{pod.pod_name}:{remote}", local]


def parse_running_pod(pods_json: dict) -> str | None:
    """Return the lexicographically-first Running pod name, or None.

    Pods whose deletion has started still report Running until their containers
    stop; skipping them keeps a retry from resolving the dying pod.
    """
    items = pods_json.get("items", [])
    running = [
        p
        for p in items
        if p.get("status", {}).get("phase") == "Running" and not p.get("metadata", {}).get("deletionTimestamp")
    ]
    if not running:
        return None
    return min(running, key=lambda p: p.get("metadata", {}).get("name", ""))["metadata"]["name"]


def run_logged(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    logger.info("Running command: %s", shlex.join(cmd))
    return subprocess.run(cmd, **kwargs)


def find_workspace_root(start: Path) -> Path | None:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return None


@contextmanager
def controller_client(config_file: str) -> Iterator[IrisClient]:
    iris_config = load_config(config_file)
    controller_address = iris_config.controller_address()
    providers = provider_bundle(iris_config)
    controller = providers.controller
    workspace = find_workspace_root(Path.cwd())
    if not controller_address:
        controller_address = controller.discover_controller(iris_config.controller)
    with controller.tunnel(address=controller_address) as tunneled:
        client = IrisClient.remote(tunneled, workspace=workspace)
        try:
            yield client
        finally:
            client.shutdown()


def state_path(state_dir: Path, session_name: str) -> Path:
    return state_dir / f"{session_name}.json"


def is_job_active(client: IrisClient, job_id: str) -> bool:
    return client.job_state(JobName.from_wire(job_id)) not in TERMINAL_JOB_STATES


def release_holder_job(config_file: str, job_id: str, state_file: Path, *, force: bool) -> None:
    """Terminate a dev holder job, then delete its local session file.

    The file stays when termination fails, so the job id is not lost while the pod may still
    run; ``force`` deletes it anyway for jobs that are already gone.
    """
    try:
        with controller_client(config_file) as client:
            client.cancel_job(JobName.from_wire(job_id))
    except Exception as exc:
        if not force:
            raise click.ClickException(
                f"Failed to terminate holder job {job_id}; the pod may still be running. "
                f"Retry `release`, or pass --force to drop local state anyway (then confirm the job is gone "
                f"with `iris job list`). Error: {exc}"
            ) from exc
        logger.warning("Terminate failed but --force given; deleting local session state anyway.", exc_info=True)
    state_file.unlink(missing_ok=True)


def wait_for_running_tasks(job: Job, *, node_count: int, timeout: float) -> list[str]:
    """Block until every holder task is RUNNING; return their task_ids ordered by task index.

    A gang either lands whole or not at all, so this waits for all ``node_count`` tasks
    rather than shelling into a partially-placed session.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = job.state_only()
        if state in FAILED_JOB_STATES:
            error = job.status().error_message or state.name
            raise click.ClickException(f"Dev pod allocation failed: {error}")
        tasks = job.tasks()
        if len(tasks) == node_count and all(task.status().state is TaskState.RUNNING for task in tasks):
            return [str(task.task_id) for task in sorted(tasks, key=lambda task: task.task_index)]
        time.sleep(5)
    raise click.ClickException(f"Timed out waiting for {node_count} dev task(s) after {int(timeout)}s")


def wait_for_running_pod(target: CoreweaveTarget, task_id: str, *, timeout: float) -> PodRef:
    """Poll kubectl until the task's backing pod is Running."""
    selector = pod_label_selector(task_id)
    logger.info("Polling for a Running pod matching selector %s", selector)
    deadline = time.monotonic() + timeout
    last_err = ""
    while time.monotonic() < deadline:
        result = subprocess.run(kubectl_get_pods_cmd(target, selector), capture_output=True, text=True)
        if result.returncode == 0:
            pod_name = parse_running_pod(json.loads(result.stdout or "{}"))
            if pod_name:
                return PodRef(namespace=target.namespace, pod_name=pod_name, container=TASK_CONTAINER)
        else:
            last_err = result.stderr.strip()
        time.sleep(3)
    raise click.ClickException(
        f"Timed out resolving a Running pod for selector {selector!r} after {int(timeout)}s. {last_err}"
    )
