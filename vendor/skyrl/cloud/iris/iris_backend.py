#!/usr/bin/env python3
"""Derive and submit an Iris GPU job from one validated Hydra launch config.

The resolved document supplies allocation, routing, task environment, storage,
and Ray settings. This module projects those values into the Iris SDK and sends
the same document to every task; it has no user-facing option parser.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
from urllib.parse import unquote, urlparse

import yaml
from iris.client.client import IrisClient, Job
from iris.cluster.constraints import CLUSTER_CONSTRAINT_KEY, Constraint, ConstraintOp, infer_preemptible_constraint
from iris.cluster.platforms.k8s.coreweave_topology import gpu_gang_coscheduling_level
from iris.cluster.types import CoschedulingConfig, ResourceSpec, gpu_device
from iris.resources.state import JobState
from iris.rpc import job_pb2
from omegaconf import DictConfig, OmegaConf

from cloud.iris.launch_config import SubmissionMode, load_launch_config
from cloud.iris.launch import LaunchOutcome
from cloud.iris.export_hf_checkpoint import export_terminal_policy
from cloud.iris.ray_storage import (
    RaySpillBackend,
    resolve_ray_spill_target,
)
from cloud.iris.rl_config_translation import (
    RL_CONFIG_PAYLOAD_ENV,
    RL_CONFIG_TASK_DIR,
    RL_ENTRYPOINTS,
    RLEntrypoint,
)
from marinskyrl.resource_locator import (
    is_cloud_uri,
    join_resource_path,
)
from skyrl_train.rollouts.harbor_settings import EnvironmentType
from cloud.iris.runtime_bundle import build_runtime_bundle
from cloud.iris.ray_ports import RayPort
from marinskyrl.environment_contract import (
    DEBUG_ARTIFACT_DIR_ENV,
    DEBUG_MODE_ENV,
    DebugMode,
    EnvVarManager,
    EnvVarScope,
    wandb_launch_environment,
)
from cloud.iris.runtime_environment import (
    CHECKPOINT_EXPORT_ENTRYPOINT,
    IRIS_TASK_IMAGE,
    MARINSKYRL_ACTIVATION_FILE,
    MARINSKYRL_TASK_ROOT,
    RuntimeProfile,
    task_setup_script,
)

DAYTONA_RL_SECRET_PROJECT = "hai-gcp-models"
DAYTONA_RL_SECRET_NAME = "DAYTONA_RL_API_KEY"
DAYTONA_RL_SECRET_VERSION = "1"


RUNTIME_BUNDLE_INCLUDES = ("**/*",)
AUTOMATIC_RESOURCE_REQUEST = "auto"
MEMORY_RESOURCE = "memory"
DISK_RESOURCE = "ephemeral-storage"
# Leave the remainder of live allocatable RAM and disk to kubelet, daemonsets,
# and filesystem overhead.
NODE_RESOURCE_FRACTION = 0.80


@dataclass(frozen=True)
class RLStoragePaths:
    """Resolved storage paths used for launch diagnostics and retention."""

    checkpoint_root: str
    export_root: str
    trajectory_root: str
    ray_log_root: str
    resume_checkpoint_count: int


def _is_checkpoint_export(args: SimpleNamespace) -> bool:
    return getattr(args, "entrypoint", None) == CHECKPOINT_EXPORT_ENTRYPOINT


def _parse_quantity_to_gib(q: str) -> float:
    """Parse a Kubernetes or Iris byte quantity to GiB."""
    q = q.strip()
    for suf, mult in (("Ki", 2**10), ("Mi", 2**20), ("Gi", 2**30), ("Ti", 2**40), ("Pi", 2**50)):
        if q.endswith(suf):
            return float(q[: -len(suf)]) * mult / 2**30
    for suf, mult in (
        ("KB", 1e3),
        ("MB", 1e6),
        ("GB", 1e9),
        ("TB", 1e12),
        ("PB", 1e15),
        ("k", 1e3),
        ("M", 1e6),
        ("G", 1e9),
        ("T", 1e12),
        ("P", 1e15),
    ):
        if q.endswith(suf):
            return float(q[: -len(suf)]) * mult / 2**30
    return float(q) / 2**30  # plain bytes


@dataclass(frozen=True)
class ResourceQuantities:
    """Per-node memory and disk quantities in GiB."""

    memory_gib: float
    disk_gib: float


@dataclass(frozen=True)
class NodeResourceBudget:
    """Live headroom and policy-capped automatic requests for one node."""

    headroom: ResourceQuantities
    automatic: ResourceQuantities


@dataclass(frozen=True)
class ClusterResourceSnapshot:
    """Available matching nodes observed in one Kubernetes query."""

    context: str
    gpu_variant: str
    gpus_per_node: int
    nodes: tuple[NodeResourceBudget, ...]


@dataclass(frozen=True)
class ResolvedResourceRequests:
    """Per-node memory and disk requests ready for Iris submission."""

    memory: str
    disk: str


class _LauncherTermination(BaseException):
    """A process termination signal converted into supervised job cancellation."""

    def __init__(self, signum: int):
        self.signum = signum
        super().__init__(f"launcher received signal {signum}")


@contextlib.contextmanager
def _supervised_termination_signals() -> Iterator[None]:
    previous = signal.getsignal(signal.SIGTERM)

    def request_termination(signum, _frame) -> None:
        raise _LauncherTermination(signum)

    signal.signal(signal.SIGTERM, request_termination)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def _cancel_iris_job_tree(job: Job, job_id: str, cause: BaseException) -> None:
    print(f"[rl-iris] Cancelling job tree {job_id}...", file=sys.stderr, flush=True)
    try:
        job.cancel()
    except BaseException as cleanup_error:
        cause.add_note(f"Failed to confirm cancellation of Iris job tree {job_id}: {cleanup_error}")
        raise cause from cleanup_error
    print(f"[rl-iris] Cancelled job tree {job_id}.", file=sys.stderr, flush=True)


def supervise_iris_job(job: Job) -> LaunchOutcome:
    """Wait for a terminal state, cancelling the job tree if supervision cannot continue."""
    job_id = str(job.job_id)
    try:
        with _supervised_termination_signals():
            status = job.wait(stream_logs=True, timeout=float("inf"), raise_on_failure=False)
    except (KeyboardInterrupt, _LauncherTermination) as interruption:
        _cancel_iris_job_tree(job, job_id, interruption)
        exit_code = 130 if isinstance(interruption, KeyboardInterrupt) else 128 + interruption.signum
        return LaunchOutcome(job_id=job_id, job_state=JobState.KILLED.value, exit_code=exit_code)
    except BaseException as error:
        _cancel_iris_job_tree(job, job_id, error)
        raise
    return LaunchOutcome(
        job_id=job_id,
        job_state=status.state.value,
        exit_code=0 if status.state is JobState.SUCCEEDED else 1,
    )


def _gpu_resources(gpu_variant: str, gpu_count: int, *, cpu: float, memory: str, disk: str) -> ResourceSpec:
    resources = ResourceSpec(cpu=cpu, memory=memory, disk=disk)
    resources.device = gpu_device(gpu_variant, gpu_count)
    return resources


def _gpu_multinode(gpu_variant: str, gpu_count: int, replicas: int) -> CoschedulingConfig | None:
    if replicas <= 1:
        return None
    level = gpu_gang_coscheduling_level(gpu_variant, gpu_count, replicas)
    return CoschedulingConfig(group_by=level)


def _gpu_constraints(
    resources: job_pb2.ResourceSpecProto,
    *,
    replicas: int,
    target_cluster: str | None,
) -> list[Constraint]:
    constraints = []
    if target_cluster:
        constraints.append(Constraint.create(key=CLUSTER_CONSTRAINT_KEY, op=ConstraintOp.EQ, value=target_cluster))
    inferred = infer_preemptible_constraint(resources, replicas, constraints)
    if inferred is not None:
        constraints.append(inferred)
    return constraints


def _model_path(uri: str) -> str:
    """Resolve a typed model URI into the task runtime's model reference."""
    if is_cloud_uri(uri):
        return uri
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        if parsed.netloc not in ("", "localhost"):
            raise ValueError(f"Model file URI must be local: {uri!r}")
        return unquote(parsed.path)
    return uri


def _iris_submission_state(config_path: Path, config: DictConfig) -> SimpleNamespace:
    """Build the in-memory Iris launch state from one validated Hydra config."""
    raw = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    assert isinstance(raw, dict)
    run = raw["run"]
    runtime = raw["runtime"]
    iris = raw["iris"]
    allocation = iris["allocation"]
    ray = raw["ray"]
    artifacts = raw["artifacts"]
    inputs = raw["inputs"]
    skyrl = raw["skyrl"]
    model = inputs["model"]
    model_path = _model_path(model["uri"])
    trajectory_retention = (skyrl.get("generator") or {}).get("trajectory_retention") or {}
    contents = config_path.read_bytes()
    digest = hashlib.sha256(contents).hexdigest()[:16]
    suffix = config_path.suffix or ".yaml"
    task_config = BundledLaunchConfig(
        task_path=f"{RL_CONFIG_TASK_DIR}/{digest}{suffix}",
        payload=base64.b64encode(contents).decode("ascii"),
    )
    submission = run["submission"]
    args = SimpleNamespace(
        launch_config=str(config_path),
        launch_config_bundle=task_config,
        entrypoint=runtime["entrypoint"],
        model_path=model_path,
        resume_checkpoints_to_keep=int(artifacts["resume_checkpoint_count"]),
        num_nodes=int(allocation["num_nodes"]),
        gpus_per_node=int(allocation["gpus_per_node"]),
        gpu_variant=allocation["gpu_variant"],
        cpu=float(allocation["cpu"]),
        memory=allocation["memory"],
        disk=allocation["disk"],
        ray_spill_dir=ray["spill_dir"],
        ray_spill_backend=RaySpillBackend(ray["spill_backend"]),
        rendezvous_dir=ray["rendezvous_dir"],
        cluster=iris["cluster"],
        cluster_config=iris["cluster_config"],
        source_digest=runtime["source_digest"],
        runtime_profile=RuntimeProfile(runtime["profile"]),
        job_name=iris["job_name"],
        priority=iris["priority"],
        max_retries=int(iris["max_retries"]),
        timeout=int(iris["timeout"]),
        no_wait=submission == SubmissionMode.DETACH,
        dry_run=submission == SubmissionMode.PREPARE,
        target_cluster=iris["target_cluster"],
        parent_cluster_config=iris["parent_cluster_config"],
        container_profile=(
            job_pb2.CONTAINER_PROFILE_UNSPECIFIED
            if iris["container_profile"] is None
            else job_pb2.ContainerProfile.Value(iris["container_profile"])
        ),
        wandb_entity=iris["wandb_entity"],
        config=config,
    )
    args.storage_paths = RLStoragePaths(
        checkpoint_root=artifacts["checkpoint_root"],
        export_root=artifacts["export_root"],
        trajectory_root=trajectory_retention.get("output_path")
        or join_resource_path(artifacts["attempts_root"], "trajectories"),
        ray_log_root=ray["log_dir"],
        resume_checkpoint_count=args.resume_checkpoints_to_keep,
    )
    return args


class IrisBackend:
    """Validate and submit one resolved Hydra launch config."""

    def validate(self, config_path: Path) -> None:
        _iris_submission_state(config_path, load_launch_config(config_path))

    def launch(self, config_path: Path) -> LaunchOutcome:
        config = load_launch_config(config_path)
        args = _iris_submission_state(config_path, config)
        if config.run.submission == SubmissionMode.PREPARE:
            raise ValueError("Prepare mode validates a launch without submitting it")
        with (
            build_runtime_bundle(config.runtime.source_digest) as workspace,
            contextlib.redirect_stdout(sys.stderr),
        ):
            return launch(args, workspace)

    def export_terminal_policy(self, config_path: Path) -> None:
        export_terminal_policy(config_path)


def _pod_resource_request_gib(pod: dict[str, Any], resource: str) -> float:
    """Return the scheduler-effective request for one pod resource."""
    spec = pod.get("spec", {})

    def request(container: dict[str, Any]) -> float:
        quantity = container.get("resources", {}).get("requests", {}).get(resource)
        return _parse_quantity_to_gib(quantity) if isinstance(quantity, str) else 0.0

    application_request = sum(request(container) for container in spec.get("containers", []))
    init_request = max((request(container) for container in spec.get("initContainers", [])), default=0.0)
    overhead = spec.get("overhead", {}).get(resource)
    overhead_request = _parse_quantity_to_gib(overhead) if isinstance(overhead, str) else 0.0
    return max(application_request, init_request) + overhead_request


def _node_is_ready(node: dict[str, Any]) -> bool:
    conditions = node.get("status", {}).get("conditions", [])
    return any(condition.get("type") == "Ready" and condition.get("status") == "True" for condition in conditions)


def _node_resource_budgets(
    items: list[dict[str, Any]],
    *,
    gpu_variant: str,
    gpus_per_node: int,
) -> list[NodeResourceBudget]:
    """Return live headroom and automatic request budgets for matching Ready nodes."""
    pod_requests: dict[str, ResourceQuantities] = {}
    for item in items:
        if item.get("kind") != "Pod" or item.get("status", {}).get("phase") in {"Succeeded", "Failed"}:
            continue
        node_name = item.get("spec", {}).get("nodeName")
        if not isinstance(node_name, str):
            continue
        requested = pod_requests.get(node_name, ResourceQuantities(memory_gib=0.0, disk_gib=0.0))
        pod_requests[node_name] = ResourceQuantities(
            memory_gib=requested.memory_gib + _pod_resource_request_gib(item, MEMORY_RESOURCE),
            disk_gib=requested.disk_gib + _pod_resource_request_gib(item, DISK_RESOURCE),
        )

    available = []
    for item in items:
        if item.get("kind") != "Node" or item.get("spec", {}).get("unschedulable") or not _node_is_ready(item):
            continue
        metadata = item.get("metadata", {})
        allocatable = item.get("status", {}).get("allocatable", {})
        product = metadata.get("labels", {}).get("nvidia.com/gpu.product", "")
        if allocatable.get("nvidia.com/gpu") != str(gpus_per_node) or gpu_variant.lower() not in product.lower():
            continue
        node_name = metadata.get("name")
        if not isinstance(node_name, str):
            continue
        memory_allocatable = _parse_quantity_to_gib(allocatable[MEMORY_RESOURCE])
        disk_allocatable = _parse_quantity_to_gib(allocatable[DISK_RESOURCE])
        requested = pod_requests.get(node_name, ResourceQuantities(memory_gib=0.0, disk_gib=0.0))
        headroom = ResourceQuantities(
            memory_gib=memory_allocatable - requested.memory_gib,
            disk_gib=disk_allocatable - requested.disk_gib,
        )
        available.append(
            NodeResourceBudget(
                headroom=headroom,
                automatic=ResourceQuantities(
                    memory_gib=min(memory_allocatable * NODE_RESOURCE_FRACTION, headroom.memory_gib),
                    disk_gib=min(disk_allocatable * NODE_RESOURCE_FRACTION, headroom.disk_gib),
                ),
            )
        )
    return available


def _inspect_cluster_resources(
    cluster_config_path: str, *, gpu_variant: str, gpus_per_node: int
) -> ClusterResourceSnapshot:
    """Read one live node-and-pod resource snapshot from the selected cluster."""
    cluster_config = _load_cluster_config(cluster_config_path)
    platform = cluster_config.get("platform")
    coreweave = platform.get("coreweave") if isinstance(platform, dict) else None
    kubeconfig = coreweave.get("kubeconfig_path") if isinstance(coreweave, dict) else None
    context = coreweave.get("kube_context") if isinstance(coreweave, dict) else None
    if not isinstance(kubeconfig, str) or not isinstance(context, str):
        raise SystemExit(
            "The selected cluster config needs platform.coreweave.kubeconfig_path and kube_context "
            "to derive automatic memory and disk requests; pass both explicitly."
        )

    try:
        out = subprocess.run(
            [
                "kubectl",
                "--kubeconfig",
                str(Path(kubeconfig).expanduser()),
                "--context",
                context,
                "get",
                "nodes,pods",
                "--all-namespaces",
                "-o",
                "json",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout
    except Exception as exc:  # noqa: BLE001 - convert cluster/tool failures into launch guidance
        raise SystemExit(
            f"Could not inspect {gpus_per_node}x{gpu_variant} nodes in kube context {context!r}: {exc}. "
            "Pass explicit --memory and --disk values."
        ) from exc
    items = json.loads(out)["items"]
    if not isinstance(items, list):
        raise ValueError("kubectl node-and-pod snapshot has a non-list items field")
    nodes = _node_resource_budgets(items, gpu_variant=gpu_variant, gpus_per_node=gpus_per_node)
    return ClusterResourceSnapshot(
        context=context,
        gpu_variant=gpu_variant,
        gpus_per_node=gpus_per_node,
        nodes=tuple(nodes),
    )


def _resource_request_is_automatic(request: str) -> bool:
    return request.strip().lower() == AUTOMATIC_RESOURCE_REQUEST


def _resolve_gang_resource_requests(
    snapshot: ClusterResourceSnapshot,
    *,
    num_nodes: int,
    memory_request: str,
    disk_request: str,
) -> ResolvedResourceRequests:
    """Select one resource pair that fits the requested gang in a live snapshot."""
    automatic_memory = _resource_request_is_automatic(memory_request)
    automatic_disk = _resource_request_is_automatic(disk_request)
    requested_memory_gib = None if automatic_memory else _parse_quantity_to_gib(memory_request)
    requested_disk_gib = None if automatic_disk else _parse_quantity_to_gib(disk_request)
    candidates = [
        resources
        for resources in snapshot.nodes
        if (requested_memory_gib is None or resources.headroom.memory_gib >= requested_memory_gib)
        and (requested_disk_gib is None or resources.headroom.disk_gib >= requested_disk_gib)
    ]
    if len(candidates) < num_nodes:
        raise SystemExit(
            f"Automatic resources need {num_nodes} Ready, schedulable "
            f"{snapshot.gpus_per_node}x{snapshot.gpu_variant} nodes, "
            f"but only {len(candidates)} of {len(snapshot.nodes)} matching nodes in kube context "
            f"{snapshot.context!r} fit the constraints memory={memory_request}, disk={disk_request}; "
            "adjust the resource requests."
        )

    if automatic_memory:
        selected = sorted(candidates, key=lambda resources: resources.automatic.memory_gib, reverse=True)[:num_nodes]
    else:
        selected = sorted(candidates, key=lambda resources: resources.automatic.disk_gib, reverse=True)[:num_nodes]
    memory_gib = int(min(resources.automatic.memory_gib for resources in selected))
    disk_gib = int(min(resources.automatic.disk_gib for resources in selected))
    if (automatic_memory and memory_gib < 1) or (automatic_disk and disk_gib < 1):
        automatic_resources = " and ".join(
            resource for resource, automatic in (("memory", automatic_memory), ("disk", automatic_disk)) if automatic
        )
        raise SystemExit(
            f"No positive automatic {automatic_resources} request fits {num_nodes} "
            f"{snapshot.gpus_per_node}x{snapshot.gpu_variant} nodes "
            f"in kube context {snapshot.context!r}; pass explicit --memory and --disk values."
        )
    resolved_memory = f"{memory_gib}Gi" if automatic_memory else memory_request
    resolved_disk = f"{disk_gib}Gi" if automatic_disk else disk_request
    return ResolvedResourceRequests(memory=resolved_memory, disk=resolved_disk)


def resolve_node_resource_requests(
    cluster_config_path: str,
    *,
    gpu_variant: str,
    gpus_per_node: int,
    num_nodes: int,
    memory_request: str,
    disk_request: str,
) -> ResolvedResourceRequests:
    """Return live admission-aware resource requests for a GPU gang."""
    snapshot = _inspect_cluster_resources(
        cluster_config_path,
        gpu_variant=gpu_variant,
        gpus_per_node=gpus_per_node,
    )
    resolved = _resolve_gang_resource_requests(
        snapshot,
        num_nodes=num_nodes,
        memory_request=memory_request,
        disk_request=disk_request,
    )
    print(
        f"[rl-iris] automatic node resources: largest requests with live headroom on {num_nodes} of "
        f"{len(snapshot.nodes)} matching nodes, capped at {NODE_RESOURCE_FRACTION:.0%} allocatable = "
        f"memory {resolved.memory}, disk {resolved.disk}",
        flush=True,
    )
    return resolved


RL_PYTHON = "python"
SKYRL_HOME = MARINSKYRL_TASK_ROOT
# Iris synchronizes the project and local modules below /app. Setup verifies
# their source identity and installs the independent frozen profile.
APP_DIR = SKYRL_HOME


@dataclass(frozen=True)
class BundledLaunchConfig:
    """Task path and environment payload for one resolved launch config."""

    task_path: str
    payload: str

    def task_environment(self) -> dict[str, str]:
        """Return the environment needed to materialize the config."""
        return {RL_CONFIG_PAYLOAD_ENV: self.payload}


def _resolve_parent_cluster_config(cluster_config: Optional[str]) -> Optional[str]:
    """Path to the PARENT (marin) cluster YAML for federated submission.

    The marin meta-scheduler config (marin.yaml) that owns iris.oa.dev and lists the
    CoreWeave clusters as delegation peers. Defaults to the ``marin.yaml`` sibling of
    ``--cluster-config`` (they live in the same ``lib/iris/config/`` dir); falls back
    to the same search roots as :func:`_resolve_cluster_config_default`.
    """
    if cluster_config:
        sib = Path(cluster_config).with_name("marin.yaml")
        if sib.exists():
            return str(sib)
    rel = "lib/iris/config/marin.yaml"
    for c in (
        Path.home() / "Documents/marin" / rel,
        Path("/Users/benjaminfeuer/Documents/marin") / rel,
        Path(os.environ.get("MARIN_ROOT", "")) / rel,
    ):
        if c.exists():
            return str(c)
    return None


def _load_cluster_config(cluster_config: str) -> dict[str, Any]:
    """Load the selected Iris cluster configuration for launch-time defaults."""
    try:
        with open(cluster_config) as f:
            loaded = yaml.safe_load(f)
    except OSError as exc:
        raise SystemExit(
            f"Could not load --cluster-config {cluster_config!r} to resolve RL launch defaults: {exc}. "
            "Pass an existing cluster config or explicit --cpu and --rendezvous-dir values."
        ) from exc
    if not isinstance(loaded, dict):
        raise SystemExit(f"--cluster-config {cluster_config!r} must contain a YAML mapping.")
    return loaded


def _daytona_rl_api_key_from_secret_manager() -> Optional[str]:
    """The pinned RL Daytona key from Secret Manager, or ``None`` if gcloud is
    missing/denied or the secret is empty. Never logged."""
    command = [
        "gcloud",
        "secrets",
        "versions",
        "access",
        DAYTONA_RL_SECRET_VERSION,
        f"--secret={DAYTONA_RL_SECRET_NAME}",
        f"--project={DAYTONA_RL_SECRET_PROJECT}",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _resolve_daytona_rl_api_key() -> str:
    """Resolve the RL-org Daytona key from the launch environment or Secret Manager.

    ``--secrets-env`` is loaded before this function runs. Keep its RL-specific name at
    the launch boundary, then let the caller expose the resolved value as
    ``DAYTONA_API_KEY`` because that is the name expected by Daytona and Harbor.
    """
    env_key = os.environ.get(DAYTONA_RL_SECRET_NAME)
    if env_key:
        print(
            f"[rl-iris] Daytona: using {DAYTONA_RL_SECRET_NAME} from --secrets-env.",
            flush=True,
        )
        return env_key
    value = _daytona_rl_api_key_from_secret_manager()
    if value:
        print(
            "[rl-iris] Daytona: agentic run uses canonical Google Secret Manager "
            f"{DAYTONA_RL_SECRET_NAME} version {DAYTONA_RL_SECRET_VERSION}.",
            flush=True,
        )
        return value
    raise SystemExit(
        "[rl-iris] no Daytona RL key available: provide DAYTONA_RL_API_KEY via --secrets-env "
        "or authenticate gcloud for the Marin project, then retry."
    )


def _effective_gdn_backend(args: SimpleNamespace) -> str:
    return str(args.config.skyrl.get("generator", {}).get("gdn_backend", "torch")).lower()


def _rl_config_uses_daytona(config: DictConfig) -> bool:
    """Return whether the configured task backend requires Daytona credentials."""
    uses_harbor = str(config.runtime.entrypoint) in {
        RL_ENTRYPOINTS[RLEntrypoint.TERMINAL_BENCH],
        RL_ENTRYPOINTS[RLEntrypoint.TERMINAL_BENCH_GENERATE],
    } or bool(config.skyrl.get("data", {}).get("terminal_bench_data"))
    backend = OmegaConf.select(
        config,
        "skyrl.terminal_bench_config.harbor.environment_type",
        default=EnvironmentType.DAYTONA.value,
    )
    return uses_harbor and backend == EnvironmentType.DAYTONA.value


def build_debug_launch_env(args: SimpleNamespace) -> dict[str, str]:
    """Resolve the effective debug preset after the job name and RL config exist."""
    trainer = args.config.skyrl.get("trainer", {})
    mode = str(trainer.get("debug_mode", DebugMode.LIGHT.value))
    try:
        resolved = DebugMode(mode)
    except ValueError as error:
        choices = ", ".join(item.value for item in DebugMode)
        raise ValueError(f"trainer.debug_mode must be one of: {choices}; got {mode!r}") from error
    if resolved is DebugMode.OFF:
        return {}
    phase_diagnostics = trainer.get("collective_phase_diagnostics") is not False
    return EnvVarManager.for_debug_launch(
        mode=resolved,
        job_name=args.job_name,
        collective_phase_diagnostics=phase_diagnostics,
    ).environment_for(EnvVarScope.TASK_RUNTIME)


def config_task_environment(config: DictConfig) -> dict[str, str]:
    """Resolve task settings and explicitly named backend secrets at submission."""
    extra = dict(config.runtime.task_env)
    skyrl_controls = sorted(str(key) for key in extra if str(key).startswith("SKYRL_"))
    if skyrl_controls:
        raise ValueError(
            "RL config extra_env may not define MarinSkyRL behavior; use typed Hydra settings instead: "
            + ", ".join(skyrl_controls)
        )
    out: dict[str, str] = {}
    for k, v in extra.items():
        if v is None:
            continue
        if isinstance(v, bool):
            v = int(v)
        out[str(k)] = str(v)
    if config.runtime.entrypoint != CHECKPOINT_EXPORT_ENTRYPOINT:
        for backend in ("remote_judge", "daytona"):
            key_env = OmegaConf.select(config, f"skyrl.environment.task_sessions.{backend}.api_key_env")
            if key_env is not None:
                out[key_env] = os.environ[key_env]
        for callback in OmegaConf.select(config, "skyrl.trainer.callbacks", default=[]) or []:
            if callback.get("type") == "harbor_evaluation":
                key_env = callback["daytona_api_key_env"]
                out[key_env] = os.environ[key_env]
    return out


def _job_scope_fr_dump_path(prefix: str, job_name: str) -> str:
    """Rewrite a JOB-SCOPED NCCL flight-recorder dump path so its slug segment is the
    ACTUAL job name, e.g. ``/tmp/fr_dumps/<slug>/nccl_fr_rank`` -> ``/tmp/fr_dumps/
    <job_name>/nccl_fr_rank``.

    WHY (2026-07-11 FR-slug bug): the 80B configs hardcode
    ``TORCH_NCCL_DEBUG_INFO_TEMP_FILE: /tmp/fr_dumps/80b-next-cp1/nccl_fr_rank`` in
    their ``extra_env:``, so a run launched under a DIFFERENT ``--job-name`` (e.g.
    ``80b-next-cp1-r3d2``) still wrote its FR dumps under the stale ``80b-next-cp1``
    slug (harmless there, but wrong — a future FR dump would land under the wrong
    slug). Deriving the slug from the live job name keeps the dump under the right
    per-job dir. The controller's ``ensure_fr_dump_dir`` mkdir -p's whatever dirname
    the cvar carries, so overriding the cvar here is sufficient.

    ONLY rewrites the job-scoped ``.../fr_dumps/<slug>/<file>`` pattern; a bare
    generic path (``/tmp/nccl_fr_rank``, which every non-80B iris config uses) has no
    slug segment and is returned UNCHANGED (byte-identical for those configs)."""
    parent = os.path.dirname(prefix)  # e.g. /tmp/fr_dumps/<slug>
    grandparent = os.path.dirname(parent)  # e.g. /tmp/fr_dumps
    if os.path.basename(grandparent) != "fr_dumps":
        return prefix  # not a job-scoped fr_dumps path; leave it
    return os.path.join(grandparent, job_name, os.path.basename(prefix))


def _build_task_shell(
    args: SimpleNamespace,
    controller_cmd: list[str],
    pythonpath: str,
) -> list[str]:
    """Wrap the node controller with per-pod storage and cache setup."""
    ctrl = shlex.join(controller_cmd)
    # Run outside task_runtime so every pod establishes Ray's filesystem contract
    # before importing or starting any MarinSkyRL controller code. The controller
    # repeats this check immediately before `ray start` as defense in depth.
    spill_preflight = resolve_ray_spill_target(
        args.rendezvous_dir, args.ray_spill_backend, args.ray_spill_dir
    ).shell_preflight()
    # TileLang JIT-cache warm-start shim for GDN/FlashQLA runs only.
    # generator.gdn_backend=flashqla lazily JIT-compiles the FlashQLA GatedDeltaNet TileLang
    # kernels on the first GPU forward into the node-local, ephemeral TileLang cache.
    # This brackets the train command with a per-pod, per-NODE cache sync (the bash
    # runs once per task pod / node, and
    # TileLang's cache is node-local, so one --down warms all 8 local GPU workers):
    #   --down BEFORE the controller -> pulls the keyed warm cache (seed cache.tgz +
    #          incremental per-hash-dir objects) into TILELANG_CACHE_DIR so TileLang
    #          hash-matches and skips the cold compile. A miss is a warn+continue no-op.
    #   --up   at EXIT (bash EXIT trap; fires on normal completion AND a `set -e`/crash
    #          exit) -> uploads NEWLY-compiled hash-dirs as per-hash objects (race-free
    #          across the ~16 writers — content-addressed, no cache.tgz overwrite).
    # The shim never fails the job (best-effort; exits 0 even on S3 error).
    # TILELANG_CACHE_DIR is exported (defaulting to TileLang's own default) so the shim
    # and the trainer's TileLang agree on the location; a config-set value wins.
    # TILELANG_CACHE_MODEL_PATH lets the shim derive the model component of the key.
    sync_py = "cloud/iris/tilelang_cache_sync.py"
    tl_down = f"{RL_PYTHON} {sync_py} --down || true"
    tl_up = f"{RL_PYTHON} {sync_py} --up || true"
    # The controller is run as a BACKGROUND child + `wait` (not `exec`) so we can
    # (a) run --up at exit via the bash EXIT trap and (b) FORWARD SIGTERM/SIGINT to
    # the controller — preserving the old `exec` graceful-shutdown path (rank-0's Ray
    # teardown + nonzero exit on preemption) that a plain child would lose. `wait` is
    # interrupted by the trapped signal (rc>128); we re-`wait` to reap the child's
    # real exit code after its forwarded-TERM shutdown.
    flashqla_enabled = _effective_gdn_backend(args) == "flashqla"
    gdn_branch = (
        f'export TILELANG_CACHE_DIR="${{TILELANG_CACHE_DIR:-/root/.tilelang/cache}}"; '
        f"export TILELANG_CACHE_MODEL_PATH={shlex.quote(args.model_path)}; "
        f"{tl_down}; "
        f"trap {shlex.quote(tl_up)} EXIT; "
        f'trap \'[ -n "$_child" ] && kill -TERM "$_child" 2>/dev/null\' TERM INT; '
        f"set +e; {ctrl} & _child=$!; "
        f'wait "$_child"; _rc=$?; '
        f'if [ $_rc -gt 128 ]; then wait "$_child" 2>/dev/null; _rc=$?; fi; '
        f"exit $_rc"
    )
    if not flashqla_enabled:
        gdn_branch = f"exec {ctrl}"
    bash = (
        f"set -e; {spill_preflight}cd {APP_DIR}; "
        f"export SKYRL_HOME={shlex.quote(SKYRL_HOME)}; "
        f"source {shlex.quote(MARINSKYRL_ACTIVATION_FILE)}; "
        f"export PYTHONPATH={shlex.quote(pythonpath)}:${{PYTHONPATH:-}}; "
        f"export VLLM_USE_V1=1; "
        f"{gdn_branch}"
    )
    return ["bash", "-c", bash]


def build_config_task_command(args: SimpleNamespace) -> list[str]:
    """Build the replica-controller command for the resolved launch config."""
    launch_config = args.launch_config_bundle
    if not isinstance(launch_config, BundledLaunchConfig):
        raise RuntimeError("resolved launch config payload is missing")
    controller_cmd = [
        RL_PYTHON,
        "cloud/iris/task_runtime.py",
        "--config",
        launch_config.task_path,
    ]
    pythonpath = f"{APP_DIR}:{SKYRL_HOME}:{SKYRL_HOME}/skyrl-train"
    return _build_task_shell(args, controller_cmd, pythonpath)


def launch(args: SimpleNamespace, workspace: Path) -> LaunchOutcome:
    """Submit a resolved launch config and, unless detached, wait for its terminal state."""
    if not args.job_name:
        args.job_name = f"rl-iris-{time.strftime('%Y%m%d-%H%M%S')}"

    if not _is_checkpoint_export(args) and _rl_config_uses_daytona(args.config):
        daytona_api_key = _resolve_daytona_rl_api_key()
        os.environ[DAYTONA_RL_SECRET_NAME] = daytona_api_key
        os.environ["DAYTONA_API_KEY"] = daytona_api_key

    command = build_config_task_command(args)

    # Per-task resources: one whole selected GPU node per task.
    gpu_spec = f"{args.gpu_variant}x{args.gpus_per_node}"

    automatic_memory = _resource_request_is_automatic(str(args.memory))
    automatic_disk = _resource_request_is_automatic(str(args.disk))
    if automatic_memory or automatic_disk:
        resolved_resources = resolve_node_resource_requests(
            args.cluster_config,
            gpu_variant=args.gpu_variant,
            gpus_per_node=args.gpus_per_node,
            num_nodes=args.num_nodes,
            memory_request=str(args.memory),
            disk_request=str(args.disk),
        )
        args.memory = resolved_resources.memory
        args.disk = resolved_resources.disk

    user = os.environ.get("USER") or os.environ.get("USERNAME") or "user"
    print(f"[rl-iris] Job:        /{user}/{args.job_name}", flush=True)
    print(f"[rl-iris] Cluster:    {args.cluster}  ({args.cluster_config})", flush=True)
    print(f"[rl-iris] Runtime:    {args.source_digest} ({args.runtime_profile.value})", flush=True)
    print(
        f"[rl-iris] Topology:   {args.num_nodes} node(s) x {gpu_spec}  "
        f"(= {args.num_nodes * args.gpus_per_node} GPUs, exclusive, gang/leafgroup)",
        flush=True,
    )
    print(f"[rl-iris] Per node:   cpu={args.cpu} memory={args.memory} disk={args.disk}", flush=True)
    print(f"[rl-iris] Priority:   {args.priority}", flush=True)
    print(f"[rl-iris] Launch config:  {args.launch_config}  model={args.model_path}", flush=True)
    storage_paths = args.storage_paths
    if not _is_checkpoint_export(args):
        print(
            f"[rl-iris] Resume:     {storage_paths.checkpoint_root} (keep {storage_paths.resume_checkpoint_count})",
            flush=True,
        )
        print(f"[rl-iris] Canonical:  {storage_paths.export_root}", flush=True)
        print(f"[rl-iris] Trajectory: {storage_paths.trajectory_root}", flush=True)
        print(f"[rl-iris] Ray logs:   {storage_paths.ray_log_root}", flush=True)
        print(f"[rl-iris] Ray spill:  {args.ray_spill_backend.value}:{args.ray_spill_dir}", flush=True)
    print(f"[rl-iris] Rendezvous: {args.rendezvous_dir}", flush=True)
    print(f"[rl-iris] Command:    {shlex.join(command)}", flush=True)

    if args.dry_run:
        print("[rl-iris] --dry-run: not submitting", flush=True)
        return LaunchOutcome(job_id="", job_state="prepared", exit_code=0)

    # Defer heavy iris imports so --dry-run / --help stay snappy.
    #
    # NOTE: post iris PR #6652 (pydantic config parsing) + #6730 (multi-backend
    # controller) the old submit API moved. The config is now a pydantic
    # ``IrisClusterConfig`` loaded via the MODULE-LEVEL ``load_config(path)``
    # (the ``IrisConfig`` class + its ``.load()`` / ``.provider_bundle()`` /
    # ``.proto`` are gone). The provider bundle is now built by the module-level
    # ``iris.cluster.composer.provider_bundle(config)``, and ``LocalCluster``
    # moved to ``iris.cluster.local_cluster``. The job-build helpers
    # (ResourceSpec / constraints / EnvironmentSpec / Entrypoint / job_pb2) and
    # the ``IrisClient.remote(...)`` /
    # ``client.submit(...)`` surface are UNCHANGED — see how the marin CLI itself
    # now submits in iris/cli/job.py + iris/cli/connect.py, which this mirrors.
    from iris.cluster.types import EnvironmentSpec, Entrypoint

    # Per-task resources: whole node, all GPUs (no co-tenant → exclusive).
    resources = _gpu_resources(
        args.gpu_variant,
        args.gpus_per_node,
        cpu=args.cpu,
        memory=args.memory,
        disk=args.disk,
    )

    # Multi-node gang: replicas=num_nodes; for GPUs with replicas>1 this returns
    # CoschedulingConfig(group_by="leafgroup") — co-schedule all nodes on one IB
    # leaf fabric, atomically (Kueue gang admission on cw-us-east-02a).
    replicas = args.num_nodes
    coscheduling = _gpu_multinode(args.gpu_variant, args.gpus_per_node, replicas)

    resources_proto = resources.to_proto()
    # --target-cluster (federated submission) appends a `cluster EQ <peer>` constraint
    # so the marin meta-scheduler DELEGATES the whole job to the peer child (see the
    # submission block below). None on the default direct path (byte-identical).
    constraints = _gpu_constraints(
        resources_proto,
        replicas=replicas,
        target_cluster=args.target_cluster,
    )

    priority_band = job_pb2.PriorityBand.Value(f"PRIORITY_BAND_{args.priority.upper()}")

    # Env: secrets file values + the standard RL/iris-serve signals. iris injects
    # IRIS_TASK_ID / IRIS_NUM_TASKS / IRIS_ADVERTISE_HOST per task automatically.
    env_vars: dict[str, str] = {}
    # Forward runtime.task_env; launcher-owned rendezvous and secret values below
    # win on collision.
    config_extra_env = config_task_environment(args.config)
    if config_extra_env:
        env_vars.update(config_extra_env)
        print(f"[rl-iris] Config extra_env: {', '.join(sorted(config_extra_env))}", flush=True)
    debug_env = build_debug_launch_env(args)
    if debug_env:
        env_vars.update(debug_env)
        print(
            f"[rl-iris] Debug mode {debug_env[DEBUG_MODE_ENV]}: artifacts -> {debug_env[DEBUG_ARTIFACT_DIR_ENV]}",
            flush=True,
        )
    env_vars.update(args.launch_config_bundle.task_environment())
    # ── Per-cluster infra-env DEFAULTS (fill-gap belt for cluster-specific footguns) ──────────
    # Some clusters need a specific network/NCCL interface that a cluster-AGNOSTIC RL config
    # won't (and shouldn't) carry. Fill it in here, keyed on --target-cluster, only if the
    # config's extra_env did not set it (lowest precedence — an explicit
    # value always wins). cw-rno2a: its host_network:true nodes expose IB/IPoIB (ibs*/ibp*) +
    # virtual ifaces, so NCCL AND Ray's raylet/GCS mis-detect the bootstrap interface and the
    # multi-node gang SILENTLY never forms (keep-6 2026-07-19: 6 arms idle-heartbeated 3.4h at
    # zero progress before this was diagnosed). Pin the bootstrap socket to the host ethernet PF
    # via the exclude pattern (value = cw-rno2a.yaml:128). No-op on cw-us-east-02a (auto-detect
    # already lands on the PF). Add a cluster row here rather than editing every RL config.
    _target_cluster = str(getattr(args, "target_cluster", "") or "")
    # Do NOT add GLOO_SOCKET_IFNAME here with this value. Gloo does not accept
    # NCCL's `^exclude` syntax — it wants a literal interface name and fails with
    # `Unable to find address for: ^ibs` at engine init. Gloo picking loopback on
    # multi-node gangs is still an open problem; the fix needs a real interface
    # name, which differs per node, so the exclusion trick does not transfer.
    _CLUSTER_ENV_DEFAULTS: dict[str, dict[str, str]] = {
        "cw-rno2a": {"NCCL_SOCKET_IFNAME": "^ibs,ibp,lo,docker,veth,cilium,lxc"},
    }
    for _k, _v in _CLUSTER_ENV_DEFAULTS.get(_target_cluster, {}).items():
        if _k not in env_vars:
            env_vars[_k] = _v
            print(f"[rl-iris] Cluster infra-env default for {_target_cluster}: {_k}={_v}", flush=True)
    # FR-slug fix: a config may hardcode a JOB-SCOPED NCCL flight-recorder dump path
    # (/tmp/fr_dumps/<slug>/nccl_fr_rank) with a STALE slug from the config it was
    # copied from. Re-scope the slug to the live --job-name so a future FR dump lands
    # under the right per-job dir (the controller mkdir -p's the cvar's dirname). No-op
    # for the bare generic /tmp/nccl_fr_rank path every non-80B config uses.
    for _fr_cvar in ("TORCH_NCCL_DEBUG_INFO_TEMP_FILE", "TORCH_FR_DUMP_TEMP_FILE"):
        _old = env_vars.get(_fr_cvar)
        if _old:
            _new = _job_scope_fr_dump_path(_old, args.job_name)
            if _new != _old:
                env_vars[_fr_cvar] = _new
                print(f"[rl-iris] FR-slug re-scope: {_fr_cvar} {_old} -> {_new}", flush=True)
    if args.rendezvous_dir:
        env_vars["OT_AGENT_IRIS_RENDEZVOUS_DIR"] = args.rendezvous_dir
    # Forward the launch host's secrets (mirrors launch_eval_iris.py passthrough).
    #
    # IMPORTANT — do NOT forward AWS_*/R2_* here. The cw-us-east-02a cluster
    # projects an `iris-task-env` k8s Secret into EVERY task pod via `envFrom`
    # (because storage.remote_state_dir is an s3:// URI), and that secret already
    # carries the correct in-cluster R2 credentials + endpoint
    # (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_ENDPOINT_URL / AWS_REGION /
    # FSSPEC_S3). In K8s, explicit container `env` entries take precedence over
    # `envFrom`, so forwarding the launch host's AWS_* (which point at a
    # DIFFERENT account and lack AWS_ENDPOINT_URL) would CLOBBER the pod's
    # injected creds and make the s3://marin-us-east-02a rendezvous (multi-node)
    # silently target real AWS S3 instead of the cluster store. NOTE: the default
    # object store moved R2 (s3://marin-na) -> CW (s3://marin-us-east-02a) on
    # 2026-07-05 (marin c7caecc95a) — pods now inject CW creds+AWS_ENDPOINT_URL and
    # can no longer reach R2. Let the cluster-injected creds win; the
    # fsspec rendezvous in task_runtime.py uses default credential
    # discovery and picks them up.
    #
    # Daytona credentials MUST be forwarded: agentic RL (terminal_bench / Harbor)
    # builds a Daytona sandbox per trial, and iris injects only HF/WANDB into the
    # task pod — nothing else. Without DAYTONA_API_KEY the worker's harbor client
    # raises DaytonaAuthenticationError on every env build, so no sandbox comes
    # up, the verifier never runs, and EVERY trajectory finalizes as
    # VerificationNotCompletedError with reward 0 (observed zeroing an entire
    # reverify rollout). Mirror the base IrisLauncher passthrough set
    # so the same creds reach the RL worker.
    #
    # SkyRL passes trainer.project_name directly to wandb.init but does not pass an
    # entity. Resolve that missing dimension here: the explicit launch argument wins,
    # followed by the launch-host value and the Marin team default.
    for k in (
        "HF_TOKEN",
        "WANDB_API_KEY",
        "DAYTONA_API_KEY",
        "DAYTONA_JWT_TOKEN",
        "DAYTONA_ORGANIZATION_ID",
        "DAYTONA_API_URL",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "TOGETHER_API_KEY",
    ):
        v = os.environ.get(k)
        if v:
            env_vars[k] = v
    env_vars.update(wandb_launch_environment(entity=args.wandb_entity))
    if not _is_checkpoint_export(args) and _rl_config_uses_daytona(args.config):
        for field in ("daytona_api_key_env", "eval_daytona_api_key_env"):
            name = OmegaConf.select(args.config, f"skyrl.terminal_bench_config.harbor.{field}")
            if name is not None:
                value = os.environ.get(name)
                if not value:
                    raise ValueError(f"Missing Daytona credential environment variable: {name}")
                env_vars[name] = value

    # Load the cluster config (pydantic IrisClusterConfig) and build the provider
    # bundle, then discover + tunnel to the controller. This mirrors the marin
    # CLI's own path (iris/cli/connect.py::require_controller_url): for a local
    # controller start an in-process LocalCluster; otherwise use the config's
    # controller_address() (defaults.worker.controller_address) if set, else fall
    # back to the backend's discover_controller(). cw-us-east-02a's controller
    # kind is "coreweave" (non-local, no IAP auth) → the discover path.
    #
    # FEDERATED submission (--target-cluster set): submit through the PARENT (marin)
    # meta-scheduler instead of the peer's own controller. We load marin.yaml (whose
    # dashboard_url is the IAP-gated iris.oa.dev) and tunnel THERE; the `cluster EQ
    # <peer>` constraint appended above makes marin delegate the whole job to the peer
    # child.
    # Reaching iris.oa.dev requires IAP creds (`iris login` with an @openathena.ai
    # account, or an allowlisted service account); tunnel()/IrisClient handle the auth.
    submit_cluster_config = args.cluster_config
    if args.target_cluster:
        parent_cfg = args.parent_cluster_config or _resolve_parent_cluster_config(args.cluster_config)
        if not parent_cfg:
            raise SystemExit(
                "[rl-iris] --target-cluster set but no parent (marin) cluster config "
                "could be resolved. Pass --parent-cluster-config <path to marin.yaml>."
            )
        submit_cluster_config = parent_cfg
        print(
            f"[rl-iris] Federated submission: delegating to peer '{args.target_cluster}' "
            f"via the marin meta-scheduler ({parent_cfg}).",
            flush=True,
        )
    from contextlib import contextmanager as _contextmanager

    @_contextmanager
    def _direct_client():
        if ambient_client := _ambient_in_cluster_client(workspace):
            with ambient_client as client:
                yield client
            return
        from iris.cluster.composer import provider_bundle
        from iris.cluster.config import load_config
        from iris.cluster.local_cluster import LocalCluster

        # Direct submission to --cluster's own controller. On CoreWeave the loopback SSH
        # tunnel presents as the trusted local_admin identity (no IAP login needed) —
        # byte-identical to before.
        iris_config = load_config(submit_cluster_config)
        bundle = provider_bundle(iris_config)
        if iris_config.controller.controller_kind() == "local":
            controller_address = LocalCluster(iris_config).start()
        else:
            controller_address = iris_config.controller_address() or bundle.controller.discover_controller(
                iris_config.controller
            )
        with bundle.controller.tunnel(address=controller_address) as controller_url:
            # The sealed workspace already contains only the declared runtime files.
            yield IrisClient.remote(controller_url, workspace=workspace, extra_bundle_includes=RUNTIME_BUNDLE_INCLUDES)

    if args.target_cluster:
        # Federated submission MUST carry the IAP *user* identity: the controller rejects
        # a loopback/local_admin tunnel identity for a federated job ("a local_admin
        # (CIDR/loopback) identity cannot submit a federated job"), because delegation
        # forwards the submitter's identity to the peer for its owner check. Connect to
        # the marin parent exactly as the `iris job run` CLI does — open_iris_client
        # threads the IAP ClientCredentials (iris JWT + IAP OIDC token) from the cached
        # `iris login`, so the submission carries the user identity rather than loopback.
        # Requires a completed `iris --cluster=marin login` (@openathena.ai).
        from iris.cli.connect import open_iris_client

        client_cm = open_iris_client(
            config_file=Path(submit_cluster_config), workspace=workspace, extra_bundle_includes=RUNTIME_BUNDLE_INCLUDES
        )
    else:
        client_cm = _direct_client()

    with client_cm as client:
        entrypoint = Entrypoint.from_command(*command)
        job = client.submit(
            entrypoint=entrypoint,
            name=args.job_name,
            task_image=IRIS_TASK_IMAGE,
            container_profile=args.container_profile,
            resources=resources,
            environment=EnvironmentSpec(
                env_vars=env_vars,
                extras=["gpu"],
                setup_scripts=[task_setup_script(args.source_digest, args.runtime_profile)],
            ),
            constraints=constraints or None,
            coscheduling=coscheduling,
            replicas=replicas,
            ports=[port.value for port in RayPort],
            max_retries_failure=args.max_retries,
            max_task_failures=args.max_retries,
            priority_band=priority_band,
            timeout=None if args.timeout == 0 else _seconds_to_duration(args.timeout),
        )
        full_job_id = str(job.job_id)
        print(
            f"[rl-iris] Submitted: {full_job_id}  (replicas={replicas}, "
            f"coscheduling={getattr(coscheduling, 'group_by', None)})",
            flush=True,
        )

        if args.no_wait:
            return LaunchOutcome(
                job_id=full_job_id,
                job_state="submitted",
                exit_code=0,
            )
        print(
            f"[rl-iris] Now streaming logs for {full_job_id}. This process runs until the job ends.\n"
            "[rl-iris] SIGINT and SIGTERM cancel the complete job tree; signals do not detach.\n"
            "[rl-iris] Use --no-wait to submit and return instead.",
            file=sys.stderr,
            flush=True,
        )
        outcome = supervise_iris_job(job)
        print(f"[rl-iris] Job exit: {outcome.exit_code}", flush=True)
        return outcome


def _ambient_in_cluster_client(workspace: Path) -> IrisClient | None:
    """Connect a nested launch to the controller assigned to its coordinator task."""
    controller_url = os.environ.get("IRIS_CONTROLLER_URL")
    if not controller_url:
        return None
    return IrisClient.in_cluster(controller_url, workspace=workspace, extra_bundle_includes=RUNTIME_BUNDLE_INCLUDES)


def _seconds_to_duration(secs: int):
    from rigging.timing import Duration

    return Duration.from_seconds(secs)
