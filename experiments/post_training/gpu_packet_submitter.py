# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit a JSON GPU packet through a target-cluster CPU coordinator.

The coordinator route is the production default while federated GPU availability is
not propagated reliably.  The CPU coordinator is pinned to the requested cluster and
submits an unpinned GPU child through its ambient Iris client, so the target cluster's
own Kueue admission state decides capacity.  The direct route is retained as an
explicit switch for use after federation is qualified.
"""

import argparse
import hashlib
import json
import os
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from fray.client import Client
from fray.current_client import current_client
from fray.iris_backend import FrayIrisClient
from fray.types import Entrypoint as FrayEntrypoint
from fray.types import EnvironmentConfig, JobRequest, JobStatus, ResourceConfig
from iris.cli.connect import open_iris_client
from iris.cli.job import build_resources
from iris.client.client import iris_ctx
from iris.cluster.constraints import CLUSTER_CONSTRAINT_KEY, Constraint, ConstraintOp
from iris.cluster.types import Entrypoint as IrisEntrypoint
from iris.cluster.types import EnvironmentSpec, JobName
from iris.rpc import job_pb2
from rigging.filesystem.storage_path import StoragePath

PACKET_FILENAME = "gpu-packet.json"
Route = Literal["coordinator", "direct"]
QueueReader = Callable[[str], Mapping[str, Any]]
ReceiptWriter = Callable[[str, Mapping[str, Any]], None]
RETRYABLE_SUBMISSION_CODES = {Code.FAILED_PRECONDITION}
NONTERMINAL_STATUS_BACKOFF_SECONDS = 1.0
ENV_REFERENCE_PATTERN = re.compile(r"\$\{ENV:([A-Za-z_][A-Za-z0-9_]*)\}")
ENV_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class GpuJobPacket:
    """A complete, reproducible request for one GPU child job family."""

    schema_version: int
    job_name: str
    command: tuple[str, ...]
    gpu_variant: str
    gpus_per_task: int
    replicas: int
    cpu: float
    ram: str
    disk: str
    receipt_uri: str | None
    timeout_seconds: int
    credential_placeholders: tuple[str, ...] = ()
    environment: Mapping[str, Any] | None = None
    processes_per_task: int = 1
    max_retries_failure: int = 0
    max_retries_preemption: int = 0
    max_task_failures: int = 0
    priority: str = "batch"
    preemptible: bool = True

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        job_name: str | None = None,
        receipt_uri: str | None = None,
    ) -> "GpuJobPacket":
        packet = cls(
            schema_version=raw.get("schema_version", 1),
            job_name=job_name or raw.get("job_name", ""),
            command=tuple(raw["command"]),
            gpu_variant=raw.get("gpu_variant", "H100"),
            gpus_per_task=raw.get("gpus_per_task", 1),
            replicas=raw.get("replicas", 1),
            cpu=raw.get("cpu", 4),
            ram=raw.get("ram", "32g"),
            disk=raw.get("disk", "64g"),
            receipt_uri=receipt_uri or raw.get("receipt_uri"),
            timeout_seconds=raw.get("timeout_seconds", 3600),
            credential_placeholders=tuple(raw.get("credential_placeholders", ())),
            environment=raw.get("environment"),
            processes_per_task=raw.get("processes_per_task", 1),
            max_retries_failure=raw.get("max_retries_failure", 0),
            max_retries_preemption=raw.get("max_retries_preemption", 0),
            max_task_failures=raw.get("max_task_failures", 0),
            priority=raw.get("priority", "batch"),
            preemptible=raw.get("preemptible", True),
        )
        packet.validate()
        return packet

    @classmethod
    def read(
        cls,
        path: Path,
        *,
        job_name: str | None = None,
        receipt_uri: str | None = None,
    ) -> "GpuJobPacket":
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict):
            raise ValueError("GPU packet must be a JSON object")
        return cls.from_mapping(raw, job_name=job_name, receipt_uri=receipt_uri)

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported GPU packet schema_version: {self.schema_version}")
        if not self.job_name or "/" in self.job_name or " " in self.job_name:
            raise ValueError("job_name must be a non-empty Iris job name")
        if not self.command or not all(isinstance(value, str) and value for value in self.command):
            raise ValueError("command must contain non-empty strings")
        if not self.gpu_variant or self.gpus_per_task < 1 or self.replicas < 1:
            raise ValueError("GPU variant, gpus_per_task, and replicas must request GPUs")
        if self.cpu <= 0 or not self.ram or not self.disk:
            raise ValueError("cpu, ram, and disk must be positive resource requests")
        if self.timeout_seconds < 1:
            raise ValueError("timeout_seconds must be positive")
        if self.processes_per_task < 1:
            raise ValueError("processes_per_task must be positive")
        if self.priority not in {"batch", "interactive"}:
            raise ValueError("priority must be batch or interactive")
        if self.receipt_uri is not None and not self.receipt_uri.startswith(("s3://", "gs://", "file://", "/")):
            raise ValueError("receipt_uri must be an absolute local or object-store URI")
        if len(set(self.credential_placeholders)) != len(self.credential_placeholders):
            raise ValueError("credential_placeholders must not contain duplicates")
        if any(
            not isinstance(name, str) or ENV_NAME_PATTERN.fullmatch(name) is None
            for name in self.credential_placeholders
        ):
            raise ValueError("credential_placeholders must contain valid environment variable names")
        if self.environment is not None:
            allowed = {"env_vars", "extras", "setup_scripts", "sync_packages"}
            unknown = sorted(set(self.environment) - allowed)
            if unknown:
                raise ValueError(f"unknown environment fields: {unknown}")
        references = self._environment_references()
        if set(references) != set(self.credential_placeholders):
            raise ValueError(
                "credential_placeholders must exactly name the environment ${ENV:VAR} references; "
                f"declared={sorted(self.credential_placeholders)}, referenced={sorted(references)}"
            )

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()

    def _environment_references(self) -> dict[str, str]:
        env_vars = self.environment.get("env_vars", {}) if self.environment is not None else {}
        references: dict[str, str] = {}
        for key, value in env_vars.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("environment.env_vars keys and values must be strings")
            match = ENV_REFERENCE_PATTERN.fullmatch(value)
            if "${ENV:" in value and match is None:
                raise ValueError(f"environment reference for {key} must use exact ${{ENV:VAR}} syntax")
            if match is not None:
                referenced_name = match.group(1)
                if key != referenced_name:
                    raise ValueError(f"environment reference {key} must name itself, not {referenced_name}")
                references[key] = referenced_name
        return references

    def _resolved_credentials(self) -> dict[str, str]:
        references = self._environment_references()
        missing = sorted(name for name in references.values() if not os.environ.get(name))
        if missing:
            raise ValueError(f"GPU packet requires ambient environment variables: {', '.join(missing)}")
        return {key: os.environ[name] for key, name in references.items()}

    def _environment_config(self) -> EnvironmentConfig | None:
        if self.environment is None:
            return None
        env_vars = dict(self.environment.get("env_vars", {}))
        env_vars.update(self._resolved_credentials())
        return EnvironmentConfig(
            workspace=".",
            env_vars=env_vars,
            extras=tuple(self.environment.get("extras", ())),
            setup_scripts=self.environment.get("setup_scripts"),
            sync_packages=tuple(self.environment.get("sync_packages", ())),
        )

    def child_request(self, *, target_cluster: str | None) -> JobRequest:
        priority = job_pb2.PRIORITY_BAND_BATCH if self.priority == "batch" else job_pb2.PRIORITY_BAND_INTERACTIVE
        return JobRequest(
            name=self.job_name,
            entrypoint=FrayEntrypoint.from_binary(self.command[0], self.command[1:]),
            resources=ResourceConfig.with_gpu(
                self.gpu_variant,
                count=self.gpus_per_task,
                cpu=self.cpu,
                ram=self.ram,
                disk=self.disk,
                preemptible=self.preemptible,
                target_cluster=target_cluster,
            ),
            environment=self._environment_config(),
            replicas=self.replicas,
            processes_per_task=self.processes_per_task,
            max_retries_failure=self.max_retries_failure,
            max_retries_preemption=self.max_retries_preemption,
            max_task_failures=self.max_task_failures,
            priority=priority,
            timeout_seconds=self.timeout_seconds,
        )


def _write_new_receipt(uri: str, receipt: Mapping[str, Any]) -> None:
    path = StoragePath(uri)
    if path.exists():
        raise FileExistsError(f"refusing to replace receipt: {uri}")
    path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")


def _enum_name(value: Any) -> str:
    return getattr(value, "name", str(value))


def read_target_local_queue(job_id: str, *, timeout_seconds: float = 30) -> Mapping[str, Any]:
    """Read initial task admission from the coordinator's local Iris controller."""

    deadline = time.monotonic() + timeout_seconds
    while True:
        status = iris_ctx().client.job(JobName.from_wire(job_id)).status()
        if status.task_count:
            return {
                "job_state": _enum_name(status.state),
                "task_count": status.task_count,
                "tasks": [
                    {
                        "task_id": str(task.task_id),
                        "state": _enum_name(task.state),
                        "pending_reason": task.pending_reason,
                        "status_message": task.status_message,
                        "execution_cluster_id": task.execution_cluster_id,
                    }
                    for task in status.tasks
                ],
            }
        if time.monotonic() >= deadline:
            raise TimeoutError(f"target-local controller did not expose tasks for {job_id}")
        time.sleep(0.25)


def _wait_for_terminal(child: Any) -> JobStatus:
    """Wait through queue delay and transient nonterminal returns."""

    while True:
        status = child.wait(timeout=None, raise_on_failure=False, stream_logs=True)
        if JobStatus.finished(status):
            return status
        time.sleep(NONTERMINAL_STATUS_BACKOFF_SECONDS)


def coordinate_packet(
    packet: GpuJobPacket,
    *,
    target_cluster: str,
    fallback_cluster: str | None,
    selected_cluster: str | None = None,
    source_packet_sha256: str | None = None,
    coordinator_job_id: str | None = None,
    client: Client | None = None,
    queue_reader: QueueReader = read_target_local_queue,
    receipt_writer: ReceiptWriter = _write_new_receipt,
) -> Mapping[str, Any]:
    """Submit the GPU child in-cluster and persist its local queue admission."""

    # Deliberately omit target_cluster: the coordinator's ambient client points at
    # the target cluster and must consult that cluster's own Kueue state.
    child = (client or current_client()).submit(packet.child_request(target_cluster=None), adopt_existing=False)
    queue = queue_reader(child.job_id)
    task_count = cast(int, queue.get("task_count", 0))
    if task_count != packet.replicas:
        raise RuntimeError(f"target-local queue exposed {task_count} tasks; expected {packet.replicas}")
    receipt = {
        "schema_version": 1,
        "submitter_interface": "generic-gpu-coordinator/v1",
        "event": "gpu_child_terminal",
        "route": "coordinator",
        "target_cluster": target_cluster,
        "fallback_cluster": fallback_cluster,
        "selected_cluster": selected_cluster or target_cluster,
        "fallback_used": (selected_cluster or target_cluster) != target_cluster,
        "capacity_source": "cluster_queue",
        "capacity_observation_source": "target_cluster_local_queue",
        "packet_sha256": source_packet_sha256 or packet.sha256,
        "resolved_request_sha256": packet.sha256,
        "coordinator_job_id": coordinator_job_id,
        "child_job_id": child.job_id,
        "queue": queue,
    }
    fallback = fallback_cluster or "none"
    print(
        "GPU_PACKET_COORDINATOR_ADMISSION_PASS "
        f"route=coordinator target={target_cluster} fallback={fallback} "
        f"selected={selected_cluster or target_cluster} child={child.job_id} tasks={task_count} "
        "capacity_source=cluster_queue",
        flush=True,
    )
    # Queue delay is not part of the child execution limit. Iris enforces the
    # packet timeout on the child after admission; the parent must outlive it.
    terminal = _wait_for_terminal(child)
    receipt["terminal_result"] = terminal.value
    if packet.receipt_uri is None:
        raise ValueError("receipt_uri is required for coordinator execution")
    receipt_writer(packet.receipt_uri, receipt)
    if terminal is not JobStatus.SUCCEEDED:
        raise RuntimeError(f"GPU child {child.job_id} finished with {terminal.value}")
    return receipt


def _coordinator_command() -> list[str]:
    return [
        "bash",
        "-c",
        'export PYTHONPATH="$PWD/lib/marin/src:$PWD/lib/iris/src:$PWD/lib/fray/src:$PWD/lib/rigging/src:$PWD"; '
        "exec python -m experiments.post_training.gpu_packet_submitter --coordinate",
    ]


def _coordinator_environment(
    target_cluster: str,
    fallback_cluster: str | None,
    selected_cluster: str,
    source_packet_sha256: str,
    packet: GpuJobPacket,
) -> EnvironmentSpec:
    if packet.receipt_uri is None:
        raise ValueError("receipt_uri is required for coordinator submission")
    credential_env = packet._resolved_credentials()
    return EnvironmentSpec(
        env_vars={
            "GPU_PACKET_ROUTE": "coordinator",
            "GPU_PACKET_TARGET_CLUSTER": target_cluster,
            "GPU_PACKET_FALLBACK_CLUSTER": fallback_cluster or "",
            "GPU_PACKET_SELECTED_CLUSTER": selected_cluster,
            "GPU_PACKET_SOURCE_SHA256": source_packet_sha256,
            "GPU_PACKET_JOB_NAME": packet.job_name,
            "GPU_PACKET_RECEIPT_URI": packet.receipt_uri,
            **credential_env,
        },
        extras=[],
        setup_scripts=None,
    )


def _submit_coordinator(
    client: Any,
    packet: GpuJobPacket,
    target_cluster: str,
    fallback_cluster: str | None,
    selected_cluster: str,
    name: str,
    source_packet_sha256: str,
    source_packet_bytes: bytes,
):
    return client.submit(
        entrypoint=IrisEntrypoint(command=_coordinator_command(), workdir_files={PACKET_FILENAME: source_packet_bytes}),
        name=name,
        resources=build_resources(None, None, cpu=2, memory="8GB", disk="16GB"),
        environment=_coordinator_environment(
            target_cluster, fallback_cluster, selected_cluster, source_packet_sha256, packet
        ),
        constraints=[Constraint.create(key=CLUSTER_CONSTRAINT_KEY, op=ConstraintOp.EQ, value=selected_cluster)],
        max_retries_failure=0,
        max_retries_preemption=0,
        max_task_failures=0,
        # The GPU child owns the packet's execution timeout. Bounding this CPU
        # parent by that duration would kill a child that waited in Kueue first.
        timeout=None,
        scheduling_timeout=None,
        priority_band=job_pb2.PRIORITY_BAND_BATCH,
    )


def submit_packet(
    packet: GpuJobPacket,
    *,
    target_cluster: str,
    fallback_cluster: str | None,
    route: Route,
    coordinator_job_name: str,
    client: Any,
    direct_client: Client | None = None,
    source_packet_sha256: str | None = None,
    route_receipt: Mapping[str, Any] | None = None,
    coordinator_submitter: Callable[
        [Any, GpuJobPacket, str, str | None, str, str, str, bytes], Any
    ] = _submit_coordinator,
    source_packet_bytes: bytes | None = None,
    receipt_writer: ReceiptWriter = _write_new_receipt,
) -> Mapping[str, Any]:
    """Submit the packet by its selected route and return a devbox receipt."""

    if fallback_cluster == target_cluster:
        raise ValueError("fallback_cluster must differ from target_cluster")
    if route == "direct" and route_receipt is None:
        raise ValueError("direct route requires a qualified 1xH100 route_receipt")
    packet_sha256 = source_packet_sha256 or packet.sha256
    direct_job: Any | None = None

    def submit(cluster: str) -> str:
        nonlocal direct_job
        if route == "coordinator":
            job = coordinator_submitter(
                client,
                packet,
                target_cluster,
                fallback_cluster,
                cluster,
                coordinator_job_name,
                packet_sha256,
                source_packet_bytes or packet.canonical_json().encode(),
            )
            return str(job.job_id)
        direct_job = (direct_client or FrayIrisClient.from_iris_client(client)).submit(
            packet.child_request(target_cluster=cluster), adopt_existing=False
        )
        return direct_job.job_id

    selected_cluster = target_cluster
    try:
        submitted_job_id = submit(selected_cluster)
    except ConnectError as error:
        if fallback_cluster is None or error.code not in RETRYABLE_SUBMISSION_CODES:
            raise
        selected_cluster = fallback_cluster
        submitted_job_id = submit(selected_cluster)
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "submitter_interface": "generic-gpu-coordinator/v1",
        "event": "gpu_packet_submitted",
        "route": route,
        "target_cluster": target_cluster,
        "fallback_cluster": fallback_cluster,
        "selected_cluster": selected_cluster,
        "fallback_used": selected_cluster != target_cluster,
        "capacity_source": "cluster_queue",
        "capacity_observation_source": (
            "target_cluster_local_queue" if route == "coordinator" else "direct_route_admission_receipt"
        ),
        "packet_sha256": packet_sha256,
        "resolved_request_sha256": packet.sha256,
        "route_receipt": route_receipt,
        "submitted_job_id": submitted_job_id,
    }
    if route == "direct":
        assert direct_job is not None
        terminal = _wait_for_terminal(direct_job)
        receipt.update(
            event="gpu_child_terminal",
            coordinator_job_id=None,
            child_job_id=direct_job.job_id,
            terminal_result=terminal.value,
        )
        if packet.receipt_uri is None:
            raise ValueError("receipt_uri is required for direct execution")
        receipt_writer(packet.receipt_uri, receipt)
        if terminal is not JobStatus.SUCCEEDED:
            raise RuntimeError(f"GPU child {direct_job.job_id} finished with {terminal.value}")
    return receipt


def _coordinate_from_environment() -> None:
    if os.environ.get("GPU_PACKET_ROUTE") != "coordinator":
        raise RuntimeError("coordinator task is missing GPU_PACKET_ROUTE=coordinator")
    target = os.environ["GPU_PACKET_TARGET_CLUSTER"]
    fallback = os.environ.get("GPU_PACKET_FALLBACK_CLUSTER") or None
    selected = os.environ["GPU_PACKET_SELECTED_CLUSTER"]
    source_sha256 = os.environ["GPU_PACKET_SOURCE_SHA256"]
    packet = GpuJobPacket.read(
        Path(os.environ["IRIS_WORKDIR"]) / PACKET_FILENAME,
        job_name=os.environ["GPU_PACKET_JOB_NAME"],
        receipt_uri=os.environ["GPU_PACKET_RECEIPT_URI"],
    )
    coordinate_packet(
        packet,
        target_cluster=target,
        fallback_cluster=fallback,
        selected_cluster=selected,
        source_packet_sha256=source_sha256,
        coordinator_job_id=str(iris_ctx().job_id),
    )


def _read_launch_action(
    path: Path,
) -> tuple[Path, str, str, str, str | None, Route, Mapping[str, Any] | None]:
    action = json.loads(path.read_text())
    if action.get("action") != "launch_gpu" or action.get("handoff") != "generic_gpu_submitter":
        raise ValueError("action must be a launch_gpu generic_gpu_submitter handoff")
    launch = action.get("launch")
    if not isinstance(launch, dict):
        raise ValueError("action.launch must be an object")
    if launch.get("submitter_interface") != "generic-gpu-coordinator/v1":
        raise ValueError("unsupported submitter_interface")
    if launch.get("capacity_source") != "cluster_queue":
        raise ValueError("capacity_source must be cluster_queue")
    if launch.get("receipt_required") is not True:
        raise ValueError("generic-gpu-coordinator/v1 requires a receipt")
    route = launch.get("route")
    if route not in {"coordinator", "direct"}:
        raise ValueError("route must be coordinator or direct")
    route_receipt = launch.get("route_receipt")
    if route == "direct" and route_receipt is None:
        raise ValueError("direct route requires a 1xH100 route_receipt")
    if route_receipt is not None:
        if not isinstance(route_receipt, dict) or set(route_receipt) != {"path", "sha256", "line"}:
            raise ValueError("route_receipt must contain path, sha256, and line")
        if not all(isinstance(route_receipt[key], str) and route_receipt[key] for key in route_receipt):
            raise ValueError("route_receipt values must be non-empty strings")
        if len(route_receipt["sha256"]) != 64 or any(
            character not in "0123456789abcdef" for character in route_receipt["sha256"]
        ):
            raise ValueError("route_receipt.sha256 must be a lowercase SHA-256 digest")
    packet_binding = launch.get("packet")
    if not isinstance(packet_binding, dict):
        raise ValueError("launch.packet must be an object")
    packet_path = Path(packet_binding.get("path", ""))
    if not packet_path.is_absolute():
        raise ValueError("launch.packet.path must be absolute")
    packet_sha256 = packet_binding.get("sha256")
    if not isinstance(packet_sha256, str) or len(packet_sha256) != 64:
        raise ValueError("launch.packet.sha256 must be a SHA-256 hex digest")
    actual_sha256 = hashlib.sha256(packet_path.read_bytes()).hexdigest()
    if actual_sha256 != packet_sha256:
        raise ValueError(f"packet SHA-256 mismatch: expected {packet_sha256}, got {actual_sha256}")
    target = launch.get("target_cluster")
    fallback = launch.get("fallback_cluster")
    if not isinstance(target, str) or not target:
        raise ValueError("launch.target_cluster must be a non-empty string")
    if not isinstance(fallback, str) or not fallback:
        raise ValueError("launch.fallback_cluster must be a non-empty string")
    job_name = action.get("job_name")
    if not isinstance(job_name, str) or not job_name:
        raise ValueError("action.job_name must be a non-empty string")
    return packet_path, packet_sha256, job_name.rsplit("/", 1)[-1], target, fallback, route, route_receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", type=Path, help="experimentctl launch_gpu action JSON")
    parser.add_argument("--packet", type=Path)
    parser.add_argument("--target-cluster")
    parser.add_argument("--fallback-cluster")
    parser.add_argument("--route", choices=("coordinator", "direct"), default="coordinator")
    parser.add_argument("--coordinator-job-name")
    parser.add_argument("--submission-receipt")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--coordinate", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.coordinate:
        _coordinate_from_environment()
        return
    source_packet_sha256 = None
    route_receipt = None
    if args.action is not None:
        if args.packet is not None or args.target_cluster is not None or args.fallback_cluster is not None:
            parser.error("--action cannot be combined with packet or cluster arguments")
        (
            args.packet,
            source_packet_sha256,
            action_job_name,
            args.target_cluster,
            args.fallback_cluster,
            args.route,
            route_receipt,
        ) = _read_launch_action(args.action)
    else:
        action_job_name = None
    required = {"--packet or --action": args.packet, "--coordinator-job-name": args.coordinator_job_name}
    missing = [flag for flag, value in required.items() if value is None]
    if missing:
        parser.error(f"required arguments: {', '.join(missing)}")
    source_packet_bytes = args.packet.read_bytes()
    packet = GpuJobPacket.read(args.packet, job_name=action_job_name)
    if source_packet_sha256 is None:
        source_packet_sha256 = hashlib.sha256(source_packet_bytes).hexdigest()
    if args.route == "direct" and route_receipt is None:
        parser.error("direct route requires an experimentctl action with a 1xH100 route_receipt")
    preview = {
        "route": args.route,
        "target_cluster": args.target_cluster,
        "fallback_cluster": args.fallback_cluster,
        "coordinator_job_name": args.coordinator_job_name,
        "packet_sha256": source_packet_sha256,
        "job_name": packet.job_name,
        "gpu": f"{packet.gpu_variant}x{packet.gpus_per_task}",
        "replicas": packet.replicas,
    }
    print(json.dumps(preview, sort_keys=True, indent=2), flush=True)
    if not args.execute:
        return
    if args.submission_receipt is None:
        parser.error("--submission-receipt is required with --execute")
    packet = replace(packet, receipt_uri=args.submission_receipt)
    if os.environ.get("IRIS_USER") != "atqamar":
        raise RuntimeError("IRIS_USER must be atqamar")
    with open_iris_client(
        config_file=Path("lib/iris/config/marin.yaml"), cluster_name="marin", workspace=Path.cwd()
    ) as client:
        receipt = submit_packet(
            packet,
            target_cluster=args.target_cluster,
            fallback_cluster=args.fallback_cluster,
            route=args.route,
            coordinator_job_name=args.coordinator_job_name,
            client=client,
            source_packet_sha256=source_packet_sha256,
            route_receipt=route_receipt,
            source_packet_bytes=source_packet_bytes,
        )
    print(
        "GPU_PACKET_SUBMITTED "
        f"route={args.route} target={args.target_cluster} fallback={args.fallback_cluster or 'none'} "
        f"job={receipt['submitted_job_id']} packet_sha256={source_packet_sha256}",
        flush=True,
    )


if __name__ == "__main__":
    main()
