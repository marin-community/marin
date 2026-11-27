#!/usr/bin/env python3
"""Derive and submit an Iris GPU job from a validated Hydra launch config."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import os
import shlex
import signal
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote, urlparse

from iris.client.client import Job
from iris.cluster.client.job_info import get_job_info
from iris.cluster.constraints import Constraint, infer_preemptible_constraint
from iris.cluster.platforms.k8s.coreweave_topology import gpu_gang_coscheduling_level
from iris.cluster.types import CoschedulingConfig, ResourceSpec, gpu_device
from iris.resources.state import JobState
from iris.rpc import job_pb2
from omegaconf import DictConfig, OmegaConf
from rigging.config_discovery import find_project_root

from cloud.iris.resource_requests import _resource_request_is_automatic, resolve_node_resource_requests
from cloud.iris.launch_environment import (
    BundledLaunchConfig,
    _is_checkpoint_export,
    build_debug_launch_env,
    config_task_environment,
    _job_scope_fr_dump_path,
    build_config_task_command,
)
from cloud.iris.launch_config import SubmissionMode, derive_launch_topology, load_launch_config
from cloud.iris.launch import LaunchOutcome
from cloud.iris.export_hf_checkpoint import export_terminal_policy
from cloud.iris.submission import submission_client
from cloud.iris.ray_storage import (
    RaySpillBackend,
)
from cloud.iris.rl_config_translation import RL_CONFIG_TASK_DIR
from marinskyrl.resource_locator import (
    is_cloud_uri,
    join_resource_path,
)
from cloud.iris.paths import PROJECT_ROOT
from cloud.iris.ray_ports import RayPort
from marinskyrl.environment_contract import DEBUG_ARTIFACT_DIR_ENV, DEBUG_MODE_ENV, wandb_launch_environment
from cloud.iris.runtime_environment import IRIS_TASK_IMAGE, RuntimeProfile, task_setup_command


@dataclass(frozen=True)
class RLStoragePaths:
    """Resolved storage paths used for launch diagnostics and retention."""

    checkpoint_root: str
    export_root: str
    trajectory_root: str
    ray_log_root: str
    resume_checkpoint_count: int


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
        started_at=status.started_at.as_formatted_date() if status.started_at else None,
        finished_at=status.finished_at.as_formatted_date() if status.finished_at else None,
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
) -> list[Constraint]:
    constraints = []
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
        role_topology=derive_launch_topology(raw).role_topology,
        cpu=float(allocation["cpu"]),
        memory=allocation["memory"],
        disk=allocation["disk"],
        ray_spill_dir=ray["spill_dir"],
        ray_spill_backend=RaySpillBackend(ray["spill_backend"]),
        rendezvous_dir=ray["rendezvous_dir"],
        cluster=iris["cluster"],
        cluster_config=iris["cluster_config"],
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


def _forwarded_environment(names: tuple[str, ...]) -> dict[str, str]:
    missing = [name for name in names if name not in os.environ]
    if missing:
        raise ValueError(f"Missing forwarded environment variables: {', '.join(missing)}")
    return {name: os.environ[name] for name in names}


class IrisBackend:
    """Validate and submit one resolved Hydra launch config."""

    def __init__(self, *, forward_env: tuple[str, ...] = ()):
        _forwarded_environment(forward_env)
        self.forward_env = forward_env

    def validate(self, config_path: Path) -> None:
        _iris_submission_state(config_path, load_launch_config(config_path))

    def launch(self, config_path: Path) -> LaunchOutcome:
        config = load_launch_config(config_path)
        args = _iris_submission_state(config_path, config)
        if config.run.submission == SubmissionMode.PREPARE:
            raise ValueError("Prepare mode validates a launch without submitting it")
        workspace = find_project_root(PROJECT_ROOT)
        if workspace is None:
            raise RuntimeError("SkyRL launches require a Marin checkout")
        with contextlib.redirect_stdout(sys.stderr):
            return launch(args, workspace, forward_env=self.forward_env)

    def export_terminal_policy(self, config_path: Path) -> None:
        export_terminal_policy(config_path, forward_env=self.forward_env)


def launch(args: SimpleNamespace, workspace: Path, *, forward_env: tuple[str, ...] = ()) -> LaunchOutcome:
    """Submit a resolved launch config and, unless detached, wait for its terminal state."""
    if get_job_info() is not None and (args.target_cluster or args.parent_cluster_config):
        print(
            f"[rl-iris] In-task launch: ignoring target_cluster={args.target_cluster!r} and "
            f"parent_cluster_config={args.parent_cluster_config!r}. Child jobs run on the "
            "parent's cluster; submitting through IRIS_CONTROLLER_URL.",
            flush=True,
        )
        args.target_cluster = None
        args.parent_cluster_config = None
    if not args.job_name:
        args.job_name = f"rl-iris-{time.strftime('%Y%m%d-%H%M%S')}"

    config_extra_env = config_task_environment(args.config)
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
    print(f"[rl-iris] Runtime:    {args.runtime_profile.value}", flush=True)
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

    replicas = args.num_nodes
    if args.role_topology is not None:
        coscheduling = CoschedulingConfig(group_by="nvlink.domain", group_sizes=args.role_topology.locality.group_sizes)
    else:
        coscheduling = _gpu_multinode(args.gpu_variant, args.gpus_per_node, replicas)

    resources_proto = resources.to_proto()
    # --target-cluster (federated submission) appends a `cluster EQ <peer>` constraint
    # so the marin meta-scheduler DELEGATES the whole job to the peer child (see the
    # submission block below). None on the default direct path (byte-identical).
    constraints = _gpu_constraints(
        resources_proto,
        replicas=replicas,
    )

    priority_band = job_pb2.PriorityBand.Value(f"PRIORITY_BAND_{args.priority.upper()}")

    # Env: secrets file values + the standard RL/iris-serve signals. iris injects
    # IRIS_TASK_ID / IRIS_NUM_TASKS / IRIS_ADVERTISE_HOST per task automatically.
    env_vars: dict[str, str] = {}
    # Forward runtime.task_env; launcher-owned rendezvous and secret values below
    # win on collision.
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
    # won't (and shouldn't) carry. Fill it in here, keyed on the execution cluster, only if the
    # config's extra_env did not set it (lowest precedence — an explicit
    # value always wins). cw-rno2a: its host_network:true nodes expose IB/IPoIB (ibs*/ibp*) +
    # virtual ifaces, so NCCL AND Ray's raylet/GCS mis-detect the bootstrap interface and the
    # multi-node gang SILENTLY never forms (keep-6 2026-07-19: 6 arms idle-heartbeated 3.4h at
    # zero progress before this was diagnosed). Pin the bootstrap socket to the host ethernet PF
    # via the exclude pattern (value = cw-rno2a.yaml:128). No-op on cw-us-east-02a (auto-detect
    # already lands on the PF). Add a cluster row here rather than editing every RL config.
    # Do NOT add GLOO_SOCKET_IFNAME here with this value. Gloo does not accept
    # NCCL's `^exclude` syntax — it wants a literal interface name and fails with
    # `Unable to find address for: ^ibs` at engine init. Gloo picking loopback on
    # multi-node gangs is still an open problem; the fix needs a real interface
    # name, which differs per node, so the exclusion trick does not transfer.
    _CLUSTER_ENV_DEFAULTS: dict[str, dict[str, str]] = {
        "cw-rno2a": {"NCCL_SOCKET_IFNAME": "^ibs,ibp,lo,docker,veth,cilium,lxc"},
    }
    for _k, _v in _CLUSTER_ENV_DEFAULTS.get(args.cluster, {}).items():
        if _k not in env_vars:
            env_vars[_k] = _v
            print(f"[rl-iris] Cluster infra-env default for {args.cluster}: {_k}={_v}", flush=True)
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
    # Keep launch-host AWS overrides out of cluster jobs. The read-only
    # submitted CoreWeave files and AWS_SHARED_CREDENTIALS_FILE supply storage
    # authentication. Iris bundles HF/W&B credentials as files; these provider
    # clients still consume explicitly forwarded environment values.
    for k in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "TOGETHER_API_KEY",
    ):
        v = os.environ.get(k)
        if v:
            env_vars[k] = v
    env_vars.update(_forwarded_environment(forward_env))
    env_vars.update(wandb_launch_environment(entity=args.wandb_entity))
    with submission_client(args.cluster_config, args.target_cluster, args.parent_cluster_config, workspace) as (
        client,
        cluster_pin,
    ):
        if cluster_pin is not None:
            constraints.append(cluster_pin)
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
                setup_scripts=[task_setup_command(args.runtime_profile)],
            ),
            constraints=constraints or None,
            coscheduling=coscheduling,
            replicas=replicas,
            ports=[port.value for port in RayPort],
            max_retries_failure=args.max_retries,
            max_task_failures=args.max_retries,
            **(
                {"max_retries_preemption": 0}
                if "records" in args.config.skyrl.trainer.get("logger", ()) and not _is_checkpoint_export(args)
                else {}
            ),
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


def _seconds_to_duration(secs: int):
    from rigging.timing import Duration

    return Duration.from_seconds(secs)
