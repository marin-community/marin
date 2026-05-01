# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Job management via command passthrough (replaces ``iris-run``).

Usage:
    iris --config cluster.yaml job run -- python train.py --epochs 10
    iris --config cluster.yaml job run --tpu v5litepod-16 -e WANDB_API_KEY $WANDB_API_KEY -- python train.py
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import click
import humanfriendly
import yaml
from google.protobuf import json_format
from rigging.timing import Duration, Timestamp
from tabulate import tabulate

from iris.cli.bug_report import file_github_issue, format_bug_report, gather_bug_report
from iris.cli.main import require_controller_url
from iris.client import IrisClient
from iris.client.client import Job, JobFailedError
from iris.cluster.constraints import (
    Constraint,
    WellKnownAttribute,
    device_variant_constraint,
    infer_preemptible_constraint,
    preemptible_constraint,
    region_constraint,
    zone_constraint,
)
from iris.cluster.redaction import redact_submit_argv
from iris.cluster.types import (
    TERMINAL_TASK_STATES,
    CoschedulingConfig,
    Entrypoint,
    EnvironmentSpec,
    JobName,
    ReservationEntry,
    ResourceSpec,
    get_tpu_topology,
    gpu_device,
    tpu_device,
)
from iris.rpc import job_pb2
from iris.rpc.auth import TokenProvider
from iris.rpc.proto_utils import PRIORITY_BAND_NAMES, job_state_friendly, priority_band_value, task_state_friendly
from iris.time_proto import timestamp_from_proto

logger = logging.getLogger(__name__)

_STATE_MAP: dict[str, job_pb2.JobState] = {
    "pending": job_pb2.JOB_STATE_PENDING,
    "building": job_pb2.JOB_STATE_BUILDING,
    "running": job_pb2.JOB_STATE_RUNNING,
    "succeeded": job_pb2.JOB_STATE_SUCCEEDED,
    "failed": job_pb2.JOB_STATE_FAILED,
    "killed": job_pb2.JOB_STATE_KILLED,
    "worker_failed": job_pb2.JOB_STATE_WORKER_FAILED,
    "unschedulable": job_pb2.JOB_STATE_UNSCHEDULABLE,
}


def _format_resources(resources: job_pb2.ResourceSpecProto | None) -> str:
    """Format job resources as a compact human-readable string."""
    if not resources:
        return "-"

    parts = []

    # CPU
    if resources.cpu_millicores:
        parts.append(f"{resources.cpu_millicores / 1000:g}cpu")

    # Memory
    if resources.memory_bytes:
        parts.append(humanfriendly.format_size(resources.memory_bytes, binary=True))

    # Disk
    if resources.disk_bytes:
        parts.append(f"{humanfriendly.format_size(resources.disk_bytes, binary=True)} disk")

    # Device (TPU/GPU)
    if resources.HasField("device"):
        device = resources.device
        if device.HasField("tpu"):
            parts.append(device.tpu.variant)
        elif device.HasField("gpu"):
            gpu = device.gpu
            gpu_str = f"{gpu.count}x{gpu.variant}" if gpu.variant else f"{gpu.count}gpu"
            parts.append(gpu_str)

    return ", ".join(parts) if parts else "-"


def _terminate_jobs(
    client: IrisClient,
    job_ids: tuple[str, ...],
    include_children: bool,
) -> list[JobName]:
    terminated: list[JobName] = []
    for raw in job_ids:
        name = JobName.from_wire(raw)
        if include_children:
            terminated.extend(client.terminate_prefix(name, exclude_finished=True))
        else:
            client.terminate(name)
            terminated.append(name)
    return terminated


def _print_terminated(terminated: list[JobName]) -> None:
    if terminated:
        click.echo("Terminated jobs:")
        for job_name in terminated:
            click.echo(f"  {job_name}")
    else:
        click.echo("No running jobs matched.")


def load_env_vars(env_flags: tuple[tuple[str, ...], ...] | list | None) -> dict[str, str]:
    """Load environment variables from .marin.yaml and merge with flags.

    Args:
        env_flags: Tuple/list of (KEY,) or (KEY, VALUE) tuples from Click

    Returns:
        Merged environment variables
    """
    env_vars: dict[str, str] = {}
    marin_yaml = Path(".marin.yaml")
    if marin_yaml.exists():
        with open(marin_yaml) as f:
            cfg = yaml.safe_load(f) or {}
        if isinstance(cfg.get("env"), dict):
            for k, v in cfg["env"].items():
                env_vars[str(k)] = "" if v is None else str(v)

    for key in ("HF_TOKEN", "WANDB_API_KEY"):
        if key not in env_vars and os.environ.get(key):
            env_vars[key] = os.environ[key]

    if env_flags:
        for item in env_flags:
            if len(item) > 2:
                raise ValueError(f"Too many values for env var: {' '.join(item)}")
            if "=" in item[0]:
                raise ValueError(
                    f"Key cannot contain '=': {item[0]}\nYou probably meant to do '-e {' '.join(item[0].split('='))}'"
                )
            env_vars[item[0]] = item[1] if len(item) == 2 else ""

    return env_vars


def add_standard_env_vars(env_vars: dict[str, str]) -> dict[str, str]:
    """Add standard environment variables used by Marin jobs."""
    result = dict(env_vars)

    defaults = {
        "PYTHONPATH": ".",
        "PYTHONUNBUFFERED": "1",
        "HF_HOME": "~/.cache/huggingface",
        "HF_HUB_ENABLE_HF_TRANSFER": "0",
    }

    for key, value in defaults.items():
        if key not in result:
            result[key] = value

    for key in ("GCS_RESOLVE_REFRESH_SECS",):
        if key not in result and os.environ.get(key):
            result[key] = os.environ[key]

    return result


KNOWN_GPU_VARIANTS: frozenset[str] = frozenset(
    {
        "A100",
        "A10G",
        "B100",
        "B200",
        "GB200",
        "H100",
        "H200",
        "L4",
        "L40",
        "L40S",
        "RTX4090",
        "T4",
        "V100",
    }
)

_GPU_VARIANT_LOOKUP: dict[str, str] = {v.lower(): v for v in KNOWN_GPU_VARIANTS}


def parse_gpu_spec(spec: str) -> tuple[str, int]:
    """Parse a GPU spec string into (variant, count).

    Accepts: 'H100x8' → ("H100", 8), '8' → ("", 8), 'H100' → ("H100", 1).
    The variant must be a known GPU name from KNOWN_GPU_VARIANTS (case-insensitive).
    """
    if not spec:
        raise ValueError("GPU spec must not be empty")

    if spec.isdigit():
        count = int(spec)
        if count <= 0:
            raise ValueError(f"GPU count must be positive, got {count}")
        return "", count

    spec_lower = spec.lower()
    for known_lower, canonical in _GPU_VARIANT_LOOKUP.items():
        if not spec_lower.startswith(known_lower):
            continue
        rest = spec[len(known_lower) :]
        if not rest:
            return canonical, 1
        if rest[0] == "x" and rest[1:].isdigit():
            count = int(rest[1:])
            if count <= 0:
                raise ValueError(f"GPU count must be positive, got {count}")
            return canonical, count

    known = ", ".join(sorted(KNOWN_GPU_VARIANTS))
    raise ValueError(
        f"Unknown GPU spec: {spec!r}. "
        f"Expected a known variant (e.g., H100), VARIANTxCOUNT (e.g., H100x8), "
        f"or a bare count (e.g., 8). Known variants: {known}"
    )


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        return _levenshtein(b, a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1] + [0] * len(b)
        for j, cb in enumerate(b):
            curr[j + 1] = min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb))
        prev = curr
    return prev[-1]


def _find_closest(value: str, known: set[str], max_distance: int = 5) -> str | None:
    """Return the closest match from *known* by edit distance, or None."""
    best, best_dist = None, max_distance + 1
    for candidate in sorted(known):
        dist = _levenshtein(value, candidate)
        if dist < best_dist:
            best, best_dist = candidate, dist
    return best if best_dist <= max_distance else None


def _known_regions_and_zones(config) -> tuple[set[str], set[str]]:
    """Extract known regions and zones from an IrisClusterConfig proto.

    Returns:
        (regions, zones) sets derived from scale group worker attributes.
    """
    regions: set[str] = set()
    zones: set[str] = set()
    for sg in config.scale_groups.values():
        attrs = sg.worker.attributes
        if WellKnownAttribute.REGION in attrs:
            regions.add(attrs[WellKnownAttribute.REGION])
        if WellKnownAttribute.ZONE in attrs:
            zones.add(attrs[WellKnownAttribute.ZONE])
    return regions, zones


def validate_region_zone(
    regions: tuple[str, ...] | None,
    zone: str | None,
    config,
) -> None:
    """Validate --region/--zone CLI values against the cluster config.

    Raises click.BadParameter if a value doesn't match any known region/zone.
    Only validates when a config is available (i.e. --config was passed).
    """
    if config is None:
        return

    known_regions, known_zones = _known_regions_and_zones(config)

    if not known_regions and not known_zones:
        return

    if regions:
        for r in regions:
            if r not in known_regions:
                suggestion = _find_closest(r, known_regions)
                hint = f" Did you mean '{suggestion}'?" if suggestion else ""
                raise click.BadParameter(
                    f"'{r}' is not a known region in the cluster config.{hint}"
                    f" Known regions: {', '.join(sorted(known_regions))}",
                    param_hint="'--region'",
                )

    if zone:
        if zone not in known_zones:
            suggestion = _find_closest(zone, known_zones)
            hint = f" Did you mean '{suggestion}'?" if suggestion else ""
            raise click.BadParameter(
                f"'{zone}' is not a known zone in the cluster config.{hint}"
                f" Known zones: {', '.join(sorted(known_zones))}",
                param_hint="'--zone'",
            )


def build_resources(
    tpu: str | None,
    gpu: str | None,
    cpu: float = 0.5,
    memory: str = "1GB",
    disk: str = "5GB",
) -> ResourceSpec:
    """Build ResourceSpec from CLI arguments.

    When ``tpu`` contains multiple comma-separated variants, the first one is
    used as the canonical device (its chip count drives resource accounting),
    and the alternatives are surfaced separately via ``build_tpu_alternatives``
    so the caller can attach a ``device_variant_constraint`` accepting any of
    them. All variants must have the same ``vm_count``.
    """
    spec = ResourceSpec(cpu=cpu, memory=memory, disk=disk)

    if tpu:
        primary, _ = _parse_tpu_alternatives(tpu)
        spec.device = tpu_device(primary)
    elif gpu:
        variant, count = parse_gpu_spec(gpu)
        spec.device = gpu_device(variant, count)

    return spec


def _parse_tpu_alternatives(tpu_arg: str) -> tuple[str, list[str]]:
    """Split a ``--tpu`` value into (primary, alternatives).

    The CLI accepts a comma-separated list (e.g. ``v6e-4,v5litepod-4``) so a
    single job can be schedulable on any of the listed variants. The first
    variant is canonical; the rest are alternatives. All listed variants must
    share the same ``vm_count`` so multinode coscheduling stays consistent.
    """
    variants = [v.strip() for v in tpu_arg.split(",") if v.strip()]
    if not variants:
        raise click.BadParameter("--tpu must specify at least one TPU variant")
    if len(variants) == 1:
        return variants[0], []

    primary = variants[0]
    alternatives = variants[1:]
    primary_topo = get_tpu_topology(primary)
    for alt in alternatives:
        alt_topo = get_tpu_topology(alt)
        if alt_topo.vm_count != primary_topo.vm_count:
            raise click.BadParameter(
                f"TPU alternative {alt!r} has vm_count={alt_topo.vm_count} "
                f"but primary {primary!r} has vm_count={primary_topo.vm_count}. "
                f"All TPU alternatives must share the same vm_count."
            )
    return primary, alternatives


def build_tpu_alternatives(tpu_arg: str | None) -> list[str]:
    """Return the list of all TPU variants requested via ``--tpu``."""
    if not tpu_arg:
        return []
    primary, alternatives = _parse_tpu_alternatives(tpu_arg)
    return [primary, *alternatives]


# Thresholds above which the entrypoint job is considered "extra-resource-heavy"
# and requires --enable-extra-resources to proceed.
_LARGE_MEMORY_THRESHOLD_BYTES: int = humanfriendly.parse_size("4GB")
_LARGE_DISK_THRESHOLD_BYTES: int = humanfriendly.parse_size("10GB")

_ACCELERATOR_HINT = (
    "The top-level entrypoint (coordinator) job only needs CPU to schedule and "
    "dispatch work; accelerators are attached to worker tasks spawned by the job. "
    "If you truly need an accelerator on this entrypoint, pass --enable-extra-resources."
)
_LARGE_RESOURCE_HINT = (
    "The top-level entrypoint (coordinator) job typically needs only modest CPU/RAM/disk "
    "to schedule and dispatch work. "
    "If this large resource request is intentional, pass --enable-extra-resources."
)


def validate_extra_resources(
    tpu: str | None,
    gpu: str | None,
    memory: str,
    disk: str,
    enable_extra_resources: bool,
) -> None:
    """Raise UsageError if heavy resources are requested without --enable-extra-resources.

    Guards against common mistakes where users attach accelerators or request
    large RAM/disk on the entrypoint (coordinator) job instead of on worker tasks.

    Args:
        tpu: TPU type string, or None.
        gpu: GPU spec string, or None.
        memory: Memory size string (e.g. "8GB").
        disk: Disk size string (e.g. "64GB").
        enable_extra_resources: True if the user explicitly opted in.
    """
    if enable_extra_resources:
        return

    if tpu:
        raise click.UsageError(f"--tpu requires --enable-extra-resources.\n{_ACCELERATOR_HINT}")

    if gpu:
        raise click.UsageError(f"--gpu requires --enable-extra-resources.\n{_ACCELERATOR_HINT}")

    try:
        memory_bytes = humanfriendly.parse_size(memory)
    except humanfriendly.InvalidSize:
        memory_bytes = 0  # let build_resources surface the parse error

    if memory_bytes >= _LARGE_MEMORY_THRESHOLD_BYTES:
        raise click.UsageError(f"--memory {memory} (>= 4 GB) requires --enable-extra-resources.\n{_LARGE_RESOURCE_HINT}")

    try:
        disk_bytes = humanfriendly.parse_size(disk)
    except humanfriendly.InvalidSize:
        disk_bytes = 0  # let build_resources surface the parse error

    if disk_bytes >= _LARGE_DISK_THRESHOLD_BYTES:
        raise click.UsageError(f"--disk {disk} (>= 10 GB) requires --enable-extra-resources.\n{_LARGE_RESOURCE_HINT}")


def parse_reservation_spec(spec: str) -> list[ReservationEntry]:
    """Parse a reservation spec like '4:H100x8' or 'v5litepod-16'.

    Format: [COUNT:]DEVICE_SPEC
    Tries to resolve DEVICE_SPEC as a known TPU variant first, then falls back
    to GPU parsing via parse_gpu_spec.
    """
    count = 1
    device_spec = spec
    if ":" in spec:
        count_str, device_spec = spec.split(":", 1)
        count = int(count_str)
        if count < 1:
            raise ValueError(f"Reservation count must be >= 1, got {count}")

    try:
        get_tpu_topology(device_spec)
        device = tpu_device(device_spec)
    except ValueError:
        variant, gpu_count = parse_gpu_spec(device_spec)
        device = gpu_device(variant, gpu_count)

    resources = ResourceSpec(device=device)
    return [ReservationEntry(resources=resources) for _ in range(count)]


def generate_job_name(command: list[str]) -> str:
    """Generate a job name from the command."""
    script_name = "job"
    for arg in command:
        path = Path(arg)
        if path.suffix == ".py":
            script_name = path.stem
            break

    timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    return f"iris-run-{script_name}-{timestamp}"


def resolve_multinode_defaults(
    tpu: str | None,
    gpu: str | None,
    replicas: int | None,
) -> tuple[int, CoschedulingConfig | None]:
    """Auto-detect multinode topology and set replicas/coscheduling.

    For TPUs with vm_count > 1, infers replicas from the topology and enables
    coscheduling by ``tpu-name`` so that all tasks land on workers in the same
    TPU slice. For GPUs with replicas > 1, enables coscheduling by ``pool`` so
    that all replicas are scheduled together.

    Args:
        tpu: TPU type string (e.g. ``"v6e-32"``), or ``None``.
        gpu: GPU type string (e.g. ``"H100"``), or ``None``.
        replicas: Explicit replica count from the caller, or ``None`` if not
            specified (meaning the default should be inferred).

    Returns:
        A ``(replicas, coscheduling)`` tuple.  ``coscheduling`` is ``None``
        for single-host or non-multinode jobs.
    """
    if not tpu:
        if gpu and replicas is not None and replicas > 1:
            return replicas, CoschedulingConfig(group_by="pool")
        return replicas or 1, None

    try:
        topo = get_tpu_topology(tpu)
    except ValueError:
        return replicas or 1, None

    if topo.vm_count <= 1:
        return replicas or 1, None

    # Multinode TPU: auto-set replicas and coscheduling.
    if replicas is None:
        replicas = topo.vm_count
        logger.info(
            f"Multinode TPU '{tpu}' detected (vm_count={topo.vm_count}). "
            f"Auto-setting replicas={replicas} and coscheduling by tpu-name."
        )
    else:
        logger.info(
            f"Multinode TPU '{tpu}' detected (vm_count={topo.vm_count}). "
            f"Using explicit replicas={replicas} with coscheduling by tpu-name."
        )

    coscheduling = CoschedulingConfig(group_by=WellKnownAttribute.TPU_NAME)
    return replicas, coscheduling


def build_job_constraints(
    resources_proto: job_pb2.ResourceSpecProto,
    tpu_variants: list[str],
    replicas: int,
    regions: tuple[str, ...] | None = None,
    zone: str | None = None,
    preemptible: bool | None = None,
) -> list[Constraint]:
    """Assemble the constraint list for a submitted job.

    An explicit ``preemptible`` value wins over the executor heuristic:
    ``infer_preemptible_constraint`` short-circuits when any preemptible
    constraint is already present, so we append the user's choice first.
    """
    constraints: list[Constraint] = []
    if regions:
        constraints.append(region_constraint(list(regions)))
    if zone:
        constraints.append(zone_constraint(zone))
    if len(tpu_variants) > 1:
        constraints.append(device_variant_constraint(tpu_variants))
    if preemptible is not None:
        constraints.append(preemptible_constraint(preemptible))

    # Executor heuristic: small CPU-only CLI jobs (no accelerators, 1 replica,
    # CPU ≤ 0.5 cores, RAM ≤ 4 GiB) are auto-tagged as non-preemptible so
    # coordinators survive spot reclamation. Skipped when the user supplied
    # --preemptible / --no-preemptible.
    inferred = infer_preemptible_constraint(resources_proto, replicas, constraints)
    if inferred is not None:
        constraints.append(inferred)
        logger.info("Executor heuristic: auto-tagging job as non-preemptible")
    return constraints


def run_iris_job(
    command: list[str],
    env_vars: dict[str, str],
    controller_url: str,
    tpu: str | None = None,
    gpu: str | None = None,
    cpu: float = 0.5,
    memory: str = "1GB",
    disk: str = "5GB",
    wait: bool = True,
    job_name: str | None = None,
    replicas: int | None = None,
    max_retries: int = 0,
    timeout: int = 0,
    extras: list[str] | None = None,
    terminate_on_exit: bool = True,
    regions: tuple[str, ...] | None = None,
    zone: str | None = None,
    user: str | None = None,
    reserve: tuple[str, ...] | None = None,
    priority: str | None = None,
    preemptible: bool | None = None,
    token_provider: TokenProvider | None = None,
    submit_argv: list[str] | None = None,
) -> int:
    """Core job submission logic.

    Args:
        controller_url: Controller URL (from parent context tunnel).
        terminate_on_exit: If True, terminate the job on any non-normal exit
            (KeyboardInterrupt, unexpected exceptions). Normal completion is unaffected.
        regions: If provided, restrict the job to workers in these regions.
        zone: If provided, restrict the job to workers in this zone.
        reserve: Reservation specs (e.g., ("4:H100x8", "v5litepod-16")).
        preemptible: If True/False, force scheduling on (non-)preemptible workers
            and bypass the executor heuristic. If None (default), the heuristic runs.

    Returns:
        Exit code: 0 for success, 1 for failure
    """
    env_vars = add_standard_env_vars(env_vars)
    resources = build_resources(tpu, gpu, cpu=cpu, memory=memory, disk=disk)
    job_name = job_name or generate_job_name(command)
    extras = extras or []

    tpu_variants = build_tpu_alternatives(tpu)
    primary_tpu = tpu_variants[0] if tpu_variants else None

    replicas, coscheduling = resolve_multinode_defaults(primary_tpu, gpu, replicas)

    resources_proto = resources.to_proto()
    constraints = build_job_constraints(
        resources_proto=resources_proto,
        tpu_variants=tpu_variants,
        replicas=replicas,
        regions=regions,
        zone=zone,
        preemptible=preemptible,
    )

    reservation: list[ReservationEntry] | None = None
    if reserve:
        # --reserve is mutually exclusive with --region/--zone: the controller's
        # claim loop only evaluates each reservation entry's own constraints, so
        # job-level routing constraints would not gate worker claims (#4988).
        # A caller who needs a specific region/zone should name it directly; a
        # caller who uses a reservation is by definition not picking the region.
        if regions or zone:
            raise click.UsageError(
                "--reserve cannot be combined with --region or --zone. "
                "Use --region/--zone to target a specific location, or --reserve "
                "to claim from a reservation (which chooses the location for you)."
            )
        reservation = []
        for spec in reserve:
            reservation.extend(parse_reservation_spec(spec))

    logger.info(f"Submitting job: {job_name}")
    logger.info(f"Command: {' '.join(command)}")
    logger.info(f"Resources: cpu={resources.cpu:g}, memory={resources.memory}, disk={resources.disk}")
    if resources.device and resources.device.HasField("tpu"):
        if len(tpu_variants) > 1:
            logger.info(f"TPU: {resources.device.tpu.variant} (alternatives: {', '.join(tpu_variants[1:])})")
        else:
            logger.info(f"TPU: {resources.device.tpu.variant}")
    if resources.device and resources.device.HasField("gpu"):
        gpu_dev = resources.device.gpu
        logger.info(f"GPU: {gpu_dev.count}x {gpu_dev.variant or 'any'}")
    if replicas > 1:
        logger.info(f"Replicas: {replicas}")
    if coscheduling:
        logger.info(f"Coscheduling: group_by={coscheduling.group_by}")
    if regions:
        logger.info(f"Region constraint: {', '.join(regions)}")
    if zone:
        logger.info(f"Zone constraint: {zone}")
    if preemptible is not None:
        logger.info(f"Preemptible constraint: {preemptible}")
    if reservation:
        logger.info(f"Reservation: {len(reservation)} entries")

    logger.info(f"Using controller: {controller_url}")
    priority_band = job_pb2.PRIORITY_BAND_UNSPECIFIED
    if priority is not None:
        priority_band = priority_band_value(priority)
        logger.info(f"Priority band: {priority}")

    return _submit_and_wait_job(
        controller_url=controller_url,
        job_name=job_name,
        command=command,
        resources=resources,
        env_vars=env_vars,
        replicas=replicas,
        max_retries=max_retries,
        timeout=timeout,
        wait=wait,
        extras=extras,
        terminate_on_exit=terminate_on_exit,
        constraints=constraints or None,
        coscheduling=coscheduling,
        user=user,
        reservation=reservation,
        priority_band=priority_band,
        token_provider=token_provider,
        submit_argv=submit_argv,
    )


def _submit_and_wait_job(
    controller_url: str,
    job_name: str,
    command: list[str],
    resources: ResourceSpec,
    env_vars: dict[str, str],
    replicas: int,
    max_retries: int,
    timeout: int,
    wait: bool,
    extras: list[str] | None = None,
    terminate_on_exit: bool = True,
    constraints: list[Constraint] | None = None,
    coscheduling: CoschedulingConfig | None = None,
    user: str | None = None,
    reservation: list[ReservationEntry] | None = None,
    priority_band: job_pb2.PriorityBand = job_pb2.PRIORITY_BAND_UNSPECIFIED,
    token_provider: TokenProvider | None = None,
    submit_argv: list[str] | None = None,
) -> int:
    """Submit job and optionally wait for completion.

    Only KeyboardInterrupt terminates the remote job; connection failures
    are logged and re-raised without killing the job.
    """
    client = IrisClient.remote(controller_url, workspace=Path.cwd(), token_provider=token_provider)
    entrypoint = Entrypoint.from_command(*command)

    job = client.submit(
        entrypoint=entrypoint,
        name=job_name,
        resources=resources,
        environment=EnvironmentSpec(env_vars=env_vars, extras=extras or []),
        constraints=constraints,
        coscheduling=coscheduling,
        replicas=replicas,
        max_retries_failure=max_retries,
        timeout=Duration.from_seconds(timeout) if timeout else None,
        user=user,
        reservation=reservation,
        priority_band=priority_band,
        submit_argv=submit_argv,
    )

    logger.info(f"Job submitted: {job.job_id}")
    click.echo(str(job.job_id))

    if not wait:
        return 0

    logger.info(
        "Streaming logs (Ctrl+C to stop). If disconnected, reconnect with: iris job logs -f %s",
        job.job_id,
    )
    try:
        try:
            status = job.wait(stream_logs=True, timeout=float("inf"))
            logger.info(f"Job completed with state: {status.state}")
            return 0 if status.state == job_pb2.JOB_STATE_SUCCEEDED else 1
        except JobFailedError as e:
            logger.error(f"Job failed: {e}")
            return 1
    except KeyboardInterrupt:
        if terminate_on_exit:
            logger.info(f"Terminating job {job.job_id}...")
            terminated = _terminate_jobs(client, (str(job.job_id),), include_children=True)
            for t in terminated:
                logger.info(f"  Terminated: {t}")
        return 130
    except Exception:
        logger.warning(
            "Connection lost; job %s is still running. Reconnect with: iris job logs -f %s",
            job.job_id,
            job.job_id,
        )
        raise


@click.group("job")
def job() -> None:
    """Manage Iris jobs."""


@job.command(
    "run",
    context_settings={"ignore_unknown_options": True},
    help="""Submit jobs to Iris clusters.

Examples:

  \b
  # Simple CPU job
  iris --config cluster.yaml job run -- python script.py

  \b
  # TPU job with environment variables
  iris --config cluster.yaml job run --tpu v5litepod-16 \\
    -e WANDB_API_KEY $WANDB_API_KEY -- python train.py

  \b
  # Submit and detach
  iris --config cluster.yaml job run --no-wait -- python long_job.py
""",
)
@click.option(
    "-e",
    "--env-vars",
    "env_vars",
    multiple=True,
    type=(str, str),
    help="Set environment variables for the job (KEY VALUE). Can be repeated.",
)
@click.option(
    "--tpu",
    type=str,
    help=(
        "TPU type to request (e.g., v5litepod-16). Pass a comma-separated list "
        "(e.g., v6e-4,v5litepod-4) to allow scheduling on any of the listed "
        "variants — useful when capacity is contested. All variants must share "
        "the same vm_count. Requires --enable-extra-resources."
    ),
)
@click.option(
    "--gpu",
    type=str,
    help="GPU spec: VARIANTxCOUNT (e.g., H100x8), COUNT (e.g., 8), or VARIANT (e.g., H100). Needs --enable-extra-resources.",  # noqa: E501
)
@click.option(
    "--enable-extra-resources",
    is_flag=True,
    default=False,
    help=(
        "Allow accelerators (--tpu/--gpu) and large resource requests (>= 4 GB RAM or >= 10 GB disk) "
        "on the entrypoint job. Not needed for typical coordinator jobs — accelerators should be "
        "requested by worker tasks spawned by the job."
    ),
)
@click.option("--cpu", type=float, default=0.5, show_default=True, help="Number of CPUs to request")
@click.option("--memory", type=str, default="1GB", show_default=True, help="Memory size to request (e.g., 8GB, 512MB)")
@click.option(
    "--disk", type=str, default="5GB", show_default=True, help="Ephemeral disk size to request (e.g., 64GB, 1TB)"
)
@click.option("--no-wait", is_flag=True, help="Don't wait for job completion")
@click.option("--job-name", type=str, help="Custom job name (default: auto-generated)")
@click.option("--user", type=str, help="Override the user prefix for the submitted job.")
@click.option(
    "--replicas", type=int, default=None, help="Number of tasks for gang scheduling (auto-detected for multinode TPUs)"
)
@click.option("--max-retries", type=int, default=0, help="Max retries on failure (default: 0)")
@click.option("--timeout", type=int, default=0, show_default=True, help="Job timeout in seconds (0 = no timeout)")
@click.option("--region", multiple=True, help="Restrict to region(s) (e.g., --region us-central2). Can be repeated.")
@click.option("--zone", type=str, help="Restrict to zone (e.g., --zone us-central2-b).")
@click.option("--extra", multiple=True, help="UV extras to install (e.g., --extra cpu). Can be repeated.")
@click.option(
    "--reserve",
    multiple=True,
    help=(
        "Reserve workers before scheduling. Format: [COUNT:]DEVICE "
        "(e.g., 4:H100x8, v5litepod-16). Can be repeated. Reservation does not "
        "attach accelerator devices to the task; use --tpu/--gpu for accelerator jobs."
    ),
)
@click.option(
    "--priority",
    type=click.Choice(PRIORITY_BAND_NAMES, case_sensitive=False),
    default=None,
    help="Priority band for scheduling (default: interactive). Lower bands run first; batch jobs yield to interactive.",
)
@click.option(
    "--preemptible/--no-preemptible",
    "preemptible",
    default=None,
    help=(
        "Force scheduling on preemptible (--preemptible) or non-preemptible "
        "(--no-preemptible) workers. Overrides the executor heuristic. "
        "Default: heuristic-based (small CPU-only jobs pinned to non-preemptible)."
    ),
)
@click.option(
    "--terminate-on-exit/--no-terminate-on-exit",
    default=True,
    help="Terminate the job on Ctrl+C (default: terminate). Tunnel failures never kill the job.",
)
@click.argument("cmd", nargs=-1, type=click.UNPROCESSED, required=True)
@click.pass_context
def run(
    ctx,
    env_vars: tuple[tuple[str, str], ...],
    tpu: str | None,
    gpu: str | None,
    cpu: float,
    memory: str,
    disk: str,
    enable_extra_resources: bool,
    no_wait: bool,
    job_name: str | None,
    user: str | None,
    replicas: int | None,
    max_retries: int,
    timeout: int,
    region: tuple[str, ...],
    zone: str | None,
    extra: tuple[str, ...],
    reserve: tuple[str, ...],
    priority: str | None,
    preemptible: bool | None,
    terminate_on_exit: bool,
    cmd: tuple[str, ...],
):
    """Submit jobs to Iris clusters."""
    controller_url = require_controller_url(ctx)
    validate_extra_resources(tpu, gpu, memory, disk, enable_extra_resources)
    validate_region_zone(region or None, zone, ctx.obj.get("config"))

    command = list(cmd)
    if not command:
        raise click.UsageError("No command provided after --")

    submit_argv = redact_submit_argv(list(sys.argv))

    # ignore_unknown_options silently passes typo'd flags (e.g. --reservation
    # instead of --reserve) into cmd. Catch any flags that leaked through
    # before the actual command starts — these were meant for iris, not the
    # user's program.
    for arg in command:
        if not arg.startswith("-"):
            break
        raise click.UsageError(
            f"Unknown option {arg!r}. Iris options must come before '--'. Did you mean a different flag?"
        )

    env_vars_dict = load_env_vars(env_vars)

    try:
        exit_code = run_iris_job(
            command=command,
            env_vars=env_vars_dict,
            controller_url=controller_url,
            tpu=tpu,
            gpu=gpu,
            cpu=cpu,
            memory=memory,
            disk=disk,
            wait=not no_wait,
            job_name=job_name,
            user=user,
            replicas=replicas,
            max_retries=max_retries,
            timeout=timeout,
            extras=list(extra),
            terminate_on_exit=terminate_on_exit,
            regions=region or None,
            zone=zone,
            reserve=reserve or None,
            priority=priority,
            preemptible=preemptible,
            token_provider=ctx.obj.get("token_provider"),
            submit_argv=submit_argv,
        )
    except Exception:
        bundle = ctx.obj.get("provider_bundle")
        if bundle is not None:
            try:
                bundle.controller.debug_report()
            except Exception:
                logger.debug("Controller post-mortem failed", exc_info=True)
        raise

    sys.exit(exit_code)


@job.command("stop")
@click.argument("job_id", nargs=-1, required=True)
@click.option(
    "--include-children/--no-include-children",
    default=True,
    help="Terminate child jobs under the given job ID prefix (default: include).",
)
@click.pass_context
def stop(ctx, job_id: tuple[str, ...], include_children: bool) -> None:
    """Terminate one or more jobs."""
    controller_url = require_controller_url(ctx)
    client = IrisClient.remote(controller_url, workspace=Path.cwd(), token_provider=ctx.obj.get("token_provider"))
    terminated = _terminate_jobs(client, job_id, include_children)
    _print_terminated(terminated)


@job.command("kill")
@click.argument("job_id", nargs=-1, required=True)
@click.option(
    "--include-children/--no-include-children",
    default=True,
    help="Terminate child jobs under the given job ID prefix (default: include).",
)
@click.pass_context
def kill(ctx, job_id: tuple[str, ...], include_children: bool) -> None:
    """Terminate one or more jobs (alias for stop)."""
    controller_url = require_controller_url(ctx)
    client = IrisClient.remote(controller_url, workspace=Path.cwd(), token_provider=ctx.obj.get("token_provider"))
    terminated = _terminate_jobs(client, job_id, include_children)
    _print_terminated(terminated)


@job.command("list")
@click.option("--state", type=str, default=None, help="Filter by state (e.g., running, pending, failed)")
@click.option("--prefix", type=str, default=None, help="Filter by job name prefix")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_context
def list_jobs(ctx, state: str | None, prefix: str | None, json_output: bool) -> None:
    """List jobs with optional filtering."""
    controller_url = require_controller_url(ctx)
    client = IrisClient.remote(controller_url, workspace=Path.cwd(), token_provider=ctx.obj.get("token_provider"))

    state_value: job_pb2.JobState | None = None
    if state is not None:
        state_lower = state.lower()
        if state_lower not in _STATE_MAP:
            valid = ", ".join(sorted(_STATE_MAP.keys()))
            raise click.UsageError(f"Unknown state '{state}'. Valid states: {valid}")
        state_value = _STATE_MAP[state_lower]

    prefix_name = JobName.from_wire(prefix) if prefix else None
    jobs = client.list_jobs(state=state_value, prefix=prefix_name)

    # Sort by submitted_at descending (most recent first)
    jobs.sort(key=lambda j: j.submitted_at.epoch_ms, reverse=True)

    if json_output:
        serialized = [json_format.MessageToDict(j, preserving_proto_field_name=True) for j in jobs]
        click.echo(json.dumps(serialized, indent=2))
        return

    if not jobs:
        click.echo("No jobs found.")
        return

    # Build table rows
    rows: list[list[str]] = []
    has_reasons = False

    for j in jobs:
        job_id = j.job_id
        state_name = job_state_friendly(j.state)
        submitted = timestamp_from_proto(j.submitted_at).as_formatted_date() if j.submitted_at.epoch_ms else "-"
        resources = _format_resources(j.resources) if j.HasField("resources") else "-"

        # Show error for failed jobs, pending_reason for pending/unschedulable
        reason = j.error or j.pending_reason or ""
        if reason:
            has_reasons = True
            # Truncate long reasons
            reason = (reason[:60] + "...") if len(reason) > 63 else reason

        rows.append([job_id, state_name, resources, submitted, reason])

    # Build headers - only include REASON column if there are any reasons
    if has_reasons:
        headers = ["JOB ID", "STATE", "RESOURCES", "SUBMITTED", "REASON"]
    else:
        headers = ["JOB ID", "STATE", "RESOURCES", "SUBMITTED"]
        rows = [row[:4] for row in rows]

    click.echo(tabulate(rows, headers=headers, tablefmt="plain"))


def _task_index(task_id: str) -> str:
    last = task_id.rsplit("/", 1)[-1]
    return last or task_id


def _task_duration_ms(task: job_pb2.TaskStatus) -> int | None:
    if not task.started_at.epoch_ms:
        return None
    end_ms = task.finished_at.epoch_ms or Timestamp.now().epoch_ms()
    return max(0, end_ms - task.started_at.epoch_ms)


def _format_duration_ms(ms: int | None) -> str:
    if ms is None:
        return "-"
    return humanfriendly.format_timespan(ms / 1000)


def _format_memory_mb(mb: int) -> str:
    if not mb:
        return "-"
    return humanfriendly.format_size(mb * 1_000_000)


def build_job_summary(
    job_status: job_pb2.JobStatus,
    tasks: list[job_pb2.TaskStatus],
) -> dict:
    """Build a structured job/task summary (CLI + test entry point).

    Returns a dict with job-level fields and a per-task list including
    peak memory, final state, exit code, and duration. Pure function over
    protos — no RPC calls — so it can be unit-tested without a cluster.
    """
    task_summaries = []

    def _sort_key(t: job_pb2.TaskStatus) -> tuple[int, str]:
        idx = _task_index(t.task_id)
        try:
            return (int(idx), "")
        except ValueError:
            return (2**31, idx)

    for t in sorted(tasks, key=_sort_key):
        usage = t.resource_usage
        task_summaries.append(
            {
                "task_id": t.task_id,
                "index": _task_index(t.task_id),
                "state": task_state_friendly(t.state),
                # Only surface exit_code once the task is terminal. Proto scalar
                # defaults mean a RUNNING/ASSIGNED/BUILDING task would otherwise
                # report exit=0 and look like a clean success.
                "exit_code": int(t.exit_code) if t.state in TERMINAL_TASK_STATES else None,
                "duration_ms": _task_duration_ms(t),
                "memory_mb": int(usage.memory_mb) if usage.memory_mb else 0,
                "memory_peak_mb": int(usage.memory_peak_mb) if usage.memory_peak_mb else 0,
                "cpu_millicores": int(usage.cpu_millicores) if usage.cpu_millicores else 0,
                "disk_mb": int(usage.disk_mb) if usage.disk_mb else 0,
                "worker_id": t.worker_id,
                "error": t.error,
            }
        )

    return {
        "job_id": job_status.job_id,
        "name": job_status.name,
        "state": job_state_friendly(job_status.state),
        "exit_code": int(job_status.exit_code),
        "error": job_status.error,
        "failure_count": int(job_status.failure_count),
        "preemption_count": int(job_status.preemption_count),
        "task_count": int(job_status.task_count),
        "completed_count": int(job_status.completed_count),
        "task_state_counts": dict(job_status.task_state_counts),
        "tasks": task_summaries,
    }


def _render_job_summary_text(summary: dict) -> str:
    lines = [
        f"Job: {summary['job_id']}" + (f" ({summary['name']})" if summary["name"] else ""),
        f"State: {summary['state']}  exit={summary['exit_code']}  "
        f"failures={summary['failure_count']}  preemptions={summary['preemption_count']}",
        f"Tasks: {summary['completed_count']}/{summary['task_count']} completed  "
        + "  ".join(f"{k}={v}" for k, v in sorted(summary["task_state_counts"].items()) if v),
    ]
    if summary["error"]:
        lines.append(f"Error: {summary['error']}")
    lines.append("")

    rows = []
    for t in summary["tasks"]:
        rows.append(
            [
                t["index"],
                t["state"],
                "-" if t["exit_code"] is None else t["exit_code"],
                _format_duration_ms(t["duration_ms"]),
                _format_memory_mb(t["memory_peak_mb"]),
                _format_memory_mb(t["memory_mb"]),
                (t["error"] or "")[:50] + ("..." if len(t["error"] or "") > 50 else ""),
            ]
        )
    headers = ["TASK", "STATE", "EXIT", "DURATION", "PEAK MEM", "CUR MEM", "ERROR"]
    lines.append(tabulate(rows, headers=headers, tablefmt="plain"))
    return "\n".join(lines)


@job.command("summary")
@click.argument("job_id")
@click.option("--json", "json_output", is_flag=True, help="Emit structured JSON instead of a text table.")
@click.pass_context
def summary(ctx, job_id: str, json_output: bool) -> None:
    """Print a per-task summary (peak memory, state, exit, duration) for a job.

    Works for both running and completed jobs. Data is read from the controller's
    existing ``GetJobStatus`` / ``ListTasks`` RPCs (no checkpoint scraping).
    """
    controller_url = require_controller_url(ctx)
    client = IrisClient.remote(controller_url, workspace=Path.cwd(), token_provider=ctx.obj.get("token_provider"))
    job_name = JobName.from_wire(job_id)
    job_status = client.status(job_name)
    tasks = client.list_tasks(job_name)
    result = build_job_summary(job_status, tasks)
    if json_output:
        click.echo(json.dumps(result, indent=2, default=str))
        return
    click.echo(_render_job_summary_text(result))


@job.command("logs")
@click.argument("job_id")
@click.option("--since-ms", type=int, default=None, help="Only show logs after this epoch millisecond timestamp.")
@click.option(
    "--since-seconds",
    type=int,
    default=None,
    help="Only show logs from the last N seconds.",
)
@click.option("--follow", "-f", is_flag=True, help="Stream logs continuously.")
@click.option(
    "--max-lines",
    type=int,
    default=0,
    help="Maximum number of log lines to return (0 = server default, currently 1000).",
)
@click.option("--tail/--no-tail", default=True, help="Return the most recent lines instead of the earliest.")
@click.option(
    "--level",
    type=click.Choice(["debug", "info", "warning", "error", "critical"], case_sensitive=False),
    default=None,
    help="Minimum log level to display (e.g., --level warning).",
)
@click.pass_context
def logs(
    ctx,
    job_id: str,
    since_ms: int | None,
    since_seconds: int | None,
    follow: bool,
    max_lines: int,
    tail: bool,
    level: str | None,
) -> None:
    """Stream task logs for a job using batch log fetching."""
    if since_ms is not None and since_seconds is not None:
        raise click.UsageError("Specify only one of --since-ms or --since-seconds.")

    controller_url = require_controller_url(ctx)
    client = IrisClient.remote(controller_url, workspace=Path.cwd(), token_provider=ctx.obj.get("token_provider"))

    if since_seconds is not None:
        since_ms = Timestamp.now().epoch_ms() - (since_seconds * 1000)

    start_since_ms = since_ms or 0
    job_name = JobName.from_wire(job_id)

    min_level = level.upper() if level else ""

    if follow:
        job = Job(client, job_name)
        job.wait(
            stream_logs=True,
            timeout=float("inf"),
            raise_on_failure=False,
            since_ms=start_since_ms,
            min_level=min_level,
        )
        return

    entries = client.fetch_task_logs(
        job_name,
        start=Timestamp.from_ms(start_since_ms) if start_since_ms > 0 else None,
        max_lines=max_lines,
        tail=tail,
        min_level=min_level,
    )
    for entry in entries:
        ts = entry.timestamp.as_short_time()
        click.echo(f"[{ts}] task={entry.task_id} | {entry.data}")


@job.command("bug-report")
@click.argument("job_id")
@click.option("--file-issue", is_flag=True, help="File a GitHub issue with the report")
@click.option("--repo", type=str, default=None, help="GitHub repo (default: auto-detect from git remote)")
@click.option("--tail", type=int, default=50, help="Recent log lines per task to include")
@click.option("--labels", type=str, default="bug", help="Comma-separated labels for the GitHub issue")
@click.pass_context
def bug_report(ctx, job_id: str, file_issue: bool, repo: str | None, tail: int, labels: str):
    """Generate a diagnostic bug report for a job."""
    controller_url = require_controller_url(ctx)
    report = gather_bug_report(
        controller_url, JobName.from_wire(job_id), tail=tail, token_provider=ctx.obj.get("token_provider")
    )
    markdown = format_bug_report(report)

    if file_issue:
        title = f"[Iris] Job {report.job_id} {report.state_name}: {report.error_summary}"
        url = file_github_issue(title, markdown, repo=repo, labels=labels.split(","))
        if url:
            click.echo(f"Filed issue: {url}")
        else:
            click.echo("Failed to file issue. Report printed below:\n")
            click.echo(markdown)
    else:
        click.echo(markdown)
