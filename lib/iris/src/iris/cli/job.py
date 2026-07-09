# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Job management via command passthrough (replaces ``iris-run``).

Usage:
    iris --config cluster.yaml job run -- python train.py --epochs 10
    iris --config cluster.yaml job run --tpu v5litepod-16 -e WANDB_API_KEY $WANDB_API_KEY -- python train.py
"""

import csv
import difflib
import logging
import os
import sys
import time
from pathlib import Path

import click
import humanfriendly
import yaml
from rigging.credentials import ClientCredentials
from rigging.timing import Duration, Timestamp
from tabulate import tabulate

from iris.cli.connect import iris_client_for_ctx, require_controller_url
from iris.client import IrisClient
from iris.client.client import Job, JobFailedError
from iris.cluster.constraints import (
    CLUSTER_CONSTRAINT_KEY,
    Constraint,
    ConstraintOp,
    WellKnownAttribute,
    availability_constraint,
    device_variant_constraint,
    get_device_variant,
    infer_preemptible_constraint,
    preemptible_constraint,
    region_constraint,
    zone_constraint,
)
from iris.cluster.redaction import redact_submit_argv
from iris.cluster.tpu_topology import get_tpu_topology
from iris.cluster.types import (
    TERMINAL_TASK_STATES,
    CoschedulingConfig,
    Entrypoint,
    EnvironmentSpec,
    JobName,
    ResourceSpec,
    gpu_device,
    tpu_device,
)
from iris.rpc import job_pb2
from iris.rpc.proto_display import (
    CONTAINER_PROFILE_NAMES,
    PRIORITY_BAND_NAMES,
    job_state_friendly,
    priority_band_value,
    task_state_friendly,
)
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


def _remote_client(ctx: click.Context) -> IrisClient:
    return iris_client_for_ctx(ctx, workspace=Path.cwd())


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


def _read_targets_from_stdin() -> list[str]:
    """Read Iris job/task ids from stdin, one per line.

    Parses each line as a CSV record (symmetric with ``iris query -f csv``, which
    quotes fields through ``csv.writer``) and keeps the first field when it looks
    like an Iris id (leading ``/``). This consumes ``iris query -f csv`` output
    directly — a header row and trailing columns are dropped, and ids that
    contain a comma (quoted by the writer) or a space survive intact, since a
    JobName component may hold either:

        iris query -f csv "SELECT task_id FROM tasks WHERE ..." | iris job kick --stdin
    """
    targets: list[str] = []
    for row in csv.reader(sys.stdin):
        if not row:
            continue
        field = row[0].strip()
        if field.startswith("/"):
            targets.append(field)
    return targets


def _collect_targets(targets: tuple[str, ...], use_stdin: bool) -> list[str]:
    """Merge positional targets with stdin ids for a bulk action.

    A literal ``-`` among the positionals, or ``use_stdin``, appends the ids read
    from stdin to the positional ids, letting a query pipe straight into an
    action. The ``-`` sentinel is consumed, not returned as a target.
    """
    read_stdin = use_stdin or "-" in targets
    collected = [t for t in targets if t != "-"]
    if read_stdin:
        collected.extend(_read_targets_from_stdin())
    return collected


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
        "GH200",
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


def _find_closest(value: str, known: set[str]) -> str | None:
    """Return the closest match from *known* by sequence similarity, or None."""
    matches = difflib.get_close_matches(value, sorted(known), n=1, cutoff=0.6)
    return matches[0] if matches else None


def _known_regions_and_zones(config) -> tuple[set[str], set[str]]:
    """Extract known regions and zones from an IrisClusterConfig.

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


def reserve_spec_to_availability(spec: str) -> Constraint:
    """Turn a ``--reserve`` spec like ``4:H100x8`` or ``v5litepod-16`` into a hard
    ``availability:<variant>`` constraint.

    Format: ``[COUNT:]DEVICE_SPEC``. The count is ignored — availability is a
    zone-level placement constraint ("schedule me where this accelerator can be
    found"), not a capacity hold, so the number of workers is meaningless.
    DEVICE_SPEC resolves as a known TPU variant first, then falls back to a GPU
    spec.
    """
    device_spec = spec.split(":", 1)[1] if ":" in spec else spec

    try:
        get_tpu_topology(device_spec)
        device = tpu_device(device_spec)
    except ValueError:
        variant, gpu_count = parse_gpu_spec(device_spec)
        device = gpu_device(variant, gpu_count)

    variant = get_device_variant(device)
    if not variant:
        raise click.UsageError(f"--reserve {spec!r} does not name an accelerator variant.")
    return availability_constraint(variant)


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
    TPU slice. For GPUs with replicas > 1, enables coscheduling by ``leafgroup``
    (the H100 InfiniBand multi-node colocation level) so all replicas are
    scheduled together.

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
            return replicas, CoschedulingConfig(group_by="leafgroup")
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
    target_cluster: str | None = None,
) -> list[Constraint]:
    """Assemble the constraint list for a submitted job.

    An explicit ``preemptible`` value wins over the executor heuristic:
    ``infer_preemptible_constraint`` short-circuits when any preemptible
    constraint is already present, so we append the user's choice first.

    ``target_cluster``, if set, appends a ``cluster EQ <peer>`` federation
    pin (``CLUSTER_CONSTRAINT_KEY``) that routes the whole job to the named
    federation peer instead of scheduling it locally.
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
    if target_cluster:
        constraints.append(Constraint.create(key=CLUSTER_CONSTRAINT_KEY, op=ConstraintOp.EQ, value=target_cluster))

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
    processes_per_task: int = 1,
    max_retries: int = 0,
    timeout: int = 0,
    extras: list[str] | None = None,
    setup_scripts: list[str] | None = None,
    sync_packages: list[str] | None = None,
    terminate_on_exit: bool = True,
    regions: tuple[str, ...] | None = None,
    zone: str | None = None,
    user: str | None = None,
    reserve: tuple[str, ...] | None = None,
    priority: str | None = None,
    preemptible: bool | None = None,
    task_image: str | None = None,
    container_profile: str | None = None,
    credentials: ClientCredentials | None = None,
    submit_argv: list[str] | None = None,
    dashboard_url: str | None = None,
    target_cluster: str | None = None,
) -> int:
    """Core job submission logic.

    Args:
        controller_url: Controller URL (from parent context tunnel).
        dashboard_url: Public dashboard origin (e.g. https://iris.oa.dev). When
            set, a clickable job URL is logged on submit.
        terminate_on_exit: If True, terminate the job on any non-normal exit
            (KeyboardInterrupt, unexpected exceptions). Normal completion is unaffected.
        regions: If provided, restrict the job to workers in these regions.
        zone: If provided, restrict the job to workers in this zone.
        reserve: Hard availability constraints (e.g., ("4:H100x8", "v5litepod-16"))
            that confine the job to a zone where the named accelerator can be found.
        preemptible: If True/False, force scheduling on (non-)preemptible workers
            and bypass the executor heuristic. If None (default), the heuristic runs.
        target_cluster: If provided, federate the whole job to this peer cluster
            instead of scheduling it locally. Distinct from the connection-level
            ``--cluster`` option, which only selects which controller the CLI talks to.
        task_image: Optional task container image override. When None, workers use
            their cluster-configured default task image.

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
        target_cluster=target_cluster,
    )

    if reserve:
        # --reserve is now a hard, zone-level availability constraint: "schedule me
        # only in a zone where this accelerator can be found." It filters candidate
        # zones rather than holding capacity, so it composes with --region/--zone.
        availability = [reserve_spec_to_availability(spec) for spec in reserve]
        constraints = [*(constraints or []), *availability]

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
    if target_cluster:
        logger.info(f"Federating to peer cluster: {target_cluster}")
    if reserve:
        logger.info(f"Availability constraint: {', '.join(reserve)}")
    if task_image:
        logger.info(f"Task image: {task_image}")

    logger.info(f"Using controller: {controller_url}")
    priority_band = job_pb2.PRIORITY_BAND_UNSPECIFIED
    if priority is not None:
        priority_band = priority_band_value(priority)
        logger.info(f"Priority band: {priority}")

    profile = job_pb2.CONTAINER_PROFILE_UNSPECIFIED
    if container_profile is not None:
        profile = job_pb2.ContainerProfile.Value(container_profile)
        logger.info(f"Container profile: {container_profile}")

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
        setup_scripts=setup_scripts,
        sync_packages=sync_packages,
        terminate_on_exit=terminate_on_exit,
        constraints=constraints or None,
        coscheduling=coscheduling,
        processes_per_task=processes_per_task,
        user=user,
        priority_band=priority_band,
        container_profile=profile,
        credentials=credentials,
        submit_argv=submit_argv,
        dashboard_url=dashboard_url,
        task_image=task_image,
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
    setup_scripts: list[str] | None = None,
    sync_packages: list[str] | None = None,
    terminate_on_exit: bool = True,
    constraints: list[Constraint] | None = None,
    coscheduling: CoschedulingConfig | None = None,
    processes_per_task: int = 1,
    user: str | None = None,
    priority_band: job_pb2.PriorityBand = job_pb2.PRIORITY_BAND_UNSPECIFIED,
    container_profile: job_pb2.ContainerProfile = job_pb2.CONTAINER_PROFILE_UNSPECIFIED,
    credentials: ClientCredentials | None = None,
    submit_argv: list[str] | None = None,
    dashboard_url: str | None = None,
    task_image: str | None = None,
) -> int:
    """Submit job and optionally wait for completion.

    Only KeyboardInterrupt terminates the remote job; connection failures
    are logged and re-raised without killing the job.
    """
    client = IrisClient.remote(controller_url, workspace=Path.cwd(), credentials=credentials)
    entrypoint = Entrypoint.from_command(*command)

    job = client.submit(
        entrypoint=entrypoint,
        name=job_name,
        resources=resources,
        environment=EnvironmentSpec(
            env_vars=env_vars, extras=extras or [], setup_scripts=setup_scripts, sync_packages=sync_packages or []
        ),
        constraints=constraints,
        coscheduling=coscheduling,
        replicas=replicas,
        processes_per_task=processes_per_task,
        max_retries_failure=max_retries,
        max_task_failures=max_retries,
        timeout=Duration.from_seconds(timeout) if timeout else None,
        user=user,
        priority_band=priority_band,
        container_profile=container_profile,
        submit_argv=submit_argv,
        task_image=task_image,
    )

    logger.info(f"Job submitted: {job.job_id}")
    if dashboard_url:
        logger.info(f"Dashboard: {job.job_id.dashboard_url(dashboard_url)}")
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
@click.option("--cpu", type=float, default=0.1, show_default=True, help="Number of CPUs to request")
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
@click.option(
    "--processes-per-task",
    type=int,
    default=1,
    help="GPU processes to run inside each task (default 1). >1 fans each task into N JAX processes per GPU group.",
)
@click.option("--max-retries", type=int, default=0, help="Max retries on failure (default: 0)")
@click.option("--timeout", type=int, default=0, show_default=True, help="Job timeout in seconds (0 = no timeout)")
@click.option("--region", multiple=True, help="Restrict to region(s) (e.g., --region us-central2). Can be repeated.")
@click.option("--zone", type=str, help="Restrict to zone (e.g., --zone us-central2-b).")
@click.option(
    "--target-cluster",
    type=str,
    default=None,
    help=(
        "Federate the whole job to this peer cluster instead of scheduling it locally. "
        "This is distinct from the top-level --cluster option, which only selects which "
        "controller the CLI connects to; --target-cluster stays connected to that "
        "controller and asks it to hand the job off to the named peer."
    ),
)
@click.option("--extra", multiple=True, help="UV extras to install (e.g., --extra cpu). Can be repeated.")
@click.option(
    "--sync-package",
    multiple=True,
    help=(
        "Scope the default `uv sync` to specific workspace members instead of "
        "syncing all of them (e.g., --sync-package marin-core). Can be repeated."
    ),
)
@click.option(
    "--no-sync",
    is_flag=True,
    help="Skip environment setup entirely: run the command in the task image as-is (no uv sync).",
)
@click.option(
    "--reserve",
    multiple=True,
    help=(
        "Availability constraint: schedule only in a zone where this accelerator can "
        "be found (the job waits otherwise). Format: [COUNT:]DEVICE (e.g., 4:H100x8, "
        "v5litepod-16); the count is ignored. Can be repeated. This only constrains "
        "placement — it holds no capacity and attaches no device. Use --tpu/--gpu to "
        "actually request an accelerator."
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
    "--task-image",
    type=str,
    default=None,
    help=(
        "Override the task container image for this job. The image must already exist in a registry visible to workers."
    ),
)
@click.option(
    "--container-profile",
    type=click.Choice(CONTAINER_PROFILE_NAMES, case_sensitive=False),
    default=None,
    help=(
        "Container security profile (default: CONTAINER_PROFILE_DEFAULT). RESTRICTED hardens "
        "the container; DOCKER_ACCESS and PRIVILEGED are elevated and require admin."
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
    processes_per_task: int,
    max_retries: int,
    timeout: int,
    region: tuple[str, ...],
    zone: str | None,
    target_cluster: str | None,
    extra: tuple[str, ...],
    sync_package: tuple[str, ...],
    no_sync: bool,
    reserve: tuple[str, ...],
    priority: str | None,
    preemptible: bool | None,
    task_image: str | None,
    container_profile: str | None,
    terminate_on_exit: bool,
    cmd: tuple[str, ...],
):
    """Submit jobs to Iris clusters."""
    controller_url = require_controller_url(ctx)
    config = ctx.obj.get("config") if ctx.obj else None
    dashboard_url = config.dashboard_url if config else None
    validate_extra_resources(tpu, gpu, memory, disk, enable_extra_resources)
    validate_region_zone(region or None, zone, ctx.obj.get("config"))
    if no_sync and sync_package:
        raise click.UsageError("--no-sync skips setup entirely; it cannot be combined with --sync-package.")

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
            processes_per_task=processes_per_task,
            max_retries=max_retries,
            timeout=timeout,
            extras=list(extra),
            setup_scripts=[] if no_sync else None,
            sync_packages=list(sync_package),
            terminate_on_exit=terminate_on_exit,
            regions=region or None,
            zone=zone,
            target_cluster=target_cluster,
            reserve=reserve or None,
            priority=priority,
            preemptible=preemptible,
            task_image=task_image,
            container_profile=container_profile,
            credentials=ctx.obj.get("credentials"),
            submit_argv=submit_argv,
            dashboard_url=dashboard_url or None,
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


def _stop_jobs(ctx, job_id: tuple[str, ...], include_children: bool, stdin: bool, dry_run: bool) -> None:
    targets = _collect_targets(job_id, stdin)
    if not targets:
        raise click.UsageError("No jobs given. Pass job ids, or --stdin (or '-') to read them from stdin.")
    if dry_run:
        click.echo(f"[dry-run] would terminate {len(targets)} job(s):")
        for t in targets:
            click.echo(f"  {t}")
        return
    client = _remote_client(ctx)
    terminated = _terminate_jobs(client, tuple(targets), include_children)
    _print_terminated(terminated)


@job.command("stop")
@click.argument("job_id", nargs=-1, required=False)
@click.option(
    "--include-children/--no-include-children",
    default=True,
    help="Terminate child jobs under the given job ID prefix (default: include).",
)
@click.option("--stdin", is_flag=True, default=False, help="Also read job ids from stdin (one per line; CSV-tolerant).")
@click.option("--dry-run", is_flag=True, default=False, help="Print the jobs that would be terminated without sending.")
@click.pass_context
def stop(ctx, job_id: tuple[str, ...], include_children: bool, stdin: bool, dry_run: bool) -> None:
    """Terminate one or more jobs.

    Pass ``-`` or ``--stdin`` to read job ids from stdin so a query can pipe in:

    \b
      iris query -f csv "SELECT job_id FROM jobs WHERE user_id='alice' AND state=3" \\
        | iris job stop --stdin
    """
    _stop_jobs(ctx, job_id, include_children, stdin, dry_run)


@job.command("kill")
@click.argument("job_id", nargs=-1, required=False)
@click.option(
    "--include-children/--no-include-children",
    default=True,
    help="Terminate child jobs under the given job ID prefix (default: include).",
)
@click.option("--stdin", is_flag=True, default=False, help="Also read job ids from stdin (one per line; CSV-tolerant).")
@click.option("--dry-run", is_flag=True, default=False, help="Print the jobs that would be terminated without sending.")
@click.pass_context
def kill(ctx, job_id: tuple[str, ...], include_children: bool, stdin: bool, dry_run: bool) -> None:
    """Terminate one or more jobs (alias for stop)."""
    _stop_jobs(ctx, job_id, include_children, stdin, dry_run)


_KICK_STATE_MAP = {
    "preempted": job_pb2.TASK_STATE_PREEMPTED,
    "failed": job_pb2.TASK_STATE_FAILED,
}


@job.command("kick")
@click.argument("target", nargs=-1, required=False)
@click.option(
    "--state",
    "-s",
    type=click.Choice(sorted(_KICK_STATE_MAP)),
    default="preempted",
    show_default=True,
    help="Terminal state to force: 'preempted' retries if budget remains; 'failed' does not retry.",
)
@click.option("--reason", type=str, default="", help="Reason recorded on the kicked task attempts.")
@click.option(
    "--stdin", is_flag=True, default=False, help="Also read target ids from stdin (one per line; CSV-tolerant)."
)
@click.option("--dry-run", is_flag=True, default=False, help="Print the targets that would be kicked without sending.")
@click.pass_context
def kick(ctx, target: tuple[str, ...], state: str, reason: str, stdin: bool, dry_run: bool) -> None:
    """Force task attempts into a terminal state (emergency override).

    Each TARGET is a task id (/user/job/0), task-attempt id (/user/job/0:3), or a
    job id (/user/job) that expands to the job's active tasks. Pass ``-`` or
    ``--stdin`` to read ids from stdin, so a query can pipe straight in:

    \b
      iris query -f csv "SELECT t.task_id FROM tasks t JOIN workers w
        ON t.current_worker_id=w.worker_id
        WHERE w.slice_id='<slice>' AND t.state IN (2,3,9)
        AND t.job_id NOT LIKE '/keep/%'" | iris job kick --stdin --reason "drain slice"
    """
    targets = _collect_targets(target, stdin)
    if not targets:
        raise click.UsageError("No targets given. Pass task/job ids, or --stdin (or '-') to read them from stdin.")

    if dry_run:
        click.echo(f"[dry-run] would kick {len(targets)} target(s) to {state}:")
        for t in targets:
            click.echo(f"  {t}")
        return

    client = _remote_client(ctx)
    results = client.kick_tasks(targets, desired_state=_KICK_STATE_MAP[state], reason=reason)

    queued = [r for r in results if r.queued]
    rejected = [r for r in results if not r.queued]
    if queued:
        click.echo(f"Queued kick to {state} for {len(queued)} task(s):")
        for r in queued:
            click.echo(f"  {r.task_id}")
    if rejected:
        click.echo("Rejected:")
        for r in rejected:
            label = r.task_id or r.target
            click.echo(f"  {label}: {r.detail}")
    if not results:
        click.echo("No tasks matched.")
    if rejected and not queued:
        ctx.exit(1)


@job.command("list")
@click.option("--state", type=str, default=None, help="Filter by state (e.g., running, pending, failed)")
@click.option(
    "--prefix",
    type=str,
    default=None,
    help="Anchored prefix match against the wire-form job_id (e.g. '/alice/exp-').",
)
@click.pass_context
def list_jobs(ctx, state: str | None, prefix: str | None) -> None:
    """List jobs with optional filtering."""
    client = _remote_client(ctx)

    state_value: job_pb2.JobState | None = None
    if state is not None:
        state_lower = state.lower()
        if state_lower not in _STATE_MAP:
            valid = ", ".join(sorted(_STATE_MAP.keys()))
            raise click.UsageError(f"Unknown state '{state}'. Valid states: {valid}")
        state_value = _STATE_MAP[state_lower]

    jobs = client.list_jobs(state=state_value, prefix=prefix)

    # Sort by submitted_at descending (most recent first)
    jobs.sort(key=lambda j: j.submitted_at.epoch_ms, reverse=True)

    if not jobs:
        click.echo("No jobs found.")
        return

    rows: list[list[str]] = []
    has_reasons = False

    for j in jobs:
        job_id = j.job_id
        state_name = job_state_friendly(j.state)
        submitted = timestamp_from_proto(j.submitted_at).as_formatted_date() if j.submitted_at.epoch_ms else "-"

        reason = j.error or j.pending_reason or ""
        if reason:
            has_reasons = True
            reason = (reason[:60] + "...") if len(reason) > 63 else reason

        rows.append([job_id, state_name, submitted, reason])

    if has_reasons:
        headers = ["JOB ID", "STATE", "SUBMITTED", "REASON"]
    else:
        headers = ["JOB ID", "STATE", "SUBMITTED"]
        rows = [row[:3] for row in rows]

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
@click.pass_context
def summary(ctx, job_id: str) -> None:
    """Print a per-task summary (peak memory, state, exit, duration) for a job.

    Works for both running and completed jobs. Data is read from the controller's
    existing ``GetJobStatus`` / ``ListTasks`` RPCs (no checkpoint scraping).
    """
    client = _remote_client(ctx)
    job_name = JobName.from_wire(job_id)
    job_status = client.status(job_name)
    tasks = client.list_tasks(job_name)
    result = build_job_summary(job_status, tasks)
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
    """Stream task logs for a job and its descendants using batch log fetching."""
    if since_ms is not None and since_seconds is not None:
        raise click.UsageError("Specify only one of --since-ms or --since-seconds.")

    client = _remote_client(ctx)

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
