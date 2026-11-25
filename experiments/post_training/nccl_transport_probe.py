# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Probe which network transport NCCL picks between two Iris GPU tasks.

Run as a two-task gang Iris job (``--replicas 2``). Every task runs the same
command. Task 0 publishes a rendezvous endpoint through the Iris endpoint
registry; both tasks then run the same sequence of environment variants, each
spawning one NCCL worker per local GPU for an all-reduce and an all-to-all
benchmark with ``NCCL_DEBUG=INFO``. The launcher parses each worker's captured
stdout and stderr for the transport NCCL selected (``via NET/IB/...`` versus
``via NET/Socket``) and reports bandwidth and pass/fail per variant as one JSON
object on stdout. A failing variant records the worker log tail instead of
aborting the run.

Example submission (the vendored SkyRL runtime image and its megatron profile
provide torch with the production NCCL build; the pinned image digest is the
`IRIS_TASK_IMAGE` from `vendor/skyrl/cloud/iris/runtime_environment.py`):

    IMAGE=ghcr.io/marin-community/iris-task@sha256:13519c59442bd70ed5ec2902869f15094992d6849d7e34e353f1338dec7ff5f8

    iris --config lib/iris/config/marin.yaml job run \
        --target-cluster cw-rno2a --gpu H100x1 --enable-extra-resources \
        --replicas 2 --priority production \
        --job-name ib-audit-nccl-probe-rno2a-1 --timeout 5400 --no-wait \
        --task-image "$IMAGE" \
        --cpu 8 --memory 32GB --disk 200GB \
        --no-sync -- \
        bash /app/vendor/skyrl/cloud/iris/setup_task_runtime.sh megatron \
        python /app/experiments/post_training/nccl_transport_probe.py

The variants cover the recipe workaround (`NCCL_NET=Socket`,
`NCCL_IB_DISABLE=1`, `NCCL_SOCKET_FAMILY=AF_INET`) as one overlay and each
variable alone, the exact production overlay that also pins
`NCCL_SOCKET_IFNAME`, and InfiniBand hints (`NCCL_IB_HCA`,
`NCCL_IB_GID_INDEX`).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import os
import re
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.actor.resolver import Resolver
from iris.client.client import iris_ctx
from iris.cluster.client.job_info import JobInfo, get_job_info
from iris.cluster.types import EndpointAccess
from rigging.filesystem.factory import open_url
from rigging.network import interface_for_ipv4

logger = logging.getLogger("nccl_transport_probe")

RESULT_DIR = Path("/tmp/nccl_probe")
VARIANT_TIMEOUT_SECONDS = 420
RENDEZVOUS_TIMEOUT_SECONDS = 300
PROCESS_GROUP_TIMEOUT_SECONDS = 300
ALLREDUCE_BYTES = 64 * 2**20
ALLTOALL_BYTES = 64 * 2**20
WARMUP_ITERATIONS = 3
ALLREDUCE_ITERATIONS = 10
ALLTOALL_ITERATIONS = 8
MAX_LOCAL_GPUS = 2
ERROR_TAIL_LINES = 12
TRACEBACK_MAX_LINES = 80
ENDPOINT_BASE_NAME = "nccl-probe"

NCCL_VERSION_RE = re.compile(r"NCCL version ([0-9.]+)")
TRANSPORT_RE = re.compile(r"via (NET/(?:IB|Socket)\S*|NET/\S+)")
IB_DEVICE_SELECTION_RE = re.compile(r"\bNET/IB\s*:\s*Using\b")

DEFAULT_SOCKET_IFNAME = "^ibs,ibp,lo,docker,veth,cilium,lxc"

# Endpoint states that are expected while task 0 is still starting.
_RENDEZVOUS_PENDING_CODES = frozenset({Code.NOT_FOUND, Code.UNIMPLEMENTED, Code.UNAVAILABLE})


@dataclass(frozen=True)
class Variant:
    """One environment overlay applied on top of the inherited task env."""

    name: str
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class NcclCensus:
    """Transport facts parsed from one worker's NCCL INFO output."""

    nccl_version: str
    transports: list[str]
    ib_device_lines: list[str] = field(default_factory=list)


@dataclass
class WorkerProcess:
    """A spawned worker process waiting to be collected."""

    process: subprocess.Popen
    variant_name: str
    rank: int
    world_size: int
    started: float
    log_path: Path
    output_path: Path
    result_path: Path
    log_file: IO[bytes]
    output_file: IO[bytes]


@dataclass(frozen=True)
class RankResult:
    """Outcome of one rank's benchmark within a variant."""

    variant: str
    rank: int
    passed: bool
    returncode: int
    duration: float
    error_tail: list[str]
    worker: dict | None
    census: NcclCensus


def build_variants(extra_env: dict[str, str] | None = None) -> list[Variant]:
    """Build the environment matrix: the workaround, each variable alone, and IB hints.

    ``extra_env`` appends one ``env_overlay`` variant carrying caller-supplied
    variables, for example a measured per-host ``NCCL_IB_HCA`` set.
    """
    all_three = {
        "NCCL_NET": "Socket",
        "NCCL_IB_DISABLE": "1",
        "NCCL_SOCKET_FAMILY": "AF_INET",
    }
    variants = [
        Variant("baseline_default_env"),
        Variant("net_socket_only", {"NCCL_NET": "Socket"}),
        Variant("ib_disable_only", {"NCCL_IB_DISABLE": "1"}),
        Variant("af_inet_only", {"NCCL_SOCKET_FAMILY": "AF_INET"}),
        Variant("all_three", dict(all_three)),
        Variant("prod_exact", {**all_three, "NCCL_SOCKET_IFNAME": DEFAULT_SOCKET_IFNAME}),
        Variant("ib_hca_mlx5_0", {"NCCL_IB_HCA": "mlx5_0"}),
        Variant("ib_hca_mlx5_1", {"NCCL_IB_HCA": "mlx5_1"}),
        Variant("ib_gid_index_0", {"NCCL_IB_GID_INDEX": "0"}),
    ]
    if extra_env:
        variants.append(Variant("env_overlay", dict(extra_env)))
    return variants


def parse_nccl_census(lines: list[str]) -> NcclCensus:
    """Extract the version, transports, and InfiniBand device selection from NCCL INFO output."""
    version = ""
    transports: list[str] = []
    ib_device_lines: list[str] = []
    for line in lines:
        if not version:
            match = NCCL_VERSION_RE.search(line)
            if match:
                version = match.group(1)
        match = TRANSPORT_RE.search(line)
        if match and match.group(1) not in transports:
            transports.append(match.group(1))
        if IB_DEVICE_SELECTION_RE.search(line) and line not in ib_device_lines:
            ib_device_lines.append(line)
    return NcclCensus(nccl_version=version, transports=transports, ib_device_lines=ib_device_lines)


def allocate_port_block(count: int) -> int:
    """Find a base port where the next ``count`` ports are all free."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("0.0.0.0", 0))
    base = probe.getsockname()[1]
    probe.close()
    for offset in range(count):
        check = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            check.bind(("0.0.0.0", base + offset))
        except OSError:
            check.close()
            return allocate_port_block(count)
        check.close()
    return base


def wait_for_rendezvous(
    resolver: Resolver, endpoint_name: str, timeout: int = RENDEZVOUS_TIMEOUT_SECONDS
) -> tuple[str, int]:
    """Poll the Iris endpoint registry until task 0's master address appears."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resolved = resolver.resolve(endpoint_name)
            if not resolved.is_empty:
                url = resolved.first().url
                host, _, port = url.rpartition(":")
                return host, int(port)
        except ConnectError as error:
            # A not-yet-registered endpoint and a briefly unreachable controller
            # are both expected while task 0 starts; everything else is real.
            if error.code not in _RENDEZVOUS_PENDING_CODES:
                raise
        time.sleep(2)
    raise TimeoutError(f"No rendezvous endpoint {endpoint_name!r} after {timeout}s")


def spawn_worker(
    variant: Variant,
    index: int,
    local_rank: int,
    master: str,
    port: int,
    world_size: int,
    log_dir: Path,
) -> WorkerProcess:
    """Start one benchmark worker process without waiting for it.

    Ranks must all start before any of them can pass ``init_process_group``,
    so the caller spawns every local rank first and collects afterwards.
    """
    log_path = log_dir / f"{variant.name}-rank{index}.log"
    output_path = log_dir / f"{variant.name}-rank{index}.out"
    result_path = log_dir / f"{variant.name}-rank{index}.json"
    env = dict(os.environ)
    env.update(
        {
            "MASTER_ADDR": master,
            "MASTER_PORT": str(port),
            "WORLD_SIZE": str(world_size),
            "RANK": str(index),
            "LOCAL_RANK": str(local_rank),
            "NCCL_DEBUG": "INFO",
            "PROBE_RESULT_FILE": str(result_path),
            **variant.env,
        }
    )
    if os.environ.get("PROBE_INIT_TIMEOUT"):
        env["PROBE_INIT_TIMEOUT"] = os.environ["PROBE_INIT_TIMEOUT"]
    if os.environ.get("PROBE_BUFFER_BYTES"):
        env["PROBE_BUFFER_BYTES"] = os.environ["PROBE_BUFFER_BYTES"]
    if os.environ.get("PROBE_PYNCCL_INIT"):
        env["PROBE_PYNCCL_INIT"] = os.environ["PROBE_PYNCCL_INIT"]
    cmd = [sys.executable, str(Path(__file__).resolve()), "--mode", "worker", "--variant", variant.name]
    log_file: IO[bytes] = open(log_path, "wb")  # noqa: SIM115 - closed in collect_worker after the child exits
    output_file: IO[bytes] = open(output_path, "wb")  # noqa: SIM115 - closed in collect_worker after the child exits
    process = subprocess.Popen(cmd, env=env, stdout=output_file, stderr=log_file)
    return WorkerProcess(
        process=process,
        variant_name=variant.name,
        rank=index,
        world_size=world_size,
        started=time.time(),
        log_path=log_path,
        output_path=output_path,
        result_path=result_path,
        log_file=log_file,
        output_file=output_file,
    )


def write_worker_result(result_path: Path, payload: dict) -> None:
    """Write one worker's machine result atomically (temp file plus rename)."""
    temp_path = result_path.with_name(result_path.name + ".tmp")
    temp_path.write_text(json.dumps(payload))
    os.replace(temp_path, result_path)


def parse_worker_result(
    result_path: Path,
    expected_rank: int,
    expected_variant: str,
    expected_world_size: int,
) -> tuple[dict | None, str | None]:
    """Load one worker's machine result, validating its identity fields.

    The per-rank result file replaces stdout payload scanning: NCCL INFO
    writes from NCCL's C threads interleave with the Python print and can
    embed the JSON mid-line, so a line-start scan silently loses payloads
    (the .66 3-rack run lost 95 of 192 this way). Returns ``(payload,
    problem)``; a missing, invalid, or mismatched file yields a problem
    string and no payload, which fails the rank closed.
    """
    if not result_path.exists():
        return None, f"worker result file missing: {result_path}"
    try:
        payload = json.loads(result_path.read_text())
    except json.JSONDecodeError as error:
        return None, f"worker result file invalid JSON ({error})"
    expected_fields = (
        ("rank", expected_rank),
        ("variant", expected_variant),
        ("world_size", expected_world_size),
    )
    mismatches = [
        f"{field}={payload.get(field)!r} != {expected!r}"
        for field, expected in expected_fields
        if payload.get(field) != expected
    ]
    if mismatches:
        return None, f"worker result identity mismatch: {'; '.join(mismatches)}"
    return payload, None


def worker_traceback(lines: list[str]) -> list[str]:
    """Return the first Python traceback block in merged worker output.

    NCCL INFO spam fills the tail window for a hung or failed collective, so
    the failing worker's traceback is pulled out separately: it starts at
    ``Traceback (most recent call last):`` and ends at the unindented
    exception line.
    """
    start = next(
        (index for index, line in enumerate(lines) if line.startswith("Traceback (most recent call last):")),
        None,
    )
    if start is None:
        return []
    block = [lines[start]]
    for line in lines[start + 1 :]:
        block.append(line)
        if line and not line.startswith((" ", "\t")):
            break
    return block[:TRACEBACK_MAX_LINES]


def collect_worker(handle: WorkerProcess) -> RankResult:
    """Wait for one spawned worker and parse its logs and result payload."""
    try:
        returncode = handle.process.wait(timeout=VARIANT_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        handle.process.kill()
        returncode = handle.process.wait()
    handle.log_file.close()
    handle.output_file.close()
    duration = time.time() - handle.started
    # NCCL writes its INFO output to stdout and traces to stderr depending on
    # build and NCCL_DEBUG_FILE, so the census parses both streams.
    lines = handle.log_path.read_text(errors="replace").splitlines()
    lines += handle.output_path.read_text(errors="replace").splitlines()
    census = parse_nccl_census(lines)
    payload, result_problem = parse_worker_result(
        handle.result_path, handle.rank, handle.variant_name, handle.world_size
    )
    if result_problem:
        lines.append(result_problem)
    pynccl_failed = bool(
        payload is not None
        and isinstance(payload.get("pynccl"), dict)
        and payload["pynccl"].get("pynccl_init") == "failed"
    )
    if pynccl_failed:
        # A failed engine-side communicator init fails the rank closed.
        lines.append(f"pynccl init failed: {payload['pynccl'].get('error')}")
    traceback_lines = worker_traceback(lines) if returncode != 0 else []
    tail = lines[-ERROR_TAIL_LINES:]
    error_tail = [*traceback_lines, "--- last output ---", *tail] if traceback_lines else tail
    return RankResult(
        variant=handle.variant_name,
        rank=handle.rank,
        passed=returncode == 0 and payload is not None and not pynccl_failed,
        returncode=returncode,
        duration=round(duration, 2),
        error_tail=error_tail,
        worker=payload,
        census=census,
    )


def run_variant(
    variant: Variant,
    rank_offset: int,
    local_gpus: int,
    world_size: int,
    master: str,
    port: int,
    log_dir: Path,
) -> dict:
    """Run one variant on this task's local GPUs and summarize the ranks."""
    handles = [
        spawn_worker(variant, rank_offset + local, local, master, port, world_size, log_dir)
        for local in range(local_gpus)
    ]
    ranks = [collect_worker(handle) for handle in handles]
    return {
        "variant": variant.name,
        "env_overlay": variant.env,
        "pass": all(rank.passed for rank in ranks),
        "world_size": world_size,
        "ranks": [asdict_without_worker_noise(rank) for rank in ranks],
        "transports": sorted({transport for rank in ranks for transport in rank.census.transports}),
    }


def asdict_without_worker_noise(rank: RankResult) -> dict:
    """Flatten a rank result into the report schema."""
    return {
        "variant": rank.variant,
        "rank": rank.rank,
        "pass": rank.passed,
        "returncode": rank.returncode,
        "duration": rank.duration,
        "error_tail": rank.error_tail,
        "worker": rank.worker,
        "nccl_version": rank.census.nccl_version,
        "transports": rank.census.transports,
        "ib_device_lines": rank.census.ib_device_lines,
    }


def resolve_gloo_socket_ifname(advertise_host: str) -> str:
    """Pick the interface workers must use for cross-host Gloo bootstrap.

    Gloo resolves peer addresses from the hostname and binds loopback in the
    task containers, so the Gloo control group behind the PyNccl stage fails
    connectFullMesh cross-host while NCCL itself works (Snowball .64).
    Production vLLM solves this with ``GLOO_SOCKET_IFNAME``: honor an ambient
    value first, otherwise resolve the interface owning the advertise
    address, the same interface NCCL's rendezvous already rides. Setting it
    in the launcher environment propagates to spawned workers.
    """
    ambient = os.environ.get("GLOO_SOCKET_IFNAME")
    if ambient:
        return ambient
    interface = interface_for_ipv4(advertise_host)
    os.environ["GLOO_SOCKET_IFNAME"] = interface
    return interface


def run_launcher(args: argparse.Namespace) -> None:
    """Run every variant on this task and print the aggregate JSON report."""
    import torch  # noqa: PLC0415 - torch is an optional dependency outside the vendor runtime env

    job_info = get_job_info()
    if job_info is None:
        raise RuntimeError("This script must run inside an Iris job (no IRIS_TASK_ID in the environment)")
    num_tasks = job_info.num_tasks
    task_index = job_info.task_index
    local_gpus = min(torch.cuda.device_count(), args.max_local_gpus) or 1
    world_size = args.world_size if args.world_size else num_tasks * local_gpus
    rank_offset = args.rank_offset if args.rank_offset is not None else task_index * local_gpus
    gloo_interface = resolve_gloo_socket_ifname(job_info.advertise_host)
    logger.info("Gloo bootstrap interface %s for %s", gloo_interface, job_info.advertise_host)
    if num_tasks < 2 and args.endpoint_name is None:
        logger.warning("num_tasks=%d: single-task run exercises intra-node transport only", num_tasks)

    ctx = iris_ctx()
    if args.endpoint_name:
        # An absolute name bypasses the job namespace, so two separate jobs can
        # share one rendezvous; the low ranks register it with link access so
        # the peer job may resolve it.
        endpoint_name = args.endpoint_name
        coordinator = rank_offset == 0
    else:
        endpoint_name = f"{ENDPOINT_BASE_NAME}-attempt-{job_info.attempt_id}"
        coordinator = task_index == 0
    if coordinator:
        base_port = allocate_port_block(len(build_variants()) + 1)
        master_host = job_info.advertise_host
        access = EndpointAccess.ENDPOINT_ACCESS_LINK if args.endpoint_name else EndpointAccess.ENDPOINT_ACCESS_PRIVATE
        endpoint_id = ctx.registry.register(endpoint_name, f"{master_host}:{base_port}", access=access)
        logger.info("Registered rendezvous %s at %s:%d", endpoint_name, master_host, base_port)
    else:
        rendezvous_timeout = (
            args.rendezvous_timeout_seconds
            if args.rendezvous_timeout_seconds
            else (args.init_timeout_seconds or RENDEZVOUS_TIMEOUT_SECONDS)
        )
        master_host, base_port = wait_for_rendezvous(ctx.resolver, endpoint_name, timeout=rendezvous_timeout)
        logger.info("Resolved rendezvous at %s:%d", master_host, base_port)

    log_dir = RESULT_DIR / f"task{task_index}"
    log_dir.mkdir(parents=True, exist_ok=True)
    selected = set(args.variants.split(",")) if args.variants else None
    results = []
    for offset, variant in enumerate(build_variants(extra_env=args.env_pairs)):
        if selected is not None and variant.name not in selected:
            continue
        logger.info("[task %s] variant %s starting", task_index, variant.name)
        results.append(
            run_variant(variant, rank_offset, local_gpus, world_size, master_host, base_port + 1 + offset, log_dir)
        )
        logger.info("[task %s] variant %s finished pass=%s", task_index, variant.name, results[-1]["pass"])

    if coordinator:
        ctx.registry.unregister(endpoint_id)
    report = {
        "job": {
            "task_index": task_index,
            "num_tasks": num_tasks,
            "local_gpus": local_gpus,
            "host": job_info.advertise_host,
            "gloo_socket_ifname": gloo_interface,
            "ib_ports": collect_ib_port_inventory(),
        },
        "results": results,
    }
    print(json.dumps(report))
    (RESULT_DIR / f"results-task{task_index}.json").write_text(json.dumps(report, indent=2))


def gate_endpoint_name(job_info: JobInfo) -> str:
    """Give each GPU gang its own rendezvous within the coordinator namespace."""
    job = hashlib.sha256(str(job_info.job_id).encode()).hexdigest()[:16]
    return f"{ENDPOINT_BASE_NAME}-gate-{job}-attempt-{job_info.attempt_id}"


def run_gate(args: argparse.Namespace) -> None:
    """Launch-time interconnect check: fail closed unless cross-node NCCL rides InfiniBand.

    With ``--gate-coverage first-pair`` (the default), tasks 0 and 1 each run one
    NCCL rank over the default environment; any other task exits immediately.
    With ``--gate-coverage all-gpus``, every task joins with all of its local
    GPUs, so communicator init exercises every GPU-to-NIC pairing (a 64-rank
    data-parallel engine can fail on one GPU's HCA even when the first pair is
    healthy). The check passes when the census shows NET/IB or no NET transport
    at all (intra-node NVLink), and fails with the NCCL error tail otherwise.
    """
    import torch  # noqa: PLC0415 - torch is an optional dependency outside the vendor runtime env

    job_info = get_job_info()
    if job_info is None:
        raise RuntimeError("This script must run inside an Iris job (no IRIS_TASK_ID in the environment)")
    num_tasks = job_info.num_tasks
    task_index = job_info.task_index
    if num_tasks < 2:
        logger.info("gate: single-node job (%d task); NVLink only, nothing to assert", num_tasks)
        return
    if args.gate_coverage == "first-pair":
        if task_index > 1:
            logger.info("gate: task %d of %d does not participate; exiting", task_index, num_tasks)
            return
        local_gpus = 1
        world_size = 2
    else:
        local_gpus = torch.cuda.device_count()
        world_size = num_tasks * local_gpus
    gloo_interface = resolve_gloo_socket_ifname(job_info.advertise_host)
    logger.info("gate: Gloo bootstrap interface %s for %s", gloo_interface, job_info.advertise_host)
    ctx = iris_ctx()
    endpoint_name = gate_endpoint_name(job_info)
    gate_variants = [
        variant
        for variant in build_variants(extra_env=args.env_pairs or None)
        if variant.name in ("baseline_default_env", "env_overlay")
    ]
    if task_index == 0:
        base_port = allocate_port_block(len(gate_variants))
        master_host = job_info.advertise_host
        endpoint_id = ctx.registry.register(endpoint_name, f"{master_host}:{base_port}")
        logger.info("gate: registered rendezvous %s at %s:%d", endpoint_name, master_host, base_port)
    else:
        master_host, base_port = wait_for_rendezvous(ctx.resolver, endpoint_name)
        logger.info("gate: resolved rendezvous at %s:%d", master_host, base_port)

    log_dir = RESULT_DIR / f"gate-task{task_index}"
    log_dir.mkdir(parents=True, exist_ok=True)
    rank_offset = task_index if args.gate_coverage == "first-pair" else task_index * local_gpus
    results = []
    for offset, variant in enumerate(gate_variants):
        results.append(
            run_variant(variant, rank_offset, local_gpus, world_size, master_host, base_port + offset, log_dir)
        )
    if task_index == 0:
        ctx.registry.unregister(endpoint_id)
    verdicts = [gate_verdict(result["pass"], [rank["transports"] for rank in result["ranks"]]) for result in results]
    verdict = next((item for item in verdicts if item != "pass"), "pass")
    report = {
        "gate": {
            "task_index": task_index,
            "verdict": verdict,
            "gloo_socket_ifname": gloo_interface,
            "results": results,
            "ib_ports": collect_ib_port_inventory(),
        }
    }
    report_json = json.dumps(report)
    print(report_json)
    # Persist alongside stdout: the pod-local copy is the durable evidence for
    # an offline postmortem when the task exits nonzero.
    (RESULT_DIR / f"gate-results-task{task_index}.json").write_text(report_json)
    if args.gate_report_prefix:
        # The pod-local copy is deleted with the terminal pod; publish to the
        # regional evidence prefix before the nonzero exit. A publish failure
        # is reported loudly but never changes the verdict.
        try:
            report_uri = publish_gate_report(args.gate_report_prefix, job_info.attempt_id, task_index, report_json)
            logger.info("gate: published report to %s", report_uri)
        except Exception as publish_error:
            print(
                f"gate: report publish under {args.gate_report_prefix} failed: "
                f"{type(publish_error).__name__}: {publish_error}",
                file=sys.stderr,
            )
    if verdict != "pass":
        for result in results:
            for rank in result["ranks"]:
                if rank["pass"]:
                    continue
                for line in rank["error_tail"]:
                    print(line, file=sys.stderr)
        raise SystemExit(f"gate: interconnect check failed ({verdict}); refusing to start the trainer")


def publish_gate_report(report_prefix: str, attempt_id: str, task_index: int, report_json: str) -> str:
    """Publish the full gate report to the regional evidence prefix.

    The pod-local copy is deleted with the terminal GPU pod, so the report is
    written to the launch-provided prefix (``rendezvous_dir``/interconnect-gate)
    before the task exits. Returns the published URI.
    """
    uri = f"{report_prefix.rstrip('/')}/attempt-{attempt_id}/gate-results-task{task_index}.json"
    with open_url(uri, "wb") as report_file:
        report_file.write(report_json.encode())
    return uri


def collect_ib_port_inventory() -> list[dict[str, str]]:
    """Read the host's InfiniBand device/port states from sysfs.

    Returns one record per port with the device name, port number, state and
    link layer, so a launch record carries the measured link-layer inventory.
    """
    inventory: list[dict[str, str]] = []
    for device_path in sorted(Path("/sys/class/infiniband").glob("*")):
        device = device_path.name
        for port_path in sorted((device_path / "ports").iterdir()):
            if not port_path.is_dir():
                continue
            record = {"device": device, "port": port_path.name}
            for field_name in ("state", "link_layer", "rate"):
                value = port_path / field_name
                if value.exists():
                    record[field_name] = value.read_text().strip()
            inventory.append(record)
    return inventory


def gate_verdict(passed: bool, rank_transports: list[list[str]]) -> str:
    """Map one probe variant's outcome onto the launch-gate verdict.

    NET/IB or an empty transport list (intra-node NVLink) passes; a socket
    transport or a failed benchmark fails closed.
    """
    if not passed:
        return "collective_failed"
    flat = {transport for transports in rank_transports for transport in transports}
    if any(transport.startswith("NET/Socket") for transport in flat):
        return "socket_transport"
    return "pass"


def mapped_nccl_libraries(maps_path: Path = Path("/proc/self/maps")) -> list[dict[str, str]]:
    """Identify the mapped libnccl files in this process with their hashes.

    Maps lines are ``addr perms offset dev inode pathname``; the path field
    may itself contain spaces, so split at most five times.
    """
    records: list[dict[str, str]] = []
    if not maps_path.exists():
        return records
    seen: set[str] = set()
    for line in maps_path.read_text(errors="replace").splitlines():
        fields = line.split(None, 5)
        if len(fields) < 6:
            continue
        path = fields[5].strip()
        if "libnccl" not in path or path in seen or not path.startswith("/"):
            continue
        seen.add(path)
        digest = hashlib.sha256()
        with open(path, "rb") as library_file:
            for block in iter(lambda: library_file.read(8 * 2**20), b""):
                digest.update(block)
        records.append({"path": path, "sha256": digest.hexdigest()})
    return records


def run_pynccl_init_check(device) -> dict:
    """Initialize vLLM's PyNcclCommunicator exactly as the engine does.

    The pinned vLLM requires a non-NCCL control group: the constructor asserts
    the group's backend is not NCCL, broadcasts the unique id over that group,
    and runs ncclCommInitRank plus a warm-up all_reduce itself — the stage
    whose failure aborted the Snowball .60 restart. The wrapper returns
    silently unavailable when it cannot load its NCCL library, so pass
    requires available and not disabled. The mapped libnccl path and hash go
    next to the version as qualification evidence; the NCCL guard that
    produced the .60 warning compares remote and local link layers, not
    library identity.
    """
    info: dict[str, object] = {"stage": "pynccl_init"}
    try:
        import torch.distributed as dist  # noqa: PLC0415 - torch is an optional dependency

        # Optional vLLM imports: the probe runs in environments without it.
        from vllm import envs as vllm_envs  # noqa: PLC0415
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator  # noqa: PLC0415

        control_group = dist.new_group(backend="gloo")
        communicator = PyNcclCommunicator(group=control_group, device=device)
        if not communicator.available or communicator.disabled:
            # The engine skips the PyNccl stage when its own parsed flag says
            # so (vllm.envs, not string truthiness — "0" is not a disable);
            # a recipe that disables it in the engine has nothing to assert.
            if vllm_envs.VLLM_DISABLE_PYNCCL:
                info["pynccl_init"] = "skipped"
                return info
            info["pynccl_init"] = "failed"
            info["error"] = (
                "PyNcclCommunicator reported unavailable "
                f"(available={communicator.available}, disabled={communicator.disabled})"
            )
            return info
        info["pynccl_init"] = "pass"
        info["vllm_disable_pynccl"] = vllm_envs.VLLM_DISABLE_PYNCCL
        info["nccl_library_version"] = communicator.nccl.ncclGetVersion()
        info["nccl_library_raw_version"] = communicator.nccl_version
        try:
            info["mapped_libraries"] = mapped_nccl_libraries()
        except Exception as inventory_error:
            info["mapped_libraries_error"] = f"{type(inventory_error).__name__}: {inventory_error}"
    except Exception as error:
        info["pynccl_init"] = "failed"
        info["error"] = f"{type(error).__name__}: {error}"
    return info


def all_to_all_element_count(buffer_bytes: int, world_size: int) -> int:
    """Largest bf16 element count within ``buffer_bytes`` divisible by ``world_size``.

    ``dist.all_to_all_single`` with equal splits requires dim0 divisible by
    world_size; the 3-rack 192-rank gate run failed at this collective because
    the nominal power-of-two count left a remainder. Round down so the message
    never exceeds the requested buffer, and refuse a buffer too small for one
    element per rank.
    """
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    elements = buffer_bytes // 2
    count = elements - elements % world_size
    if count == 0:
        raise ValueError(
            f"buffer_bytes={buffer_bytes} holds {elements} bf16 elements, too few for world_size={world_size}"
        )
    return count


def run_worker_mode(args: argparse.Namespace) -> None:
    """Run one rank's collectives and print its benchmark JSON on stdout."""
    import torch  # noqa: PLC0415
    import torch.distributed as dist  # noqa: PLC0415

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    init_timeout = int(os.environ.get("PROBE_INIT_TIMEOUT", PROCESS_GROUP_TIMEOUT_SECONDS))
    buffer_bytes = int(os.environ.get("PROBE_BUFFER_BYTES", ALLREDUCE_BYTES))
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=init_timeout))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    device = torch.device("cuda", torch.cuda.current_device())

    def bench(collective, iterations: int) -> float:
        for _ in range(WARMUP_ITERATIONS):
            collective()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            collective()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) / iterations

    tensor = torch.empty(buffer_bytes // 2, dtype=torch.bfloat16, device=device)

    def run_all_reduce() -> None:
        dist.all_reduce(tensor)

    all_reduce_seconds = bench(run_all_reduce, ALLREDUCE_ITERATIONS)
    all_reduce_bandwidth = buffer_bytes / all_reduce_seconds * 2 * (world_size - 1) / world_size / 1e9

    a2a_elements = all_to_all_element_count(buffer_bytes, world_size)
    a2a_bytes = a2a_elements * 2
    a2a_input = torch.empty(a2a_elements, dtype=torch.bfloat16, device=device)
    a2a_output = torch.empty_like(a2a_input)

    def run_all_to_all() -> None:
        dist.all_to_all_single(a2a_output, a2a_input)

    all_to_all_seconds = bench(run_all_to_all, ALLTOALL_ITERATIONS)
    all_to_all_bandwidth = a2a_bytes / all_to_all_seconds * (world_size - 1) / world_size / 1e9

    payload = {
        "variant": args.variant,
        "rank": rank,
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(device),
        "world_size": world_size,
        "allreduce_busbw_gigabytes_per_second": round(all_reduce_bandwidth, 2),
        "alltoall_busbw_gigabytes_per_second": round(all_to_all_bandwidth, 2),
        "alltoall_bytes": a2a_bytes,
    }
    if os.environ.get("PROBE_PYNCCL_INIT"):
        payload["pynccl"] = run_pynccl_init_check(device)
    dist.barrier()
    dist.destroy_process_group()
    write_worker_result(Path(os.environ["PROBE_RESULT_FILE"]), payload)
    print(json.dumps(payload))


def main() -> None:
    parser = argparse.ArgumentParser(description="NCCL transport probe for two-node Iris jobs")
    parser.add_argument("--mode", choices=("launcher", "worker", "gate"), default="launcher")
    parser.add_argument("--variant", help="Worker mode: variant name carried into the result JSON")
    parser.add_argument("--variants", help="Launcher: comma-separated variant names to run")
    parser.add_argument("--max-local-gpus", type=int, default=MAX_LOCAL_GPUS)
    parser.add_argument(
        "--endpoint-name",
        help="Launcher: absolute registry name shared by two separate jobs; the low ranks register it public",
    )
    parser.add_argument(
        "--rank-offset",
        type=int,
        help="Launcher: global rank of this job's first worker (default: task_index * local_gpus)",
    )
    parser.add_argument(
        "--world-size", type=int, help="Launcher: total number of ranks (default: num_tasks * local_gpus)"
    )
    parser.add_argument(
        "--init-timeout-seconds",
        type=int,
        help="Launcher: process-group init timeout for the workers (default 300)",
    )
    parser.add_argument(
        "--gate-coverage",
        choices=("first-pair", "all-gpus"),
        default="first-pair",
        help="Gate mode: first-pair checks tasks 0+1 one GPU each; all-gpus joins every task with all local GPUs",
    )
    parser.add_argument(
        "--gate-report-prefix",
        help="Gate mode: regional evidence prefix (rendezvous_dir/interconnect-gate) for the full report JSON",
    )
    parser.add_argument(
        "--env",
        action="append",
        metavar="KEY=VALUE",
        help="Extra environment overlay run as the 'env_overlay' variant (repeatable)",
    )
    parser.add_argument(
        "--buffer-bytes",
        type=int,
        help="Worker buffer size in bytes for both collectives (default 64 MiB)",
    )
    parser.add_argument(
        "--pynccl-init",
        action="store_true",
        help="Workers also initialize vLLM's PyNcclCommunicator (the engine-side comm stage)"
        " and report the loaded NCCL version",
    )
    parser.add_argument(
        "--rendezvous-timeout-seconds",
        type=int,
        help="Launcher: how long a peer waits for the coordinator's registry entry (default: init timeout)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args.env_pairs = {key: value for key, _, value in (item.partition("=") for item in args.env or [])}
    if not args.env_pairs:
        args.env_pairs = None
    if args.buffer_bytes:
        os.environ["PROBE_BUFFER_BYTES"] = str(args.buffer_bytes)
    if args.pynccl_init:
        os.environ["PROBE_PYNCCL_INIT"] = "1"
    if args.init_timeout_seconds:
        os.environ["PROBE_INIT_TIMEOUT"] = str(args.init_timeout_seconds)
    if args.mode == "launcher":
        run_launcher(args)
    elif args.mode == "gate":
        run_gate(args)
    else:
        run_worker_mode(args)


if __name__ == "__main__":
    main()
