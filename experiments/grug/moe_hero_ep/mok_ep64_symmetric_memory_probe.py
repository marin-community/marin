# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Probe CUDA symmetric memory in the one-process-per-GPU EP64 topology."""

import argparse
import gc
import json
import logging
import os
import socket
import subprocess
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from xml.etree import ElementTree as ET

from iris.cluster.client.job_info import get_job_info
from iris.hooks.multigpu import (
    IRIS_MULTIGPU_LOCAL_DEVICE_IDS_ENV,
    IRIS_MULTIGPU_PROCESS_COUNT_ENV,
    IRIS_MULTIGPU_PROCESS_INDEX_ENV,
)
from iris.runtime.jax_init import initialize_jax
from levanter.utils.jax_utils import multihost_broadcast_sync
from rigging.network import interface_for_ipv4

logger = logging.getLogger(__name__)

_IMEX_DEVICE_ROOT = Path("/dev/nvidia-caps-imex-channels")
_DIRECT_SYMMETRIC_MEMORY_ENV = "TORCH_SYMMMEM_IMPLICIT_POOL"
_DISABLE_SYMMETRIC_MEMORY_MULTICAST_ENV = "TORCH_SYMM_MEM_DISABLE_MULTICAST"
_DEFAULT_SIGNAL_PAD_BYTES = 1024 * 1024


@dataclass(frozen=True)
class ProbeRank:
    global_rank: int
    world_size: int
    local_device_id: int


@dataclass(frozen=True)
class FabricEnvironment:
    imex_channels: tuple[str, ...]
    cluster_uuids: tuple[str, ...]
    clique_ids: tuple[str, ...]
    fabric_states: tuple[str, ...]
    fabric_statuses: tuple[str, ...]


@dataclass(frozen=True)
class ProbeResult:
    rank: ProbeRank
    backend: str
    arena_bytes: int
    iterations: int
    peer_pointer_count: int
    signal_pointer_count: int
    remote_reads_checked: int
    remote_writes_checked: int
    fabric: FabricEnvironment


@dataclass(frozen=True)
class ProbeConfig:
    expected_world_size: int
    arena_bytes: int = 4096
    iterations: int = 3
    timeout: float = 180.0


def probe_rank_from_env(env: dict[str, str]) -> ProbeRank:
    """Resolve the exact one-GPU supervised-process identity."""
    missing = [
        key
        for key in (
            IRIS_MULTIGPU_PROCESS_INDEX_ENV,
            IRIS_MULTIGPU_PROCESS_COUNT_ENV,
            IRIS_MULTIGPU_LOCAL_DEVICE_IDS_ENV,
        )
        if key not in env
    ]
    if missing:
        raise ValueError(f"EP64 probe requires the Iris multi-GPU supervisor; missing {missing}")

    device_ids = [int(value) for value in env[IRIS_MULTIGPU_LOCAL_DEVICE_IDS_ENV].split(",")]
    if len(device_ids) != 1:
        raise ValueError(f"EP64 probe requires one GPU per process, got device ids {device_ids}")

    rank = ProbeRank(
        global_rank=int(env[IRIS_MULTIGPU_PROCESS_INDEX_ENV]),
        world_size=int(env[IRIS_MULTIGPU_PROCESS_COUNT_ENV]),
        local_device_id=device_ids[0],
    )
    if not 0 <= rank.global_rank < rank.world_size:
        raise ValueError(f"Invalid supervised rank {rank.global_rank} for world size {rank.world_size}")
    return rank


def pattern_byte(rank: int, iteration: int) -> int:
    """Return a deterministic nonzero byte for one rank and iteration."""
    return ((rank * 37 + iteration * 17) % 255) + 1


def sample_offsets(arena_bytes: int) -> tuple[int, ...]:
    """Choose boundary and interior offsets without duplicates."""
    if arena_bytes < 2:
        raise ValueError(f"arena_bytes must be at least 2, got {arena_bytes}")
    return tuple(sorted({0, arena_bytes // 3, arena_bytes // 2, arena_bytes - 1}))


def _fabric_xml_text(root: ET.Element, field: str) -> tuple[str, ...]:
    values: set[str] = set()
    normalized_field = field.lower().replace("_", "")
    for fabric in root.iter():
        normalized_tag = fabric.tag.lower().replace("_", "")
        if normalized_tag not in {"fabric", "gpufabric"}:
            continue
        for element in fabric.iter():
            if element.tag.lower().replace("_", "") != normalized_field:
                continue
            if (element.text or "").strip():
                values.add((element.text or "").strip())
    return tuple(sorted(values))


def fabric_environment() -> FabricEnvironment:
    """Capture pod-visible IMEX devices and NVLink-fabric identity."""
    imex_channels = tuple(str(path) for path in sorted(_IMEX_DEVICE_ROOT.glob("channel*")))
    completed = subprocess.run(
        ["nvidia-smi", "-q", "-x"],
        check=True,
        capture_output=True,
        text=True,
    )
    root = ET.fromstring(completed.stdout)
    return FabricEnvironment(
        imex_channels=imex_channels,
        cluster_uuids=_fabric_xml_text(root, "cluster_uuid"),
        clique_ids=_fabric_xml_text(root, "clique_id"),
        fabric_states=_fabric_xml_text(root, "state"),
        fabric_statuses=_fabric_xml_text(root, "status"),
    )


def _coordinator_endpoint(rank: ProbeRank) -> str:
    endpoint: str | None = None
    if rank.global_rank == 0:
        job_info = get_job_info()
        if job_info is None:
            raise RuntimeError("EP64 probe requires Iris job metadata")
        family = socket.AF_INET6 if ":" in job_info.advertise_host else socket.AF_INET
        with socket.socket(family) as listener:
            listener.bind(("", 0))
            port = listener.getsockname()[1]
        endpoint = (
            f"[{job_info.advertise_host}]:{port}" if family == socket.AF_INET6 else f"{job_info.advertise_host}:{port}"
        )
    resolved = multihost_broadcast_sync(endpoint, is_source=rank.global_rank == 0)
    if not isinstance(resolved, str):
        raise RuntimeError(f"Invalid Torch Store endpoint {resolved!r}")
    return resolved


def _validate_fabric(fabric: FabricEnvironment) -> None:
    if not fabric.imex_channels:
        raise RuntimeError(f"No IMEX channel devices are visible under {_IMEX_DEVICE_ROOT}")
    if not fabric.fabric_states or any(value.lower() != "completed" for value in fabric.fabric_states):
        raise RuntimeError(f"NVLink fabric is not completed: {fabric.fabric_states}")
    valid_statuses = {"success", "nvml_success"}
    if not fabric.fabric_statuses or any(value.lower() not in valid_statuses for value in fabric.fabric_statuses):
        raise RuntimeError(f"NVLink fabric status is not successful: {fabric.fabric_statuses}")
    if not fabric.cluster_uuids or any(value.lower() in {"n/a", "none", "0"} for value in fabric.cluster_uuids):
        raise RuntimeError(f"NVLink fabric has no usable cluster UUID: {fabric.cluster_uuids}")
    if not fabric.clique_ids or any(value.lower() in {"n/a", "none"} for value in fabric.clique_ids):
        raise RuntimeError(f"NVLink fabric has no usable clique ID: {fabric.clique_ids}")


def _validate_world_fabric(fabrics: list[FabricEnvironment]) -> None:
    cluster_uuids = {value for fabric in fabrics for value in fabric.cluster_uuids}
    clique_ids = {value for fabric in fabrics for value in fabric.clique_ids}
    if len(cluster_uuids) != 1:
        raise RuntimeError(f"Ranks do not share one NVLink cluster UUID: {sorted(cluster_uuids)}")
    if len(clique_ids) != 1:
        raise RuntimeError(f"Ranks do not share one NVLink clique ID: {sorted(clique_ids)}")


def _device_phase_barrier(
    *, torch: object, dist: object, group: object, handle: object, channel: int, timeout_ms: int
) -> None:
    """Align host progress before exercising the device-side fabric barrier.

    The probe intentionally validates remote values from Python, whose hundreds of scalar reads at
    EP64 can skew rank arrival by minutes. Production MoK does not use this host barrier; its fused
    kernel publishes readiness directly on device.
    """

    torch.cuda.synchronize()  # type: ignore[attr-defined]
    dist.barrier(group=group)  # type: ignore[attr-defined]
    handle.barrier(channel=channel, timeout_ms=timeout_ms)  # type: ignore[attr-defined]
    torch.cuda.synchronize()  # type: ignore[attr-defined]


def _jax_device_round_trip(rank: ProbeRank, expected: int) -> None:
    import jax  # noqa: PLC0415  # optional GPU dependency, imported only by the live probe
    import jax.numpy as jnp  # noqa: PLC0415

    if jax.process_index() != rank.global_rank or jax.process_count() != rank.world_size:
        raise RuntimeError(
            f"JAX rank/world mismatch: got {jax.process_index()}/{jax.process_count()}, "
            f"expected {rank.global_rank}/{rank.world_size}"
        )
    devices = jax.local_devices()
    if len(devices) != 1:
        raise RuntimeError(f"EP64 probe requires one local JAX device, got {devices}")
    device = devices[0]
    if device.platform != "gpu":
        raise RuntimeError(f"EP64 probe requires a JAX GPU device, got {device}")
    local_hardware_id = getattr(device, "local_hardware_id", None)
    if local_hardware_id is not None and int(local_hardware_id) != rank.local_device_id:
        raise RuntimeError(
            f"JAX local hardware id {local_hardware_id} does not match supervised GPU {rank.local_device_id}"
        )
    observed = int(jax.device_get(jax.jit(lambda value: value + 1)(jnp.asarray(expected - 1, dtype=jnp.int32))))
    if observed != expected:
        raise RuntimeError(f"JAX device round trip expected {expected}, got {observed}")


def run_probe(*, expected_world_size: int, arena_bytes: int, iterations: int, timeout: float) -> ProbeResult:
    """Rendezvous one CUDA arena per rank and validate all peer mappings."""
    rank = probe_rank_from_env(dict(os.environ))
    if rank.world_size != expected_world_size:
        raise ValueError(f"Expected {expected_world_size} processes, got {rank.world_size}")
    if iterations <= 0:
        raise ValueError(f"iterations must be positive, got {iterations}")
    if arena_bytes < rank.world_size:
        raise ValueError(f"arena_bytes must be at least the world size ({rank.world_size}), got {arena_bytes}")
    offsets = sample_offsets(arena_bytes)

    os.environ[_DIRECT_SYMMETRIC_MEMORY_ENV] = "0"
    os.environ[_DISABLE_SYMMETRIC_MEMORY_MULTICAST_ENV] = "1"
    os.environ["LOCAL_RANK"] = str(rank.local_device_id)
    initialize_jax()
    _jax_device_round_trip(rank, expected=rank.global_rank + 1)
    job_info = get_job_info()
    if job_info is None:
        raise RuntimeError("EP64 probe requires Iris job metadata")
    gloo_interface = interface_for_ipv4(job_info.advertise_host)
    os.environ["GLOO_SOCKET_IFNAME"] = gloo_interface
    logger.info("Gloo metadata group uses interface %s for %s", gloo_interface, job_info.advertise_host)

    import torch  # noqa: PLC0415  # optional GPU dependency, imported only by the live probe
    import torch.distributed as dist  # noqa: PLC0415
    import torch.distributed._symmetric_memory as symm_mem  # noqa: PLC0415

    if not torch.cuda.is_available():
        raise RuntimeError("EP64 symmetric-memory probe requires CUDA")
    if not dist.is_gloo_available():
        raise RuntimeError("PyTorch Gloo is required for symmetric-memory metadata rendezvous")

    torch.cuda.set_device(rank.local_device_id)
    device = torch.device("cuda", rank.local_device_id)
    fabric = fabric_environment()
    _validate_fabric(fabric)
    endpoint = _coordinator_endpoint(rank)

    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://{endpoint}",
        rank=rank.global_rank,
        world_size=rank.world_size,
        timeout=timedelta(seconds=timeout),
    )
    group = dist.group.WORLD
    arena = None
    handle = None
    remote_tensors = []
    try:
        world_fabrics: list[FabricEnvironment | None] = [None] * rank.world_size
        dist.all_gather_object(world_fabrics, fabric, group=group)
        if any(item is None for item in world_fabrics):
            raise RuntimeError(f"Fabric metadata gather was incomplete: {world_fabrics}")
        _validate_world_fabric([item for item in world_fabrics if item is not None])
        symm_mem.set_signal_pad_size(_DEFAULT_SIGNAL_PAD_BYTES)
        backend = str(symm_mem.get_backend(device))
        if backend.upper() != "CUDA":
            raise RuntimeError(f"Expected the CUDA symmetric-memory backend, got {backend!r}")
        arena = symm_mem.empty(arena_bytes, dtype=torch.uint8, device=device)
        handle = symm_mem.rendezvous(arena, group)
        if handle.world_size != rank.world_size or handle.rank != rank.global_rank:
            raise RuntimeError(
                f"Symmetric-memory identity mismatch: handle rank/world={handle.rank}/{handle.world_size}, "
                f"expected {rank.global_rank}/{rank.world_size}"
            )

        peer_pointers = tuple(int(pointer) for pointer in handle.buffer_ptrs)
        signal_pointers = tuple(int(pointer) for pointer in handle.signal_pad_ptrs)
        if len(peer_pointers) != rank.world_size or not all(peer_pointers):
            raise RuntimeError(f"Invalid peer pointer table of length {len(peer_pointers)}")
        if len(signal_pointers) != rank.world_size or not all(signal_pointers):
            raise RuntimeError(f"Invalid signal pointer table of length {len(signal_pointers)}")

        remote_tensors = [handle.get_remote_tensor(peer, arena.size(), arena.dtype) for peer in range(rank.world_size)]
        remote_reads_checked = 0
        remote_writes_checked = 0
        barrier_timeout_ms = int(timeout * 1000)
        for iteration in range(iterations):
            arena.fill_(pattern_byte(rank.global_rank, iteration))
            _device_phase_barrier(
                torch=torch,
                dist=dist,
                group=group,
                handle=handle,
                channel=0,
                timeout_ms=barrier_timeout_ms,
            )
            for peer, remote in enumerate(remote_tensors):
                observed = [int(remote[offset].item()) for offset in offsets]
                expected = pattern_byte(peer, iteration)
                if observed != [expected] * len(offsets):
                    raise AssertionError(
                        f"Remote read mismatch rank={rank.global_rank} peer={peer} iteration={iteration}: "
                        f"expected {expected}, got {observed}"
                    )
                remote_reads_checked += len(offsets)

            expected_write = pattern_byte(rank.global_rank, iteration + 101)
            for remote in remote_tensors:
                remote[rank.global_rank] = expected_write
            _device_phase_barrier(
                torch=torch,
                dist=dist,
                group=group,
                handle=handle,
                channel=1,
                timeout_ms=barrier_timeout_ms,
            )
            for source in range(rank.world_size):
                observed = int(arena[source].item())
                expected = pattern_byte(source, iteration + 101)
                if observed != expected:
                    raise AssertionError(
                        f"Remote write mismatch rank={rank.global_rank} source={source} iteration={iteration}: "
                        f"expected {expected}, got {observed}"
                    )
                remote_writes_checked += 1
            _device_phase_barrier(
                torch=torch,
                dist=dist,
                group=group,
                handle=handle,
                channel=0,
                timeout_ms=barrier_timeout_ms,
            )

        _device_phase_barrier(
            torch=torch,
            dist=dist,
            group=group,
            handle=handle,
            channel=1,
            timeout_ms=barrier_timeout_ms,
        )
        _jax_device_round_trip(rank, expected=rank.global_rank + iterations + 1)
        result = ProbeResult(
            rank=rank,
            backend=backend,
            arena_bytes=arena_bytes,
            iterations=iterations,
            peer_pointer_count=len(peer_pointers),
            signal_pointer_count=len(signal_pointers),
            remote_reads_checked=remote_reads_checked,
            remote_writes_checked=remote_writes_checked,
            fabric=fabric,
        )
    finally:
        remote_tensors.clear()
        del handle
        del arena
        gc.collect()
        if dist.is_initialized():
            dist.barrier(group=group)
            dist.destroy_process_group(group)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-world-size", type=int, required=True)
    parser.add_argument("--arena-bytes", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    probe_entrypoint(
        ProbeConfig(
            expected_world_size=args.expected_world_size,
            arena_bytes=args.arena_bytes,
            iterations=args.iterations,
            timeout=args.timeout,
        )
    )


def probe_entrypoint(config: ProbeConfig) -> None:
    """Run the probe as a Fray callable and close the JAX world explicitly."""
    logging.basicConfig(level=logging.INFO)

    import jax  # noqa: PLC0415  # optional GPU dependency, initialized by run_probe

    try:
        result = run_probe(
            expected_world_size=config.expected_world_size,
            arena_bytes=config.arena_bytes,
            iterations=config.iterations,
            timeout=config.timeout,
        )
        print(json.dumps(asdict(result), sort_keys=True), flush=True)
    finally:
        if jax.distributed.is_initialized():
            jax.distributed.shutdown()


if __name__ == "__main__":
    main()
