"""GPU NUMA affinity utilities for multi-socket and unified memory architectures (e.g., GH200).

On GH200 nodes, each GPU has its own NUMA node for HBM memory (e.g., nodes 4, 12, 20, 28)
which is separate from the CPU NUMA node (0, 1, 2, 3). This module detects the correct
CPU NUMA affinity for each GPU and binds the calling process accordingly.

Detection strategy (in priority order):
1. nvidia-smi topo -m (most direct, but fails under proxychains/CUDA_VISIBLE_DEVICES)
2. Pure sysfs + numactl (no nvidia-smi needed — enumerates NVIDIA PCI devices, reads
   their NUMA nodes, then maps GPU NUMA nodes to CPU NUMA nodes via distance matrix)

Activation: Set ``trainer.placement.enable_numa_affinity=true`` to install CPU and memory
affinity. Topology queries and placement diagnostics remain available when it is unset.
"""

import os
import re
import subprocess
from collections import Counter
from ctypes import POINTER, c_int, c_ulong, c_void_p, get_errno
from functools import lru_cache
from typing import Collection, Dict, List, Optional, Tuple

from loguru import logger

from skyrl_train.numa_policy import (
    cpu_numa_topology,
    install_host_memory_policy,
    is_numa_affinity_enabled,
    load_libnuma,
    parse_linux_range_list,
)


def _nvidia_smi_env() -> Dict[str, str]:
    """Get environment for nvidia-smi subprocesses with CUDA_VISIBLE_DEVICES removed.

    nvidia-smi respects CUDA_VISIBLE_DEVICES on some driver versions, which remaps
    GPU indices (e.g., physical GPU 2 becomes GPU0). We need the real physical
    indices to map GPUs to their correct NUMA nodes, so we strip the variable.
    """
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


# ---------------------------------------------------------------------------
# Method 1: nvidia-smi topo -m (fastest, but unreliable under proxychains)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _parse_nvidia_smi_topo() -> Optional[Dict[int, Tuple[List[int], int]]]:
    """Parse nvidia-smi topo -m to get GPU -> (cpu_list, numa_node) mapping.

    Returns:
        Dict mapping GPU index to (cpu_affinity_list, cpu_numa_node), or None on failure.
        Example on GH200: {0: ([0..71], 0), 1: ([72..143], 1), ...}
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True,
            text=True,
            timeout=10,
            env=_nvidia_smi_env(),
        )
        if result.returncode != 0:
            return None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None

    gpu_map = {}
    for line in result.stdout.splitlines():
        # Match lines starting with "GPU0", "GPU1", etc.
        match = re.match(r"^GPU(\d+)\s+", line)
        if not match:
            continue

        gpu_idx = int(match.group(1))

        # Split by whitespace — the CPU Affinity and NUMA Affinity columns
        # are the last two (or three with GPU NUMA ID) tab-separated fields.
        # Format: GPU0 <connections...> <CPU Affinity> <NUMA Affinity> [<GPU NUMA ID>]
        parts = line.split()

        # Find CPU Affinity field: looks like "0-71" or "0-71,144-215"
        cpu_affinity_str = None
        numa_node = None
        for i, part in enumerate(parts):
            if re.match(r"^\d+(-\d+)?(,\d+(-\d+)?)*$", part) and "-" in part:
                cpu_affinity_str = part
                # Next numeric field after CPU Affinity is NUMA Affinity
                if i + 1 < len(parts) and parts[i + 1].isdigit():
                    numa_node = int(parts[i + 1])
                break

        if cpu_affinity_str is not None and numa_node is not None:
            cpus = parse_linux_range_list(cpu_affinity_str)
            gpu_map[gpu_idx] = (cpus, numa_node)

    return gpu_map if gpu_map else None


# ---------------------------------------------------------------------------
# Placement diagnostics
# ---------------------------------------------------------------------------


def memory_nodes_for_range(address: int, length: int, *, max_pages: int = 256) -> Dict[int, int]:
    """Return sampled physical NUMA node IDs mapped to their page counts."""
    if address <= 0 or length <= 0:
        raise ValueError("address and length must be positive")
    if max_pages <= 0:
        raise ValueError("max_pages must be positive")

    page_size = os.sysconf("SC_PAGE_SIZE")
    first_page = address - address % page_size
    last_page = (address + length - 1) // page_size * page_size
    page_count = (last_page - first_page) // page_size + 1
    if page_count <= max_pages:
        page_indices = list(range(page_count))
    elif max_pages == 1:
        page_indices = [page_count // 2]
    else:
        page_indices = sorted({index * (page_count - 1) // (max_pages - 1) for index in range(max_pages)})

    pages = (c_void_p * len(page_indices))(*(first_page + index * page_size for index in page_indices))
    statuses = (c_int * len(page_indices))()
    libnuma = load_libnuma()
    libnuma.move_pages.argtypes = [c_int, c_ulong, POINTER(c_void_p), POINTER(c_int), POINTER(c_int), c_int]
    libnuma.move_pages.restype = c_int
    if libnuma.move_pages(0, len(page_indices), pages, None, statuses, 0) != 0:
        errno = get_errno()
        raise OSError(errno, os.strerror(errno))
    failed_statuses = [status for status in statuses if status < 0]
    if failed_statuses:
        raise OSError(-failed_statuses[0], os.strerror(-failed_statuses[0]))
    return dict(Counter(int(status) for status in statuses))


# ---------------------------------------------------------------------------
# Method 2: Pure sysfs + numactl (no nvidia-smi needed)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _enumerate_gpus_from_sysfs() -> Optional[Dict[int, int]]:
    """Enumerate NVIDIA GPUs and their NUMA nodes directly from sysfs.

    Scans /sys/bus/pci/devices/ for NVIDIA GPUs (vendor 0x10de, class 0x03xxxx)
    and reads their numa_node. GPUs are ordered by PCI bus ID, matching nvidia-smi
    GPU index ordering.

    Returns:
        Dict mapping GPU index (0-based, ordered by bus ID) to NUMA node ID,
        or None on failure.
    """
    pci_base = "/sys/bus/pci/devices"
    nvidia_gpus = []
    try:
        for dev in sorted(os.listdir(pci_base)):
            dev_path = os.path.join(pci_base, dev)
            try:
                with open(os.path.join(dev_path, "vendor")) as f:
                    vendor = f.read().strip()
                with open(os.path.join(dev_path, "class")) as f:
                    dev_class = f.read().strip()
                # NVIDIA vendor = 0x10de, display/3D controller class = 0x03xxxx
                if vendor == "0x10de" and dev_class.startswith("0x03"):
                    with open(os.path.join(dev_path, "numa_node")) as f:
                        numa = int(f.read().strip())
                    nvidia_gpus.append((dev, numa))
            except (FileNotFoundError, ValueError):
                continue
    except FileNotFoundError:
        return None

    if not nvidia_gpus:
        return None

    result = {i: numa for i, (_, numa) in enumerate(nvidia_gpus)}
    logger.debug(f"NUMA: sysfs GPU enumeration: {result}")
    return result


def physical_gpu_id_for_worker(cuda_visible_devices: Optional[str], resolved_local_rank: int) -> int:
    """Resolve the physical GPU assigned to a Ray worker."""
    devices = [device for device in (cuda_visible_devices or "").split(",") if device]
    if not devices:
        return resolved_local_rank
    visible_rank = 0 if len(devices) == 1 else resolved_local_rank
    if visible_rank >= len(devices):
        raise RuntimeError(f"local rank {visible_rank} is outside CUDA_VISIBLE_DEVICES={','.join(devices)}")
    return int(devices[visible_rank])


def _find_closest_cpu_numa_node(gpu_numa_node: int, cpu_node_ids: Collection[int]) -> Optional[int]:
    """Find the CPU NUMA node closest to a GPU NUMA node using the kernel distance matrix.

    On GH200, GPU NUMA nodes (4, 12, 20, 28) have no CPUs — their HBM memory lives
    there. The closest CPU NUMA node (0, 1, 2, 3) is the one on the same SoC,
    determined by reading /sys/devices/system/node/nodeN/distance.
    """
    if gpu_numa_node in cpu_node_ids:
        return gpu_numa_node  # GPU NUMA node has CPUs directly

    # Read distance from gpu_numa_node to all other nodes
    dist_path = f"/sys/devices/system/node/node{gpu_numa_node}/distance"
    try:
        with open(dist_path) as f:
            distances = [int(d) for d in f.read().strip().split()]
    except (FileNotFoundError, ValueError):
        logger.debug(f"NUMA: could not read distance matrix for node {gpu_numa_node}")
        return None

    # Find the CPU node with the smallest distance
    best_node = None
    best_dist = float("inf")
    for node_id in cpu_node_ids:
        if node_id < len(distances) and distances[node_id] < best_dist:
            best_dist = distances[node_id]
            best_node = node_id

    if best_node is not None:
        logger.debug(f"NUMA: GPU NUMA node {gpu_numa_node} -> closest CPU NUMA node {best_node} (distance={best_dist})")
    return best_node


def _get_affinity_via_sysfs_numactl(gpu_physical_id: int) -> Optional[Tuple[List[int], int]]:
    """Get GPU CPU affinity using sysfs GPU enumeration + numactl, no nvidia-smi.

    Steps:
    1. Enumerate NVIDIA GPUs from /sys/bus/pci/devices/ to get GPU -> NUMA node map
    2. Parse numactl --hardware to get NUMA node -> CPU list map
    3. If GPU's NUMA node has no CPUs (GH200 HBM node), find the closest CPU NUMA node
       via the kernel distance matrix
    """
    gpu_numa_map = _enumerate_gpus_from_sysfs()
    if not gpu_numa_map or gpu_physical_id not in gpu_numa_map:
        return None

    gpu_numa = gpu_numa_map[gpu_physical_id]

    topology = cpu_numa_topology()

    # Direct match: GPU's NUMA node has CPUs
    if gpu_numa in topology:
        return (list(topology[gpu_numa]), gpu_numa)

    # GH200 case: GPU NUMA node is HBM-only, find closest CPU NUMA node
    closest = _find_closest_cpu_numa_node(gpu_numa, topology.keys())
    if closest is not None:
        return (list(topology[closest]), closest)

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_gpu_cpu_affinity(gpu_physical_id: int) -> Optional[Tuple[List[int], int]]:
    """Get the optimal CPU list and NUMA node for a GPU.

    Args:
        gpu_physical_id: Physical GPU index (as seen by nvidia-smi, not CUDA logical index).

    Returns:
        Tuple of (cpu_list, numa_node) or None if detection fails.
    """
    # Method 1: nvidia-smi topo (most direct, but fails under proxychains)
    topo = _parse_nvidia_smi_topo()
    if topo and gpu_physical_id in topo:
        return topo[gpu_physical_id]

    # Method 2: sysfs + numactl (no nvidia-smi needed, handles GH200 correctly)
    result = _get_affinity_via_sysfs_numactl(gpu_physical_id)
    if result is not None:
        return result

    return None


def set_numa_affinity_for_gpu(gpu_id: int) -> None:
    """Install GPU-local CPU affinity and an LPDDR-only policy when enabled.

    The disabled path is a no-op. Once enabled, topology discovery, memory-policy
    installation, and CPU binding are strict contracts: an undetectable or
    disallowed topology and an empty CPU overlap raise ``RuntimeError``.
    """
    if not is_numa_affinity_enabled():
        return None

    affinity = get_gpu_cpu_affinity(gpu_id)
    if affinity is None:
        raise RuntimeError(f"could not detect NUMA topology for physical GPU {gpu_id}")

    cpu_list, numa_node = affinity
    target_memory_nodes = install_host_memory_policy()
    if numa_node not in target_memory_nodes:
        raise RuntimeError(
            f"GPU {gpu_id}'s CPU NUMA node {numa_node} is outside the allowed host-memory nodes {target_memory_nodes}"
        )

    target_cpus = set(cpu_list)
    try:
        allowed_cpus = os.sched_getaffinity(0)
    except (OSError, AttributeError) as error:
        raise RuntimeError("could not read the worker's allowed CPU set") from error

    if not target_cpus.issubset(allowed_cpus):
        overlap = target_cpus & allowed_cpus
        logger.warning(
            f"NUMA affinity: target CPUs {cpu_list[0]}-{cpu_list[-1]} (NUMA {numa_node}) "
            f"not fully within allowed cpuset {min(allowed_cpus)}-{max(allowed_cpus)} "
            f"({len(allowed_cpus)} CPUs). This is likely a SLURM cgroup restriction. "
            f"Consider adding '--cpu-bind=none' to srun or setting "
            f"'TaskPluginParam=none' in slurm.conf."
        )
        if overlap:
            target_cpus = overlap
            logger.info(f"NUMA affinity: falling back to {len(overlap)} overlapping CPUs for GPU {gpu_id}")
        else:
            raise RuntimeError(f"GPU {gpu_id}'s CPU NUMA node {numa_node} has no CPUs in the worker's allowed CPU set")

    try:
        os.sched_setaffinity(0, target_cpus)
        logger.info(
            f"NUMA affinity: bound process to CPUs {min(target_cpus)}-{max(target_cpus)} "
            f"(NUMA node {numa_node}) for GPU {gpu_id}"
        )
    except (OSError, AttributeError) as error:
        raise RuntimeError(f"could not bind GPU {gpu_id}'s worker to CPUs {sorted(target_cpus)}") from error
