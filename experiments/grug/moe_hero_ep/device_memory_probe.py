# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Research probe: split a GPU's physical memory into the XLA pool and everything outside it.

Under `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`, XLA allocates from the device's default CUDA
memory pool. Memory outside that pool (NCCL communicators and their NVLS buffers, the collective
BFC allocator, cuBLAS/cuDNN workspaces, loaded kernel modules, the CUDA context) is
`total - free - pool_reserved`.
"""

import logging
from dataclasses import dataclass

import jax

logger = logging.getLogger(__name__)

GIB = 1024**3


@dataclass(frozen=True)
class DeviceMemoryBreakdown:
    free: int
    total: int
    pool_reserved: int
    pool_used: int
    pool_reserved_high: int
    pool_used_high: int
    xla_in_use: int
    xla_peak: int

    @property
    def non_pool(self) -> int:
        return self.total - self.free - self.pool_reserved

    def metrics(self, prefix: str) -> dict[str, float]:
        return {
            f"{prefix}/free_gib": self.free / GIB,
            f"{prefix}/pool_reserved_gib": self.pool_reserved / GIB,
            f"{prefix}/pool_used_gib": self.pool_used / GIB,
            f"{prefix}/pool_reserved_high_gib": self.pool_reserved_high / GIB,
            f"{prefix}/non_pool_gib": self.non_pool / GIB,
        }

    def describe(self) -> str:
        return (
            f"free={self.free / GIB:.3f} pool_reserved={self.pool_reserved / GIB:.3f} "
            f"pool_used={self.pool_used / GIB:.3f} pool_reserved_high={self.pool_reserved_high / GIB:.3f} "
            f"pool_used_high={self.pool_used_high / GIB:.3f} non_pool={self.non_pool / GIB:.3f} "
            f"xla_in_use={self.xla_in_use / GIB:.3f} xla_peak={self.xla_peak / GIB:.3f} GiB"
        )


def _checked(result):
    # cuda.bindings returns (CUresult, *values).
    err, *values = result
    if int(err) != 0:
        raise RuntimeError(f"CUDA driver call failed: {err}")
    return values[0] if len(values) == 1 else values


def device_memory_breakdown(device) -> DeviceMemoryBreakdown:
    # GPU-only optional dependency; the module imports on CPU hosts.
    from cuda.bindings import driver

    stats = device.memory_stats() or {}
    _checked(driver.cuInit(0))
    cu_device = _checked(driver.cuDeviceGet(device.local_hardware_id))
    context = _checked(driver.cuDevicePrimaryCtxRetain(cu_device))
    _checked(driver.cuCtxPushCurrent(context))
    try:
        free, total = _checked(driver.cuMemGetInfo())
        pool = _checked(driver.cuDeviceGetDefaultMemPool(cu_device))
        attr = driver.CUmemPool_attribute

        def pool_attr(name):
            return int(_checked(driver.cuMemPoolGetAttribute(pool, getattr(attr, name))))

        reserved = pool_attr("CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT")
        used = pool_attr("CU_MEMPOOL_ATTR_USED_MEM_CURRENT")
        reserved_high = pool_attr("CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH")
        used_high = pool_attr("CU_MEMPOOL_ATTR_USED_MEM_HIGH")
    finally:
        _checked(driver.cuCtxPopCurrent())
        _checked(driver.cuDevicePrimaryCtxRelease(cu_device))
    return DeviceMemoryBreakdown(
        free=int(free),
        total=int(total),
        pool_reserved=reserved,
        pool_used=used,
        pool_reserved_high=reserved_high,
        pool_used_high=used_high,
        xla_in_use=int(stats.get("bytes_in_use", 0)),
        xla_peak=int(stats.get("peak_bytes_in_use", 0)),
    )


def log_device_memory_breakdown(tag: str) -> None:
    # A research probe must not take down the run it instruments.
    for device in jax.local_devices():
        try:
            breakdown = device_memory_breakdown(device)
        except Exception:
            logger.exception("DEVMEM[%s] probe failed", tag)
            return
        logger.info(
            "DEVMEM[%s] process=%d device=%d local_hw=%d %s",
            tag,
            jax.process_index(),
            device.id,
            device.local_hardware_id,
            breakdown.describe(),
        )
