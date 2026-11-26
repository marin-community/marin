"""NUMA initialization helpers that do not require importing vLLM."""

from collections.abc import Callable
from typing import Any


async def set_async_worker_numa_affinity(collective_rpc: Callable[..., Any]) -> Any:
    """Apply worker affinity before asynchronous engine startup completes."""
    return await collective_rpc("set_numa_affinity")
