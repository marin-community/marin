"""Dependency-light Ray worker bootstrap.

Ray imports this module before assigning an actor-specific CUDA visibility mask.
Keep both :mod:`skyrl_train` and this module free of torch/transformers/ray imports
so CUDA cannot be initialized against the driver's device view first.
"""

import asyncio
import logging
import os

from skyrl_train.numa_policy import install_host_memory_policy, is_numa_affinity_enabled
from skyrl_train.env_vars import DEBUG_MODE_ENV, DebugMode
from marinskyrl.process_diagnostics import initialize_process_diagnostics


logger = logging.getLogger(__name__)


INCOMPATIBLE_NCCL_ENVIRONMENT = (
    "NCCL_BLOCKING_WAIT",
    "TORCH_NCCL_BLOCKING_WAIT",
    "TORCH_NCCL_BLOCKING_WAIT_TIMEOUT_MS",
)


def configure_worker_process() -> None:
    """Install process-wide prerequisites before Ray creates actor threads."""

    # MarinSkyRL uses ProcessGroupNCCL's asynchronous watchdog. Blocking wait
    # selects an incompatible wait path. The timeout-like variable is not a
    # PyTorch setting, but removing it makes the warning below expose that
    # invalid configuration instead of leaving a false deadline in the worker.
    removed_nccl_settings = [
        variable for variable in INCOMPATIBLE_NCCL_ENVIRONMENT if os.environ.pop(variable, None) is not None
    ]
    if removed_nccl_settings:
        logger.warning(
            "Ignoring incompatible NCCL worker settings before PyTorch initialization: %s",
            ", ".join(removed_nccl_settings),
        )
    if is_numa_affinity_enabled():
        install_host_memory_policy()
    # Ray has already projected the centrally managed worker environment before this hook runs.
    if os.environ.get(DEBUG_MODE_ENV) in {DebugMode.LIGHT, DebugMode.DISTRIBUTED}:
        initialize_process_diagnostics("ray-worker-bootstrap")
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

    try:
        import uvloop  # noqa: PLC0415 - optional dependency in some worker environments
    except ImportError:
        return

    def _stock_new_event_loop():
        return asyncio.SelectorEventLoop()

    def _stock_install():
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

    uvloop.new_event_loop = _stock_new_event_loop
    uvloop.install = _stock_install
    uvloop.EventLoopPolicy = asyncio.DefaultEventLoopPolicy
    if hasattr(uvloop, "Loop"):
        uvloop.Loop = asyncio.SelectorEventLoop
