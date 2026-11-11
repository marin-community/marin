"""NCCL channel limit for batch-invariant vLLM engines.

With batch invariance on, the pinned vLLM limits every NCCL communicator to one channel. It sets
the limit in ``override_envs_for_invariance``, which each worker calls just before it creates its
NCCL groups. ``generator.inference_engine_nccl_max_channels`` raises the engine's limit after that
call, so the engine's own collectives can use more channels. Weight-sync groups stay at one channel
so that each engine's channel layout matches the trainer's. Batch invariance at the raised limit
depends on NCCL using one ring on every channel; ``tests/gpu/test_engine_nccl_channels.py`` checks it.

This module runs inside the vLLM worker processes, because it is imported with the worker
extension class.
"""

import os

from loguru import logger
from vllm.model_executor.determinism import batch_invariant as vllm_batch_invariant

from skyrl_train.batch_invariant import BATCH_INVARIANT_NCCL_ENV, NCCL_MAX_CHANNELS_ENV
from skyrl_train.env_vars import ENGINE_NCCL_MAX_CHANNELS_ENV

WEIGHT_SYNC_NCCL_CHANNELS = int(BATCH_INVARIANT_NCCL_ENV[NCCL_MAX_CHANNELS_ENV])


def engine_nccl_max_channels() -> int | None:
    """The configured engine channel limit, or ``None`` to keep vLLM's limit."""
    raw = os.environ.get(ENGINE_NCCL_MAX_CHANNELS_ENV)
    if raw is None:
        return None
    channels = int(raw)
    if channels < 1:
        raise ValueError(f"{ENGINE_NCCL_MAX_CHANNELS_ENV} must be a positive integer, not {raw!r}")
    return channels


def install_engine_nccl_channel_limit() -> None:
    """Make vLLM's batch-invariant setup apply the configured engine channel limit."""
    channels = engine_nccl_max_channels()
    if channels is None:
        return
    pinned = vllm_batch_invariant.override_envs_for_invariance

    def override_envs_with_engine_channels() -> None:
        pinned()
        os.environ[NCCL_MAX_CHANNELS_ENV] = str(channels)
        logger.info(f"Batch-invariant vLLM engine {NCCL_MAX_CHANNELS_ENV} raised to {channels}")

    vllm_batch_invariant.override_envs_for_invariance = override_envs_with_engine_channels


def weight_sync_max_ctas() -> int | None:
    """The channel cap that keeps a weight-sync group at the trainer's limit.

    Returns ``None`` when the engine keeps vLLM's limit, which already matches the trainer.
    """
    if engine_nccl_max_channels() is None:
        return None
    return WEIGHT_SYNC_NCCL_CHANNELS
