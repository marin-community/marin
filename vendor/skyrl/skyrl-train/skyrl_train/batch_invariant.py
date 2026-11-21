"""Batch-invariant trainer activation and NCCL settings."""

import importlib
import logging
import os

from skyrl_train.env_vars import VLLM_BATCH_INVARIANT_ENV


logger = logging.getLogger(__name__)

# Mirror pinned vLLM as a set; replace this list via https://github.com/marin-community/vllm/issues/80.
BATCH_INVARIANT_NCCL_ENV = {
    "NCCL_LAUNCH_MODE": "GROUP",  # Match vLLM's kernel launch mode.
    "NCCL_COLLNET_ENABLE": "0",  # Avoid CollNet reduction offload.
    "NCCL_NVLS_ENABLE": "0",  # Avoid NVLink SHARP reduction offload.
    "NCCL_P2P_NET_DISABLE": "1",  # Match vLLM; NCCL does not document this switch.
    "NCCL_MIN_NCHANNELS": "1",  # Match vLLM's channel lower bound.
    "NCCL_MAX_NCHANNELS": "1",  # Prevent a multiple-channel split.
    "NCCL_PROTO": "Simple",  # Keep LL and LL128 out of collectives.
    "NCCL_ALGO": "allreduce:tree",  # Keep all-reduce on the Tree algorithm.
    "NCCL_NTHREADS": "1",  # Match vLLM; NCCL does not document 1 as valid.
    "NCCL_SOCKET_NTHREADS": "1",  # Avoid platform-dependent socket thread counts.
}


def enable_trainer_batch_invariance(enabled: bool) -> None:
    """Enable trainer CUDA overrides when configured."""

    if not enabled:
        return

    if os.environ.get(VLLM_BATCH_INVARIANT_ENV) != "1":
        raise RuntimeError(
            f"trainer.algorithm.batch_invariant=true, but {VLLM_BATCH_INVARIANT_ENV}=1 "
            "was not propagated to the trainer worker"
        )

    try:
        # Use the same installed kernels as rollout workers so the two
        # log-probability paths share reduction order.
        batch_invariant = importlib.import_module("vllm.model_executor.determinism.batch_invariant")
    except ImportError as error:
        raise RuntimeError(
            "trainer.algorithm.batch_invariant=true requires the pinned Marin vLLM runtime; "
            "launch training with the vllm extra"
        ) from error

    batch_invariant.init_batch_invariance()
    try:
        registered_ops = sorted(batch_invariant._batch_invariant_LIB._op_impls)
    except AttributeError as error:
        raise RuntimeError(
            "The pinned vLLM batch-invariant initializer completed without exposing registered CUDA overrides"
        ) from error
    logger.info(
        "Batch-invariant trainer kernels enabled from the pinned vLLM runtime; registered CUDA overrides: %s",
        ", ".join(registered_ops),
    )
