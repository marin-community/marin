"""Batch invariance of engine reduce-scatters at a raised NCCL channel limit.

``generator.inference_engine_nccl_max_channels`` lifts vLLM's one-channel limit for the engine's own
collectives. At TP=1 with the ``allgather_reducescatter`` backend, the MoE combine's reduce-scatters
are the engine's only floating-point collectives. vLLM issues ``ncclReduceScatter`` when every rank
holds the same token count and grouped ``ncclReduce`` calls otherwise. A ring sums each element in ring
order, so a token's sum stays independent of the message size only while every channel follows the
same ring. NCCL builds rings per channel from the topology, so run this on the engine's hardware.
"""

import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from skyrl_train.batch_invariant import BATCH_INVARIANT_NCCL_ENV, NCCL_MAX_CHANNELS_ENV
from tests.gpu.grug_gpu_gates import require_hoppers
from tests.gpu.utils import get_available_gpus

HIDDEN_SIZE = 2048
# Per-rank token counts from a single decode token to a large prefill chunk.
TOKENS_PER_RANK = (1, 3, 64, 257, 2048, 8192)
MAX_TOKENS = max(TOKENS_PER_RANK)


def _unequal_counts(tokens: int, world_size: int) -> list[int]:
    return [max(1, tokens - owner) for owner in range(world_size)]


def _mismatch(result: torch.Tensor, expected: torch.Tensor, case: str) -> str | None:
    if torch.equal(result, expected):
        return None
    differs = result != expected
    rows = differs.any(dim=1).nonzero().flatten().tolist()
    largest = (result.float() - expected.float()).abs().max().item()
    return f"{case}: {differs.sum().item()} elements in tokens {rows[:8]} differ, max abs {largest:.3g}"


def _check_reduce_scatters(rank: int, world_size: int, rendezvous: str, channels: int) -> None:
    os.environ.update(BATCH_INVARIANT_NCCL_ENV)
    os.environ[NCCL_MAX_CHANNELS_ENV] = str(channels)
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=world_size, device_id=device)
    try:
        # contributions[owner, token] is this rank's term in the sum that ``owner`` receives for ``token``.
        generator = torch.Generator().manual_seed(rank)
        contributions = torch.randn(world_size, MAX_TOKENS, HIDDEN_SIZE, generator=generator, dtype=torch.float32)
        contributions = contributions.to(device=device, dtype=torch.bfloat16)

        reference = torch.empty(MAX_TOKENS, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
        dist.reduce_scatter_tensor(reference, contributions.reshape(-1, HIDDEN_SIZE))

        mismatches = []
        for tokens in TOKENS_PER_RANK:
            equal = torch.empty(tokens, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
            # reduce_scatter_tensor misreads strided input; at one token the reshape is a strided view.
            dist.reduce_scatter_tensor(equal, contributions[:, :tokens].reshape(-1, HIDDEN_SIZE).contiguous())
            mismatches.append(_mismatch(equal, reference[:tokens], f"equal counts, {tokens} tokens per rank"))

            counts = _unequal_counts(tokens, world_size)
            unequal = torch.empty(counts[rank], HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
            dist.reduce_scatter(unequal, [contributions[owner, :count] for owner, count in enumerate(counts)])
            mismatches.append(_mismatch(unequal, reference[: counts[rank]], f"unequal counts {counts}"))
        torch.cuda.synchronize(device)
        mismatches = [mismatch for mismatch in mismatches if mismatch is not None]
        assert not mismatches, f"rank {rank} at {channels} channels:\n" + "\n".join(mismatches)
    finally:
        dist.destroy_process_group()


# One channel is vLLM's pinned limit and checks the test itself; 32 is the raised engine limit.
@pytest.mark.parametrize("channels", [1, 32])
def test_engine_reduce_scatter_is_batch_invariant(tmp_path: Path, channels: int) -> None:
    """A token's reduce-scatter sum must not change with how many tokens share the collective."""
    world_size = min(len(get_available_gpus()), 8)
    if world_size < 2:
        pytest.skip("needs at least two GPUs")
    require_hoppers(world_size)
    rendezvous = f"file://{tmp_path / 'rendezvous'}"
    mp.spawn(_check_reduce_scatters, args=(world_size, rendezvous, channels), nprocs=world_size, join=True)
