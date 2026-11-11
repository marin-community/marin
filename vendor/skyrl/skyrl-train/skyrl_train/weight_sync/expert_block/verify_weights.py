"""Compare replayed expert-block weights and trainer replicas byte by byte."""

from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class ReplayReport:
    participant: int
    version: int
    compared_bytes: int
    parameter_bytes: int
    mismatched_bytes: int
    quantized_bytes: int = 0


REPLAY_COMPARE_CHUNK_BYTES = 16 << 20


def mismatched_bytes(actual: torch.Tensor, expected: torch.Tensor, chunk_bytes: int) -> torch.Tensor:
    """Count byte differences between tensors of the same shape and dtype."""
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError("Replay tensors differ in shape or dtype")
    actual_bytes = actual.detach().contiguous().view(-1).view(torch.uint8)
    expected_bytes = expected.detach().contiguous().view(-1).view(torch.uint8)
    mismatched = torch.zeros((), dtype=torch.int64, device=actual.device)
    # Bound the int64 reduction temporary for large parameter tensors.
    for start in range(0, actual_bytes.numel(), chunk_bytes):
        count = min(chunk_bytes, actual_bytes.numel() - start)
        mismatched.add_(actual_bytes.narrow(0, start, count).ne(expected_bytes.narrow(0, start, count)).sum())
    return mismatched


@dataclass(frozen=True)
class ReplicaReport:
    participant: int
    version: int
    compared_bytes: int
    mismatched_bytes: int


# Bytes compared per all-reduce. Each chunk is copied to int32 twice, for the group minimum and
# maximum, so it needs eight times this much memory on a GPU the trainer already fills.
REPLICA_COMPARE_CHUNK_BYTES = 16 << 20


def compare_replicas(
    sources: dict[str, torch.Tensor],
    groups: dict[str, dist.ProcessGroup],
    participant: int,
    version: int,
    *,
    chunk_bytes: int = REPLICA_COMPARE_CHUNK_BYTES,
) -> ReplicaReport:
    """Count the bytes where this rank's parameters differ from its peers'.

    ``groups`` maps each parameter name to the process group of the ranks that hold a copy. Bytes
    are reduced as integers, so the comparison is exact.
    """
    mismatched = 0
    compared = 0
    with torch.no_grad():
        for name, source in sources.items():
            words = source.detach().contiguous().view(-1).view(torch.uint8)
            for start in range(0, words.numel(), chunk_bytes):
                low = words.narrow(0, start, min(chunk_bytes, words.numel() - start)).to(torch.int32)
                high = low.clone()
                dist.all_reduce(low, op=dist.ReduceOp.MIN, group=groups[name])
                dist.all_reduce(high, op=dist.ReduceOp.MAX, group=groups[name])
                mismatched += int(low.ne(high).sum().item())
            compared += words.numel()
    return ReplicaReport(participant, version, compared, mismatched)
