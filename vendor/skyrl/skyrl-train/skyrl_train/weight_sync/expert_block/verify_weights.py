"""Verify an expert-block sync by replaying it (``generator.expert_block_sync.verify``).

Every root sends its weights again. Each receiver receives them into scratch and counts the
bytes that differ from its installed weights. It also checks that the bytes it compared add up
to all the parameter bytes it holds, counting only the HF rows of a padded vocabulary tensor.
Each trainer rank checks that its data-parallel peers hold identical bytes, because any of
them can be a root.

A replay sends about as many bytes as a sync.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist

from skyrl_train.weight_sync.expert_block.stream import Stream


@dataclass(frozen=True)
class ReplayReport:
    participant: int
    version: int
    compared_bytes: int
    parameter_bytes: int
    mismatched_bytes: int


REPLAY_COMPARE_CHUNK_BYTES = 16 << 20


def replay(stream: Stream, version: int, *, chunk_bytes: int = REPLAY_COMPARE_CHUNK_BYTES) -> ReplayReport:
    """Run the sync's broadcasts again. Receivers compare each transfer with their installed weights."""
    if chunk_bytes <= 0:
        raise ValueError("Replay comparison chunks must contain at least one byte")
    with torch.no_grad():
        return _replay(stream, version, chunk_bytes)


def _replay(stream: Stream, version: int, chunk_bytes: int) -> ReplayReport:
    landings = [(item, landing) for item, landing in stream.transfers() if landing is not None]
    scratch = torch.empty(
        max([landing.nbytes for _, landing in landings] + [0]), dtype=torch.uint8, device=stream.device
    )
    mismatched = torch.zeros((), dtype=torch.int64, device=stream.device)
    compared = 0
    for item, landing in stream.transfers():
        if landing is None:
            stream.send(item)
            continue
        wire = scratch.narrow(0, 0, landing.nbytes).view(landing.wire_dtype).view(landing.installed.shape)
        stream.receive(item, wire)
        # The router weight is stored as FP32; compare it in its wire dtype.
        installed = landing.installed.to(landing.wire_dtype)
        wire_bytes = wire.view(-1).view(torch.uint8)
        installed_bytes = installed.view(-1).view(torch.uint8)
        # Summing bool bytes can materialize an int64 input. A vocabulary tensor
        # otherwise needs eight times its byte size in additional GPU memory.
        for start in range(0, landing.nbytes, chunk_bytes):
            count = min(chunk_bytes, landing.nbytes - start)
            mismatched.add_(wire_bytes.narrow(0, start, count).ne(installed_bytes.narrow(0, start, count)).sum())
        compared += landing.nbytes
    parameter_bytes = 0
    if not stream.trainer:
        # Count parameter bytes in the wire dtype, so the FP32 router weight counts as BF16.
        parameter_bytes = sum(value.numel() * value.element_size() for value in stream.parameters.values())
        for _, landing in landings:
            parameter_bytes -= landing.installed.numel() * (
                landing.installed.element_size() - landing.nbytes // landing.installed.numel()
            )
    if stream.device.type == "cuda":
        torch.cuda.synchronize(stream.device)
    return ReplayReport(stream.participant, version, compared, parameter_bytes, int(mismatched.item()))


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
