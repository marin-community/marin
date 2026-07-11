# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: PLC0415

import json
import multiprocessing
import os
import traceback
from dataclasses import asdict, dataclass
from queue import Empty

from cute_nvshmem_transport.pattern_kernels import PeerPattern, TransferDirection, run_pattern_probe


@dataclass(frozen=True)
class PatternRankResult:
    rank: int
    ready_epochs: tuple[int, ...]
    consumed_epochs: tuple[int, ...]
    validation_errors: int
    first_error: tuple[int, int, int, int, int, int]


def _rank_probe(
    rank: int,
    num_pes: int,
    uid: object,
    num_epochs: int,
    num_slots: int,
    payload_bytes: int,
    direction: TransferDirection,
    pattern: PeerPattern,
    results: multiprocessing.Queue,
) -> None:
    try:
        import nvshmem.core as nvshmem
        from cuda.core import Device

        device = Device(rank)
        device.set_current()
        nvshmem.init(device=device, uid=uid, rank=rank, nranks=num_pes, initializer_method="uid")
        result = run_pattern_probe(num_epochs, num_slots, payload_bytes, direction, pattern)
        nvshmem.finalize()
        results.put(PatternRankResult(rank=rank, **asdict(result)))
    except BaseException:
        results.put((rank, traceback.format_exc()))


def _peer_active(pattern: PeerPattern, rank: int, peer: int, num_pes: int) -> bool:
    if pattern is PeerPattern.ALL_TO_ALL:
        return peer != rank
    if pattern is PeerPattern.PAIR:
        return peer == (rank ^ 1)
    return peer == (rank + 1) % num_pes


def run_pattern_correctness(
    num_pes: int,
    num_epochs: int,
    num_slots: int,
    payload_bytes: int,
    direction: TransferDirection,
    pattern: PeerPattern,
) -> list[PatternRankResult]:
    if not 2 <= num_pes <= 8:
        raise ValueError("num_pes must be in [2, 8]")
    if pattern is PeerPattern.PAIR and num_pes % 2:
        raise ValueError("pair requires an even number of PEs")

    import nvshmem.core as nvshmem

    multiprocessing.set_start_method("spawn", force=True)
    uid = nvshmem.get_unique_id()
    queue: multiprocessing.Queue = multiprocessing.Queue()
    processes = [
        multiprocessing.Process(
            target=_rank_probe,
            args=(rank, num_pes, uid, num_epochs, num_slots, payload_bytes, direction, pattern, queue),
        )
        for rank in range(num_pes)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=600)
        if process.is_alive():
            process.terminate()
            raise TimeoutError(f"rank process {process.pid} did not finish")

    rank_results = []
    errors = []
    for _ in range(num_pes):
        try:
            result = queue.get(timeout=5)
        except Empty:
            errors.append("a rank exited without returning a result")
            continue
        if isinstance(result, tuple):
            errors.append(result[1])
        else:
            rank_results.append(result)
    if errors:
        raise RuntimeError("\n".join(errors))

    for result in rank_results:
        if result.validation_errors:
            raise AssertionError(
                f"rank {result.rank} recorded {result.validation_errors} errors; first={result.first_error}"
            )
        for peer in range(num_pes):
            for slot in range(num_slots):
                slot_epoch = num_epochs - ((num_epochs - 1 - slot) % num_slots)
                if slot_epoch < 1:
                    slot_epoch = 0
                index = peer * num_slots + slot
                incoming_peer = peer
                incoming_active = _peer_active(pattern, incoming_peer, result.rank, num_pes)
                outgoing_active = _peer_active(pattern, result.rank, peer, num_pes)
                if result.ready_epochs[index] != (slot_epoch if incoming_active else 0):
                    raise AssertionError(f"rank {result.rank} ready[{peer}, {slot}]={result.ready_epochs[index]}")
                if result.consumed_epochs[index] != (slot_epoch if outgoing_active else 0):
                    raise AssertionError(f"rank {result.rank} consumed[{peer}, {slot}]={result.consumed_epochs[index]}")
    return sorted(rank_results, key=lambda item: item.rank)


def main() -> None:
    num_pes = int(os.environ.get("NVTP_NUM_PES", "8"))
    num_epochs = int(os.environ.get("NVTP_NUM_EPOCHS", "1000"))
    num_slots = int(os.environ.get("NVTP_NUM_SLOTS", "4"))
    payload_bytes = int(os.environ.get("NVTP_PAYLOAD_BYTES", "256"))
    direction = TransferDirection(os.environ.get("NVTP_DIRECTION", TransferDirection.PUSH))
    pattern = PeerPattern(os.environ.get("NVTP_PATTERN", PeerPattern.ALL_TO_ALL))
    results = run_pattern_correctness(num_pes, num_epochs, num_slots, payload_bytes, direction, pattern)
    print(json.dumps([asdict(result) for result in results], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
