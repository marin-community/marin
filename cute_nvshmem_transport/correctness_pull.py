# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: PLC0415

import json
import multiprocessing
import os
import traceback
from dataclasses import asdict, dataclass
from queue import Empty

from cute_nvshmem_transport.pull_kernels import PullOperation, run_get_probe
from cute_nvshmem_transport.signals import final_slot_epochs


@dataclass(frozen=True)
class PullRankResult:
    rank: int
    ready_epochs: tuple[int, ...]
    consumed_epochs: tuple[int, ...]
    validation_errors: int
    first_error_epoch: int
    first_error_payload: tuple[int, int, int, int]
    elapsed_seconds: float


def _rank_probe(
    rank: int,
    nranks: int,
    uid: object,
    num_epochs: int,
    num_slots: int,
    payload_bytes: int,
    repetitions: int,
    operation: PullOperation,
    results: multiprocessing.Queue,
) -> None:
    try:
        import nvshmem.core as nvshmem
        from cuda.core import Device

        device = Device(rank)
        device.set_current()
        nvshmem.init(device=device, uid=uid, rank=rank, nranks=nranks, initializer_method="uid")
        result = run_get_probe(num_epochs, num_slots, operation, payload_bytes, repetitions)
        nvshmem.finalize()
        results.put(PullRankResult(rank=rank, **asdict(result)))
    except BaseException:
        results.put((rank, traceback.format_exc()))


def run_pull_correctness(
    num_pes: int,
    num_epochs: int,
    num_slots: int,
    operation: PullOperation,
    payload_bytes: int = 16,
    repetitions: int = 1,
) -> list[PullRankResult]:
    if not 2 <= num_pes <= 8:
        raise ValueError("num_pes must be in [2, 8]")
    if num_epochs < 1:
        raise ValueError("num_epochs must be positive")

    import nvshmem.core as nvshmem

    multiprocessing.set_start_method("spawn", force=True)
    uid = nvshmem.get_unique_id()
    results: multiprocessing.Queue = multiprocessing.Queue()
    processes = [
        multiprocessing.Process(
            target=_rank_probe,
            args=(rank, num_pes, uid, num_epochs, num_slots, payload_bytes, repetitions, operation, results),
        )
        for rank in range(num_pes)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=300)
        if process.is_alive():
            process.terminate()
            raise TimeoutError(f"rank process {process.pid} did not finish")

    rank_results = []
    errors = []
    for _ in range(num_pes):
        try:
            result = results.get(timeout=5)
        except Empty:
            errors.append("a rank exited without returning a result")
            continue
        if isinstance(result, tuple):
            errors.append(result[1])
        else:
            rank_results.append(result)
    if errors:
        raise RuntimeError("\n".join(errors))

    expected_epochs = final_slot_epochs(num_epochs, num_slots)
    for result in rank_results:
        if result.ready_epochs != expected_epochs:
            raise AssertionError(
                f"rank {result.rank} ready epochs are {result.ready_epochs}, expected {expected_epochs}"
            )
        if result.consumed_epochs != expected_epochs:
            raise AssertionError(
                f"rank {result.rank} consumed epochs are {result.consumed_epochs}, expected {expected_epochs}"
            )
        if result.validation_errors:
            raise AssertionError(
                f"rank {result.rank} recorded {result.validation_errors} payload errors; "
                f"first error at epoch {result.first_error_epoch} with payload {result.first_error_payload}"
            )
    return sorted(rank_results, key=lambda result: result.rank)


def main() -> None:
    num_pes = int(os.environ.get("NVTP_NUM_PES", "2"))
    num_epochs = int(os.environ.get("NVTP_NUM_EPOCHS", "1000"))
    num_slots = int(os.environ.get("NVTP_NUM_SLOTS", "1"))
    payload_bytes = int(os.environ.get("NVTP_PAYLOAD_BYTES", "16"))
    repetitions = int(os.environ.get("NVTP_REPETITIONS", "1"))
    operation = PullOperation(os.environ.get("NVTP_PULL_OPERATION", PullOperation.BLOCKING))
    results = run_pull_correctness(num_pes, num_epochs, num_slots, operation, payload_bytes, repetitions)
    print(json.dumps([asdict(result) for result in results], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
