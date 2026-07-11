# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: PLC0415

import json
import multiprocessing
import os
import traceback
from dataclasses import asdict, dataclass
from queue import Empty

from cute_nvshmem_transport.push_kernels import PushOperation, run_push_probe


@dataclass(frozen=True)
class PushRankResult:
    rank: int
    ready_epochs: tuple[int, ...]
    consumed_epochs: tuple[int, ...]
    validation_errors: int
    first_error_epoch: int
    first_error_payload: tuple[int, int, int, int]


def _rank_probe(
    rank: int,
    nranks: int,
    uid: object,
    num_epochs: int,
    num_slots: int,
    operation: PushOperation,
    results: multiprocessing.Queue,
) -> None:
    try:
        import nvshmem.core as nvshmem
        from cuda.core import Device

        device = Device(rank)
        device.set_current()
        nvshmem.init(device=device, uid=uid, rank=rank, nranks=nranks, initializer_method="uid")
        result = run_push_probe(num_epochs, num_slots, operation)
        nvshmem.finalize()
        results.put(PushRankResult(rank=rank, **asdict(result)))
    except BaseException:
        results.put((rank, traceback.format_exc()))


def run_push_correctness(
    num_pes: int,
    num_epochs: int,
    num_slots: int,
    operation: PushOperation,
) -> list[PushRankResult]:
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
            args=(rank, num_pes, uid, num_epochs, num_slots, operation, results),
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

    expected_epochs = tuple(
        num_epochs - ((num_epochs - slot - 1) % num_slots) if num_epochs > slot else 0 for slot in range(num_slots)
    )
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
    operation = PushOperation(os.environ.get("NVTP_PUSH_OPERATION", PushOperation.PUT_SIGNAL))
    results = run_push_correctness(num_pes, num_epochs, num_slots, operation)
    print(json.dumps([asdict(result) for result in results], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
