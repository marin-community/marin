# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: PLC0415

import ctypes
import json
import multiprocessing
import os
import traceback
from dataclasses import asdict, dataclass
from queue import Empty

from cute_nvshmem_transport.peer_tensor_kernels import run_peer_tensor_store_probe

PROBE_BYTES = 4096


@dataclass(frozen=True)
class RankResult:
    rank: int
    peer: int
    peer_value: int
    peer_buffer_available: bool
    peer_tensor_value: int


def _copy_first_byte_to_host(buffer: object) -> int:
    from cuda.bindings import driver

    host_value = ctypes.c_ubyte()
    error = driver.cuMemcpyDtoH(ctypes.addressof(host_value), int(buffer.handle), 1)[0]
    if int(error) != 0:
        raise RuntimeError(f"cuMemcpyDtoH failed with CUDA error {int(error)}")
    return host_value.value


def _rank_probe(rank: int, nranks: int, uid: object, results: multiprocessing.Queue) -> None:
    try:
        import nvshmem.core as nvshmem
        from cuda.core import Device

        device = Device(rank)
        device.set_current()
        stream = device.create_stream()
        nvshmem.init(device=device, uid=uid, rank=rank, nranks=nranks, initializer_method="uid")

        arena = nvshmem.buffer(PROBE_BYTES)
        scratch = nvshmem.buffer(PROBE_BYTES)
        peer = (rank + 1) % nranks
        arena.fill(rank + 1, stream=stream)
        stream.sync()
        peer_tensor_value = run_peer_tensor_store_probe()
        expected_tensor_value = ((rank - 1) % nranks) + 1
        if peer_tensor_value != expected_tensor_value:
            raise AssertionError(
                f"rank {rank} received {peer_tensor_value} through peer tensor; expected {expected_tensor_value}"
            )

        nvshmem.barrier_all(stream)
        stream.sync()

        peer_buffer = nvshmem.get_peer_buffer(arena, peer)
        peer_buffer.copy_to(scratch, stream=stream)
        stream.sync()
        peer_value = _copy_first_byte_to_host(scratch)
        expected = peer + 1
        if peer_value != expected:
            raise AssertionError(f"rank {rank} read {peer_value} from peer {peer}; expected {expected}")

        nvshmem.barrier_all(stream)
        stream.sync()
        nvshmem.free(scratch)
        nvshmem.free(arena)
        nvshmem.finalize()
        results.put(RankResult(rank, peer, peer_value, True, peer_tensor_value))
    except BaseException:
        results.put((rank, traceback.format_exc()))


def run_peer_buffer_probe(nranks: int) -> list[RankResult]:
    if nranks < 2:
        raise ValueError("nranks must be at least two")
    if nranks > 8:
        raise ValueError("the CoreWeave H100 probe supports at most eight local ranks")

    import nvshmem.core as nvshmem

    multiprocessing.set_start_method("spawn", force=True)
    uid = nvshmem.get_unique_id()
    results: multiprocessing.Queue = multiprocessing.Queue()
    processes = [
        multiprocessing.Process(target=_rank_probe, args=(rank, nranks, uid, results)) for rank in range(nranks)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=120)
        if process.is_alive():
            process.terminate()
            raise TimeoutError(f"rank process {process.pid} did not finish")

    rank_results = []
    errors = []
    for _ in range(nranks):
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
    return sorted(rank_results, key=lambda result: result.rank)


def main() -> None:
    nranks = int(os.environ.get("NVTP_NUM_PES", "2"))
    results = run_peer_buffer_probe(nranks)
    print(json.dumps([asdict(result) for result in results], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
