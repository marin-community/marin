# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: PLC0415

import ctypes
import os
import statistics
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

PAYLOAD_WORDS = 4


class PullOperation(StrEnum):
    BLOCKING = "blocking"
    NBI_QUIET = "nbi_quiet"
    PEER_LOAD = "peer_load"
    BLOCKING_WARP = "blocking_warp"
    NBI_WARP_QUIET = "nbi_warp_quiet"
    BLOCKING_BLOCK = "blocking_block"
    NBI_BLOCK_QUIET = "nbi_block_quiet"
    PEER_LOAD_WARP = "peer_load_warp"
    NBI_BATCHED_QUIET = "nbi_batched_quiet"


@dataclass(frozen=True)
class PullKernelResult:
    ready_epochs: tuple[int, ...]
    consumed_epochs: tuple[int, ...]
    validation_errors: int
    first_error_epoch: int
    first_error_payload: tuple[int, int, int, int]
    elapsed_seconds: float


def _copy_scalar_to_host(buffer: object, ctype: type[ctypes._SimpleCData]) -> int:
    from cuda.bindings import driver

    value = ctype()
    error = driver.cuMemcpyDtoH(ctypes.addressof(value), int(buffer.handle), ctypes.sizeof(value))[0]
    if int(error) != 0:
        raise RuntimeError(f"cuMemcpyDtoH failed with CUDA error {int(error)}")
    return int(value.value)


def _copy_int32_array_to_host(buffer: object, length: int) -> tuple[int, ...]:
    from cuda.bindings import driver

    values = (ctypes.c_int32 * length)()
    error = driver.cuMemcpyDtoH(ctypes.addressof(values), int(buffer.handle), ctypes.sizeof(values))[0]
    if int(error) != 0:
        raise RuntimeError(f"cuMemcpyDtoH failed with CUDA error {int(error)}")
    return tuple(int(value) for value in values)


def _copy_uint64_array_to_host(buffer: object, length: int) -> tuple[int, ...]:
    from cuda.bindings import driver

    values = (ctypes.c_uint64 * length)()
    error = driver.cuMemcpyDtoH(ctypes.addressof(values), int(buffer.handle), ctypes.sizeof(values))[0]
    if int(error) != 0:
        raise RuntimeError(f"cuMemcpyDtoH failed with CUDA error {int(error)}")
    return tuple(int(value) for value in values)


def _wait_for_start_gate(nvshmem: object) -> None:
    ready_dir = os.environ.get("NVTP_READY_DIR")
    start_file = os.environ.get("NVTP_START_FILE")
    if not ready_dir and not start_file:
        return
    if not ready_dir or not start_file:
        raise ValueError("NVTP_READY_DIR and NVTP_START_FILE must be set together")
    rank = int(nvshmem.my_pe())
    ready_path = Path(ready_dir)
    ready_path.mkdir(parents=True, exist_ok=True)
    (ready_path / f"transport-{rank}").touch()
    while not Path(start_file).exists():
        time.sleep(0.01)


def run_get_probe(
    num_epochs: int,
    num_slots: int,
    operation: PullOperation,
    payload_bytes: int = PAYLOAD_WORDS * 4,
    repetitions: int = 1,
) -> PullKernelResult:
    """Run a one-slot pull ring entirely inside one device kernel."""
    if num_epochs < 1:
        raise ValueError("num_epochs must be positive")
    if num_slots not in (1, 2, 4, 8):
        raise ValueError("num_slots must be one of 1, 2, 4, or 8")
    if payload_bytes < PAYLOAD_WORDS * 4 or payload_bytes % 4:
        raise ValueError("payload_bytes must be a multiple of four and at least 16")
    if operation is PullOperation.NBI_BATCHED_QUIET and payload_bytes % 16:
        raise ValueError("nbi_batched_quiet requires payload_bytes to be a multiple of 16")
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    payload_words = payload_bytes // 4

    import cutlass
    import cutlass.cute as cute
    import nvshmem.core as nvshmem
    import nvshmem.core.device.cute as nvshmem_cute
    import nvshmem.core.device.cute.mem as nvshmem_cute_mem
    import nvshmem.core.interop.cute as cute_interop
    from cuda.core import Device
    from cutlass.cute.arch.nvvm_wrappers import fence_acq_rel_sys, load, store, sync_warp
    from cutlass.cute.typing import Int32
    from nvshmem.bindings.device.cute import quiet

    source = cute_interop.tensor((payload_words * num_slots,), dtype=cute.Int32)
    destination = cute_interop.tensor((payload_words * num_slots,), dtype=cute.Int32)
    ready = cute_interop.tensor((num_slots,), dtype=cute.Uint64)
    consumed = cute_interop.tensor((num_slots,), dtype=cute.Uint64)
    validation = cute_interop.tensor((6,), dtype=cute.Int32)
    tensors = (source, destination, ready, consumed, validation)

    stream = Device().create_stream()
    for tensor in tensors:
        buffer, _, _ = cute_interop.tensor_get_buffer(tensor)
        buffer.fill(0, stream=stream)
    stream.sync()

    if operation is PullOperation.BLOCKING:

        @cute.jit
        def transfer(
            destination: cute.Tensor,
            source: cute.Tensor,
            peer_source: cute.Tensor,
            predecessor: Int32,
            words: Int32,
        ):
            nvshmem_cute.get(destination, source, predecessor)

    elif operation is PullOperation.NBI_QUIET:

        @cute.jit
        def transfer(
            destination: cute.Tensor,
            source: cute.Tensor,
            peer_source: cute.Tensor,
            predecessor: Int32,
            words: Int32,
        ):
            nvshmem_cute.get_nbi(destination, source, predecessor)
            quiet()

    elif operation is PullOperation.PEER_LOAD:

        @cute.jit
        def transfer(
            destination: cute.Tensor,
            source: cute.Tensor,
            peer_source: cute.Tensor,
            predecessor: Int32,
            words: Int32,
        ):
            peer_ptr = cute.make_ptr(
                cute.Int32,
                peer_source.iterator.toint(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            destination_ptr = cute.make_ptr(
                cute.Int32,
                destination.iterator.toint(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            fence_acq_rel_sys()
            for offset in range(words):
                store(
                    (destination_ptr + offset).llvm_ptr,
                    load((peer_ptr + offset).llvm_ptr, cute.Int32, cop="cv"),
                )
            fence_acq_rel_sys()

    elif operation is PullOperation.BLOCKING_WARP:

        @cute.jit
        def transfer(
            destination: cute.Tensor,
            source: cute.Tensor,
            peer_source: cute.Tensor,
            predecessor: Int32,
            words: Int32,
        ):
            nvshmem_cute.get_warp(destination, source, predecessor)

    elif operation is PullOperation.NBI_WARP_QUIET:

        @cute.jit
        def transfer(
            destination: cute.Tensor,
            source: cute.Tensor,
            peer_source: cute.Tensor,
            predecessor: Int32,
            words: Int32,
        ):
            nvshmem_cute.get_nbi_warp(destination, source, predecessor)
            quiet()

    elif operation is PullOperation.BLOCKING_BLOCK:

        @cute.jit
        def transfer(
            destination: cute.Tensor,
            source: cute.Tensor,
            peer_source: cute.Tensor,
            predecessor: Int32,
            words: Int32,
        ):
            nvshmem_cute.get_block(destination, source, predecessor)

    elif operation is PullOperation.NBI_BLOCK_QUIET:

        @cute.jit
        def transfer(
            destination: cute.Tensor,
            source: cute.Tensor,
            peer_source: cute.Tensor,
            predecessor: Int32,
            words: Int32,
        ):
            nvshmem_cute.get_nbi_block(destination, source, predecessor)
            quiet()

    elif operation is PullOperation.PEER_LOAD_WARP:

        @cute.jit
        def transfer(
            destination: cute.Tensor,
            source: cute.Tensor,
            peer_source: cute.Tensor,
            predecessor: Int32,
            words: Int32,
        ):
            thread_index, _, _ = cute.arch.thread_idx()
            lane = thread_index % 32
            peer_ptr = cute.make_ptr(cute.Int32, peer_source.iterator.toint(), cute.AddressSpace.gmem, assumed_align=16)
            destination_ptr = cute.make_ptr(
                cute.Int32, destination.iterator.toint(), cute.AddressSpace.gmem, assumed_align=16
            )
            fence_acq_rel_sys()
            for offset in range(lane, words, 32):
                store(
                    (destination_ptr + offset).llvm_ptr,
                    load((peer_ptr + offset).llvm_ptr, cute.Int32, cop="cv"),
                )
            sync_warp()

    elif operation is PullOperation.NBI_BATCHED_QUIET:

        @cute.jit
        def transfer(
            destination: cute.Tensor,
            source: cute.Tensor,
            peer_source: cute.Tensor,
            predecessor: Int32,
            words: Int32,
        ):
            chunk_words = words // 4
            for chunk in cutlass.range_constexpr(4):
                destination_chunk = cute.make_tensor(
                    destination.iterator + chunk * chunk_words, cute.make_layout(chunk_words)
                )
                source_chunk = cute.make_tensor(source.iterator + chunk * chunk_words, cute.make_layout(chunk_words))
                nvshmem_cute.get_nbi(destination_chunk, source_chunk, predecessor)
            quiet()

    else:
        raise ValueError(f"unknown pull operation {operation}")

    warp_cooperative = operation in (
        PullOperation.BLOCKING_WARP,
        PullOperation.NBI_WARP_QUIET,
        PullOperation.PEER_LOAD_WARP,
    )
    block_cooperative = operation in (PullOperation.BLOCKING_BLOCK, PullOperation.NBI_BLOCK_QUIET)

    @cute.kernel
    def pull_ring_kernel(
        source: cute.Tensor,
        destination: cute.Tensor,
        ready: cute.Tensor,
        consumed: cute.Tensor,
        validation: cute.Tensor,
        epochs: Int32,
        slots: Int32,
        words: Int32,
    ):
        thread_index, _, _ = cute.arch.thread_idx()
        rank = nvshmem.my_pe()
        num_pes = nvshmem.n_pes()
        successor = (rank + 1) % num_pes
        predecessor = (rank + num_pes - 1) % num_pes
        peer_source = nvshmem_cute_mem.get_peer_tensor(source, predecessor)

        if block_cooperative:
            for epoch in range(1, epochs + 1):
                slot = (epoch - 1) % slots
                source_slot = cute.make_tensor(source.iterator + slot * payload_words, cute.make_layout(payload_words))
                peer_source_slot = cute.make_tensor(
                    peer_source.iterator + slot * payload_words, cute.make_layout(payload_words)
                )
                destination_slot = cute.make_tensor(
                    destination.iterator + slot * payload_words, cute.make_layout(payload_words)
                )
                source_ptr = cute.make_ptr(
                    cute.Int32, source_slot.iterator.toint(), cute.AddressSpace.gmem, assumed_align=16
                )
                destination_ptr = cute.make_ptr(
                    cute.Int32, destination_slot.iterator.toint(), cute.AddressSpace.gmem, assumed_align=16
                )
                ready_slot = cute.make_tensor(ready.iterator + slot, cute.make_layout(1))
                consumed_slot = cute.make_tensor(consumed.iterator + slot, cute.make_layout(1))
                if thread_index == 0:
                    if epoch > slots:
                        nvshmem_cute.signal_wait(consumed_slot, nvshmem.ComparisonType.CMP_GE, epoch - slots)
                    store(source_ptr.llvm_ptr, cutlass.Int32(rank), sem="release", scope="sys")
                    store((source_ptr + 1).llvm_ptr, cutlass.Int32(epoch), sem="release", scope="sys")
                    store((source_ptr + 2).llvm_ptr, cutlass.Int32(slot), sem="release", scope="sys")
                    store((source_ptr + 3).llvm_ptr, cutlass.Int32(rank ^ epoch), sem="release", scope="sys")
                    fence_acq_rel_sys()
                    nvshmem_cute.signal_op(ready_slot, epoch, nvshmem.SignalOp.SIGNAL_SET, successor)
                cute.arch.sync_threads()
                if thread_index == 32:
                    nvshmem_cute.signal_wait(ready_slot, nvshmem.ComparisonType.CMP_GE, epoch)
                cute.arch.sync_threads()
                transfer(destination_slot, source_slot, peer_source_slot, predecessor, words)
                cute.arch.sync_threads()
                if thread_index == 32:
                    fence_acq_rel_sys()
                    observed_rank = load(destination_ptr.llvm_ptr, cute.Int32, cop="cv")
                    observed_epoch = load((destination_ptr + 1).llvm_ptr, cute.Int32, cop="cv")
                    observed_slot = load((destination_ptr + 2).llvm_ptr, cute.Int32, cop="cv")
                    observed_checksum = load((destination_ptr + 3).llvm_ptr, cute.Int32, cop="cv")
                    if (
                        observed_rank != predecessor
                        or observed_epoch != epoch
                        or observed_slot != slot
                        or observed_checksum != (predecessor ^ epoch)
                    ):
                        if validation[0] == 0:
                            validation[1] = epoch
                            validation[2] = observed_rank
                            validation[3] = observed_epoch
                            validation[4] = observed_slot
                            validation[5] = observed_checksum
                        validation[0] = validation[0] + 1
                    nvshmem_cute.signal_op(consumed_slot, epoch, nvshmem.SignalOp.SIGNAL_SET, predecessor)
                cute.arch.sync_threads()

        elif thread_index == 0:
            for epoch in range(1, epochs + 1):
                slot = (epoch - 1) % slots
                source_slot = cute.make_tensor(
                    source.iterator + slot * payload_words,
                    cute.make_layout(payload_words),
                )
                source_ptr = cute.make_ptr(
                    cute.Int32,
                    source_slot.iterator.toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                ready_slot = cute.make_tensor(ready.iterator + slot, cute.make_layout(1))
                consumed_slot = cute.make_tensor(consumed.iterator + slot, cute.make_layout(1))
                if epoch > slots:
                    nvshmem_cute.signal_wait(consumed_slot, nvshmem.ComparisonType.CMP_GE, epoch - slots)
                store(source_ptr.llvm_ptr, cutlass.Int32(rank), sem="release", scope="sys")
                store((source_ptr + 1).llvm_ptr, cutlass.Int32(epoch), sem="release", scope="sys")
                store((source_ptr + 2).llvm_ptr, cutlass.Int32(slot), sem="release", scope="sys")
                store((source_ptr + 3).llvm_ptr, cutlass.Int32(rank ^ epoch), sem="release", scope="sys")
                fence_acq_rel_sys()
                nvshmem_cute.signal_op(ready_slot, epoch, nvshmem.SignalOp.SIGNAL_SET, successor)

        consumer_active = (
            False if block_cooperative else (thread_index >= 32 if warp_cooperative else thread_index == 32)
        )

        if consumer_active:
            for epoch in range(1, epochs + 1):
                slot = (epoch - 1) % slots
                source_slot = cute.make_tensor(
                    source.iterator + slot * payload_words,
                    cute.make_layout(payload_words),
                )
                peer_source_slot = cute.make_tensor(
                    peer_source.iterator + slot * payload_words,
                    cute.make_layout(payload_words),
                )
                destination_slot = cute.make_tensor(
                    destination.iterator + slot * payload_words,
                    cute.make_layout(payload_words),
                )
                destination_ptr = cute.make_ptr(
                    cute.Int32,
                    destination_slot.iterator.toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                ready_slot = cute.make_tensor(ready.iterator + slot, cute.make_layout(1))
                consumed_slot = cute.make_tensor(consumed.iterator + slot, cute.make_layout(1))
                if thread_index == 32:
                    nvshmem_cute.signal_wait(ready_slot, nvshmem.ComparisonType.CMP_GE, epoch)
                if warp_cooperative:
                    sync_warp()
                transfer(destination_slot, source_slot, peer_source_slot, predecessor, words)
                if warp_cooperative:
                    sync_warp()
                if thread_index == 32:
                    fence_acq_rel_sys()
                    observed_rank = load(destination_ptr.llvm_ptr, cute.Int32, cop="cv")
                    observed_epoch = load((destination_ptr + 1).llvm_ptr, cute.Int32, cop="cv")
                    observed_slot = load((destination_ptr + 2).llvm_ptr, cute.Int32, cop="cv")
                    observed_checksum = load((destination_ptr + 3).llvm_ptr, cute.Int32, cop="cv")
                    if (
                        observed_rank != predecessor
                        or observed_epoch != epoch
                        or observed_slot != slot
                        or observed_checksum != (predecessor ^ epoch)
                    ):
                        if validation[0] == 0:
                            validation[1] = epoch
                            validation[2] = observed_rank
                            validation[3] = observed_epoch
                            validation[4] = observed_slot
                            validation[5] = observed_checksum
                        validation[0] = validation[0] + 1
                    nvshmem_cute.signal_op(consumed_slot, epoch, nvshmem.SignalOp.SIGNAL_SET, predecessor)

    @cute.jit
    def pull_ring_launcher(
        source: cute.Tensor,
        destination: cute.Tensor,
        ready: cute.Tensor,
        consumed: cute.Tensor,
        validation: cute.Tensor,
        epochs: Int32,
        slots: Int32,
        words: Int32,
    ):
        pull_ring_kernel(source, destination, ready, consumed, validation, epochs, slots, words).launch(
            grid=[1, 1, 1],
            block=[64, 1, 1],
        )

    bitcode = nvshmem.find_device_bitcode_library()
    compiled = cute.compile(
        pull_ring_launcher,
        source,
        destination,
        ready,
        consumed,
        validation,
        1,
        num_slots,
        payload_words,
        options=f" --link-libraries={bitcode}",
    )
    compiled = compiled.to(Device().device_id)
    cuda_library = compiled.jit_module.cuda_library
    kernel_object = nvshmem.NvshmemKernelObject.from_handle(int(cuda_library[0]))
    nvshmem.library_init(kernel_object)
    _wait_for_start_gate(nvshmem)

    elapsed_times = []
    for _ in range(repetitions):
        nvshmem.barrier_all(stream)
        stream.sync()
        for tensor in (ready, consumed, validation):
            buffer, _, _ = cute_interop.tensor_get_buffer(tensor)
            buffer.fill(0, stream=stream)
        stream.sync()
        nvshmem.barrier_all(stream)
        stream.sync()
        start = time.perf_counter()
        compiled(source, destination, ready, consumed, validation, num_epochs, num_slots, payload_words)
        Device().sync()
        nvshmem.barrier_all(stream)
        stream.sync()
        elapsed_times.append(time.perf_counter() - start)
    elapsed_seconds = statistics.median(elapsed_times)

    ready_buffer, _, _ = cute_interop.tensor_get_buffer(ready)
    consumed_buffer, _, _ = cute_interop.tensor_get_buffer(consumed)
    validation_buffer, _, _ = cute_interop.tensor_get_buffer(validation)
    validation_values = _copy_int32_array_to_host(validation_buffer, 6)
    result = PullKernelResult(
        ready_epochs=_copy_uint64_array_to_host(ready_buffer, num_slots),
        consumed_epochs=_copy_uint64_array_to_host(consumed_buffer, num_slots),
        validation_errors=validation_values[0],
        first_error_epoch=validation_values[1],
        first_error_payload=validation_values[2:6],
        elapsed_seconds=elapsed_seconds,
    )

    nvshmem.library_finalize(kernel_object)
    for tensor in tensors:
        cute_interop.free_tensor(tensor)
    return result
