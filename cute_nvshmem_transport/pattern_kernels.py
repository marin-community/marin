# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: PLC0415

import ctypes
from dataclasses import dataclass
from enum import StrEnum

HEADER_WORDS = 5


class TransferDirection(StrEnum):
    PUSH = "push"
    PULL = "pull"


class PeerPattern(StrEnum):
    PAIR = "pair"
    RING = "ring"
    ALL_TO_ALL = "all_to_all"


@dataclass(frozen=True)
class PatternKernelResult:
    ready_epochs: tuple[int, ...]
    consumed_epochs: tuple[int, ...]
    validation_errors: int
    first_error: tuple[int, int, int, int, int, int]


def _copy_int32_array(buffer: object, length: int) -> tuple[int, ...]:
    from cuda.bindings import driver

    values = (ctypes.c_int32 * length)()
    error = driver.cuMemcpyDtoH(ctypes.addressof(values), int(buffer.handle), ctypes.sizeof(values))[0]
    if int(error) != 0:
        raise RuntimeError(f"cuMemcpyDtoH failed with CUDA error {int(error)}")
    return tuple(int(value) for value in values)


def _copy_uint64_array(buffer: object, length: int) -> tuple[int, ...]:
    from cuda.bindings import driver

    values = (ctypes.c_uint64 * length)()
    error = driver.cuMemcpyDtoH(ctypes.addressof(values), int(buffer.handle), ctypes.sizeof(values))[0]
    if int(error) != 0:
        raise RuntimeError(f"cuMemcpyDtoH failed with CUDA error {int(error)}")
    return tuple(int(value) for value in values)


def run_pattern_probe(
    num_epochs: int,
    num_slots: int,
    payload_bytes: int,
    direction: TransferDirection,
    pattern: PeerPattern,
) -> PatternKernelResult:
    """Exercise a fully device-driven multi-peer transport pattern."""
    if num_epochs < 1:
        raise ValueError("num_epochs must be positive")
    if num_slots not in (1, 2, 4, 8):
        raise ValueError("num_slots must be one of 1, 2, 4, or 8")
    if payload_bytes < HEADER_WORDS * 4 or payload_bytes % 4:
        raise ValueError(f"payload_bytes must be a multiple of four and at least {HEADER_WORDS * 4}")

    import cutlass
    import cutlass.cute as cute
    import nvshmem.core as nvshmem
    import nvshmem.core.device.cute as nvshmem_cute
    import nvshmem.core.interop.cute as cute_interop
    from cuda.core import Device
    from cutlass.cute.arch.nvvm_wrappers import fence_acq_rel_sys, load
    from cutlass.cute.typing import Int32

    num_pes = int(nvshmem.n_pes())
    if pattern is PeerPattern.PAIR and num_pes % 2:
        raise ValueError("pair requires an even number of PEs")
    payload_words = payload_bytes // 4
    peer_slots = num_pes * num_slots
    payload = cute_interop.tensor((peer_slots * payload_words,), dtype=cute.Int32)
    scratch = cute_interop.tensor((peer_slots * payload_words,), dtype=cute.Int32)
    ready = cute_interop.tensor((peer_slots,), dtype=cute.Uint64)
    consumed = cute_interop.tensor((peer_slots,), dtype=cute.Uint64)
    validation = cute_interop.tensor((7,), dtype=cute.Int32)
    tensors = (payload, scratch, ready, consumed, validation)
    stream = Device().create_stream()
    for tensor in tensors:
        buffer, _, _ = cute_interop.tensor_get_buffer(tensor)
        buffer.fill(0, stream=stream)
    stream.sync()

    pattern_id = list(PeerPattern).index(pattern)
    direction_id = list(TransferDirection).index(direction)

    @cute.kernel
    def pattern_kernel(
        payload: cute.Tensor,
        scratch: cute.Tensor,
        ready: cute.Tensor,
        consumed: cute.Tensor,
        validation: cute.Tensor,
        epochs: Int32,
        slots: Int32,
        words: Int32,
        selected_pattern: Int32,
        selected_direction: Int32,
    ):
        thread_index, _, _ = cute.arch.thread_idx()
        my_rank = nvshmem.my_pe()
        ranks = nvshmem.n_pes()

        if thread_index == 0:
            for epoch in range(1, epochs + 1):
                slot = (epoch - 1) % slots
                for peer in range(ranks):
                    active = (
                        (selected_pattern == 2 and peer != my_rank)
                        or (selected_pattern == 1 and peer == (my_rank + 1) % ranks)
                        or (selected_pattern == 0 and peer == (my_rank ^ 1))
                    )
                    if active:
                        local_signal_index = peer * slots + slot
                        remote_signal_index = my_rank * slots + slot
                        local_payload_index = (peer * slots + slot) * words
                        remote_payload_index = (my_rank * slots + slot) * words
                        local_payload = cute.make_tensor(payload.iterator + local_payload_index, cute.make_layout(words))
                        remote_payload = cute.make_tensor(
                            scratch.iterator + remote_payload_index, cute.make_layout(words)
                        )
                        remote_ready = cute.make_tensor(ready.iterator + remote_signal_index, cute.make_layout(1))
                        local_consumed = cute.make_tensor(consumed.iterator + local_signal_index, cute.make_layout(1))
                        if epoch > slots:
                            nvshmem_cute.signal_wait(local_consumed, nvshmem.ComparisonType.CMP_GE, epoch - slots)
                        local_payload[0] = my_rank
                        local_payload[1] = peer
                        local_payload[2] = slot
                        local_payload[3] = epoch
                        for offset in range(4, words):
                            local_payload[offset] = my_rank ^ peer ^ slot ^ epoch ^ offset
                        fence_acq_rel_sys()
                        if selected_direction == 0:
                            nvshmem_cute.put_signal(
                                remote_payload,
                                local_payload,
                                remote_ready,
                                epoch,
                                nvshmem.SignalOp.SIGNAL_SET,
                                peer,
                            )
                        else:
                            nvshmem_cute.signal_op(remote_ready, epoch, nvshmem.SignalOp.SIGNAL_SET, peer)

        if thread_index == 32:
            for epoch in range(1, epochs + 1):
                slot = (epoch - 1) % slots
                for peer in range(ranks):
                    active = (
                        (selected_pattern == 2 and peer != my_rank)
                        or (selected_pattern == 1 and peer == (my_rank + ranks - 1) % ranks)
                        or (selected_pattern == 0 and peer == (my_rank ^ 1))
                    )
                    if active:
                        signal_index = peer * slots + slot
                        payload_index = (peer * slots + slot) * words
                        ready_slot = cute.make_tensor(ready.iterator + signal_index, cute.make_layout(1))
                        consumed_slot = cute.make_tensor(consumed.iterator + my_rank * slots + slot, cute.make_layout(1))
                        received = cute.make_tensor(scratch.iterator + payload_index, cute.make_layout(words))
                        nvshmem_cute.signal_wait(ready_slot, nvshmem.ComparisonType.CMP_GE, epoch)
                        if selected_direction == 1:
                            peer_source = cute.make_tensor(
                                payload.iterator + (my_rank * slots + slot) * words,
                                cute.make_layout(words),
                            )
                            nvshmem_cute.get(received, peer_source, peer)
                        fence_acq_rel_sys()
                        received_ptr = cute.make_ptr(
                            cute.Int32, received.iterator.toint(), cute.AddressSpace.gmem, assumed_align=4
                        )
                        for offset in range(words):
                            observed = load((received_ptr + offset).llvm_ptr, cute.Int32, cop="cv")
                            expected = cutlass.Int32(peer ^ my_rank ^ slot ^ epoch ^ offset)
                            if offset == 0:
                                expected = cutlass.Int32(peer)
                            elif offset == 1:
                                expected = cutlass.Int32(my_rank)
                            elif offset == 2:
                                expected = cutlass.Int32(slot)
                            elif offset == 3:
                                expected = cutlass.Int32(epoch)
                            if observed != expected:
                                if validation[0] == 0:
                                    validation[1] = peer
                                    validation[2] = my_rank
                                    validation[3] = slot
                                    validation[4] = epoch
                                    validation[5] = offset
                                    validation[6] = observed
                                validation[0] = validation[0] + 1
                        nvshmem_cute.signal_op(consumed_slot, epoch, nvshmem.SignalOp.SIGNAL_SET, peer)

    @cute.jit
    def launcher(
        payload: cute.Tensor,
        scratch: cute.Tensor,
        ready: cute.Tensor,
        consumed: cute.Tensor,
        validation: cute.Tensor,
        epochs: Int32,
        slots: Int32,
        words: Int32,
        selected_pattern: Int32,
        selected_direction: Int32,
    ):
        pattern_kernel(
            payload,
            scratch,
            ready,
            consumed,
            validation,
            epochs,
            slots,
            words,
            selected_pattern,
            selected_direction,
        ).launch(grid=[1, 1, 1], block=[64, 1, 1])

    bitcode = nvshmem.find_device_bitcode_library()
    compiled = cute.compile(
        launcher,
        payload,
        scratch,
        ready,
        consumed,
        validation,
        1,
        num_slots,
        payload_words,
        pattern_id,
        direction_id,
        options=f" --link-libraries={bitcode}",
    ).to(Device().device_id)
    cuda_library = compiled.jit_module.cuda_library
    kernel_object = nvshmem.NvshmemKernelObject.from_handle(int(cuda_library[0]))
    nvshmem.library_init(kernel_object)
    nvshmem.barrier_all(stream)
    stream.sync()
    compiled(
        payload,
        scratch,
        ready,
        consumed,
        validation,
        num_epochs,
        num_slots,
        payload_words,
        pattern_id,
        direction_id,
    )
    Device().sync()
    nvshmem.barrier_all(stream)
    stream.sync()

    ready_buffer, _, _ = cute_interop.tensor_get_buffer(ready)
    consumed_buffer, _, _ = cute_interop.tensor_get_buffer(consumed)
    validation_buffer, _, _ = cute_interop.tensor_get_buffer(validation)
    validation_values = _copy_int32_array(validation_buffer, 7)
    result = PatternKernelResult(
        ready_epochs=_copy_uint64_array(ready_buffer, peer_slots),
        consumed_epochs=_copy_uint64_array(consumed_buffer, peer_slots),
        validation_errors=validation_values[0],
        first_error=validation_values[1:7],
    )
    nvshmem.library_finalize(kernel_object)
    for tensor in tensors:
        cute_interop.free_tensor(tensor)
    return result
