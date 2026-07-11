# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: PLC0415

import ctypes
from dataclasses import dataclass
from enum import StrEnum

PAYLOAD_WORDS = 4


class PushOperation(StrEnum):
    PUT_SIGNAL = "put_signal"
    PUT_SIGNAL_NBI_QUIET = "put_signal_nbi_quiet"
    PUT_NBI_QUIET_SIGNAL = "put_nbi_quiet_signal"
    PEER_STORE_SIGNAL = "peer_store_signal"


@dataclass(frozen=True)
class PushKernelResult:
    ready_epochs: tuple[int, ...]
    consumed_epochs: tuple[int, ...]
    validation_errors: int
    first_error_epoch: int
    first_error_payload: tuple[int, int, int, int]


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


def run_push_probe(num_epochs: int, num_slots: int, operation: PushOperation) -> PushKernelResult:
    """Run a one-slot push ring entirely inside one device kernel."""
    if num_epochs < 1:
        raise ValueError("num_epochs must be positive")
    if num_slots not in (1, 2, 4, 8):
        raise ValueError("num_slots must be one of 1, 2, 4, or 8")

    import cutlass.cute as cute
    import nvshmem.core as nvshmem
    import nvshmem.core.device.cute as nvshmem_cute
    import nvshmem.core.device.cute.mem as nvshmem_cute_mem
    import nvshmem.core.interop.cute as cute_interop
    from cuda.core import Device
    from cutlass.cute.arch.nvvm_wrappers import fence_acq_rel_sys, load, store
    from cutlass.cute.typing import Int32
    from nvshmem.bindings.device.cute import quiet

    source = cute_interop.tensor((PAYLOAD_WORDS * num_slots,), dtype=cute.Int32)
    inbox = cute_interop.tensor((PAYLOAD_WORDS * num_slots,), dtype=cute.Int32)
    ready = cute_interop.tensor((num_slots,), dtype=cute.Uint64)
    consumed = cute_interop.tensor((num_slots,), dtype=cute.Uint64)
    validation = cute_interop.tensor((6,), dtype=cute.Int32)
    tensors = (source, inbox, ready, consumed, validation)

    stream = Device().create_stream()
    for tensor in tensors:
        buffer, _, _ = cute_interop.tensor_get_buffer(tensor)
        buffer.fill(0, stream=stream)
    stream.sync()

    if operation is PushOperation.PUT_SIGNAL:

        @cute.jit
        def publish(
            inbox: cute.Tensor,
            peer_inbox: cute.Tensor,
            source: cute.Tensor,
            ready: cute.Tensor,
            epoch: Int32,
            successor: Int32,
        ):
            nvshmem_cute.put_signal(
                inbox,
                source,
                ready,
                epoch,
                nvshmem.SignalOp.SIGNAL_SET,
                successor,
            )

    elif operation is PushOperation.PUT_SIGNAL_NBI_QUIET:

        @cute.jit
        def publish(
            inbox: cute.Tensor,
            peer_inbox: cute.Tensor,
            source: cute.Tensor,
            ready: cute.Tensor,
            epoch: Int32,
            successor: Int32,
        ):
            nvshmem_cute.put_signal_nbi(
                inbox,
                source,
                ready,
                epoch,
                nvshmem.SignalOp.SIGNAL_SET,
                successor,
            )
            quiet()

    elif operation is PushOperation.PUT_NBI_QUIET_SIGNAL:

        @cute.jit
        def publish(
            inbox: cute.Tensor,
            peer_inbox: cute.Tensor,
            source: cute.Tensor,
            ready: cute.Tensor,
            epoch: Int32,
            successor: Int32,
        ):
            nvshmem_cute.put_nbi(inbox, source, successor)
            quiet()
            nvshmem_cute.signal_op(ready, epoch, nvshmem.SignalOp.SIGNAL_SET, successor)

    elif operation is PushOperation.PEER_STORE_SIGNAL:

        @cute.jit
        def publish(
            inbox: cute.Tensor,
            peer_inbox: cute.Tensor,
            source: cute.Tensor,
            ready: cute.Tensor,
            epoch: Int32,
            successor: Int32,
        ):
            peer_ptr = cute.make_ptr(
                cute.Int32,
                peer_inbox.iterator.toint(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            store(peer_ptr.llvm_ptr, source[0], sem="release", scope="sys")
            store((peer_ptr + 1).llvm_ptr, source[1], sem="release", scope="sys")
            store((peer_ptr + 2).llvm_ptr, source[2], sem="release", scope="sys")
            store((peer_ptr + 3).llvm_ptr, source[3], sem="release", scope="sys")
            fence_acq_rel_sys()
            nvshmem_cute.signal_op(ready, epoch, nvshmem.SignalOp.SIGNAL_SET, successor)

    else:
        raise ValueError(f"unknown push operation {operation}")

    @cute.kernel
    def push_ring_kernel(
        source: cute.Tensor,
        inbox: cute.Tensor,
        ready: cute.Tensor,
        consumed: cute.Tensor,
        validation: cute.Tensor,
        epochs: Int32,
        slots: Int32,
    ):
        thread_index, _, _ = cute.arch.thread_idx()
        rank = nvshmem.my_pe()
        num_pes = nvshmem.n_pes()
        successor = (rank + 1) % num_pes
        predecessor = (rank + num_pes - 1) % num_pes
        peer_inbox = nvshmem_cute_mem.get_peer_tensor(inbox, successor)

        if thread_index == 0:
            for epoch in range(1, epochs + 1):
                slot = (epoch - 1) % slots
                source_slot = cute.make_tensor(
                    source.iterator + slot * PAYLOAD_WORDS,
                    cute.make_layout(PAYLOAD_WORDS),
                )
                inbox_slot = cute.make_tensor(
                    inbox.iterator + slot * PAYLOAD_WORDS,
                    cute.make_layout(PAYLOAD_WORDS),
                )
                peer_inbox_slot = cute.make_tensor(
                    peer_inbox.iterator + slot * PAYLOAD_WORDS,
                    cute.make_layout(PAYLOAD_WORDS),
                )
                ready_slot = cute.make_tensor(ready.iterator + slot, cute.make_layout(1))
                consumed_slot = cute.make_tensor(consumed.iterator + slot, cute.make_layout(1))
                if epoch > slots:
                    nvshmem_cute.signal_wait(consumed_slot, nvshmem.ComparisonType.CMP_GE, epoch - slots)
                source_slot[0] = rank
                source_slot[1] = epoch
                source_slot[2] = slot
                source_slot[3] = rank ^ epoch
                publish(inbox_slot, peer_inbox_slot, source_slot, ready_slot, epoch, successor)

        if thread_index == 32:
            for epoch in range(1, epochs + 1):
                slot = (epoch - 1) % slots
                inbox_slot = cute.make_tensor(
                    inbox.iterator + slot * PAYLOAD_WORDS,
                    cute.make_layout(PAYLOAD_WORDS),
                )
                ready_slot = cute.make_tensor(ready.iterator + slot, cute.make_layout(1))
                consumed_slot = cute.make_tensor(consumed.iterator + slot, cute.make_layout(1))
                inbox_ptr = cute.make_ptr(
                    cute.Int32,
                    inbox_slot.iterator.toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                nvshmem_cute.signal_wait(ready_slot, nvshmem.ComparisonType.CMP_GE, epoch)
                fence_acq_rel_sys()
                observed_rank = load(inbox_ptr.llvm_ptr, cute.Int32, cop="cv")
                observed_epoch = load((inbox_ptr + 1).llvm_ptr, cute.Int32, cop="cv")
                observed_slot = load((inbox_ptr + 2).llvm_ptr, cute.Int32, cop="cv")
                observed_checksum = load((inbox_ptr + 3).llvm_ptr, cute.Int32, cop="cv")
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
    def push_ring_launcher(
        source: cute.Tensor,
        inbox: cute.Tensor,
        ready: cute.Tensor,
        consumed: cute.Tensor,
        validation: cute.Tensor,
        epochs: Int32,
        slots: Int32,
    ):
        push_ring_kernel(source, inbox, ready, consumed, validation, epochs, slots).launch(
            grid=[1, 1, 1],
            block=[64, 1, 1],
        )

    bitcode = nvshmem.find_device_bitcode_library()
    compiled = cute.compile(
        push_ring_launcher,
        source,
        inbox,
        ready,
        consumed,
        validation,
        1,
        num_slots,
        options=f" --link-libraries={bitcode}",
    )
    compiled = compiled.to(Device().device_id)
    cuda_library = compiled.jit_module.cuda_library
    kernel_object = nvshmem.NvshmemKernelObject.from_handle(int(cuda_library[0]))
    nvshmem.library_init(kernel_object)

    compiled(source, inbox, ready, consumed, validation, num_epochs, num_slots)
    Device().sync()

    ready_buffer, _, _ = cute_interop.tensor_get_buffer(ready)
    consumed_buffer, _, _ = cute_interop.tensor_get_buffer(consumed)
    validation_buffer, _, _ = cute_interop.tensor_get_buffer(validation)
    validation_values = _copy_int32_array_to_host(validation_buffer, 6)
    result = PushKernelResult(
        ready_epochs=_copy_uint64_array_to_host(ready_buffer, num_slots),
        consumed_epochs=_copy_uint64_array_to_host(consumed_buffer, num_slots),
        validation_errors=validation_values[0],
        first_error_epoch=validation_values[1],
        first_error_payload=validation_values[2:6],
    )

    nvshmem.library_finalize(kernel_object)
    for tensor in tensors:
        cute_interop.free_tensor(tensor)
    return result
