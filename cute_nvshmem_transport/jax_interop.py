# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: PLC0415

import json
import multiprocessing
import os
import traceback
from dataclasses import asdict, dataclass
from queue import Empty


@dataclass(frozen=True)
class JaxInteropReport:
    nvshmem_pointer: int
    jax_pointer: int
    pointer_identity: bool
    initial_values_match: bool
    external_update_visible_to_cached_numpy: bool
    external_update_visible_to_jax_kernel: bool
    jax_output_pointer_distinct: bool
    host_put_direct_jax: str
    host_put_wrapped_jax: str
    jax_to_cuda_stream_handoff: str
    wrapped_put_values_match: bool


@dataclass(frozen=True)
class RemoteJaxPushReport:
    sender_pointer: int
    receiver_nvshmem_pointer: int
    receiver_jax_pointer: int
    receiver_pointer_identity: bool
    receiver_sum: int
    values_match: bool


@dataclass(frozen=True)
class StreamJaxPushReport:
    sender_input_pointer: int
    sender_output_pointer: int
    receiver_nvshmem_pointer: int
    receiver_input_pointer: int
    receiver_output_pointer: int
    sender_alias_identity: bool
    receiver_alias_identity: bool
    receiver_sum: int
    values_match: bool
    steady_state_host_synchronizations: int


def run_local_jax_interop_probe(size: int = 16) -> JaxInteropReport:
    """Probe local DLPack aliasing and host-RMA acceptance for JAX arrays."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    import nvshmem.core as nvshmem
    from cuda.core import Buffer, Device

    device = Device()
    stream = device.create_stream()
    symmetric_buffer = nvshmem.buffer(size)
    symmetric_buffer.fill(7, stream=stream)
    stream.sync()

    jax_view = jax.dlpack.from_dlpack(symmetric_buffer)
    jax_view.block_until_ready()
    cached_numpy = np.asarray(jax_view)
    initial_values_match = bool(np.all(cached_numpy == 7))
    nvshmem_pointer = int(symmetric_buffer.handle)
    jax_pointer = int(jax_view.unsafe_buffer_pointer())

    symmetric_buffer.fill(9, stream=stream)
    stream.sync()
    external_update_visible_to_cached_numpy = bool(np.all(cached_numpy == 9))
    updated_sum = int(jnp.sum(jax_view).block_until_ready())
    external_update_visible_to_jax_kernel = updated_sum == 9 * size
    jax_output = jax_view + jnp.uint8(1)
    jax_output.block_until_ready()
    jax_output_pointer_distinct = int(jax_output.unsafe_buffer_pointer()) != jax_pointer

    jax_source = jnp.arange(size, dtype=jnp.uint8)
    try:
        nvshmem.put(symmetric_buffer, jax_source, remote_pe=nvshmem.my_pe(), stream=stream)
        stream.sync()
        host_put_direct_jax = "accepted"
    except Exception as error:
        host_put_direct_jax = f"rejected: {type(error).__name__}: {error}"

    wrapped_source = Buffer.from_handle(
        int(jax_source.unsafe_buffer_pointer()),
        jax_source.size * jax_source.dtype.itemsize,
        owner=jax_source,
    )
    try:
        jax_source.__dlpack__(stream=int(stream.handle))
        jax_to_cuda_stream_handoff = "accepted"
    except Exception as error:
        jax_to_cuda_stream_handoff = f"rejected: {type(error).__name__}: {error}"
    try:
        nvshmem.put(symmetric_buffer, wrapped_source, remote_pe=nvshmem.my_pe(), stream=stream)
        stream.sync()
        host_put_wrapped_jax = "accepted"
    except Exception as error:
        host_put_wrapped_jax = f"rejected: {type(error).__name__}: {error}"
    wrapped_put_values_match = int(jnp.sum(jax_view).block_until_ready()) == sum(range(size))

    del jax_output, jax_view, wrapped_source
    nvshmem.free(symmetric_buffer)
    return JaxInteropReport(
        nvshmem_pointer=nvshmem_pointer,
        jax_pointer=jax_pointer,
        pointer_identity=nvshmem_pointer == jax_pointer,
        initial_values_match=initial_values_match,
        external_update_visible_to_cached_numpy=external_update_visible_to_cached_numpy,
        external_update_visible_to_jax_kernel=external_update_visible_to_jax_kernel,
        jax_output_pointer_distinct=jax_output_pointer_distinct,
        host_put_direct_jax=host_put_direct_jax,
        host_put_wrapped_jax=host_put_wrapped_jax,
        jax_to_cuda_stream_handoff=jax_to_cuda_stream_handoff,
        wrapped_put_values_match=wrapped_put_values_match,
    )


def _remote_push_rank(rank: int, uid: object, results: multiprocessing.Queue, size: int) -> None:
    try:
        import jax
        import jax.numpy as jnp
        import nvshmem.core as nvshmem
        from cuda.core import Buffer, Device

        device = Device(rank)
        device.set_current()
        stream = device.create_stream()
        nvshmem.init(device=device, uid=uid, rank=rank, nranks=2, initializer_method="uid")
        inbox = nvshmem.buffer(size)
        signal = nvshmem.buffer(8)
        inbox.fill(0, stream=stream)
        signal.fill(0, stream=stream)
        nvshmem.barrier_all(stream)
        stream.sync()

        if rank == 0:
            source = jnp.arange(size, dtype=jnp.uint8)
            source.__dlpack__(stream=int(stream.handle))
            source_buffer = Buffer.from_handle(
                int(source.unsafe_buffer_pointer()),
                source.size * source.dtype.itemsize,
                owner=source,
            )
            nvshmem.put_signal(
                inbox,
                source_buffer,
                signal,
                1,
                nvshmem.SignalOp.SIGNAL_SET,
                remote_pe=1,
                stream=stream,
            )
            stream.sync()
            sender_pointer = int(source.unsafe_buffer_pointer())
            results.put(("sender", sender_pointer))
        else:
            receiver_view = jax.dlpack.from_dlpack(inbox)
            nvshmem.signal_wait(signal, 1, nvshmem.ComparisonType.CMP_GE, stream=stream)
            stream.sync()
            receiver_sum = int(jnp.sum(receiver_view).block_until_ready())
            results.put(
                (
                    "receiver",
                    int(inbox.handle),
                    int(receiver_view.unsafe_buffer_pointer()),
                    receiver_sum,
                )
            )

        nvshmem.barrier_all(stream)
        stream.sync()
        nvshmem.free(signal)
        nvshmem.free(inbox)
        nvshmem.finalize()
    except BaseException:
        results.put(("error", rank, traceback.format_exc()))


def run_remote_jax_push_probe(size: int = 16) -> RemoteJaxPushReport:
    import nvshmem.core as nvshmem

    multiprocessing.set_start_method("spawn", force=True)
    uid = nvshmem.get_unique_id()
    results: multiprocessing.Queue = multiprocessing.Queue()
    processes = [multiprocessing.Process(target=_remote_push_rank, args=(rank, uid, results, size)) for rank in range(2)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=120)
        if process.is_alive():
            process.terminate()
            raise TimeoutError(f"rank process {process.pid} did not finish")

    sender_pointer = None
    receiver = None
    errors = []
    for _ in range(2):
        try:
            result = results.get(timeout=5)
        except Empty:
            errors.append("a rank exited without returning a result")
            continue
        if result[0] == "sender":
            sender_pointer = result[1]
        elif result[0] == "receiver":
            receiver = result[1:]
        else:
            errors.append(result[2])
    if errors:
        raise RuntimeError("\n".join(errors))
    if sender_pointer is None or receiver is None:
        raise RuntimeError("remote JAX push did not return both rank results")
    receiver_nvshmem_pointer, receiver_jax_pointer, receiver_sum = receiver
    return RemoteJaxPushReport(
        sender_pointer=sender_pointer,
        receiver_nvshmem_pointer=receiver_nvshmem_pointer,
        receiver_jax_pointer=receiver_jax_pointer,
        receiver_pointer_identity=receiver_nvshmem_pointer == receiver_jax_pointer,
        receiver_sum=receiver_sum,
        values_match=receiver_sum == sum(range(size)),
    )


def _stream_push_call(size: int, receiver: bool):
    import cutlass.cute as cute
    import cutlass.jax as cjax
    import jax
    import numpy as np
    import nvshmem.core as nvshmem
    from cuda.bindings import driver
    from nvshmem.bindings.device import cute as nvshmem_bindings

    if receiver:

        @cute.kernel
        def stream_kernel(array: cute.Tensor, signal: cute.Tensor):
            thread_index, _, _ = cute.arch.thread_idx()
            if thread_index == 0:
                signal_ptr = cute.make_ptr(
                    cute.Uint64,
                    signal.iterator.toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=8,
                )
                nvshmem_bindings.signal_wait_until(signal_ptr, nvshmem.ComparisonType.CMP_GE, 1)

    else:

        @cute.kernel
        def stream_kernel(array: cute.Tensor, inbox: cute.Tensor, signal: cute.Tensor):
            thread_index, _, _ = cute.arch.thread_idx()
            if thread_index == 0:
                signal_ptr = cute.make_ptr(
                    cute.Uint64,
                    signal.iterator.toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=8,
                )
                inbox_ptr = cute.make_ptr(
                    cute.Int8,
                    inbox.iterator.toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                source_ptr = cute.make_ptr(
                    cute.Int8,
                    array.iterator.toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                nvshmem_bindings.int8_put_signal(
                    inbox_ptr,
                    source_ptr,
                    size,
                    signal_ptr,
                    1,
                    nvshmem.SignalOp.SIGNAL_SET,
                    1,
                )

    if receiver:

        @cute.jit
        def launcher(stream: driver.CUstream, array: cute.Tensor, signal: cute.Tensor):
            stream_kernel(array, signal).launch(grid=[1, 1, 1], block=[32, 1, 1], stream=stream)

        input_spec = (
            cjax.TensorSpec(mode=(0,), ptr_assumed_align=16, static=True),
            cjax.TensorSpec(mode=(0,), ptr_assumed_align=8, static=True),
        )

    else:

        @cute.jit
        def launcher(
            stream: driver.CUstream,
            array: cute.Tensor,
            inbox: cute.Tensor,
            signal: cute.Tensor,
        ):
            stream_kernel(array, inbox, signal).launch(grid=[1, 1, 1], block=[32, 1, 1], stream=stream)

        input_spec = (
            cjax.TensorSpec(mode=(0,), ptr_assumed_align=16, static=True),
            cjax.TensorSpec(mode=(0,), ptr_assumed_align=16, static=True),
            cjax.TensorSpec(mode=(0,), ptr_assumed_align=8, static=True),
        )

    output_spec = cjax.TensorSpec(mode=(0,), ptr_assumed_align=16, static=True)
    bitcode = nvshmem.find_device_bitcode_library()
    return cjax.cutlass_call(
        launcher,
        output_shape_dtype=jax.ShapeDtypeStruct((size,), np.dtype(np.int8)),
        input_spec=input_spec,
        output_spec=output_spec,
        input_output_aliases={0: 0},
        allow_cuda_graph=False,
        compile_options=f" --link-libraries={bitcode}",
        use_static_tensors=True,
    )


def _stream_push_rank(rank: int, uid: object, results: multiprocessing.Queue, size: int) -> None:
    try:
        import cutlass.cute as cute
        import jax
        import jax.numpy as jnp
        import nvshmem.core as nvshmem
        import nvshmem.core.interop.cute as cute_interop
        from cuda.core import Device

        device = Device(rank)
        device.set_current()
        stream = device.create_stream()
        nvshmem.init(device=device, uid=uid, rank=rank, nranks=2, initializer_method="uid")
        inbox = cute_interop.tensor((size,), dtype=cute.Uint8)
        signal = cute_interop.tensor((8,), dtype=cute.Uint8)
        for tensor in (inbox, signal):
            buffer, _, _ = cute_interop.tensor_get_buffer(tensor)
            buffer.fill(0, stream=stream)
        nvshmem.barrier_all(stream)
        stream.sync()

        jax_device = jax.devices("gpu")[rank]
        with jax.default_device(jax_device):
            inbox_buffer, _, _ = cute_interop.tensor_get_buffer(inbox)
            signal_buffer, _, _ = cute_interop.tensor_get_buffer(signal)
            inbox_view = jax.dlpack.from_dlpack(inbox_buffer)
            signal_view = jax.dlpack.from_dlpack(signal_buffer)
            if rank == 0:
                source = jnp.arange(size, dtype=jnp.int8)
                call = _stream_push_call(size, receiver=False)
                ordered_source = call(source, inbox_view, signal_view)
                sender_sum = int(jnp.sum(ordered_source).block_until_ready())
                results.put(
                    (
                        "sender",
                        int(source.unsafe_buffer_pointer()),
                        int(ordered_source.unsafe_buffer_pointer()),
                        sender_sum,
                    )
                )
            else:
                call = _stream_push_call(size, receiver=True)
                ordered_view = call(inbox_view, signal_view)
                receiver_sum = int(jnp.sum(ordered_view).block_until_ready())
                results.put(
                    (
                        "receiver",
                        int(inbox_buffer.handle),
                        int(inbox_view.unsafe_buffer_pointer()),
                        int(ordered_view.unsafe_buffer_pointer()),
                        receiver_sum,
                    )
                )

        nvshmem.barrier_all(stream)
        stream.sync()
        cute_interop.free_tensor(signal)
        cute_interop.free_tensor(inbox)
        nvshmem.finalize()
    except BaseException:
        results.put(("error", rank, traceback.format_exc()))


def run_stream_jax_push_probe(size: int = 16) -> StreamJaxPushReport:
    """Run a device-side JAX→NVSHMEM→JAX chain on XLA-provided streams."""
    import nvshmem.core as nvshmem

    multiprocessing.set_start_method("spawn", force=True)
    uid = nvshmem.get_unique_id()
    results: multiprocessing.Queue = multiprocessing.Queue()
    processes = [multiprocessing.Process(target=_stream_push_rank, args=(rank, uid, results, size)) for rank in range(2)]
    for process in processes:
        process.start()

    sender = None
    receiver = None
    errors = []
    for _ in processes:
        try:
            result = results.get(timeout=300)
        except Empty:
            errors.append("a stream-interoperability rank did not return")
            break
        if result[0] == "sender":
            sender = result[1:]
        elif result[0] == "receiver":
            receiver = result[1:]
        else:
            errors.append(result[2])
            break
    for process in processes:
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
    if errors:
        raise RuntimeError("\n".join(errors))
    if sender is None or receiver is None:
        raise RuntimeError("stream JAX push did not return both rank results")
    sender_input, sender_output, sender_sum = sender
    receiver_nvshmem, receiver_input, receiver_output, receiver_sum = receiver
    return StreamJaxPushReport(
        sender_input_pointer=sender_input,
        sender_output_pointer=sender_output,
        receiver_nvshmem_pointer=receiver_nvshmem,
        receiver_input_pointer=receiver_input,
        receiver_output_pointer=receiver_output,
        sender_alias_identity=sender_input == sender_output,
        receiver_alias_identity=receiver_nvshmem == receiver_input == receiver_output,
        receiver_sum=receiver_sum,
        values_match=sender_sum == receiver_sum == sum(range(size)),
        steady_state_host_synchronizations=0,
    )


def main() -> None:
    if os.environ.get("NVTP_JAX_STREAM") == "1":
        print(json.dumps(asdict(run_stream_jax_push_probe()), indent=2, sort_keys=True))
        return
    if os.environ.get("NVTP_JAX_REMOTE") == "1":
        print(json.dumps(asdict(run_remote_jax_push_probe()), indent=2, sort_keys=True))
        return

    import nvshmem.core as nvshmem
    from cuda.core import Device

    device = Device(0)
    device.set_current()
    uid = nvshmem.get_unique_id()
    nvshmem.init(device=device, uid=uid, rank=0, nranks=1, initializer_method="uid")
    report = run_local_jax_interop_probe()
    nvshmem.finalize()
    print(json.dumps(asdict(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
