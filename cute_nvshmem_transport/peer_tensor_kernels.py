# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: PLC0415


def run_peer_tensor_store_probe() -> int:
    """Write this PE's rank into its neighbor through a CuTe peer tensor."""
    import ctypes

    import cutlass.cute as cute
    import nvshmem.core as nvshmem
    import nvshmem.core.device.cute.mem as nvshmem_cute_mem
    import nvshmem.core.interop.cute as cute_interop
    from cuda.bindings import driver
    from cuda.core import Device
    from cutlass.cute.arch.nvvm_wrappers import WARP_SIZE
    from cutlass.cute.typing import Int32

    tensor = cute_interop.tensor((4,), dtype=cute.Int32)
    buffer, _, _ = cute_interop.tensor_get_buffer(tensor)
    stream = Device().create_stream()
    buffer.fill(0, stream=stream)
    stream.sync()

    @cute.kernel
    def peer_store_kernel(array: cute.Tensor, pe: Int32):
        peer_array = nvshmem_cute_mem.get_peer_tensor(array, pe)
        thread_index, _, _ = cute.arch.thread_idx()
        if thread_index == 0:
            peer_array[0] = nvshmem.my_pe() + 1

    @cute.jit
    def peer_store_launcher(array: cute.Tensor, pe: Int32):
        peer_store_kernel(array, pe).launch(
            grid=[1, 1, 1],
            block=[cute.size(WARP_SIZE, mode=[0]), 1, 1],
        )

    bitcode = nvshmem.find_device_bitcode_library()
    compiled = cute.compile(peer_store_launcher, tensor, 0, options=f" --link-libraries={bitcode}")
    compiled = compiled.to(Device().device_id)
    cuda_library = compiled.jit_module.cuda_library
    kernel_object = nvshmem.NvshmemKernelObject.from_handle(int(cuda_library[0]))
    nvshmem.library_init(kernel_object)

    peer = (nvshmem.my_pe() + 1) % nvshmem.n_pes()
    compiled(tensor, peer)
    Device().sync()
    nvshmem.barrier(nvshmem.Teams.TEAM_WORLD, stream=stream)
    stream.sync()

    host_value = ctypes.c_int32()
    error = driver.cuMemcpyDtoH(ctypes.addressof(host_value), int(buffer.handle), ctypes.sizeof(host_value))[0]
    if int(error) != 0:
        raise RuntimeError(f"cuMemcpyDtoH failed with CUDA error {int(error)}")

    nvshmem.library_finalize(kernel_object)
    cute_interop.free_tensor(tensor)
    return host_value.value
