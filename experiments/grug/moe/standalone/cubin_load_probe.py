# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MXFP8-011: feed extracted executable blobs to cuModuleLoadData on one GPU.

Downloads the GPU binaries carved out of the failing (cute-producer) and
passing (xla-producer) jit_train_step executables and loads each with the CUDA
driver directly. If the failing run's binary reproduces
CUDA_ERROR_INVALID_VALUE here, the 16-node defect reduces to a single-GPU
artifact problem and the blob can be bisected structurally.

Usage (GB200x1 pod, --extra gpu):
    python -m experiments.grug.moe.standalone.cubin_load_probe s3://.../carve4/
"""

import struct
import sys

import fsspec
from cuda.bindings import driver as cuda


def check(res, what: str) -> bool:
    (err,) = res if isinstance(res, tuple) and len(res) == 1 else (res[0],)
    if err != cuda.CUresult.CUDA_SUCCESS:
        _, name = cuda.cuGetErrorName(err)
        print(f"    {what}: FAILED {name}")
        return False
    return True


def main() -> None:
    prefix = sys.argv[1]
    fs, root = fsspec.url_to_fs(prefix)

    check(cuda.cuInit(0), "cuInit")
    _, dev = cuda.cuDeviceGet(0)
    _, ctx = cuda.cuCtxCreate(None, 0, dev)
    _, cap_major = cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev)
    _, cap_minor = cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev)
    print(f"device compute capability sm_{cap_major}{cap_minor}")

    for path in sorted(fs.ls(root)):
        name = path.split("/")[-1]
        if not (name.endswith(".elf") or name.endswith(".ptx")):
            continue
        with fs.open(path, "rb") as f:
            blob = f.read()
        kind = "ELF" if blob[:4] == b"\x7fELF" else "PTX/other"
        extra = ""
        if kind == "ELF":
            (machine,) = struct.unpack_from("<H", blob, 18)
            (flags,) = struct.unpack_from("<I", blob, 48)
            extra = f" machine={machine} flags={hex(flags)}"
        print(f"  {name}: {len(blob)} B {kind}{extra}")
        if name.endswith(".elf"):
            err, mod = cuda.cuModuleLoadData(blob)
            if err != cuda.CUresult.CUDA_SUCCESS:
                _, ename = cuda.cuGetErrorName(err)
                print(f"    cuModuleLoadData -> {ename}")
            else:
                print("    cuModuleLoadData -> OK")
                cuda.cuModuleUnload(mod)
        else:
            err, mod = cuda.cuModuleLoadData(blob + b"\x00")
            if err != cuda.CUresult.CUDA_SUCCESS:
                _, ename = cuda.cuGetErrorName(err)
                print(f"    cuModuleLoadData(PTX) -> {ename}")
            else:
                print("    cuModuleLoadData(PTX) -> OK")
                cuda.cuModuleUnload(mod)

    cuda.cuCtxDestroy(ctx)


if __name__ == "__main__":
    main()
