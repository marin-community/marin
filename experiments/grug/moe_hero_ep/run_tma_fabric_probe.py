# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compile and run the TMA-over-fabric probe on a GPU node.

Reuses the mok_like build's CUDA toolchain materialization so the probe compiles with the same
nvcc the kernel does, rather than a second, differently-provisioned path.
"""

import pathlib
import subprocess
import sys
import tempfile

from levanter.kernels.mixture_of_kittens.build import _cuda_include_dirs, _materialize_cuda_toolchain

_SOURCE = pathlib.Path(__file__).with_name("tma_fabric_probe.cu")
_ARCH = "sm_100a"


def main() -> int:
    with tempfile.TemporaryDirectory() as scratch:
        build_dir = pathlib.Path(scratch)
        toolchain_root = _materialize_cuda_toolchain(build_dir)
        binary = build_dir / "tma_fabric_probe"
        compile_command = [
            str(toolchain_root / "bin" / "nvcc"),
            str(_SOURCE),
            "-o",
            str(binary),
            "-std=c++17",
            f"-gencode=arch=compute_{_ARCH.removeprefix('sm_')},code={_ARCH}",
            # The wheel ships libcudart_static.a and libcudadevrt.a but no libcudart.so symlink,
            # so keep nvcc's static default and just point it at the wheel's lib directory.
            "-lcuda",
        ]
        for include_dir in _cuda_include_dirs():
            compile_command.extend(("-I", str(include_dir)))
            library_dir = include_dir.parent / "lib"
            if library_dir.is_dir():
                compile_command.extend(("-L", str(library_dir), "-Xlinker", f"-rpath={library_dir}"))
        print(" ".join(compile_command), flush=True)
        compiled = subprocess.run(compile_command, check=False, capture_output=True, text=True)
        if compiled.returncode != 0:
            print((compiled.stderr or compiled.stdout).strip()[-4000:], flush=True)
            return 2

        probe = subprocess.run([str(binary)], check=False, capture_output=True, text=True)
        print(probe.stdout.strip(), flush=True)
        if probe.stderr.strip():
            print(probe.stderr.strip(), flush=True)
        return probe.returncode


if __name__ == "__main__":
    sys.exit(main())
