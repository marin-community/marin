#!/usr/bin/env python3
"""H100 runner for the upstream mixed-FP8 wgmma numeric test (PR job C).

Sets up the CUDA toolchain Mosaic-GPU needs (ptxas + libdevice) BEFORE importing jax/pytest,
then runs ``tests/mosaic/gpu_test.py::WGMMATest`` filtered to the mixed-FP8 wgmma cases. The
toolchain shim mirrors ``_ensure_cuda_toolchain`` from the marin bench scripts; it is defensive
(harmless if jax already discovers a standard nvidia-*-cu13 layout in the venv).

Usage: python run_mixed_fp8_pytest.py <path-to-jax-src> [pytest-args...]
"""
import glob
import os
import sys
import tempfile


def ensure_cuda_toolchain() -> None:
    def glob_first(name):
        for base in sys.path:
            if base and os.path.isdir(base):
                hits = glob.glob(os.path.join(base, "nvidia", "**", name), recursive=True)
                if hits:
                    return hits[0]
        return None

    import shutil

    ptxas = shutil.which("ptxas") or glob_first("ptxas")
    libdevice = glob_first("libdevice.10.bc")
    if not ptxas or not libdevice:
        print(f"cuda toolchain: incomplete (ptxas={ptxas}, libdevice={libdevice})", flush=True)
        return
    nvvm_parent = os.path.dirname(os.path.dirname(os.path.dirname(libdevice)))
    root = tempfile.mkdtemp(prefix="xla_cuda_")
    os.symlink(os.path.dirname(ptxas), os.path.join(root, "bin"))
    os.symlink(os.path.join(nvvm_parent, "nvvm"), os.path.join(root, "nvvm"))
    os.environ["PATH"] = os.path.dirname(ptxas) + os.pathsep + os.environ.get("PATH", "")
    flags = os.environ.get("XLA_FLAGS", "")
    os.environ["XLA_FLAGS"] = f"{flags} --xla_gpu_cuda_data_dir={root}".strip()
    for var in ("CUDA_DIR", "CUDA_HOME", "CUDA_PATH"):
        os.environ[var] = root
    cwd_link = os.path.join(os.getcwd(), "libdevice.10.bc")
    if not os.path.exists(cwd_link):
        os.symlink(libdevice, cwd_link)
    print(f"cuda toolchain: ptxas={ptxas}\n  libdevice={libdevice}", flush=True)


def main() -> int:
    jax_src = sys.argv[1]
    extra = sys.argv[2:]
    ensure_cuda_toolchain()
    os.chdir(jax_src)
    import pytest  # imported after toolchain env is set

    args = [
        "tests/mosaic/gpu_test.py",
        "-k",
        "test_wgmma_mixed_fp8",
        "-v",
        "--no-header",
        *extra,
    ]
    print(f"pytest {' '.join(args)}", flush=True)
    return pytest.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
