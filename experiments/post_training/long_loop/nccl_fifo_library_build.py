# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a pinned baseline/patched NCCL pair for diagnostic A/B testing only."""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join

UPSTREAM = "https://github.com/NVIDIA/nccl.git"
REVISION = "73cf112295c33aee2b895f329f592f2a9b4b0f97"
SOURCE_SHA256 = "ff940fe5950e725566ee24f9b5bc5016ed126e60e99313aec3b3d558951f5567"
PATCHED_SHA256 = "5e7aeb39cda4c4295c9297e46c7dba0167d2c0be127bceebe3f60e21d7c67ab0"
GENCODE = "-gencode=arch=compute_100,code=sm_100"
ANCHOR = """    if ((nBytes < (1 * (concurrentOps * chunkSize))) && (chunkSize > 32768)) chunkSize = 32768;
  } else if (info->algorithm == NCCL_ALGO_TREE"""
REPLACEMENT = """    if ((nBytes < (1 * (concurrentOps * chunkSize))) && (chunkSize > 32768)) chunkSize = 32768;
    // NVLSTree staging payloads must fit the SIMPLE connection FIFO slot.
    chunkSize = std::min(chunkSize, bufferMaxChunkSize);
  } else if (info->algorithm == NCCL_ALGO_TREE"""


def patch_source(source: str) -> str:
    """Apply the one-function candidate only to the pinned upstream file."""
    if hashlib.sha256(source.encode()).hexdigest() != SOURCE_SHA256 or source.count(ANCHOR) != 1:
        raise ValueError("NCCL source does not match the pinned FIFO candidate")
    result = source.replace(ANCHOR, REPLACEMENT)
    if hashlib.sha256(result.encode()).hexdigest() != PATCHED_SHA256:
        raise ValueError("Patched NCCL source hash differs from the reviewed candidate")
    return result


def publish_file(local: Path, uri: str) -> dict:
    """Upload a file and verify its bytes by reading it back within the region."""
    with local.open("rb") as stream:
        expected = hashlib.file_digest(stream, "sha256").hexdigest()
    filesystem, path = filesystem_for(uri)
    with local.open("rb") as source, filesystem.open(path, "wb") as target:
        shutil.copyfileobj(source, target, length=1 << 20)
    with filesystem.open(path, "rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    size = local.stat().st_size
    if actual != expected or filesystem.size(path) != size:
        raise ValueError(f"Regional readback differs for {uri}")
    return {"uri": uri, "sha256": actual, "bytes": size, "readback_verified": True}


def build(args) -> None:
    if platform.machine() != "aarch64":
        raise ValueError("This diagnostic requires a native ARM64 builder")
    root = Path(args.work_directory)
    root.mkdir(parents=True, exist_ok=False)
    source_root = root / "source"
    cuda_home = Path(os.environ["CUDA_HOME"])
    nvcc = cuda_home / "bin/nvcc"
    report = {
        "status": "incomplete;not_qualified",
        "source_repository": UPSTREAM,
        "source_revision": REVISION,
        "source_file_sha256": SOURCE_SHA256,
        "patched_file_sha256": PATCHED_SHA256,
        "architecture": platform.machine(),
        "gpu_use": "none;one_GPU_requested_only_for_native_ARM64_host_selection",
        "cuda_home": str(cuda_home),
        "nvcc_version": subprocess.check_output([str(nvcc), "--version"], text=True),
        "cxx_version": subprocess.check_output(["g++", "--version"], text=True),
        "gencode": GENCODE,
        "jobs": args.jobs,
        "libraries": {},
    }
    subprocess.run(
        ["git", "clone", "--depth=1", "--branch=v2.30.7-1", UPSTREAM, str(source_root)], check=True, timeout=180
    )
    actual_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source_root, text=True).strip()
    if actual_revision != REVISION:
        raise ValueError("Upstream tag moved from the pinned NCCL commit")
    deadline = time.monotonic() + args.build_seconds
    command = ["make", f"-j{args.jobs}", "src.build", f"CUDA_HOME={cuda_home}", f"NVCC_GENCODE={GENCODE}", "TRACE=1"]
    report["build_command"] = command
    try:
        for variant in ("baseline", "patched"):
            source_path = source_root / "src/enqueue.cc"
            if variant == "patched":
                source_path.write_text(patch_source(source_path.read_text()))
            elif hashlib.sha256(source_path.read_bytes()).hexdigest() != SOURCE_SHA256:
                raise ValueError("Baseline source SHA differs from pinned upstream")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("NCCL pair exceeded the build budget")
            log = root / f"{variant}.log"
            started = time.monotonic()
            with log.open("w") as stream:
                subprocess.run(
                    command, cwd=source_root, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=remaining
                )
            compile_seconds = time.monotonic() - started
            library = (source_root / "build/lib/libnccl.so.2").resolve(strict=True)
            elf_header = subprocess.check_output(["readelf", "-h", str(library)], text=True)
            if "AArch64" not in elf_header:
                raise ValueError("NCCL builder produced the wrong ELF architecture")
            snapshot = root / f"libnccl-{variant}.so.2"
            shutil.copy2(library, snapshot)
            report["libraries"][variant] = {
                **publish_file(snapshot, prefix_join(args.output_prefix, f"{variant}/libnccl.so.2")),
                "build_seconds": compile_seconds,
                "elf_header": elf_header,
                "elf_dynamic": subprocess.check_output(["readelf", "-d", str(snapshot)], text=True),
                "build_log_uri": prefix_join(args.output_prefix, f"{variant}/build.log"),
            }
        diff = root / "source.patch"
        diff.write_bytes(subprocess.check_output(["git", "diff", "--", "src/enqueue.cc"], cwd=source_root))
        report["source_patch"] = publish_file(diff, prefix_join(args.output_prefix, "source.patch"))
        if report["libraries"]["baseline"]["sha256"] == report["libraries"]["patched"]["sha256"]:
            raise ValueError("Baseline and patched NCCL binaries are identical")
        report["status"] = "built;not_qualified"
    finally:
        report["build_logs"] = {
            log.stem: publish_file(log, prefix_join(args.output_prefix, f"{log.stem}/build.log"))
            for log in sorted(root.glob("*.log"))
        }
        receipt = root / "receipt.json"
        receipt.write_text(json.dumps(report, indent=2) + "\n")
        publish_file(receipt, prefix_join(args.output_prefix, "receipt.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--work-directory", default="/tmp/nccl-fifo-library-build")
    parser.add_argument("--jobs", type=int, default=24)
    parser.add_argument("--build-seconds", type=int, default=3000)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
