# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-node survey of the CuTe-DSL compile toolchain (libNVVM heterogeneity hunt).

The MXFP8-002c quantizer fails ``NVVM_ERROR_COMPILATION`` on some GB200 nodes
and compiles on others with identical wheels (see ``mxfp8_quantizer_probe``).
This script runs on ONE node and emits everything needed to correlate the
pass/fail verdict with the toolchain that produced it:

- node identity: hostname, GPU UUIDs (stable physical-node fingerprint),
  kernel, NVIDIA kernel-module version, driver version;
- the compile ladder from ``mxfp8_quantizer_probe`` (run in a subprocess so a
  process-fatal FFI compile cannot kill the survey), with per-variant verdicts;
- the exact shared objects the DSL loaded during compilation, harvested from
  the child's ``/proc/self/maps``;
- sha256 + size for every candidate toolchain binary on the node (libnvvm,
  libnvJitLink, libcuda, ptxas) whether wheel-shipped or host-mounted, plus
  the libNVVM API/IR versions via ctypes;
- installed wheel versions for the compile-relevant packages.

Everything is printed human-readably, then condensed into one machine-readable
``SURVEY_JSON {...}`` line for harvesting across replicas. Run via a replicated
``GB200x4`` iris job (one replica per node).
"""

import ctypes
import glob
import hashlib
import importlib.metadata
import json
import os
import re
import socket
import subprocess
import sys
import sysconfig

_STANDALONE_DIR = os.path.dirname(os.path.abspath(__file__))
_LADDER_TIMEOUT = 1500
_LIB_PATTERNS = ("libnvvm", "libnvJitLink", "libnvjitlink", "libcuda.so", "ptxas", "libnvptxcompiler")
_MAPS_RE = re.compile(r"(nvvm|cuda|ptx|nvdisasm|nvjitlink|cutlass)", re.IGNORECASE)

# Child process: run the existing bisect ladder, dumping loaded .so paths after
# the first compile attempt and again at the end (compile failures may be
# process-fatal, so maps are flushed as early as possible).
_CHILD_CODE = f"""
import re, sys
sys.path.insert(0, {_STANDALONE_DIR!r})

def dump_maps(tag):
    seen = set()
    with open("/proc/self/maps") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 6 and ".so" in parts[5]:
                seen.add(parts[5])
    pat = re.compile(r"(nvvm|cuda|ptx|nvdisasm|nvjitlink|cutlass)", re.IGNORECASE)
    for p in sorted(seen):
        if pat.search(p):
            print(f"MAPS[{{tag}}]: {{p}}", flush=True)

import mxfp8_quantizer_probe as probe
probe.ensure_blackwell_arch()
probe.run_probe(1, "v1_cvt_intrinsic", use_asm=False)
dump_maps("after_v1_intrinsic")
probe.run_probe(1, "v1_asm")
probe.run_probe(2, "v2_shuffle")
probe.run_probe(3, "v3_e8m0")
probe.run_probe(4, "v4_scalebyte")
probe.run_full(512, 128)
probe.run_full(8192, 2560)
dump_maps("final")
print("LADDER_DONE", flush=True)
"""


def _run(cmd: list[str], timeout: int = 30) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception as e:
        return f"<failed: {e}>"


def _read(path: str) -> str:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError as e:
        return f"<failed: {e}>"


def _sha256(path: str) -> dict:
    h = hashlib.sha256()
    try:
        real = os.path.realpath(path)
        with open(real, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        return {"path": path, "realpath": real, "size": os.path.getsize(real), "sha256": h.hexdigest()}
    except OSError as e:
        return {"path": path, "error": str(e)}


def _nvvm_api_versions(libnvvm_path: str) -> dict:
    try:
        lib = ctypes.CDLL(libnvvm_path)
        maj, mnr = ctypes.c_int(), ctypes.c_int()
        lib.nvvmVersion(ctypes.byref(maj), ctypes.byref(mnr))
        out = {"nvvm_version": f"{maj.value}.{mnr.value}"}
        ir_maj, ir_mnr, dbg_maj, dbg_mnr = (ctypes.c_int() for _ in range(4))
        lib.nvvmIRVersion(ctypes.byref(ir_maj), ctypes.byref(ir_mnr), ctypes.byref(dbg_maj), ctypes.byref(dbg_mnr))
        out["nvvm_ir_version"] = f"{ir_maj.value}.{ir_mnr.value} (dbg {dbg_maj.value}.{dbg_mnr.value})"
        return out
    except OSError as e:
        return {"error": str(e)}


def _candidate_toolchain_files() -> list[str]:
    roots = [
        sysconfig.get_paths()["purelib"],
        "/usr/local/cuda",
        "/usr/local/nvidia",
        "/usr/lib",
        "/usr/lib64",
        "/run/nvidia",
    ]
    # Toolkit installs are versioned (/usr/local/cuda-13.x); include them.
    roots.extend(sorted(glob.glob("/usr/local/cuda-*")))
    found: set[str] = set()
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
            for name in filenames:
                if any(name.startswith(p) for p in _LIB_PATTERNS):
                    found.add(os.path.join(dirpath, name))
    # The driver userspace libs are injected by ldconfig; catch those too.
    ldconfig = _run(["ldconfig", "-p"])
    for line in ldconfig.splitlines():
        if any(p in line for p in ("libnvvm", "libcuda.so", "libnvJitLink")):
            found.add(line.split("=>")[-1].strip())
    return sorted(found)


def print_cutlass_tree_hashes():
    """Hash every installed file of the cutlass DSL distribution.

    ``nvidia-cutlass-dsl-libs-base`` and ``-libs-cu13`` ship 99 overlapping
    paths with DIFFERENT contents; whichever wheel's copy wins the install
    decides per-file which toolchain variant a venv actually runs. Emitted one
    line per file (``TREEHASH <relpath> <sha256>``) so log-line truncation
    cannot corrupt the record; classification against the wheel manifests
    happens offline.
    """
    root = os.path.join(sysconfig.get_paths()["purelib"], "nvidia_cutlass_dsl")
    if not os.path.isdir(root):
        print("TREEHASH_MISSING", flush=True)
        return
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=True):
        if "__pycache__" in dirpath:
            continue
        for name in sorted(filenames):
            p = os.path.join(dirpath, name)
            rec = _sha256(p)
            rel = os.path.relpath(p, root)
            print(f"TREEHASH {rel} {rec.get('sha256', 'ERR')}", flush=True)


def collect_env() -> dict:
    env = {
        "hostname": socket.gethostname(),
        "uname": _run(["uname", "-a"]),
        "nvidia_kmod": _read("/proc/driver/nvidia/version"),
        "nvidia_smi": _run(["nvidia-smi", "--query-gpu=driver_version,name,uuid", "--format=csv,noheader"]),
        "ptxas_which": _run(["sh", "-c", "which ptxas"]),
        "ptxas_version": _run(["sh", "-c", "ptxas --version | tail -2"]),
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""),
        "iris_env": {k: v for k, v in os.environ.items() if re.search(r"NODE|REPLICA|WORKER", k)},
    }
    pkgs = {}
    for dist in (
        "nvidia-cutlass-dsl",
        "nvidia-cutlass-dsl-libs-base",
        "nvidia-cutlass-dsl-libs-cu13",
        "nvidia-cuda-nvcc",
        "nvidia-cuda-nvcc-cu13",
        "nvidia-cuda-runtime-cu13",
        "cuda-pathfinder",
        "jax",
        "jaxlib",
        "jax-cuda13-plugin",
        "jax-cuda13-pjrt",
    ):
        try:
            pkgs[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            pkgs[dist] = None
    env["packages"] = pkgs
    return env


def run_ladder() -> tuple[dict, list[str], str]:
    """Returns (verdicts, loaded .so paths from child maps, raw child output)."""
    child_env = {**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _CHILD_CODE], capture_output=True, text=True, timeout=_LADDER_TIMEOUT, env=child_env
        )
        out = proc.stdout + ("\n[child stderr tail]\n" + proc.stderr[-2000:] if proc.returncode != 0 else "")
        status = f"exit={proc.returncode}"
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        status = "timeout"
    verdicts = {}
    maps: set[str] = set()
    for line in out.splitlines():
        if line.startswith(("PASS ", "FAIL ")):
            state, rest = line.split(" ", 1)
            name = rest.split(":", 1)[0].split(" ")[0]
            verdicts[name] = {"state": state, "detail": rest.split(":", 1)[-1].strip()[:300]}
        elif line.startswith("MAPS["):
            maps.add(line.split(": ", 1)[1])
    verdicts["_ladder_status"] = status
    verdicts["_ladder_done"] = "LADDER_DONE" in out
    return verdicts, sorted(m for m in maps if _MAPS_RE.search(m)), out


def main():
    print("=== nvvm node survey ===", flush=True)
    env = collect_env()
    print(json.dumps(env, indent=2), flush=True)
    print_cutlass_tree_hashes()

    verdicts, loaded, raw = run_ladder()
    print("--- ladder output ---", flush=True)
    print(raw, flush=True)

    hashes = [_sha256(p) for p in _candidate_toolchain_files()]
    # Hash anything the child actually loaded that the scan missed.
    scanned = {h["path"] for h in hashes}
    hashes.extend(_sha256(p) for p in loaded if p not in scanned)

    nvvm_paths = [h for h in hashes if "libnvvm" in os.path.basename(h.get("realpath", h["path"]))]
    nvvm_api = {h["path"]: _nvvm_api_versions(h["path"]) for h in nvvm_paths if "sha256" in h}

    record = {
        "env": env,
        "verdicts": verdicts,
        "loaded_libs": loaded,
        "toolchain_hashes": hashes,
        "nvvm_api": nvvm_api,
    }
    print("SURVEY_JSON " + json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
