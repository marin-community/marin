# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded compiler evidence for the opt-in native learner acceptance run."""

import argparse
import ctypes
import dataclasses
import gzip
import hashlib
import json
import math
import numbers
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import equinox as eqx
import jax
from levanter.callbacks.profiler import upload_xla_dumps
from rigging.filesystem.cluster_config import marin_temp_bucket
from rigging.filesystem.storage_path import StoragePath

MAX_FILES = 16
MAX_FILE_BYTES = 32 * 1024**2
MAX_TOTAL_BYTES = 128 * 1024**2
UPLOAD_TIMEOUT_SECONDS = 60


def scientific_xla_flags(settings: dict) -> list[str]:
    """Normalize only the dump path appended by the owned host launch boundary."""
    flags = shlex.split(os.environ.get("XLA_FLAGS", ""))
    owned = settings.get("compiler_dump_dir")
    suffix = [f"--xla_dump_to={owned}", "--xla_dump_hlo_as_text", "--xla_dump_hlo_module_re=train_step"]
    if owned is not None and flags[-3:] == suffix:
        flags[-3] = "--xla_dump_to=<native-compiler-diagnostics>"
    return flags


def evidence_destination(settings: dict, process_id: int, pid: int) -> str:
    checkpoint = StoragePath(settings["runtime"]["checkpoint_dir"])
    # An explicit bucket prevents CoreWeave's shared-scratch routing to 08a.
    scratch = dataclasses.replace(checkpoint, segments=()) if checkpoint.scheme in ("s3", "gs") else checkpoint.parent
    return marin_temp_bucket(
        14, prefix=f"xla-dumps/{settings['run_id']}/process-{process_id}/pid-{pid}", temp_prefix=str(scratch)
    )


def _memory_stats() -> list[dict]:
    rows = []
    for device in jax.local_devices():
        row = {"device_id": device.id}
        try:
            stats = device.memory_stats()
            row["stats"] = (
                None
                if stats is None
                else {
                    key: int(value) if isinstance(value, numbers.Integral) else float(value)
                    for key, value in stats.items()
                    if isinstance(value, numbers.Number) and math.isfinite(value)
                }
            )
        except Exception as error:
            row["error_type"] = type(error).__name__
        rows.append(row)
    return rows


def _state_bytes(state) -> dict:
    result = {}
    for name in ("model", "opt_state"):
        per_device = {}
        for value in jax.tree.leaves(getattr(state, name)):
            if eqx.is_array(value):
                for shard in value.addressable_shards:
                    key = str(shard.device.id)
                    per_device[key] = per_device.get(key, 0) + shard.data.size * shard.data.dtype.itemsize
        result[name] = per_device
    return result


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def _inventory(stage: Path) -> list[Path]:
    files = list(stage.iterdir())
    if (
        len(files) > MAX_FILES
        or any(
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size > MAX_FILE_BYTES
            or not (path.name.endswith(".json") or path.name.endswith(".txt.gz"))
            for path in files
        )
        or sum(path.stat().st_size for path in files) > MAX_TOTAL_BYTES
    ):
        raise ValueError("Compiler evidence exceeds the bounded text/metadata inventory")
    return files


def _archive(stage: Path) -> None:
    output = os.environ.get("IRIS_OUTPUT_DIR")
    if output:
        destination = Path(output) / "native-compile" / stage.name
        destination.mkdir(parents=True, exist_ok=True)
        for path in _inventory(stage):
            target = destination / path.name
            if not target.exists():
                shutil.copyfile(path, target)


def _publish(stage: Path, settings: dict, phase: str) -> dict:
    destination = evidence_destination(settings, jax.process_index(), os.getpid())
    environment = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "CUDA_VISIBLE_DEVICES": "",
        "JAX_NUM_CPU_DEVICES": "1",
        "OMP_NUM_THREADS": "1",
        "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false",
    }
    for name in ("JAX_PLATFORM_NAME", "JAX_CUDA_VISIBLE_DEVICES", "JAX_LOCAL_DEVICE_IDS"):
        environment.pop(name, None)
    started = time.monotonic()
    result: dict = {"destination": destination, "timeout_seconds": UPLOAD_TIMEOUT_SECONDS}
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "marin.rl.levanter_compile_diagnostics",
                str(stage),
                destination,
                "--phase",
                phase,
                "--parent-pid",
                str(os.getpid()),
            ],
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=UPLOAD_TIMEOUT_SECONDS,
        )
        result["uploaded"] = True
    except (subprocess.SubprocessError, OSError) as error:
        result.update(uploaded=False, error_type=type(error).__name__)
        if isinstance(error, subprocess.CalledProcessError):
            result["returncode"] = error.returncode
    result["seconds"] = time.monotonic() - started
    identity = stage / f"uploader-{phase}.json"
    if identity.is_file():
        result["uploader"] = json.loads(identity.read_text())
    _write_json(stage / f"upload-{phase}.json", result)
    _archive(stage)
    print(
        "native_compiler_evidence " + json.dumps({"phase": phase, "path": str(stage), **result}),
        file=sys.stderr,
        flush=True,
    )
    return result


def capture_compile(trainer, state, batch, projection_plan, settings: dict) -> Path:
    """Compile the exact no-hook step, record evidence, and release the executable before normal execution."""
    if trainer.hooks.jit_hooks:
        raise ValueError("Native compiler evidence requires the unchanged no-hook train step")
    stage = Path(tempfile.mkdtemp(prefix=f"native-compile-process-{jax.process_index()}-pid-{os.getpid()}-"))
    receipt: dict = {
        "process_id": jax.process_index(),
        "pid": os.getpid(),
        "optimizer_step": int(state.step),
        "time_ns": time.time_ns(),
        "backend": jax.default_backend(),
        "state_logical_bytes_per_device": _state_bytes(state),
        "before_compile": _memory_stats(),
    }
    _write_json(stage / "before-compile.json", receipt)
    compiled = None
    try:
        kwargs = {} if projection_plan is None else {"projection_plan": projection_plan}
        compiled = trainer._jit_train_step_fn_no_hook.lower(state, (batch,), kwargs).compile()
        stats = compiled.memory_analysis()
        receipt["compiled_memory"] = (
            None
            if stats is None
            else {
                name: getattr(stats, name)
                for name in (
                    "argument_size_in_bytes",
                    "output_size_in_bytes",
                    "alias_size_in_bytes",
                    "temp_size_in_bytes",
                )
            }
        )
        text = compiled.as_text().encode()
        receipt["hlo_is_scheduled"] = b"is_scheduled=true" in text.partition(b"\n")[0]
        if len(text) <= MAX_FILE_BYTES:
            with gzip.open(stage / "train.optimized.hlo.txt.gz", "wb") as output:
                output.write(text)
            receipt["optimized_hlo_source_bytes"] = len(text)
        else:
            receipt["optimized_hlo_skipped_bytes"] = len(text)
        del text
    except BaseException as error:
        receipt["compile_error_type"] = type(error).__name__
        raise
    finally:
        del compiled
        _finish_capture(stage, receipt, settings)
    return stage


def _finish_capture(stage: Path, receipt: dict, settings: dict) -> None:
    try:
        dump = Path(settings["compiler_dump_dir"])
        selected, skipped = [], []
        total = receipt.get("optimized_hlo_source_bytes", 0)
        if dump.is_dir():
            for path in sorted(dump.glob("*after_optimizations*.txt")):
                if (
                    not path.is_file()
                    or path.is_symlink()
                    or not any(part in path.name for part in ("buffer-assignment", "memory-usage-report"))
                ):
                    continue
                size = path.stat().st_size
                if len(selected) >= MAX_FILES - 8 or size > MAX_FILE_BYTES or total + size > MAX_TOTAL_BYTES - 65536:
                    skipped.append({"name": path.name, "bytes": size})
                    continue
                with path.open("rb") as source, gzip.open(stage / f"{path.name}.gz", "wb") as output:
                    shutil.copyfileobj(source, output)
                selected.append(path.name)
                total += size
        receipt.update(
            after_compile=_memory_stats(),
            selected_reports=selected,
            skipped_reports=skipped,
            source_bytes=total,
            files=[
                {
                    "name": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in sorted(stage.iterdir())
            ],
        )
        _write_json(stage / "before-update.json", receipt)
        _archive(stage)
        _publish(stage, settings, "before-update")
    except Exception as error:
        print(
            "native_compiler_evidence "
            + json.dumps({"phase": "before-update", "diagnostic_error_type": type(error).__name__}),
            file=sys.stderr,
            flush=True,
        )


def capture_failure(stage: Path, settings: dict, error: BaseException) -> None:
    """Preserve failure-final stats separately; diagnostic errors never replace the original training exception."""
    try:
        _write_json(
            stage / "failure-final.json",
            {
                "time_ns": time.time_ns(),
                "error_type": type(error).__name__,
                "memory": _memory_stats(),
                "before_update_sha256": hashlib.sha256((stage / "before-update.json").read_bytes()).hexdigest(),
            },
        )
        _archive(stage)
        _publish(stage, settings, "failure-final")
    except Exception as diagnostic_error:
        print(
            "native_compiler_evidence "
            + json.dumps({"phase": "failure-final", "diagnostic_error_type": type(diagnostic_error).__name__}),
            file=sys.stderr,
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    parser.add_argument("destination")
    parser.add_argument("--phase", required=True, choices=("before-update", "failure-final"))
    parser.add_argument("--parent-pid", type=int, required=True)
    args = parser.parse_args()
    if sys.platform == "linux":
        if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM) != 0:
            raise OSError(ctypes.get_errno(), "Could not bind uploader lifetime to its native learner")
        if os.getppid() != args.parent_pid:
            return
    _inventory(args.stage)
    backend = jax.default_backend()
    if backend != "cpu":
        raise RuntimeError("Compiler evidence uploader must use only the CPU backend")
    _write_json(args.stage / f"uploader-{args.phase}.json", {"backend": backend, "pid": os.getpid()})
    _inventory(args.stage)
    index = args.stage.with_name(args.stage.name + ".uploaded.json")
    uploaded = (
        {Path(name): tuple(signature) for name, signature in json.loads(index.read_text()).items()}
        if index.exists()
        else {}
    )
    try:
        StoragePath(args.destination).mkdirs()
        upload_xla_dumps(args.stage, args.destination, uploaded)
    finally:
        _write_json(index, {str(path): signature for path, signature in uploaded.items()})


if __name__ == "__main__":
    main()
