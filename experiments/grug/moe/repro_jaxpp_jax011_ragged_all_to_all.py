# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Minimal JaxPP and JAX 0.11 ragged-all-to-all regression boundary.

The two-stage combined case uses two MPMD ranks with two devices per rank.
Stage 0 produces and transfers a four-row int32 payload. Stage 1 exchanges its
two local rows over a two-device ``ragged_all_to_all`` and checks every returned
value.

Run the direct ragged-all-to-all and JaxPP transfer-only positive controls
before the combined ``jaxpp-ragged`` case. The four-stage cases add a
forward/backward transfer chain across four MPMD ranks, with either identity
stage tasks or one exact ragged exchange per stage task. The companion README
pins the exact H100 environment and command.
"""

from __future__ import annotations

import argparse
import contextlib
import faulthandler
import importlib.metadata
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxpp import jax_compat as jaxpp_compat  # pyrefly: ignore[missing-import]
from jaxpp.experimental import mpmd as jaxpp_mpmd  # pyrefly: ignore[missing-import]

JAX_VERSION = "0.11.1.dev20260725"
JAXPP_VERSION = "0.10.2"
JAXPP_REVISION = "7091a9b5ce02cd1a6bdc905f6a36e89370a5fba9"
NCCL_VERSION = "2.30.7"
TWO_STAGE_MPMD_RANKS = 2
FOUR_STAGE_MPMD_RANKS = 4
DEVICES_PER_STAGE = 2
GLOBAL_SHAPE = (4, 1)
LOCAL_ROWS = 2
PEERS = 2
EXPECTED_CHECKSUM = 202

Case = Literal[
    "direct-ragged",
    "jaxpp-transfer",
    "jaxpp-ragged",
    "jaxpp-four-stage-transfer",
    "jaxpp-four-stage-ragged",
]


def event(name: str, **fields: Any) -> None:
    print(
        json.dumps(
            {
                "time": time.time(),
                "event": name,
                "pid": os.getpid(),
                **fields,
            },
            sort_keys=True,
            default=str,
        ),
        flush=True,
    )


@contextlib.contextmanager
def phase(name: str, *, process_id: int) -> Iterator[None]:
    event("phase_entered", phase=name, process_id=process_id)
    started = time.perf_counter()
    try:
        yield
    except BaseException as error:
        event(
            "phase_failed",
            phase=name,
            process_id=process_id,
            elapsed=time.perf_counter() - started,
            error_type=type(error).__name__,
            error=str(error),
        )
        raise
    event("phase_returned", phase=name, process_id=process_id, elapsed=time.perf_counter() - started)


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def jaxpp_revision() -> str:
    source = os.environ.get("JAXPP_SOURCE")
    if source:
        return subprocess.check_output(
            ("git", "-C", source, "rev-parse", "HEAD"),
            text=True,
        ).strip()

    distribution = importlib.metadata.distribution("jaxpp")
    direct_url = distribution.read_text("direct_url.json")
    if direct_url is None:
        return "unknown"
    return json.loads(direct_url).get("vcs_info", {}).get("commit_id", "unknown")


def environment() -> dict[str, Any]:
    inline_value = jaxpp_compat.canonicalize_pjit_inline(False)
    backend = jax.extend.backend.get_backend()
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "backend": backend.platform,
        "backend_platform_version": backend.platform_version,
        "devices": [str(device) for device in jax.devices()],
        "jax": jax.__version__,
        "jaxlib": package_version("jaxlib"),
        "jax_cuda13_plugin": package_version("jax-cuda13-plugin"),
        "jax_cuda13_pjrt": package_version("jax-cuda13-pjrt"),
        "jaxpp": package_version("jaxpp"),
        "jaxpp_revision": jaxpp_revision(),
        "jaxpp_inline_value": repr(inline_value),
        "jaxpp_inline_type": type(inline_value).__qualname__,
        "nvidia_nccl_cu13": package_version("nvidia-nccl-cu13"),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "xla_python_client_mem_fraction": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", ""),
    }


def check_environment(case: Case) -> dict[str, Any]:
    actual = environment()
    expected = {
        "jax": JAX_VERSION,
        "jaxlib": JAX_VERSION,
        "jax_cuda13_plugin": JAX_VERSION,
        "jax_cuda13_pjrt": JAX_VERSION,
        "jaxpp": JAXPP_VERSION,
        "jaxpp_revision": JAXPP_REVISION,
        "nvidia_nccl_cu13": NCCL_VERSION,
    }
    mismatches = {
        name: {"expected": value, "actual": actual[name]} for name, value in expected.items() if actual[name] != value
    }
    if mismatches:
        raise RuntimeError(f"package version mismatch: {mismatches}")
    if actual["backend"] != "gpu":
        raise RuntimeError(f"{case} requires GPU JAX, got {actual['backend']}")
    if actual["jaxpp_inline_type"] == "bool":
        raise RuntimeError("the JaxPP JAX 0.11 inline compatibility patch is not active")
    return actual


@dataclass(frozen=True)
class Config:
    case: Case
    coordinator_port: int
    stack_after: float
    timeout: float


def start_watchdog(config: Config, *, process_id: int) -> threading.Event:
    faulthandler.enable()
    stop = threading.Event()

    def watch() -> None:
        started = time.monotonic()
        next_stack = config.stack_after
        while True:
            elapsed = time.monotonic() - started
            deadline = min(next_stack, config.timeout)
            if stop.wait(max(0.0, deadline - elapsed)):
                return
            elapsed = time.monotonic() - started
            if elapsed >= config.timeout:
                event("watchdog_timeout", process_id=process_id, timeout=config.timeout)
                faulthandler.dump_traceback(all_threads=True)
                os._exit(124)
            event("watchdog_stack", process_id=process_id, elapsed=elapsed)
            faulthandler.dump_traceback(all_threads=True)
            next_stack += config.stack_after

    thread = threading.Thread(target=watch, name=f"watchdog-rank-{process_id}", daemon=True)
    thread.start()
    event(
        "watchdog_started",
        process_id=process_id,
        stack_after=config.stack_after,
        timeout=config.timeout,
    )
    return stop


def payload() -> np.ndarray:
    return np.asarray([[0], [1], [100], [101]], dtype=np.int32)


def mpmd_ranks(case: Case) -> int:
    if case.startswith("jaxpp-four-stage-"):
        return FOUR_STAGE_MPMD_RANKS
    return TWO_STAGE_MPMD_RANKS


def local_transfer_check(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    source = jax.lax.axis_index("fsdp")
    expected = source * 100 + jnp.arange(LOCAL_ROWS, dtype=jnp.int32)
    mismatches = jnp.sum(x[:, 0] != expected, dtype=jnp.int32)
    checksum = jnp.sum(x, dtype=jnp.int32)
    return jax.lax.psum(mismatches, "fsdp"), jax.lax.psum(checksum, "fsdp")


def local_ragged_transform(x: jax.Array) -> jax.Array:
    source = jax.lax.axis_index("fsdp")
    offsets = jnp.arange(PEERS, dtype=jnp.int32)
    sizes = jnp.ones((PEERS,), dtype=jnp.int32)
    output_offsets = jnp.full((PEERS,), source, dtype=jnp.int32)
    return jax.lax.ragged_all_to_all(
        x,
        jnp.zeros_like(x),
        offsets,
        sizes,
        output_offsets,
        sizes,
        axis_name="fsdp",
    )


def local_ragged_check(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    output = local_ragged_transform(x)
    destination = jax.lax.axis_index("fsdp")
    expected = jnp.arange(PEERS, dtype=jnp.int32) * 100 + destination
    mismatches = jnp.sum(output[:, 0] != expected, dtype=jnp.int32)
    checksum = jnp.sum(output, dtype=jnp.int32)
    return jax.lax.psum(mismatches, "fsdp"), jax.lax.psum(checksum, "fsdp")


def exact_check(result: tuple[jax.Array, jax.Array], *, case: Case, process_id: int) -> None:
    mismatches, checksum = (int(value) for value in jax.device_get(result))
    event(
        "exact_check",
        case=case,
        process_id=process_id,
        mismatch_count=mismatches,
        checksum=checksum,
        expected_checksum=EXPECTED_CHECKSUM,
    )
    if mismatches != 0:
        raise AssertionError(f"{case}: {mismatches} payload elements differ")
    if checksum != EXPECTED_CHECKSUM:
        raise AssertionError(f"{case}: expected checksum {EXPECTED_CHECKSUM}, got {checksum}")


def run_direct_ragged(config: Config) -> None:
    process_id = 0
    watchdog = start_watchdog(config, process_id=process_id)
    try:
        devices = jax.devices()
        if len(devices) != DEVICES_PER_STAGE:
            raise ValueError(f"direct-ragged requires {DEVICES_PER_STAGE} visible GPUs, got {len(devices)}")
        mesh = Mesh(
            np.asarray(devices, dtype=object),
            ("fsdp",),
            axis_types=(AxisType.Explicit,),
        )
        rows = NamedSharding(mesh, P("fsdp", None))
        replicated = NamedSharding(mesh, P())
        transfer = jax.jit(
            jax.shard_map(
                local_ragged_check,
                mesh=mesh,
                in_specs=P("fsdp", None),
                out_specs=(P(), P()),
                check_vma=False,
            ),
            out_shardings=(replicated, replicated),
        )
        x = jax.device_put(payload(), rows)
        with phase("direct_ragged_lower_compile_execute", process_id=process_id):
            result = transfer(x)
            jax.block_until_ready(result)
        exact_check(result, case=config.case, process_id=process_id)
        event("case_passed", case=config.case, process_id=process_id)
    finally:
        watchdog.set()


def empty_array(shape: tuple[int, ...], dtype: Any, sharding: NamedSharding) -> jax.Array:
    return jax.make_array_from_single_device_arrays(shape, sharding, [], dtype=dtype)


def initialize_mpmd_payload(sharding: NamedSharding, *, owns_stage: bool) -> jax.Array:
    if owns_stage:
        return jax.device_put(payload(), sharding)
    return empty_array(GLOBAL_SHAPE, jnp.int32, sharding)


def stage_payload_task(mesh: Mesh, *, ragged: bool):
    def local_stage(x: jax.Array) -> jax.Array:
        if ragged:
            return local_ragged_transform(x)
        return x

    def stage(x: jax.Array) -> jax.Array:
        return jax.shard_map(
            local_stage,
            mesh=mesh,
            in_specs=P("fsdp", None),
            out_specs=P("fsdp", None),
            check_vma=False,
        )(x)

    return stage


def four_stage_program(
    stage_meshes: tuple[Mesh, ...],
    stage_rows: tuple[NamedSharding, ...],
    stage_scalars: tuple[NamedSharding, ...],
    x: jax.Array,
    *,
    ragged: bool,
) -> tuple[jax.Array, jax.Array]:
    stage_tasks = tuple(stage_payload_task(mesh, ragged=ragged) for mesh in stage_meshes)
    value = jaxpp_mpmd.task(
        stage_tasks[0],
        name="stage0_forward",
        out_shardings=stage_rows[0],
    )(x)
    for stage_index in range(1, len(stage_meshes)):
        value = jaxpp_mpmd.transfer(value, out_shardings=stage_rows[stage_index]).done()
        value = jaxpp_mpmd.task(
            stage_tasks[stage_index],
            name=f"stage{stage_index}_forward",
            out_shardings=stage_rows[stage_index],
        )(value)

    value = jaxpp_mpmd.task(
        stage_tasks[-1],
        name=f"stage{len(stage_meshes) - 1}_backward",
        out_shardings=stage_rows[-1],
    )(value)
    for stage_index in reversed(range(len(stage_meshes) - 1)):
        value = jaxpp_mpmd.transfer(value, out_shardings=stage_rows[stage_index]).done()
        value = jaxpp_mpmd.task(
            stage_tasks[stage_index],
            name=f"stage{stage_index}_backward",
            out_shardings=stage_rows[stage_index],
        )(value)

    def stage0_check(value: jax.Array) -> tuple[jax.Array, jax.Array]:
        return jax.shard_map(
            local_transfer_check,
            mesh=stage_meshes[0],
            in_specs=P("fsdp", None),
            out_specs=(P(), P()),
            check_vma=False,
        )(value)

    return jaxpp_mpmd.task(
        stage0_check,
        name="stage0_exact_check",
        out_shardings=(stage_scalars[0], stage_scalars[0]),
    )(value)


def run_jaxpp_worker(config: Config, process_id: int, local_device_ids: list[int]) -> None:
    event("worker_started", case=config.case, process_id=process_id, local_device_ids=local_device_ids)
    watchdog: threading.Event | None = None
    try:
        num_mpmd_ranks = mpmd_ranks(config.case)
        with phase("distributed_initialize", process_id=process_id):
            jax.distributed.initialize(
                coordinator_address=f"127.0.0.1:{config.coordinator_port}",
                num_processes=num_mpmd_ranks,
                process_id=process_id,
                local_device_ids=local_device_ids,
                cluster_detection_method="deactivate",
            )
        watchdog = start_watchdog(config, process_id=process_id)

        devices = np.asarray(jax.devices(), dtype=object).reshape(num_mpmd_ranks, DEVICES_PER_STAGE)
        mpmd_mesh = jaxpp_mpmd.MpmdMesh(
            Mesh(
                devices,
                ("pp", "fsdp"),
                axis_types=(AxisType.Explicit, AxisType.Explicit),
            ),
            "pp",
        )
        stage_meshes = tuple(mpmd_mesh.unstack)
        stage_rows = tuple(NamedSharding(mesh, P("fsdp", None)) for mesh in stage_meshes)
        stage_scalars = tuple(NamedSharding(mesh, P()) for mesh in stage_meshes)
        stage0_rows = stage_rows[0]

        if config.case.startswith("jaxpp-four-stage-"):

            @jaxpp_mpmd.mpmd(
                mpmd_mesh,
                in_shardings=(stage0_rows,),
                infer_donation=False,
            )
            def program(x: jax.Array) -> tuple[jax.Array, jax.Array]:
                return four_stage_program(
                    stage_meshes,
                    stage_rows,
                    stage_scalars,
                    x,
                    ragged=config.case == "jaxpp-four-stage-ragged",
                )

            result_stage_index = 0
        else:
            stage1_mesh = stage_meshes[1]
            stage1_rows = stage_rows[1]
            stage1_scalar = stage_scalars[1]
            check = local_ragged_check if config.case == "jaxpp-ragged" else local_transfer_check

            def stage1_check(x: jax.Array) -> tuple[jax.Array, jax.Array]:
                return jax.shard_map(
                    check,
                    mesh=stage1_mesh,
                    in_specs=P("fsdp", None),
                    out_specs=(P(), P()),
                    check_vma=False,
                )(x)

            @jaxpp_mpmd.mpmd(
                mpmd_mesh,
                in_shardings=(stage0_rows,),
                infer_donation=False,
            )
            def program(x: jax.Array) -> tuple[jax.Array, jax.Array]:
                produced = jaxpp_mpmd.task(
                    lambda value: value,
                    name="stage0_produce",
                    out_shardings=stage0_rows,
                )(x)
                transferred = jaxpp_mpmd.transfer(produced, out_shardings=stage1_rows).done()
                return jaxpp_mpmd.task(
                    stage1_check,
                    name=f"stage1_{config.case}",
                    out_shardings=(stage1_scalar, stage1_scalar),
                )(transferred)

            result_stage_index = 1
        input_struct = jax.ShapeDtypeStruct(GLOBAL_SHAPE, jnp.int32, sharding=stage0_rows)
        with phase("jaxpp_lower", process_id=process_id):
            lowered = program.lower(input_struct)

        x = initialize_mpmd_payload(stage0_rows, owns_stage=mpmd_mesh.my_mpmd_axis_index == 0)
        with phase("jaxpp_eval_local", process_id=process_id):
            result = lowered(x)
            jax.block_until_ready(result)
        if mpmd_mesh.my_mpmd_axis_index == result_stage_index:
            exact_check(result, case=config.case, process_id=process_id)

        with phase("completion_barrier", process_id=process_id):
            multihost_utils.sync_global_devices(f"{config.case}_complete")
        event("case_passed", case=config.case, process_id=process_id)
    except BaseException as error:
        event(
            "worker_failed",
            case=config.case,
            process_id=process_id,
            error_type=type(error).__name__,
            error=str(error),
        )
        raise
    finally:
        if jax.distributed.is_initialized():
            with phase("distributed_shutdown", process_id=process_id):
                jax.distributed.shutdown()
        if watchdog is not None:
            watchdog.set()


def run_jaxpp(config: Config) -> None:
    num_mpmd_ranks = mpmd_ranks(config.case)
    expected_devices = num_mpmd_ranks * DEVICES_PER_STAGE
    if len(jax.devices()) != expected_devices:
        raise ValueError(f"{config.case} requires {expected_devices} visible GPUs, got {len(jax.devices())}")

    context = mp.get_context("spawn")
    processes: list[mp.Process] = []
    try:
        for process_id in range(num_mpmd_ranks):
            local_device_ids = list(
                range(
                    process_id * DEVICES_PER_STAGE,
                    (process_id + 1) * DEVICES_PER_STAGE,
                )
            )
            process = context.Process(
                target=run_jaxpp_worker,
                args=(config, process_id, local_device_ids),
                name=f"jaxpp-rank-{process_id}",
            )
            process.start()
            processes.append(process)
            event("worker_spawned", process_id=process_id, child_pid=process.pid)

        deadline = time.monotonic() + config.timeout + 30
        while any(process.is_alive() for process in processes):
            failed = next((process for process in processes if process.exitcode not in (None, 0)), None)
            if failed is not None:
                code = failed.exitcode
                event("worker_nonzero_exit", child_pid=failed.pid, exitcode=code)
                raise SystemExit(128 - code if code is not None and code < 0 else code)
            if time.monotonic() >= deadline:
                raise TimeoutError("parent cleanup deadline exceeded")
            threading.Event().wait(0.2)

        failed = next((process for process in processes if process.exitcode), None)
        if failed is not None:
            raise SystemExit(failed.exitcode)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=10)
            event("worker_exited", child_pid=process.pid, exitcode=process.exitcode)
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join()


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=(
            "direct-ragged",
            "jaxpp-transfer",
            "jaxpp-ragged",
            "jaxpp-four-stage-transfer",
            "jaxpp-four-stage-ragged",
        ),
        required=True,
    )
    parser.add_argument("--coordinator-port", type=int, default=5831)
    parser.add_argument("--stack-after", type=float, default=30.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    return Config(**vars(args))


def main() -> None:
    config = parse_args()
    actual_environment = check_environment(config.case)
    event(
        "environment",
        config=asdict(config),
        environment=actual_environment,
        script=str(Path(__file__).resolve()),
        topology={
            "mpmd_ranks": mpmd_ranks(config.case) if config.case.startswith("jaxpp-") else 1,
            "devices_per_stage": DEVICES_PER_STAGE,
            "global_shape": GLOBAL_SHAPE,
            "dtype": "int32",
        },
    )
    if config.case == "direct-ragged":
        run_direct_ragged(config)
    else:
        run_jaxpp(config)


if __name__ == "__main__":
    main()
