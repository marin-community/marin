# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Four-GPU feasibility probe for XLA-owned peer-visible FFI buffers."""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import Protocol, TypedDict

import click
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.kernels.mixture_of_kittens.collective_memory_probe import (
    COLLECTIVE_MEMORY_PROBE_TARGET,
    CollectiveMemoryProbeBuildConfig,
    collective_memory_ring_u32,
    initialize_collective_memory_probe,
    memory_space_frontend_attributes,
)

WORLD_SIZE = 4
DEFAULT_ELEMENTS_PER_RANK = 1024
DEFAULT_BUILD_ROOT = "/tmp/marin-collective-memory-probe/build"


class HloEvidence(TypedDict):
    custom_call_lines: list[str]
    collective_memory_line_count: int
    copy_line_count: int
    copy_lines: list[str]
    zero_copy: bool


class CompiledExecutable(Protocol):
    def as_text(self) -> str: ...


def _global_input(elements_per_rank: int, execution: int) -> np.ndarray:
    indices = np.arange(elements_per_rank, dtype=np.uint32)
    return np.concatenate([np.uint32(execution << 24) | np.uint32(rank << 16) | indices for rank in range(WORLD_SIZE)])


def _expected_outputs(elements_per_rank: int, execution: int) -> tuple[np.ndarray, np.ndarray]:
    input_chunks = _global_input(elements_per_rank, execution).reshape(WORLD_SIZE, elements_per_rank)
    remote_read = np.concatenate([input_chunks[(rank + 1) % WORLD_SIZE] for rank in range(WORLD_SIZE)])
    indices = np.arange(elements_per_rank, dtype=np.uint32)
    remote_written = np.concatenate(
        [np.uint32(0xA5000000 | (((rank - 1) % WORLD_SIZE) << 20)) | indices for rank in range(WORLD_SIZE)]
    )
    return remote_read, remote_written


def _assert_exact_outputs(
    actual: tuple[jax.Array, jax.Array],
    *,
    elements_per_rank: int,
    execution: int,
) -> dict[str, object]:
    expected_read, expected_written = _expected_outputs(elements_per_rank, execution)
    actual_read, actual_written = (np.asarray(jax.device_get(value)) for value in actual)
    read_equal = bool(np.array_equal(actual_read, expected_read))
    written_equal = bool(np.array_equal(actual_written, expected_written))
    metrics = {
        "execution": execution,
        "remote_read_exact": read_equal,
        "remote_written_exact": written_equal,
        "remote_read_first_per_rank": actual_read.reshape(WORLD_SIZE, elements_per_rank)[:, 0].tolist(),
        "remote_written_first_per_rank": actual_written.reshape(WORLD_SIZE, elements_per_rank)[:, 0].tolist(),
    }
    if not read_equal or not written_equal:
        raise AssertionError(f"collective-memory ring outputs do not match the peer reference: {metrics}")
    return metrics


def _hlo_evidence(compiled: CompiledExecutable) -> HloEvidence:
    text = compiled.as_text()
    lines = text.splitlines()
    target_lines = [line.strip() for line in lines if COLLECTIVE_MEMORY_PROBE_TARGET in line]
    collective_memory_lines = [line for line in lines if "S(1)" in line]
    copy_lines = [line.strip() for line in lines if re.search(r"\bcopy(?:-start|-done)?\(", line)]
    return {
        "custom_call_lines": target_lines,
        "collective_memory_line_count": len(collective_memory_lines),
        "copy_line_count": len(copy_lines),
        "copy_lines": copy_lines,
        "zero_copy": not copy_lines,
    }


def _dump_evidence(dump_directory: Path | None) -> dict[str, object]:
    if dump_directory is None:
        return {"dump_directory": None, "buffer_assignment_files": []}
    files = sorted(
        str(path) for path in dump_directory.rglob("*") if path.is_file() and "buffer-assignment" in path.name
    )
    return {"dump_directory": str(dump_directory), "buffer_assignment_files": files}


@click.command()
@click.option("--elements-per-rank", type=click.IntRange(min=1, max=(1 << 20) - 1), default=1024, show_default=True)
@click.option("--memory-space", type=click.IntRange(min=0), default=1, show_default=True)
@click.option("--cuda-arch", type=click.Choice(("sm_100a", "sm_103a")), default="sm_100a", show_default=True)
@click.option("--build-root", type=click.Path(path_type=Path), default=Path(DEFAULT_BUILD_ROOT), show_default=True)
@click.option("--dump-directory", type=click.Path(path_type=Path))
@click.option("--concurrent", is_flag=True, help="Invoke the same compiled executable from two host threads.")
def main(
    elements_per_rank: int,
    memory_space: int,
    cuda_arch: str,
    build_root: Path,
    dump_directory: Path | None,
    concurrent: bool,
) -> None:
    """Run an isolated colored-memory ring read/write on four local GPUs."""

    handle = initialize_collective_memory_probe(
        CollectiveMemoryProbeBuildConfig(cache_root=str(build_root), cuda_arch=cuda_arch)
    )
    devices = jax.devices()
    if len(devices) != WORLD_SIZE or any(device.platform != "gpu" for device in devices):
        raise RuntimeError(f"collective-memory probe requires four visible GPUs, got {devices}")
    mesh = Mesh(
        np.asarray(devices),
        ("expert",),
        axis_types=(AxisType.Explicit,),
    )
    sharding = NamedSharding(mesh, P("expert"))

    def local_ring(local_input: jax.Array) -> tuple[jax.Array, jax.Array]:
        return collective_memory_ring_u32(local_input, memory_space=memory_space)

    ring = jax.shard_map(
        local_ring,
        mesh=mesh,
        in_specs=P("expert"),
        out_specs=(P("expert"), P("expert")),
        check_vma=False,
    )
    first_input = jax.device_put(jnp.asarray(_global_input(elements_per_rank, 1)), sharding)
    lowered = jax.jit(ring).lower(first_input)
    stablehlo = str(lowered.compiler_ir("stablehlo"))
    expected_attributes = memory_space_frontend_attributes(memory_space)
    missing_attributes = [value for value in expected_attributes.values() if value not in stablehlo]
    if missing_attributes:
        raise AssertionError(f"StableHLO dropped collective-memory attributes {missing_attributes}: {stablehlo}")
    compiled = lowered.compile()
    hlo_evidence = _hlo_evidence(compiled)
    if memory_space == 1 and not any(line.count("S(1)") >= 2 for line in hlo_evidence["custom_call_lines"]):
        raise AssertionError(
            f"optimized HLO did not assign the custom-call results to collective memory: {hlo_evidence}"
        )
    if memory_space == 0 and not hlo_evidence["zero_copy"]:
        raise AssertionError(f"default-space probe retained optimized-HLO boundary copies: {hlo_evidence}")
    first_metrics = _assert_exact_outputs(
        compiled(first_input),
        elements_per_rank=elements_per_rank,
        execution=1,
    )

    concurrent_metrics: list[dict[str, object]] = []
    if concurrent:
        inputs = tuple(
            jax.device_put(jnp.asarray(_global_input(elements_per_rank, execution)), sharding) for execution in (2, 3)
        )
        start = Barrier(2)

        def execute(execution: int, value: jax.Array) -> dict[str, object]:
            start.wait()
            output = compiled(value)
            jax.block_until_ready(output)
            return _assert_exact_outputs(output, elements_per_rank=elements_per_rank, execution=execution)

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(execute, execution, value) for execution, value in zip((2, 3), inputs, strict=True)
            ]
            concurrent_metrics = [future.result() for future in futures]

    metrics = {
        "probe": "xla_collective_memory_ring_u32",
        "memory_space": memory_space,
        "frontend_attributes": expected_attributes,
        "elements_per_rank": elements_per_rank,
        "library_path": str(handle.library_path),
        "stablehlo_attributes_present": True,
        "single_execution": first_metrics,
        "concurrent_executions": concurrent_metrics,
        "optimized_hlo": hlo_evidence,
        "xla_dump": _dump_evidence(dump_directory),
    }
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
