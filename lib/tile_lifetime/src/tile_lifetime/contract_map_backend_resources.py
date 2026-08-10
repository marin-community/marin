# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Artifact and logical-boundary records for Contract/Map GPU evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tile_lifetime.cuda_contract_map_backend_codegen import (
    ContractMapBackendBuffer,
    GeneratedCudaContractMapBackendFfi,
)

_FUNCTION = re.compile(r"Function properties for (?P<name>[^\s]+)")
_SPILLS = re.compile(
    r"(?P<stack>\d+) bytes stack frame, (?P<stores>\d+) bytes spill stores, (?P<loads>\d+) bytes spill loads"
)
_REGISTERS = re.compile(r"Used (?P<registers>\d+) registers")
_SHARED = re.compile(r"(?P<bytes>\d+) bytes smem")


@dataclass(frozen=True)
class ContractMapCompilePlan:
    """Deterministic commands that retain every required CUDA artifact."""

    source_path: Path
    shared_library_path: Path
    ptx_path: Path
    cubin_path: Path
    sass_path: Path
    shared_library_command: tuple[str, ...]
    ptx_command: tuple[str, ...]
    cubin_command: tuple[str, ...]
    sass_command: tuple[str, ...]


@dataclass(frozen=True)
class PtxasKernelResources:
    """Compiler resource facts for one generated CUDA kernel."""

    kernel_name: str
    registers_per_thread: int
    spill_load_bytes: int
    spill_store_bytes: int
    stack_frame_bytes: int
    static_shared_bytes: int


@dataclass(frozen=True)
class ContractMapLogicalBoundary:
    """Expected logical accounting, kept separate from measured evidence."""

    input_layouts: tuple[str, ...]
    output_layouts: tuple[str, ...]
    layout_adapters: tuple[dict[str, str | bool], ...]
    materialized_copies: tuple[dict[str, str | int], ...]
    saved_state_names_and_bytes: dict[str, int]
    recompute_operations: tuple[dict[str, str | int], ...]
    transposes: tuple[dict[str, str | bool | list[int]], ...]
    bitcasts: tuple[dict[str, str | list[int]], ...]

    def to_evidence(self) -> dict[str, Any]:
        """Return the exact closed logical-boundary JSON shape."""
        return {
            "input_layouts": list(self.input_layouts),
            "output_layouts": list(self.output_layouts),
            "layout_adapters": list(self.layout_adapters),
            "materialized_copies": list(self.materialized_copies),
            "saved_state_names_and_bytes": dict(self.saved_state_names_and_bytes),
            "recompute_operations": list(self.recompute_operations),
            "transposes": list(self.transposes),
            "bitcasts": list(self.bitcasts),
        }


def contract_map_compile_plan(
    generated: GeneratedCudaContractMapBackendFfi,
    *,
    artifact_directory: Path,
    nvcc: Path,
    include_directory: Path,
    architecture: str = "sm_90a",
) -> ContractMapCompilePlan:
    """Form compile/disassembly commands without running a tool or GPU."""
    if architecture != "sm_90a":
        raise ValueError("the reviewed backend slice is limited to sm_90a")
    stem = f"contract_map_{generated.policy.value}_{generated.physical_digest}"
    source_path = artifact_directory / f"{stem}.cu"
    shared_library_path = artifact_directory / f"{stem}.so"
    ptx_path = artifact_directory / f"{stem}.ptx"
    cubin_path = artifact_directory / f"{stem}.cubin"
    sass_path = artifact_directory / f"{stem}.sass"
    common = (
        str(nvcc),
        "-std=c++17",
        "-O3",
        "-lineinfo",
        "--ptxas-options=-v",
        f"-arch={architecture}",
        "-I",
        str(include_directory),
        str(source_path),
    )
    shared = (*common, "-shared", "-Xcompiler", "-fPIC", "-o", str(shared_library_path))
    ptx = (*common, "--ptx", "-o", str(ptx_path))
    cubin = (*common, "--cubin", "-o", str(cubin_path))
    sass = (str(nvcc.with_name("cuobjdump")), "--dump-sass", str(cubin_path))
    return ContractMapCompilePlan(
        source_path=source_path,
        shared_library_path=shared_library_path,
        ptx_path=ptx_path,
        cubin_path=cubin_path,
        sass_path=sass_path,
        shared_library_command=shared,
        ptx_command=ptx,
        cubin_command=cubin,
        sass_command=sass,
    )


def parse_ptxas_kernel_resources(
    output: str,
    *,
    expected_kernel_names: tuple[str, ...],
) -> tuple[PtxasKernelResources, ...]:
    """Parse ptxas resource text and require every generated kernel exactly once."""
    records: dict[str, PtxasKernelResources] = {}
    sections = _ptxas_sections(output)
    for name, section in sections:
        if name not in expected_kernel_names:
            continue
        if name in records:
            raise ValueError(f"ptxas output repeats generated kernel {name!r}")
        spills = _SPILLS.search(section)
        registers = _REGISTERS.search(section)
        if spills is None or registers is None:
            raise ValueError(f"ptxas output omits register or spill resources for {name!r}")
        shared = _SHARED.search(section)
        records[name] = PtxasKernelResources(
            kernel_name=name,
            registers_per_thread=int(registers.group("registers")),
            spill_load_bytes=int(spills.group("loads")),
            spill_store_bytes=int(spills.group("stores")),
            stack_frame_bytes=int(spills.group("stack")),
            static_shared_bytes=int(shared.group("bytes")) if shared is not None else 0,
        )
    missing = tuple(name for name in expected_kernel_names if name not in records)
    if missing:
        raise ValueError(f"ptxas output omits generated kernels: {missing}")
    return tuple(records[name] for name in expected_kernel_names)


def expected_contract_map_logical_boundary(
    generated: GeneratedCudaContractMapBackendFfi,
    *,
    kernel_only: bool,
) -> ContractMapLogicalBoundary:
    """Describe the pre-layout versus full logical step boundary before execution."""

    def layout(buffer: ContractMapBackendBuffer) -> str:
        return f"bf16[{','.join(str(dimension) for dimension in buffer.shape)}]{{1,0}}"

    inputs = tuple(layout(buffer) for buffer in generated.physical_abi.forward_inputs)
    logical_outputs = (*generated.physical_abi.forward_outputs[:1], *generated.physical_abi.reverse_outputs)
    outputs = tuple(layout(buffer) for buffer in logical_outputs)
    bytes_per_bf16 = 2
    saved = {
        buffer.role: bytes_per_bf16 * buffer.shape[0] * buffer.shape[1]
        for buffer in generated.physical_abi.forward_outputs[1:]
    }
    if kernel_only:
        return ContractMapLogicalBoundary(
            input_layouts=inputs,
            output_layouts=outputs,
            layout_adapters=(),
            materialized_copies=(),
            saved_state_names_and_bytes=saved,
            recompute_operations=(),
            transposes=(),
            bitcasts=(),
        )
    return ContractMapLogicalBoundary(
        input_layouts=inputs,
        output_layouts=outputs,
        layout_adapters=(),
        materialized_copies=(),
        saved_state_names_and_bytes=saved,
        recompute_operations=(),
        transposes=(),
        bitcasts=(),
    )


def _ptxas_sections(output: str) -> tuple[tuple[str, str], ...]:
    matches = tuple(_FUNCTION.finditer(output))
    return tuple(
        (
            match.group("name"),
            output[match.start() : matches[index + 1].start() if index + 1 < len(matches) else len(output)],
        )
        for index, match in enumerate(matches)
    )
