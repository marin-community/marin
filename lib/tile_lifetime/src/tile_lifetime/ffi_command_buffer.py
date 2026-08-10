# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate bounded XLA FFI command-buffer candidates."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from enum import StrEnum

_HANDLER_TRAITS_PLACEHOLDER = "__SHUTTLE_FFI_HANDLER_TRAITS__"
_CUSTOM_CALL_COMMAND_BUFFER_FLAG = "--xla_gpu_enable_command_buffer"

_FORBIDDEN_SOURCE_OPERATIONS = {
    "runtime scratch allocation": ("ffi::ScratchAllocator",),
    "runtime device allocation": (
        "cudaMalloc(",
        "cudaMallocAsync(",
        "cudaFree(",
        "cudaFreeAsync(",
        "malloc(",
        "calloc(",
        "realloc(",
        "std::make_unique",
        "std::make_shared",
    ),
    "lazy library handle creation": (
        "cublasCreate(",
        "cublasLtCreate(",
        "cudnnCreate(",
        "std::call_once(",
        "std::once_flag",
    ),
    "runtime autotuning": ("autotun", "algorithm selection"),
    "runtime launch-status query": (
        "cudaGetLastError(",
        "cudaPeekAtLastError(",
        "cudaGetErrorString(",
    ),
    "runtime synchronization": (
        "cudaDeviceSynchronize(",
        "cudaStreamSynchronize(",
    ),
}


class DirectLaunchFfiPhysicalCandidate(StrEnum):
    """Host-side error checking or capture-safe replay for a direct launch."""

    LAUNCH_CHECKED = "launch_checked"
    COMMAND_BUFFER_CAPTURE_SAFE = "command_buffer_capture_safe"

    @property
    def command_buffer_compatible(self) -> bool:
        """Whether the handler may carry XLA's command-buffer trait."""
        return self is DirectLaunchFfiPhysicalCandidate.COMMAND_BUFFER_CAPTURE_SAFE


def direct_launch_status_check(candidate: DirectLaunchFfiPhysicalCandidate, *, operation: str) -> str:
    """Render a launch-status check only for the launch-checked candidate."""
    if candidate is DirectLaunchFfiPhysicalCandidate.COMMAND_BUFFER_CAPTURE_SAFE:
        return ""
    if candidate is DirectLaunchFfiPhysicalCandidate.LAUNCH_CHECKED:
        return f"""  const cudaError_t status = cudaPeekAtLastError();
  if (status != cudaSuccess) {{
    return ffi::Error::Internal("{operation} launch failed: " + std::string(cudaGetErrorString(status)));
  }}"""
    raise ValueError(f"unsupported direct-launch FFI physical candidate: {candidate}")


@dataclass(frozen=True)
class FfiCommandBufferEligibility:
    """Static evidence for one generated FFI handler source."""

    eligible: bool
    forbidden_operations: tuple[str, ...]


@dataclass(frozen=True)
class FfiCommandBufferFlagAudit:
    """Parsed process-start command-buffer selection."""

    uses_xla_default: bool
    selected_entries: tuple[str, ...]


def audit_ffi_command_buffer_eligibility(source: str) -> FfiCommandBufferEligibility:
    """Reject host behavior that cannot safely be replayed as a GPU command graph."""
    lowered = source.lower()
    forbidden = [
        operation
        for operation, tokens in _FORBIDDEN_SOURCE_OPERATIONS.items()
        if any(token.lower() in lowered for token in tokens)
    ]
    if "ffi::platformstream" not in lowered or "<<<" not in source:
        forbidden.append("missing direct stream kernel launch")
    return FfiCommandBufferEligibility(not forbidden, tuple(forbidden))


def finalize_ffi_handler_source(
    template: str,
    *,
    command_buffer_compatible: bool,
    expected_handler_count: int = 1,
) -> str:
    """Fill FFI trait placeholders after validating the generated handler source."""
    if expected_handler_count <= 0:
        raise ValueError("generated FFI source must contain at least one handler")
    if template.count(_HANDLER_TRAITS_PLACEHOLDER) != expected_handler_count:
        raise ValueError(
            f"generated FFI source must contain exactly {expected_handler_count} handler-traits placeholders"
        )
    source_without_traits = template.replace(_HANDLER_TRAITS_PLACEHOLDER, "")
    if not command_buffer_compatible:
        return source_without_traits
    audit = audit_ffi_command_buffer_eligibility(source_without_traits)
    if not audit.eligible:
        raise ValueError("FFI handler is not command-buffer compatible: " + ", ".join(audit.forbidden_operations))
    traits = ",\n    {ffi::Traits::kCmdBufferCompatible}"
    return template.replace(_HANDLER_TRAITS_PLACEHOLDER, traits)


def require_custom_call_command_buffers_enabled(xla_flags: str) -> FfiCommandBufferFlagAudit:
    """Reject startup flags that override XLA's default CUSTOM_CALL support."""
    tokens = shlex.split(xla_flags)
    matching = [
        (index, token) for index, token in enumerate(tokens) if token.startswith(f"{_CUSTOM_CALL_COMMAND_BUFFER_FLAG}=")
    ]
    if len(matching) > 1:
        raise ValueError("XLA_FLAGS contains multiple xla_gpu_enable_command_buffer settings")
    if not matching:
        return FfiCommandBufferFlagAudit(True, ())

    _, token = matching[0]
    value = token.split("=", 1)[1]
    entries = tuple(part.strip() for part in value.split(",") if part.strip())
    if "-CUSTOM_CALL" in entries:
        raise ValueError("XLA_FLAGS explicitly disables CUSTOM_CALL command buffers")

    signed = tuple(entry.startswith(("+", "-")) for entry in entries)
    if entries and any(signed) and not all(signed):
        raise ValueError("xla_gpu_enable_command_buffer mixes absolute and incremental command categories")
    if entries and all(signed):
        return FfiCommandBufferFlagAudit(False, entries)
    if "CUSTOM_CALL" not in entries:
        raise ValueError("XLA_FLAGS command-buffer selection excludes CUSTOM_CALL")
    return FfiCommandBufferFlagAudit(False, entries)
