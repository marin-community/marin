# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Encode typed profiling configuration at legacy RPC boundaries."""

from iris.resources.endpoint import (
    CpuProfileConfiguration,
    CpuProfileFormat,
    MemoryProfileConfiguration,
    MemoryProfileFormat,
    ProfileConfiguration,
    ThreadsProfileConfiguration,
)
from iris.rpc import job_pb2

_CPU_FORMAT_FROM_PROTO = {
    job_pb2.CpuProfile.FORMAT_UNSPECIFIED: CpuProfileFormat.UNSPECIFIED,
    job_pb2.CpuProfile.FLAMEGRAPH: CpuProfileFormat.FLAMEGRAPH,
    job_pb2.CpuProfile.SPEEDSCOPE: CpuProfileFormat.SPEEDSCOPE,
    job_pb2.CpuProfile.RAW: CpuProfileFormat.RAW,
}
_CPU_FORMAT_TO_PROTO = {value: key for key, value in _CPU_FORMAT_FROM_PROTO.items()}
_MEMORY_FORMAT_FROM_PROTO = {
    job_pb2.MemoryProfile.FORMAT_UNSPECIFIED: MemoryProfileFormat.UNSPECIFIED,
    job_pb2.MemoryProfile.FLAMEGRAPH: MemoryProfileFormat.FLAMEGRAPH,
    job_pb2.MemoryProfile.TABLE: MemoryProfileFormat.TABLE,
    job_pb2.MemoryProfile.STATS: MemoryProfileFormat.STATS,
    job_pb2.MemoryProfile.RAW: MemoryProfileFormat.RAW,
}
_MEMORY_FORMAT_TO_PROTO = {value: key for key, value in _MEMORY_FORMAT_FROM_PROTO.items()}


def profile_configuration_from_proto(profile: job_pb2.ProfileType) -> ProfileConfiguration | None:
    """Decode a legacy profile oneof into its typed resource representation."""
    match profile.WhichOneof("profiler"):
        case "cpu":
            return CpuProfileConfiguration(
                format=_CPU_FORMAT_FROM_PROTO.get(profile.cpu.format, CpuProfileFormat.UNSPECIFIED),
                rate_hz=profile.cpu.rate_hz,
                native=profile.cpu.native if profile.cpu.HasField("native") else None,
            )
        case "memory":
            return MemoryProfileConfiguration(
                format=_MEMORY_FORMAT_FROM_PROTO.get(profile.memory.format, MemoryProfileFormat.UNSPECIFIED),
                leaks=profile.memory.leaks,
            )
        case "threads":
            return ThreadsProfileConfiguration(include_locals=profile.threads.locals)
        case _:
            return None


def profile_configuration_to_proto(profile: ProfileConfiguration | None) -> job_pb2.ProfileType:
    """Encode typed profiling configuration for a legacy worker/controller RPC."""
    if isinstance(profile, CpuProfileConfiguration):
        cpu = job_pb2.CpuProfile(format=_CPU_FORMAT_TO_PROTO[profile.format], rate_hz=profile.rate_hz)
        if profile.native is not None:
            cpu.native = profile.native
        return job_pb2.ProfileType(cpu=cpu)
    if isinstance(profile, MemoryProfileConfiguration):
        return job_pb2.ProfileType(
            memory=job_pb2.MemoryProfile(format=_MEMORY_FORMAT_TO_PROTO[profile.format], leaks=profile.leaks)
        )
    if isinstance(profile, ThreadsProfileConfiguration):
        return job_pb2.ProfileType(threads=job_pb2.ThreadsProfile(locals=profile.include_locals))
    return job_pb2.ProfileType()
