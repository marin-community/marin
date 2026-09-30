"""Bounded cgroup telemetry for Daytona sandboxes.

This module intentionally reads only fixed cgroup files.  It never collects an
environment, process arguments, ``nproc``, or ``/proc/meminfo``: host-visible
CPU/RAM values are not resource-limit evidence.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable

from .daytona_resources import DaytonaResourceProfile

_MAX_RAW_BYTES = 4096
CGROUP_PATHS = (
    "/sys/fs/cgroup/cpu.max",
    "/sys/fs/cgroup/memory.max",
    "/sys/fs/cgroup/memory.current",
    "/sys/fs/cgroup/memory.peak",
    "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
    "/sys/fs/cgroup/cpu/cpu.cfs_period_us",
    "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    "/sys/fs/cgroup/memory/memory.usage_in_bytes",
    "/sys/fs/cgroup/memory/memory.max_usage_in_bytes",
)

# Tab-delimited records allow a POSIX shell to return a bounded, parseable probe
# without requiring Python or another image dependency inside a candidate image.
CGROUP_PROBE_COMMAND = r"""
for cap_path in \
  /sys/fs/cgroup/cpu.max \
  /sys/fs/cgroup/memory.max \
  /sys/fs/cgroup/memory.current \
  /sys/fs/cgroup/memory.peak \
  /sys/fs/cgroup/cpu/cpu.cfs_quota_us \
  /sys/fs/cgroup/cpu/cpu.cfs_period_us \
  /sys/fs/cgroup/memory/memory.limit_in_bytes \
  /sys/fs/cgroup/memory/memory.usage_in_bytes \
  /sys/fs/cgroup/memory/memory.max_usage_in_bytes
do
  if [ -r "$cap_path" ]; then
    cap_value=$(cat "$cap_path" 2>/dev/null) || continue
    printf '%s\t%s\n' "$cap_path" "$cap_value"
  fi
done
""".strip()


_V1_MEMORY_UNLIMITED_FLOOR = 1 << 60


def _finite_cpu_limit(value: str | None) -> bool:
    """Accept only a positive cgroup quota and period, never max/unlimited."""
    if not isinstance(value, str):
        return False
    quota, *rest = value.split()
    if len(rest) != 1:
        return False
    period = rest[0]
    try:
        return int(quota) > 0 and int(period) > 0
    except ValueError:
        return False


def _finite_memory_limit(value: str | None) -> bool:
    """Accept bounded positive byte counts; v1's huge sentinel remains unknown."""
    if not isinstance(value, str) or not value.isdecimal():
        return False
    amount = int(value)
    return 0 < amount < _V1_MEMORY_UNLIMITED_FLOOR


def _records(raw: str) -> dict[str, str]:
    records: dict[str, str] = {}
    for line in raw.splitlines():
        path, separator, value = line.partition("\t")
        if not separator or path not in CGROUP_PATHS:
            continue
        value = value.strip()
        if value and len(value) <= 128:
            records[path] = value
    return records


def parse_cgroup_telemetry(raw: str, profile: DaytonaResourceProfile) -> dict:
    """Turn the fixed probe output into evidence without asserting enforcement."""
    if not isinstance(raw, str):
        raise TypeError("cgroup telemetry must be text")
    encoded = raw.encode()
    truncated = len(encoded) > _MAX_RAW_BYTES
    raw = encoded[:_MAX_RAW_BYTES].decode(errors="replace")
    records = _records(raw)
    v2 = any(
        path in records
        for path in (
            "/sys/fs/cgroup/cpu.max",
            "/sys/fs/cgroup/memory.max",
            "/sys/fs/cgroup/memory.current",
            "/sys/fs/cgroup/memory.peak",
        )
    )
    v1 = any("/sys/fs/cgroup/cpu/" in path or "/memory/" in path for path in records)
    if v2:
        cpu_limit = records.get("/sys/fs/cgroup/cpu.max")
        memory_limit = records.get("/sys/fs/cgroup/memory.max")
        memory_current = records.get("/sys/fs/cgroup/memory.current")
        memory_peak = records.get("/sys/fs/cgroup/memory.peak")
        version = "v2"
    elif v1:
        quota = records.get("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        period = records.get("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        cpu_limit = f"{quota} {period}" if quota and period else None
        memory_limit = records.get("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        memory_current = records.get("/sys/fs/cgroup/memory/memory.usage_in_bytes")
        memory_peak = records.get("/sys/fs/cgroup/memory/memory.max_usage_in_bytes")
        version = "v1"
    else:
        cpu_limit = memory_limit = memory_current = memory_peak = None
        version = "unavailable"

    finite_cpu = _finite_cpu_limit(cpu_limit)
    finite_memory = _finite_memory_limit(memory_limit)
    cpu_kind = (
        "cgroup_v2_quota_period"
        if version == "v2"
        else "cgroup_v1_quota_period"
        if version == "v1"
        else "unavailable"
    )
    memory_kind = (
        "cgroup_v2_bytes"
        if version == "v2"
        else "cgroup_v1_bytes"
        if version == "v1"
        else "unavailable"
    )
    return {
        "schema_version": "capability-daytona-cgroup-telemetry-v1",
        "requested_resource_profile": profile.receipt(),
        "collection": {
            "command": "fixed_cgroup_file_probe",
            "raw": raw,
            "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "raw_truncated": truncated,
            "host_visibility": "not_collected_not_used_for_limits",
        },
        "cgroup_version": version,
        "observed": {
            "cpu_max": cpu_limit,
            "memory_max": memory_limit,
            "memory_current": memory_current,
            "memory_peak": memory_peak,
        },
        "comparison": {
            "cpu": {
                "requested_cores": profile.cpu,
                "observed_kind": cpu_kind,
                "observed_finite": finite_cpu,
                "state": "observed_not_provider_verified",
            },
            "memory": {
                "requested_gb": profile.memory_gb,
                "observed_kind": memory_kind,
                "observed_finite": finite_memory,
                "state": "observed_not_provider_verified",
            },
            "resource_envelope": "not_verified_by_telemetry_plumbing",
        },
        "limit_evidence": {
            "cpu": "finite_cgroup_limit" if finite_cpu else "unavailable_or_unlimited",
            "memory": "finite_cgroup_limit"
            if finite_memory
            else "unavailable_or_unlimited",
            # Requested GB cannot be equated to a cgroup byte count without a
            # provider-unit attestation.  A later resource campaign may make
            # that comparison explicitly, but this plumbing does not.
            "requested_profile_verified": False,
        },
    }


def collect_cgroup_telemetry(
    execute: Callable[[str], dict], profile: DaytonaResourceProfile
) -> dict:
    """Collect once through the sandbox executor; provider errors stay explicit."""
    started = time.monotonic()
    try:
        result = execute(CGROUP_PROBE_COMMAND)
    except Exception as error:  # noqa: BLE001 - provider failures are receipt data
        return {
            **parse_cgroup_telemetry("", profile),
            "collection_error_type": type(error).__name__,
            "collection_seconds": round(time.monotonic() - started, 3),
        }
    stdout = result.get("stdout", "") if isinstance(result, dict) else ""
    receipt = parse_cgroup_telemetry(stdout if isinstance(stdout, str) else "", profile)
    receipt["collection_seconds"] = round(time.monotonic() - started, 3)
    if not isinstance(result, dict) or result.get("exit") != 0:
        receipt["collection_error_type"] = "ProbeCommandFailed"
    return receipt


def lifecycle_telemetry(startup: dict) -> dict:
    """Open a two-sample receipt; a startup sample is never a workload peak."""
    startup = dict(startup)
    startup["phase"] = "startup"
    return {
        "schema_version": "capability-daytona-resource-telemetry-v2",
        "startup": startup,
        "final": {
            "phase": "final",
            "state": "not_collected_stop_not_observed",
        },
        "full_trial_lifecycle_observed": False,
    }


def finalize_lifecycle_telemetry(telemetry: dict, final: dict, *, delete: bool) -> dict:
    """Attach terminal telemetry without treating a failed probe as a limit result."""
    result = dict(telemetry)
    final = dict(final)
    final["phase"] = "final"
    final["state"] = (
        "collected" if "collection_error_type" not in final else "probe_failed"
    )
    result["final"] = final
    result["full_trial_lifecycle_observed"] = True
    result["sandbox_delete_requested"] = delete
    return result
