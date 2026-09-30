from capability_pipeline.daytona_resources import DaytonaResourceProfile
from capability_pipeline.daytona_telemetry import (
    CGROUP_PROBE_COMMAND,
    collect_cgroup_telemetry,
    finalize_lifecycle_telemetry,
    lifecycle_telemetry,
    parse_cgroup_telemetry,
)

PROFILE = DaytonaResourceProfile(2, 2, 10, "adapter_kwargs")


def test_v2_cgroup_telemetry_keeps_raw_receipt_and_never_equates_request_to_limit():
    raw = "\n".join(  # noqa: FLY002
        [
            "/sys/fs/cgroup/cpu.max\t200000 100000",
            "/sys/fs/cgroup/memory.max\t2147483648",
            "/sys/fs/cgroup/memory.current\t131072",
            "/sys/fs/cgroup/memory.peak\t262144",
        ]
    )
    receipt = parse_cgroup_telemetry(raw, PROFILE)
    assert receipt["cgroup_version"] == "v2"
    assert receipt["observed"]["cpu_max"] == "200000 100000"
    assert receipt["observed"]["memory_peak"] == "262144"
    assert receipt["limit_evidence"] == {
        "cpu": "finite_cgroup_limit",
        "memory": "finite_cgroup_limit",
        "requested_profile_verified": False,
    }
    assert receipt["comparison"] == {
        "cpu": {
            "requested_cores": 2,
            "observed_kind": "cgroup_v2_quota_period",
            "observed_finite": True,
            "state": "observed_not_provider_verified",
        },
        "memory": {
            "requested_gb": 2,
            "observed_kind": "cgroup_v2_bytes",
            "observed_finite": True,
            "state": "observed_not_provider_verified",
        },
        "resource_envelope": "not_verified_by_telemetry_plumbing",
    }
    assert receipt["collection"]["raw"] == raw
    assert (
        receipt["collection"]["host_visibility"] == "not_collected_not_used_for_limits"
    )


def test_v1_cgroup_telemetry_uses_v1_equivalents():
    receipt = parse_cgroup_telemetry(
        "\n".join(  # noqa: FLY002
            [
                "/sys/fs/cgroup/cpu/cpu.cfs_quota_us\t200000",
                "/sys/fs/cgroup/cpu/cpu.cfs_period_us\t100000",
                "/sys/fs/cgroup/memory/memory.limit_in_bytes\t2147483648",
                "/sys/fs/cgroup/memory/memory.usage_in_bytes\t1024",
                "/sys/fs/cgroup/memory/memory.max_usage_in_bytes\t4096",
            ]
        ),
        PROFILE,
    )
    assert receipt["cgroup_version"] == "v1"
    assert receipt["observed"] == {
        "cpu_max": "200000 100000",
        "memory_max": "2147483648",
        "memory_current": "1024",
        "memory_peak": "4096",
    }


def test_unlimited_or_malformed_cgroup_values_are_not_finite_limits():
    for cpu, memory in (
        ("max 100000", "max"),
        ("-1 100000", "9223372036854771712"),
        ("0 100000", "0"),
        ("200000 0", "-1"),
        ("bad quota", "not-a-number"),
    ):
        receipt = parse_cgroup_telemetry(
            "\n".join(
                [
                    f"/sys/fs/cgroup/cpu.max\t{cpu}",
                    f"/sys/fs/cgroup/memory.max\t{memory}",
                ]
            ),
            PROFILE,
        )
        assert receipt["limit_evidence"] == {
            "cpu": "unavailable_or_unlimited",
            "memory": "unavailable_or_unlimited",
            "requested_profile_verified": False,
        }
        assert receipt["comparison"]["cpu"]["observed_finite"] is False
        assert receipt["comparison"]["memory"]["observed_finite"] is False


def test_missing_cgroups_are_explicitly_unavailable_not_host_measurements():
    receipt = parse_cgroup_telemetry("48\nMemTotal: 395741628 kB\n", PROFILE)
    assert receipt["cgroup_version"] == "unavailable"
    assert receipt["observed"] == {
        "cpu_max": None,
        "memory_max": None,
        "memory_current": None,
        "memory_peak": None,
    }
    assert receipt["limit_evidence"]["cpu"] == "unavailable_or_unlimited"
    assert "nproc" not in CGROUP_PROBE_COMMAND
    assert "/proc/meminfo" not in CGROUP_PROBE_COMMAND


def test_parser_ignores_any_path_outside_the_fixed_cgroup_allowlist():
    receipt = parse_cgroup_telemetry(
        "/sys/fs/cgroup/cpu.max-not-a-real-file\t200000 100000", PROFILE
    )
    assert receipt["cgroup_version"] == "unavailable"


def test_collection_keeps_provider_failure_as_unavailable_receipt():
    receipt = collect_cgroup_telemetry(
        lambda command: {"exit": 1, "stdout": "", "stderr": "not retained"},
        PROFILE,
    )
    assert receipt["collection_error_type"] == "ProbeCommandFailed"
    assert receipt["limit_evidence"]["requested_profile_verified"] is False


def test_startup_and_final_samples_remain_distinct_and_final_failure_is_explicit():
    startup = collect_cgroup_telemetry(
        lambda command: {"exit": 0, "stdout": "/sys/fs/cgroup/memory.peak\t100"},
        PROFILE,
    )
    receipt = lifecycle_telemetry(startup)
    final = collect_cgroup_telemetry(
        lambda command: {"exit": 1, "stdout": "", "stderr": "ignored"}, PROFILE
    )
    receipt = finalize_lifecycle_telemetry(receipt, final, delete=True)
    assert receipt["startup"]["phase"] == "startup"
    assert receipt["startup"]["observed"]["memory_peak"] == "100"
    assert receipt["final"] == {
        **final,
        "phase": "final",
        "state": "probe_failed",
    }
    assert receipt["full_trial_lifecycle_observed"] is True
    assert receipt["sandbox_delete_requested"] is True
