import pytest

from capability_pipeline.daytona_resources import (
    CANDIDATE_DEFAULT,
    VERIFIER_DEFAULT,
    DaytonaResourceProfile,
    profile_from_receipt,
    resolve_profile,
    snapshot_name,
)


@pytest.mark.parametrize(
    "changed",
    [
        {"cpu": 3, "memory_gb": 8, "disk_gb": 10},
        {"cpu": 4, "memory_gb": 7, "disk_gb": 10},
        {"cpu": 4, "memory_gb": 8, "disk_gb": 9},
    ],
)
def test_snapshot_cache_identity_changes_when_requested_resources_change(changed):
    recipe = "FROM registry.example/task@sha256:" + "a" * 64 + "\n"
    baseline = snapshot_name("cap-harbor", recipe, CANDIDATE_DEFAULT)
    profile = DaytonaResourceProfile(source="adapter_kwargs", **changed)
    assert baseline != snapshot_name("cap-harbor", recipe, profile)
    assert baseline == snapshot_name("cap-harbor", recipe, CANDIDATE_DEFAULT)


def test_explicit_profile_has_documented_provider_units_and_unverified_limits():
    profile = resolve_profile(
        {"cpu": 2, "memory_gb": 2, "disk_gb": 10}, default=CANDIDATE_DEFAULT
    )
    assert profile.receipt() == {
        "cpu": 2,
        "memory_gb": 2,
        "disk_gb": 10,
        "provider_units": {"cpu": "cores", "memory": "GB", "disk": "GB"},
        "source": "adapter_kwargs",
        "effective_limits": "unverified",
    }


def test_resource_receipt_round_trips_exact_units_source_and_limits():
    assert profile_from_receipt(VERIFIER_DEFAULT.receipt()) == VERIFIER_DEFAULT


@pytest.mark.parametrize(
    "receipt",
    [
        {},
        {
            **VERIFIER_DEFAULT.receipt(),
            "provider_units": {"cpu": "threads", "memory": "GB", "disk": "GB"},
        },
        {**VERIFIER_DEFAULT.receipt(), "effective_limits": "verified"},
        {**VERIFIER_DEFAULT.receipt(), "source": ""},
        {**VERIFIER_DEFAULT.receipt(), "memory_gb": True},
        {**VERIFIER_DEFAULT.receipt(), "extra": "not allowed"},
    ],
)
def test_resource_receipt_rejects_partial_or_malformed_attestation(receipt):
    with pytest.raises(ValueError):
        profile_from_receipt(receipt)


@pytest.mark.parametrize(
    "value",
    [
        {"cpu": 2, "memory": 2, "disk": 10},
        {"cpu": 2, "memory_gb": 2, "disk_gb": 10, "units": "GB"},
        {"cpu": True, "memory_gb": 2, "disk_gb": 10},
    ],
)
def test_resource_profile_rejects_ambiguous_or_invalid_units(value):
    with pytest.raises(ValueError):
        resolve_profile(value, default=CANDIDATE_DEFAULT)


def test_compatible_defaults_remain_the_existing_provider_requests():
    assert CANDIDATE_DEFAULT.receipt() == {
        "cpu": 4,
        "memory_gb": 8,
        "disk_gb": 10,
        "provider_units": {"cpu": "cores", "memory": "GB", "disk": "GB"},
        "source": "compatible_default_no_pinned_profile",
        "effective_limits": "unverified",
    }
    assert (
        VERIFIER_DEFAULT.cpu,
        VERIFIER_DEFAULT.memory_gb,
        VERIFIER_DEFAULT.disk_gb,
    ) == (
        2,
        1,
        10,
    )
