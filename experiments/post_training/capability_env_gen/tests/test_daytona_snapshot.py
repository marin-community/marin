import hashlib
from types import SimpleNamespace

import pytest

from capability_pipeline.daytona_snapshot import (
    SnapshotRecipeEvidenceError,
    provision_with_snapshot_recovery,
    snapshot_conflict,
    snapshot_not_found,
    validate_snapshot_recipe,
    wait_for_snapshot_active,
)


class DaytonaNotFoundError(RuntimeError):
    pass


class DaytonaConflictError(RuntimeError):
    pass


class Generic404(RuntimeError):
    status_code = 404


def snapshot(name, identifier):
    return SimpleNamespace(
        id=identifier,
        name=name,
        state="active",
        image_name="provider-build",
    )


def test_concurrent_snapshot_builder_is_polled_until_active_without_rebuild():
    dockerfile = "FROM python@sha256:" + "a" * 64 + "\n"
    building = snapshot("shared", "provider-id")
    building.state = "building"
    active = snapshot("shared", "provider-id")
    active.build_info = SimpleNamespace(dockerfile_content=dockerfile)
    calls, sleeps = [], []

    def get(name):
        calls.append(name)
        return active

    resolved = wait_for_snapshot_active(
        get,
        "shared",
        dockerfile,
        initial_snapshot=building,
        delays=(0, 2),
        sleeper=sleeps.append,
    )
    assert resolved is active
    assert calls == ["shared"]
    assert sleeps == [2]


def test_snapshot_registration_404_is_transient_but_other_errors_fail():
    dockerfile = "FROM immutable\n"
    active = snapshot("shared", "provider-id")
    active.build_info = SimpleNamespace(dockerfile_content=dockerfile)
    calls = 0

    def get(_name):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise DaytonaNotFoundError("snapshot shared not found")
        return active

    assert (
        wait_for_snapshot_active(
            get,
            "shared",
            dockerfile,
            delays=(0, 1),
            sleeper=lambda _delay: None,
        )
        is active
    )
    with pytest.raises(RuntimeError, match="capacity"):
        wait_for_snapshot_active(
            lambda _name: (_ for _ in ()).throw(RuntimeError("capacity")),
            "shared",
            dockerfile,
            delays=(0,),
        )


def test_snapshot_wait_fails_closed_on_terminal_unknown_or_timeout():
    dockerfile = "FROM immutable\n"
    for state in ("failed", "deleted", "mystery"):
        current = snapshot("shared", "provider-id")
        current.state = state
        with pytest.raises(RuntimeError, match="terminal or unknown"):
            wait_for_snapshot_active(
                lambda _name, observed=current: observed,
                "shared",
                dockerfile,
                initial_snapshot=current,
                delays=(0,),
            )
    building = snapshot("shared", "provider-id")
    building.state = "pending"
    with pytest.raises(RuntimeError, match="did not become active"):
        wait_for_snapshot_active(
            lambda _name: building,
            "shared",
            dockerfile,
            initial_snapshot=building,
            delays=(0, 1),
            sleeper=lambda _delay: None,
        )
    with pytest.raises(ValueError, match="total <= 300"):
        wait_for_snapshot_active(
            lambda _name: building, "shared", dockerfile, delays=(0, 301)
        )


def test_active_snapshot_requires_exact_recipe_and_provider_identity():
    active = snapshot("shared", "provider-id")
    active.build_info = SimpleNamespace(dockerfile_content="FROM wrong\n")
    with pytest.raises(SnapshotRecipeEvidenceError, match="does not match"):
        wait_for_snapshot_active(
            lambda _name: active,
            "shared",
            "FROM expected\n",
            initial_snapshot=active,
            delays=(0,),
        )
    with pytest.raises(SnapshotRecipeEvidenceError, match="wrong provider name"):
        wait_for_snapshot_active(
            lambda _name: active,
            "foreign",
            "FROM wrong\n",
            initial_snapshot=active,
            delays=(0,),
        )


def test_snapshot_not_found_reconstructs_once_before_candidate_execution():
    calls = []
    receipts = []

    def create_sandbox(name):
        calls.append(name)
        if len(calls) == 1:
            raise DaytonaNotFoundError(f"Snapshot with name {name} not found")
        return SimpleNamespace(id="sandbox-2"), [{"attempt": 1, "state": "created"}]

    value, attempts, active, events = provision_with_snapshot_recovery(
        base_name="cap-harbor-image",
        requested_image="python@sha256:" + "a" * 64,
        initial_snapshot=snapshot("cap-harbor-image", "snapshot-stale"),
        create_sandbox=create_sandbox,
        reconstruct=lambda name: snapshot(name, "snapshot-rebuilt"),
        recovery_suffix="1234abcd",
        record=lambda current: receipts.append([dict(event) for event in current]),
    )

    assert value.id == "sandbox-2"
    assert attempts == [{"attempt": 1, "state": "created"}]
    assert calls == ["cap-harbor-image", "cap-harbor-image-recovery-1234abcd"]
    assert active["id"] == "snapshot-rebuilt"
    assert active["requested_image_sha256"]
    assert [event["event"] for event in events] == [
        "snapshot_resolved",
        "sandbox_create_snapshot_not_found",
        "snapshot_reconstructed",
        "sandbox_created",
    ]
    assert receipts[-1] == events


def test_snapshot_recovery_is_bounded_and_preserves_second_failure():
    receipts = []

    def missing(_name):
        raise DaytonaNotFoundError("snapshot not found")

    with pytest.raises(DaytonaNotFoundError):
        provision_with_snapshot_recovery(
            base_name="cap-harbor-image",
            requested_image="python@sha256:" + "a" * 64,
            initial_snapshot=snapshot("cap-harbor-image", "snapshot-stale"),
            create_sandbox=missing,
            reconstruct=lambda name: snapshot(name, "snapshot-rebuilt"),
            recovery_suffix="1234abcd",
            record=lambda current: receipts.append([dict(event) for event in current]),
        )

    assert [event["event"] for event in receipts[-1]] == [
        "snapshot_resolved",
        "sandbox_create_snapshot_not_found",
        "snapshot_reconstructed",
        "snapshot_reconstruction_not_found",
    ]


def test_unrelated_provider_failure_is_not_retried():
    reconstructed = False

    def reconstruct(_name):
        nonlocal reconstructed
        reconstructed = True

    with pytest.raises(RuntimeError, match="capacity"):
        provision_with_snapshot_recovery(
            base_name="cap-harbor-image",
            requested_image="python@sha256:" + "a" * 64,
            initial_snapshot=snapshot("cap-harbor-image", "snapshot-live"),
            create_sandbox=lambda _name: (_ for _ in ()).throw(
                RuntimeError("capacity")
            ),
            reconstruct=reconstruct,
            recovery_suffix="1234abcd",
            record=lambda _events: None,
        )

    assert reconstructed is False


def test_only_same_name_conflict_allows_concurrent_creator_fallback():
    assert snapshot_conflict(DaytonaConflictError("conflict"))
    assert snapshot_conflict(RuntimeError("snapshot already exists"))
    assert not snapshot_conflict(DaytonaNotFoundError("snapshot not found"))
    assert not snapshot_conflict(RuntimeError("rate limited"))


def test_generic_not_found_does_not_trigger_snapshot_reconstruction():
    assert snapshot_not_found(DaytonaNotFoundError("snapshot named x not found"))
    assert not snapshot_not_found(DaytonaNotFoundError("sandbox x not found"))
    assert not snapshot_not_found(Generic404("route not found"))


def test_resolved_snapshot_must_be_active_and_match_content_addressed_name():
    inactive = snapshot("cap-harbor-image", "snapshot-building")
    inactive.state = "pending"
    common = {
        "base_name": "cap-harbor-image",
        "requested_image": "python@sha256:" + "a" * 64,
        "create_sandbox": lambda _name: (
            SimpleNamespace(id="unused"),
            [{"attempt": 1, "state": "created"}],
        ),
        "reconstruct": lambda name: snapshot(name, "replacement"),
        "recovery_suffix": "1234abcd",
        "record": lambda _events: None,
    }
    with pytest.raises(RuntimeError, match="not active"):
        provision_with_snapshot_recovery(initial_snapshot=inactive, **common)

    with pytest.raises(ValueError, match="wrong provider identity"):
        provision_with_snapshot_recovery(
            initial_snapshot=snapshot("wrong-name", "snapshot-live"), **common
        )


def test_cached_snapshot_with_expected_name_but_wrong_recipe_is_rejected():
    cached = snapshot("cap-verifier-same-name", "snapshot-wrong-recipe")
    cached.build_info = SimpleNamespace(dockerfile_content="FROM wrong@sha256:bad\n")

    with pytest.raises(SnapshotRecipeEvidenceError, match="does not match"):
        validate_snapshot_recipe(
            cached,
            expected_name="cap-verifier-same-name",
            expected_dockerfile="FROM expected@sha256:good\n",
        )


def test_recipe_mismatch_with_404_in_digest_is_not_snapshot_not_found():
    error = SnapshotRecipeEvidenceError(
        "cached snapshot Dockerfile mismatch: got sha256:404abc"
    )

    assert not snapshot_not_found(error)


def test_cached_snapshot_recipe_does_not_override_wrong_provider_name():
    definition = "FROM expected@sha256:good\n"
    cached = snapshot("cap-verifier-other-name", "snapshot-other-name")
    cached.build_info = SimpleNamespace(dockerfile_content=definition)

    with pytest.raises(SnapshotRecipeEvidenceError, match="wrong provider name"):
        validate_snapshot_recipe(
            cached,
            expected_name="cap-verifier-expected-name",
            expected_dockerfile=definition,
        )


@pytest.mark.parametrize(
    "build_info",
    [None, SimpleNamespace(), SimpleNamespace(dockerfile_content="")],
)
def test_cached_snapshot_without_provider_recipe_metadata_is_rejected(build_info):
    cached = snapshot("cap-verifier-cache", "snapshot-no-evidence")
    cached.build_info = build_info

    with pytest.raises(SnapshotRecipeEvidenceError, match="insufficient"):
        validate_snapshot_recipe(
            cached,
            expected_name="cap-verifier-cache",
            expected_dockerfile="FROM expected@sha256:good\n",
        )


def test_cached_snapshot_exact_provider_recipe_match_returns_hash_evidence():
    definition = "FROM expected@sha256:good\nUSER root\nRUN true\n"
    cached = snapshot("cap-verifier-cache", "snapshot-valid")
    cached.build_info = {"dockerfile_content": definition}

    evidence = validate_snapshot_recipe(
        cached,
        expected_name="cap-verifier-cache",
        expected_dockerfile=definition,
    )

    assert evidence == {
        "snapshot_name": "cap-verifier-cache",
        "dockerfile_sha256": hashlib.sha256(definition.encode()).hexdigest(),
        "evidence": "provider-build-info-exact-match",
    }
