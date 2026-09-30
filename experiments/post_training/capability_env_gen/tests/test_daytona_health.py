from types import SimpleNamespace

import pytest

from capability_pipeline.daytona_snapshot import wait_for_sandbox_deletion


class DaytonaNotFoundError(RuntimeError):
    pass


def test_deletion_lookup_allows_bounded_eventual_consistency():
    lookups = 0
    sleeps = []

    class Client:
        def get(self, sandbox_id):
            nonlocal lookups
            assert sandbox_id == "sandbox-1"
            lookups += 1
            if lookups < 4:
                return SimpleNamespace(id=sandbox_id)
            raise DaytonaNotFoundError("sandbox sandbox-1 not found")

    state, observations = wait_for_sandbox_deletion(
        Client(), "sandbox-1", sleeper=sleeps.append
    )

    assert state == "not_found"
    assert sleeps == [5, 10, 15]
    assert observations == [
        {"elapsed_seconds": 0.0, "state": "present"},
        {"elapsed_seconds": 5.0, "state": "present"},
        {"elapsed_seconds": 15.0, "state": "present"},
        {"elapsed_seconds": 30.0, "state": "not_found"},
    ]


def test_deletion_lookup_fails_closed_after_sixty_seconds():
    sleeps = []
    state, observations = wait_for_sandbox_deletion(
        SimpleNamespace(get=lambda _sandbox_id: SimpleNamespace(id="still-present")),
        "sandbox-1",
        sleeper=sleeps.append,
    )

    assert state == "present"
    assert sleeps == [5, 10, 15, 30]
    assert observations[-1] == {"elapsed_seconds": 60.0, "state": "present"}


def test_deletion_lookup_preserves_transient_lookup_errors():
    calls = 0

    def get(_sandbox_id):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary provider failure")
        raise DaytonaNotFoundError("sandbox sandbox-1 not found")

    state, observations = wait_for_sandbox_deletion(
        SimpleNamespace(get=get), "sandbox-1", delays=(0, 1), sleeper=lambda _: None
    )

    assert state == "not_found"
    assert observations == [
        {
            "elapsed_seconds": 0.0,
            "state": "lookup_error",
            "error_type": "RuntimeError",
        },
        {"elapsed_seconds": 1.0, "state": "not_found"},
    ]


@pytest.mark.parametrize("delays", [(), (1,), (0, 61)])
def test_deletion_lookup_rejects_unbounded_or_delayed_schedules(delays):
    with pytest.raises(ValueError, match="start at zero and total <= 60s"):
        wait_for_sandbox_deletion(
            SimpleNamespace(get=lambda _: None), "sandbox-1", delays=delays
        )
