# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Errors must classify correctly under the pipeline's OWN classifier.

The classifier is the vendored, unmodified capability_pipeline/daytona_snapshot.py.
Each error is checked twice: as raised locally, and after a round trip through
the HTTP wire format, since that is how the pipeline will actually receive it.
"""

import pytest
from silo.errors import SiloConflictError, SiloError, SiloNotFoundError, SiloRateLimitError, SiloRecipeError
from silo.pipeline_contract.daytona_snapshot import resource_not_found, snapshot_conflict, snapshot_not_found
from silo.wire import error_body, error_from_body


def transported(error: SiloError) -> SiloError:
    return error_from_body(error_body(error), error.status_code or 500)


@pytest.fixture(params=["local", "transported"])
def via(request):
    return (lambda e: e) if request.param == "local" else transported


def test_missing_snapshot_is_a_snapshot_not_found(via):
    error = via(SiloNotFoundError("snapshot", "cap-harbor-abc"))
    assert snapshot_not_found(error)
    assert not snapshot_conflict(error)


def test_missing_sandbox_is_a_sandbox_not_found_but_not_a_snapshot_one(via):
    # daytona_snapshot's own test pins that a sandbox 404 must NOT read as a
    # snapshot 404 -- otherwise provisioning would "recover" by rebuilding a
    # snapshot that is fine.
    error = via(SiloNotFoundError("sandbox", "slbdeadbeef"))
    assert resource_not_found(error, "sandbox")
    assert not snapshot_not_found(error)


def test_conflict_is_a_conflict_and_not_a_not_found(via):
    error = via(SiloConflictError("snapshot", "cap-harbor-abc"))
    assert snapshot_conflict(error)
    assert not snapshot_not_found(error)


@pytest.mark.parametrize(
    "error",
    [
        SiloRecipeError("unsupported Dockerfile instruction 'COPY'"),
        SiloError("no live sandbox hosts registered with the broker", status_code=503),
        SiloRateLimitError("sandbox capacity exhausted"),
    ],
)
def test_everything_else_is_neither(error, via):
    # The third row of brief section 3.4: surface it, do not turn it into
    # "create it" or "resolve theirs".
    error = via(error)
    assert not snapshot_not_found(error)
    assert not snapshot_conflict(error)


def test_rate_limit_carries_what_provider_retry_reads(via):
    error = via(SiloRateLimitError("full", retry_after_seconds=30))
    assert error.status_code == 429
    assert error.headers["Retry-After"] == "30"
    assert error.error_code == "capacity_exhausted"


def test_transport_preserves_class_name_and_message():
    original = SiloNotFoundError("snapshot", "x", "gone")
    rebuilt = transported(original)
    assert type(rebuilt) is SiloNotFoundError
    assert str(rebuilt) == str(original)
    assert rebuilt.status_code == 404
