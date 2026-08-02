# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import requests
from iac.gcp.iam import GcpRoleGrant
from iac.gcp.iam_live import GcpIamReader, ResourceTarget, _policy_url, iter_resources, parse_policy_bindings
from iac.gcp.iam_scan import Binding, Container


@pytest.mark.parametrize(
    ("target", "method", "url_fragment"),
    [
        (
            ResourceTarget(Container.PROJECT, "p"),
            "POST",
            "cloudresourcemanager.googleapis.com/v1/projects/p:getIamPolicy",
        ),
        (
            ResourceTarget(Container.KMS, "projects/p/locations/l/keyRings/r/cryptoKeys/k"),
            "GET",
            "cloudkms.googleapis.com/v1/projects/p/locations/l/keyRings/r/cryptoKeys/k:getIamPolicy",
        ),
        (
            ResourceTarget(Container.SECRET, "S"),
            "GET",
            "secretmanager.googleapis.com/v1/projects/p/secrets/S:getIamPolicy",
        ),
        (ResourceTarget(Container.BUCKET, "b1"), "GET", "storage.googleapis.com/storage/v1/b/b1/iam"),
        (
            ResourceTarget(Container.ARTIFACT_REPOSITORY, "us/repo"),
            "GET",
            "projects/p/locations/us/repositories/repo:getIamPolicy",
        ),
        (
            ResourceTarget(Container.SERVICE_ACCOUNT, "sa@p.iam.gserviceaccount.com"),
            "POST",
            "iam.googleapis.com/v1/projects/p/serviceAccounts/sa@p.iam.gserviceaccount.com:getIamPolicy",
        ),
    ],
)
def test_policy_url_targets_the_right_api_and_method(target, method, url_fragment):
    got_method, got_url = _policy_url(target, "p")

    assert got_method == method
    assert url_fragment in got_url


def test_parse_policy_bindings_flattens_members_and_drops_conditions():
    target = ResourceTarget(Container.PROJECT, "p")
    policy = {
        "bindings": [
            {"role": "roles/viewer", "members": ["serviceAccount:a@p.iam.gserviceaccount.com", "group:g@x.com"]},
            {"role": "roles/owner", "members": ["user:o@x.com"], "condition": {"title": "t", "expression": "e"}},
        ]
    }

    bindings = parse_policy_bindings(target, policy)

    assert Binding(Container.PROJECT, "p", "roles/viewer", "group:g@x.com") in bindings
    assert Binding(Container.PROJECT, "p", "roles/owner", "user:o@x.com") in bindings
    assert len(bindings) == 3


def test_iter_resources_maps_declared_resources_to_read_targets():
    grant = GcpRoleGrant(role="roles/viewer", members=())
    resources = [(Container.PROJECT, "p", (grant,)), (Container.SECRET, "S", (grant,))]

    targets = list(iter_resources(resources))

    # The grants are dropped; only (container, resource) survives into a read target.
    assert targets == [ResourceTarget(Container.PROJECT, "p"), ResourceTarget(Container.SECRET, "S")]


class _FakeResponse:
    def __init__(self, payload: dict, status_error: Exception | None) -> None:
        self._payload = payload
        self._status_error = status_error

    def raise_for_status(self) -> None:
        if self._status_error is not None:
            raise self._status_error

    def json(self) -> dict:
        return self._payload


class _FakeSession:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    def request(self, method: str, url: str, *, params=None) -> _FakeResponse:
        return self._response


def test_bindings_parses_a_successful_policy():
    payload = {"bindings": [{"role": "roles/viewer", "members": ["serviceAccount:a@p.iam.gserviceaccount.com"]}]}
    reader = GcpIamReader(session=_FakeSession(_FakeResponse(payload, status_error=None)))

    bindings = reader.bindings(ResourceTarget(Container.PROJECT, "p"), "p")

    assert bindings == [Binding(Container.PROJECT, "p", "roles/viewer", "serviceAccount:a@p.iam.gserviceaccount.com")]


def test_bindings_treats_an_http_error_as_no_bindings():
    error = requests.HTTPError("403 Forbidden")
    reader = GcpIamReader(session=_FakeSession(_FakeResponse({}, status_error=error)))

    # A single unreadable resource degrades to empty rather than aborting the whole scan.
    assert reader.bindings(ResourceTarget(Container.SECRET, "S"), "p") == []
