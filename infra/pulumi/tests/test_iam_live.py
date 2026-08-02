# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from iac.gcp.iam_live import ResourceTarget, _policy_url, parse_policy_bindings
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
