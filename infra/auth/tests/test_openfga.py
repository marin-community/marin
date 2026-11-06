# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import time
from contextlib import closing

import pytest
from rigging.grants import GrantStore, Permission

from infra.auth.openfga import AuthorizationUnavailable, OpenFGAPolicy, ProviderIdentity


@pytest.fixture
def policy(openfga_http):
    return OpenFGAPolicy(
        **{key: value for key, value in openfga_http.items() if key != "state"},
        principals=[{"principal": "researcher", "owner": "researcher", "max_ttl": 600, "delegation_depth": 2}],
        providers=[
            ProviderIdentity("reader", (Permission("cw_s3", frozenset({"s3:GetObject"}), "s3://logs/incidents/"),))
        ],
    )


def test_checks_use_pinned_model_current_relationships_and_server_derived_ancestry(policy, openfga_http):
    assert policy.check("researcher", "iris", "job.submit", "/research/job")
    request = openfga_http["state"]["requests"][-1]
    assert request["authorization"] == "Bearer synthetic-openfga-key"
    assert request["path"] == f"/stores/{openfga_http['store_id']}/check"
    assert request["payload"] == {
        "authorization_model_id": openfga_http["model_id"],
        "consistency": "HIGHER_CONSISTENCY",
        "tuple_key": {
            "user": "principal:researcher",
            "relation": "allowed",
            "object": "scope:iris/job.submit//research/job",
        },
        "contextual_tuples": {
            "tuple_keys": [
                {"user": "scope:iris/job.submit//", "relation": "parent", "object": "scope:iris/job.submit//research/"},
                {
                    "user": "scope:iris/job.submit//research/",
                    "relation": "parent",
                    "object": "scope:iris/job.submit//research/job",
                },
            ]
        },
    }
    openfga_http["state"]["allowed"] = False
    assert not policy.check("researcher", "iris", "job.submit", "/research/job")


@pytest.mark.parametrize("status,allowed", [(503, True), (302, True), (200, "true"), (200, None)])
def test_missing_or_invalid_openfga_decision_cannot_authorize_a_grant(policy, openfga_http, status, allowed):
    openfga_http["state"].update(status=status, allowed=allowed)
    with closing(GrantStore(openfga_http["token_file"].parent / "grants.db", policy.authorities, policy)) as store:
        with pytest.raises(AuthorizationUnavailable):
            store.create(
                "researcher",
                actor="session",
                permissions=(Permission("iris", frozenset({"job.submit"}), "/research/"),),
                expires_at=time.time() + 60,
            )


def test_storage_mapping_must_fit_entire_grant_even_when_openfga_permits_identity(policy):
    assert policy.allows_grant("researcher", [Permission("cw_s3", frozenset({"s3:GetObject"}), "s3://logs/incidents/")])
    assert not policy.allows_grant(
        "researcher", [Permission("cw_s3", frozenset({"s3:GetObject"}), "s3://logs/incidents/narrow/")]
    )
    assert not policy.allows_grant(
        "researcher", [Permission("cw_s3", frozenset({"s3:PutObject"}), "s3://logs/incidents/")]
    )


def test_unknown_actions_and_noncanonical_application_paths_cannot_inherit_permissions(policy):
    assert not policy.check("researcher", "iris", "job.sumbit", "/research/")
    assert not policy.check("researcher", "iris", "job.submit", "/research/../other/")
    assert not policy.check("researcher", "iris", "job.submit", "s3://logs/incidents/")
    assert not policy.check("unknown", "iris", "job.submit", "/research/")
