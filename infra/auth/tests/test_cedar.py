# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import json
from contextlib import closing

import cedarpy
import pytest
from rigging.grants import GrantDenied, GrantStore, Permission

from infra.auth.cedar import CedarPolicy, ProviderIdentity


def _record(attributes):
    return {"type": "Record", "attributes": attributes}


@pytest.fixture
def schema():
    return {
        "Marin": {
            "entityTypes": {
                "Principal": {
                    "memberOfTypes": ["Group"],
                    "shape": _record(
                        {
                            "owner": {"type": "String"},
                            "max_ttl": {"type": "Long"},
                            "delegation_depth": {"type": "Long"},
                            "multiplier": {"type": "Long"},
                        }
                    ),
                },
                "Group": {"memberOfTypes": ["Group"]},
                "Namespace": {"memberOfTypes": ["Namespace"]},
                "LoomProfile": {"memberOfTypes": ["LoomProfile"]},
                "CWIdentity": {
                    "shape": _record(
                        {
                            "resources": {"type": "Set", "element": {"type": "String"}},
                            "actions": {"type": "Set", "element": {"type": "String"}},
                            "read_only": {"type": "Boolean"},
                        }
                    )
                },
            },
            "actions": {
                action: {
                    "appliesTo": {"principalTypes": ["Principal"], "resourceTypes": [resource], "context": _record({})}
                }
                for action, resource in (
                    ("iris.job.submit", "Namespace"),
                    ("iris.job.read", "Namespace"),
                    ("loom.session.create", "LoomProfile"),
                    ("cw_s3.credentials.issue", "CWIdentity"),
                )
            },
        }
    }


@pytest.fixture
def entities():
    return [
        {"uid": {"type": "Marin::Group", "id": group}, "attrs": {}, "parents": []}
        for group in ("developers", "incident-readers")
    ] + [
        {
            "uid": {"type": "Marin::Principal", "id": principal},
            "attrs": {"owner": "user:power", "max_ttl": 600, "delegation_depth": 2, "multiplier": 2},
            "parents": [{"type": "Marin::Group", "id": group} for group in groups],
        }
        for principal, groups in (
            ("user:power", ("developers", "incident-readers")),
            ("grafana:alert", ("incident-readers",)),
        )
    ]


@pytest.fixture
def providers():
    return (
        ProviderIdentity("incident-reader", (Permission("cw_s3", frozenset({"s3:GetObject"}), "s3://logs/incidents/"),)),
        ProviderIdentity(
            "incident-writer",
            (Permission("cw_s3", frozenset({"s3:GetObject", "s3:PutObject"}), "s3://logs/incidents/"),),
        ),
    )


@pytest.fixture
def policies():
    return """
permit(principal in Marin::Group::"developers",
       action in [Marin::Action::"iris.job.submit", Marin::Action::"iris.job.read"],
       resource in Marin::Namespace::"/power/");
permit(principal == Marin::Principal::"grafana:alert",
       action == Marin::Action::"loom.session.create",
       resource == Marin::LoomProfile::"/diagnostics/");
permit(principal in Marin::Group::"incident-readers",
       action == Marin::Action::"cw_s3.credentials.issue",
       resource) when { resource.read_only };
"""


@pytest.fixture
def policy(schema, entities, providers, policies):
    return CedarPolicy(schema, policies, entities, providers)


def test_real_cedar_authorizes_group_namespace_and_alert_profile_with_different_owner(policy):
    assert policy.check("user:power", "iris", "job.submit", "/power/experiment/train")
    assert policy.check("grafana:alert", "loom", "session.create", "/diagnostics/")
    assert not policy.check("grafana:alert", "iris", "job.submit", "/power/train")
    assert not policy.check("user:power", "loom", "session.create", "/diagnostics/")
    assert not policy.check("grafana:alert", "loom", "session.create", "/other/")


@pytest.mark.parametrize(
    "resource", ["/powerful/train", "/power/../other/train", "/power/%2fother/train", "/power//train"]
)
def test_canonical_resource_hierarchy_cannot_escape_allowed_namespace(policy, resource):
    assert not policy.check("user:power", "iris", "job.submit", resource)


def test_grants_and_native_data_access_require_entire_known_provider_identity(policy):
    bundle = (
        Permission("loom", frozenset({"session.create"}), "/diagnostics/"),
        Permission("cw_s3", frozenset({"s3:GetObject"}), "s3://logs/incidents/"),
    )
    assert policy.allows_grant("grafana:alert", bundle)
    assert policy.allows_provider("grafana:alert", "incident-reader")
    assert policy.check("grafana:alert", "cw_s3", "s3:GetObject", "s3://logs/incidents/current.json")
    assert not policy.check("grafana:alert", "cw_s3", "s3:PutObject", "s3://logs/incidents/current.json")
    assert not policy.check("grafana:alert", "cw_s3", "s3:GetObject", "s3://logs/private/current.json")
    assert not policy.allows_provider("grafana:alert", "incident-writer")
    for resource in ("s3://logs/incidents/one/", "s3://logs/", "s3://other/incidents/"):
        assert not policy.allows_grant("grafana:alert", (Permission("cw_s3", frozenset({"s3:GetObject"}), resource),))


def test_forbid_on_descendant_is_enforced_at_live_operation_not_inferred_from_prefix(
    schema, entities, providers, policies
):
    policy = CedarPolicy(
        schema,
        policies
        + """
forbid(principal, action == Marin::Action::"iris.job.submit",
       resource in Marin::Namespace::"/power/private/");
""",
        entities,
        providers,
    )
    assert policy.allows_grant("user:power", (Permission("iris", frozenset({"job.submit"}), "/power/"),))
    assert policy.check("user:power", "iris", "job.submit", "/power/public/train")
    assert not policy.check("user:power", "iris", "job.submit", "/power/private/train")


@pytest.mark.parametrize("unknown", ["iris.unknown", "iris.job.submit"])
def test_unknown_policy_action_or_resource_type_is_rejected_at_load(schema, entities, providers, unknown):
    resource_type = "Namespace" if unknown == "iris.unknown" else "MissingResource"
    with pytest.raises(ValueError):
        CedarPolicy(
            schema,
            f'permit(principal, action == Marin::Action::"{unknown}", resource is Marin::{resource_type});',
            entities,
            providers,
        )


def test_unknown_runtime_principal_action_and_provider_deny(policy):
    assert not policy.check("untrusted", "iris", "job.submit", "/power/train")
    assert not policy.check("user:power", "iris", "unknown", "/power/train")
    assert not policy.allows_provider("grafana:alert", "unknown-provider")


def test_any_cedar_evaluation_error_denies_even_if_another_policy_permits(schema, entities, providers, policies):
    overflowing = (
        policies
        + """
permit(principal, action, resource)
when { principal.multiplier * 9223372036854775807 > 0 };
"""
    )
    native = cedarpy.is_authorized(
        {
            "principal": {"type": "Marin::Principal", "id": "user:power"},
            "action": {"type": "Marin::Action", "id": "iris.job.submit"},
            "resource": {"type": "Marin::Namespace", "id": "/power/"},
            "context": {},
        },
        cedarpy.PolicySet.from_str(overflowing),
        cedarpy.Entities.from_json_str(json.dumps(entities), schema),
        schema,
    )
    assert native.allowed and native.diagnostics.errors
    policy = CedarPolicy(schema, overflowing, entities, providers)
    assert not policy.check("user:power", "iris", "job.submit", "/power/train")
    assert not policy.allows_provider("grafana:alert", "incident-reader")


def test_configured_entities_cannot_override_server_derived_provider_identity(schema, entities, providers, policies):
    malicious = copy.deepcopy(entities)
    malicious.append({"uid": {"type": "Marin::CWIdentity", "id": "incident-writer"}, "attrs": {}, "parents": []})
    with pytest.raises(ValueError):
        CedarPolicy(schema, policies, malicious, providers)


def test_policy_revision_change_invalidates_durable_runtime_authority(tmp_path, schema, entities, providers, policies):
    policy = CedarPolicy(schema, policies, entities, providers)
    path = tmp_path / "grants.sqlite"
    with closing(GrantStore(path, policy.authorities, policy, now=lambda: 1000)) as store:
        grant = store.create(
            "grafana:alert",
            actor="loom:run",
            permissions=(Permission("loom", frozenset({"session.create"}), "/diagnostics/"),),
            expires_at=1200,
        )
        authorized = store.authorize(
            grant.id, actor="loom:run", service="loom", action="session.create", resource="/diagnostics/"
        )
        assert authorized.owner == "user:power"
    changed = CedarPolicy(
        schema,
        policies + 'forbid(principal, action == Marin::Action::"loom.session.create", resource);',
        entities,
        providers,
    )
    with closing(GrantStore(path, changed.authorities, changed, now=lambda: 1000)) as store:
        with pytest.raises(GrantDenied):
            store.authorize(
                grant.id, actor="loom:run", service="loom", action="session.create", resource="/diagnostics/"
            )


def test_schema_declared_app_resource_uses_the_same_evaluator(schema, entities, providers, policies):
    schema["Marin"]["entityTypes"]["App"] = {"memberOfTypes": ["App"]}
    schema["Marin"]["actions"]["marina.app.read"] = {
        "appliesTo": {"principalTypes": ["Principal"], "resourceTypes": ["App"], "context": _record({})}
    }
    policy = CedarPolicy(
        schema,
        policies
        + """
permit(principal == Marin::Principal::"user:power",
       action == Marin::Action::"marina.app.read", resource in Marin::App::"/echo/");
""",
        entities,
        providers,
    )
    assert policy.check("user:power", "marina", "app.read", "/echo/releases/current")
    assert not policy.check("user:power", "marina", "app.read", "/other/")


def test_provider_action_resource_association_is_part_of_revision(schema, entities, policies):
    first_mapping = ProviderIdentity(
        "reader",
        (
            Permission("cw_s3", frozenset({"s3:GetObject"}), "s3://logs/first/"),
            Permission("cw_s3", frozenset({"s3:GetObjectVersion"}), "s3://logs/second/"),
        ),
    )
    second_mapping = ProviderIdentity(
        "reader",
        (
            Permission("cw_s3", frozenset({"s3:GetObjectVersion"}), "s3://logs/first/"),
            Permission("cw_s3", frozenset({"s3:GetObject"}), "s3://logs/second/"),
        ),
    )
    first = CedarPolicy(schema, policies, entities, (first_mapping,))
    second = CedarPolicy(schema, policies, entities, (second_mapping,))
    assert first.check("grafana:alert", "cw_s3", "s3:GetObject", "s3://logs/first/current")
    assert not second.check("grafana:alert", "cw_s3", "s3:GetObject", "s3://logs/first/current")
    assert first.revision != second.revision
