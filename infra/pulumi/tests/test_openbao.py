# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace
from pathlib import Path

import pulumi
import pulumi_kubernetes as k8s
import pytest
from iac.auth.broker import GrantBroker, GrantBrokerArgs
from iac.auth.openbao import OpenBao, OpenBaoArgs


@pulumi.runtime.test
def test_openbao_exposes_only_verification_endpoints_and_retains_sealed_raft_servers(recording_mocks):
    mocks = recording_mocks
    pulumi.runtime.set_mocks(mocks, project="marin-auth", stack="test", preview=False)
    cluster = OpenBao(
        "openbao",
        OpenBaoArgs(
            namespace="management",
            image="openbao/openbao@sha256:" + "a" * 64,
            public_hostname="auth.example.org",
            storage_class="block",
            storage_size="10Gi",
            listener_tls_secret="listener-tls",
            listener_tls_server_name="openbao.management.svc",
            ingress_tls_secret="edge-tls",
            ingress_namespace="traefik",
            ingress_pod_labels={"app.kubernetes.io/name": "traefik"},
            client_namespace="management",
            client_pod_labels={"app.kubernetes.io/name": "marin-auth"},
            operator_groups=("auth-operators",),
            dns_cidrs=("10.16.0.10/32", "10.16.0.30/32"),
            https_egress_cidrs=(),
        ),
        k8s_provider=k8s.Provider("test", kubeconfig="", context="test"),
    )

    def check(_: list) -> None:
        resources = {resource.typ: resource.inputs for resource in mocks.resources}
        assert "kubernetes:core/v1:Secret" not in resources
        stateful = resources["kubernetes:apps/v1:StatefulSet"]["spec"]
        assert stateful["podManagementPolicy"] == "Parallel"
        assert stateful["updateStrategy"] == {"type": "OnDelete"}
        assert stateful["persistentVolumeClaimRetentionPolicy"] == {"whenDeleted": "Retain", "whenScaled": "Retain"}
        assert stateful["volumeClaimTemplates"][0]["spec"]["accessModes"] == ["ReadWriteOnce"]
        pod = stateful["template"]["spec"]
        assert pod["automountServiceAccountToken"] is False
        container = pod["containers"][0]
        assert container["args"] == ["server", "-config=/openbao/config/config.json"]
        assert container["livenessProbe"]["tcpSocket"] == {"port": "api"}
        assert container["readinessProbe"]["exec"]["command"] == ["bao", "status"]
        config = json.loads(resources["kubernetes:core/v1:ConfigMap"]["data"]["config.json"])
        assert config["storage"]["raft"]["path"] == "/openbao/data"
        assert "seal" not in config
        assert "tls_disable" not in config["listener"]["tcp"]
        assert all(
            "leader_ca_cert_file" in peer and "leader_tls_servername" in peer
            for peer in config["storage"]["raft"]["retry_join"]
        )
        ingress = resources["kubernetes:networking.k8s.io/v1:Ingress"]["spec"]
        paths = ingress["rules"][0]["http"]["paths"]
        assert {(path["path"], path["pathType"]) for path in paths} == {
            ("/v1/identity/oidc/.well-known/openid-configuration", "Exact"),
            ("/v1/identity/oidc/.well-known/keys", "Exact"),
        }
        network = resources["kubernetes:networking.k8s.io/v1:NetworkPolicy"]["spec"]
        assert set(network["policyTypes"]) == {"Ingress", "Egress"}
        assert all("podSelector" in peer for rule in network["ingress"] for peer in rule["from"])
        role = resources["kubernetes:rbac.authorization.k8s.io/v1:Role"]
        assert {(tuple(rule["resources"]), tuple(rule["verbs"])) for rule in role["rules"]} == {
            (("pods",), ("get", "list", "watch")),
            (("pods/portforward",), ("create",)),
        }

    return pulumi.Output.all(
        cluster.stateful_set.spec, cluster.network_policy.spec, cluster.discovery_ingress.spec
    ).apply(check)


@pytest.fixture
def broker_args():
    policy = {
        "bao_address": "https://openbao.management.svc:8200",
        "ca_bundle": "/var/run/marin-auth/ca/ca.crt",
        "bao_issuer": "https://auth.example.org/v1/identity/oidc",
        "issuer": "https://marin-auth.management.svc:8090",
        "database": "/var/lib/marin-auth/grants.db",
        "broker_token_file": "/var/run/marin-auth/broker-token/token",
        "signing_key_version": 1,
        "subjects": {"entity-id": "human:alice"},
        "cedar": {"schema_document": {}, "policies": "permit(principal, action, resource);", "entities": []},
        "storage": [
            {
                "name": "checkpoint-reader",
                "subject": "ad14b571-2449-4fd5-974e-f525ed965370",
                "role": "marin-coreweave",
                "token_file": "/var/run/marin-auth/storage-token/token",
                "permissions": [
                    {"service": "cw_s3", "actions": ["s3:GetObject"], "resource": "s3://checkpoints/run-1/"}
                ],
            }
        ],
        "max_token_ttl": 300,
        "loom": {
            "base_url": "https://loom.sessions.svc:8000",
            "observer_token_file": "/var/run/marin-auth/loom-token/token",
        },
    }
    return GrantBrokerArgs(
        namespace="management",
        image="marin-auth@sha256:" + "b" * 64,
        storage_class="block",
        storage_size="1Gi",
        policy=policy,
        token_secret_names=("broker-token", "loom-token", "storage-token"),
        listener_tls_secret="broker-tls",
        ca_secret="combined-trust-bundle",
        client_namespace="sessions",
        client_pod_labels={"app.kubernetes.io/name": "loom"},
        dns_cidrs=("10.16.0.10/32", "10.16.0.30/32"),
        https_egress_cidrs=(),
    )


@pulumi.runtime.test
def test_grant_broker_publishes_runtime_policy_and_keeps_credentials_in_external_mounts(recording_mocks, broker_args):
    mocks = recording_mocks
    pulumi.runtime.set_mocks(mocks, project="marin-auth", stack="test", preview=False)
    broker = GrantBroker(
        "broker",
        broker_args,
        k8s_provider=k8s.Provider("test", kubeconfig="", context="test"),
    )

    def check(_: list) -> None:
        resources = {resource.typ: resource.inputs for resource in mocks.resources}
        assert "kubernetes:core/v1:Secret" not in resources
        config = json.loads(resources["kubernetes:core/v1:ConfigMap"]["data"]["policy.yaml"])
        stateful = resources["kubernetes:apps/v1:StatefulSet"]["spec"]
        assert stateful["replicas"] == 1
        assert stateful["updateStrategy"] == {"type": "RollingUpdate"}
        assert stateful["persistentVolumeClaimRetentionPolicy"] == {"whenDeleted": "Retain", "whenScaled": "Retain"}
        pod = stateful["template"]["spec"]
        assert pod["automountServiceAccountToken"] is False
        mounts = {mount["name"]: mount for mount in pod["containers"][0]["volumeMounts"]}
        assert config["database"].startswith(mounts["data"]["mountPath"] + "/")
        assert config["broker_token_file"].startswith(mounts["token-0"]["mountPath"] + "/")
        assert mounts["token-0"]["readOnly"] is True
        volumes = {volume["name"]: volume for volume in pod["volumes"]}
        assert volumes["ca"]["secret"]["items"] == [{"key": "ca.crt", "path": "ca.crt"}]
        assert volumes["ca"]["secret"]["secretName"] == "combined-trust-bundle"
        assert all("hostPath" not in volume for volume in pod["volumes"])
        assert pod["containers"][0]["securityContext"]["allowPrivilegeEscalation"] is False
        assert resources["kubernetes:networking.k8s.io/v1:NetworkPolicy"]["spec"]["ingress"] == [
            {
                "from": [
                    {
                        "namespaceSelector": {"matchLabels": {"kubernetes.io/metadata.name": "sessions"}},
                        "podSelector": {"matchLabels": {"app.kubernetes.io/name": "loom"}},
                    }
                ],
                "ports": [{"port": 8090}],
            }
        ]
        assert {
            "to": [
                {
                    "namespaceSelector": {"matchLabels": {"kubernetes.io/metadata.name": "sessions"}},
                    "podSelector": {"matchLabels": {"app.kubernetes.io/name": "loom"}},
                }
            ],
            "ports": [{"port": 8443}],
        } in resources["kubernetes:networking.k8s.io/v1:NetworkPolicy"]["spec"]["egress"]

    return pulumi.Output.all(broker.stateful_set.spec, broker.network_policy.spec).apply(check)


@pytest.mark.parametrize("collision", ["observer-broker", "observer-storage", "storage-broker", "storage-storage"])
@pulumi.runtime.test
def test_grant_broker_rejects_reusing_credentials_across_trust_boundaries(recording_mocks, broker_args, collision):
    pulumi.runtime.set_mocks(recording_mocks, project="marin-auth", stack="test", preview=False)
    policy = broker_args.policy
    storage = policy["storage"][0]
    if collision == "observer-broker":
        policy["loom"]["observer_token_file"] = policy["broker_token_file"]
    elif collision == "observer-storage":
        policy["loom"]["observer_token_file"] = storage["token_file"]
    elif collision == "storage-broker":
        storage["token_file"] = policy["broker_token_file"]
    else:
        policy["storage"].append({**storage, "name": "other-identity"})
    # Keep the declared mounts consistent so only the credential alias is rejected.
    paths = [
        policy["broker_token_file"],
        policy["loom"]["observer_token_file"],
        *(item["token_file"] for item in policy["storage"]),
    ]
    secrets = tuple(sorted({Path(path).parent.name for path in paths}))

    with pytest.raises(ValueError, match="distinct files"):
        GrantBroker(
            "broker",
            replace(broker_args, token_secret_names=secrets),
            k8s_provider=k8s.Provider("test", kubeconfig="", context="test"),
        )
