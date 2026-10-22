# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Security contracts of the public Kubernetes and OpenBao API payloads."""

import json
from pathlib import Path

import pytest
import yaml
from iac.kubernetes.openbao import API_PORT, RAFT_PORT, Exposure, OpenBaoDeployment, openbao_manifests
from iac.openbao import CredentialProfile, credential_catalog

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def deployment() -> OpenBaoDeployment:
    stack = yaml.safe_load((REPOSITORY_ROOT / "infra/openbao/Pulumi.dev.yaml").read_text())
    return OpenBaoDeployment.model_validate(stack["config"]["marin-openbao:deployment"])


def test_openbao_bootstrap_preserves_discovery_without_serving_sealed_nodes(deployment):
    resources = {item["kind"] + "/" + item["metadata"]["name"]: item for item in openbao_manifests(deployment)}
    discovery = resources["Service/openbao-internal"]["spec"]
    api = resources["Service/openbao"]["spec"]
    pod = resources["StatefulSet/openbao"]["spec"]["template"]["spec"]
    assert discovery["publishNotReadyAddresses"] is True
    assert not api.get("publishNotReadyAddresses", False)
    assert [port["port"] for port in api["ports"]] == [API_PORT]
    probe = pod["containers"][0]["readinessProbe"]["httpGet"]
    assert probe["scheme"] == "HTTPS"
    assert probe["path"] == "/v1/sys/health?standbyok=true"
    assert "livenessProbe" not in pod["containers"][0]


@pytest.mark.parametrize("replicas", [1, 3, 5])
def test_openbao_disruptions_keep_quorum_and_persistent_identity(deployment, replicas):
    resources = openbao_manifests(deployment.model_copy(update={"replicas": replicas}))
    budget = next(item["spec"] for item in resources if item["kind"] == "PodDisruptionBudget")
    stateful = next(item["spec"] for item in resources if item["kind"] == "StatefulSet")
    assert budget["minAvailable"] > replicas / 2
    assert stateful["updateStrategy"] == {"type": "OnDelete"}
    assert set(stateful["persistentVolumeClaimRetentionPolicy"].values()) == {"Retain"}
    assert (
        stateful["template"]["spec"]["affinity"]["podAntiAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"][0][
            "topologyKey"
        ]
        == "kubernetes.io/hostname"
    )


def test_openbao_private_tls_and_bootstrap_material_stay_out_of_pulumi(deployment):
    resources = openbao_manifests(deployment)
    assert {"Secret", "Role", "RoleBinding", "ClusterRole", "ClusterRoleBinding", "Job"}.isdisjoint(
        item["kind"] for item in resources
    )
    pod = next(item["spec"]["template"]["spec"] for item in resources if item["kind"] == "StatefulSet")
    assert pod["automountServiceAccountToken"] is False
    assert not pod.get("hostNetwork", False)
    container = pod["containers"][0]
    assert container["securityContext"]["capabilities"] == {"drop": ["ALL"]}
    assert not container.get("envFrom")
    assert not any("valueFrom" in entry and "secretKeyRef" in entry["valueFrom"] for entry in container["env"])
    server = next(item["data"]["server.json"] for item in resources if item["kind"] == "ConfigMap")
    config = json.loads(server)
    assert config["listener"]["tcp"]["tls_key_file"] == "/openbao/tls/tls.key"
    assert "seal" not in config
    assert "retry_join" not in config["storage"]["raft"]


def test_openbao_external_route_preserves_tls_and_never_exposes_raft(deployment):
    internal = openbao_manifests(deployment)
    assert not any(item["kind"] == "IngressRouteTCP" for item in internal)
    public = openbao_manifests(deployment.model_copy(update={"exposure": Exposure.TLS_PASSTHROUGH}))
    route = next(item["spec"] for item in public if item["kind"] == "IngressRouteTCP")
    assert route["tls"] == {"passthrough": True}
    assert all(service["port"] == API_PORT for entry in route["routes"] for service in entry["services"])
    network = next(item["spec"] for item in public if item["kind"] == "NetworkPolicy")
    raft_rules = [rule for rule in network["ingress"] if any(port["port"] == RAFT_PORT for port in rule["ports"])]
    assert len(raft_rules) == 1
    assert all("namespaceSelector" not in source for source in raft_rules[0]["from"])


def test_openbao_controller_can_only_delegate_declared_profiles_without_inheriting_identity():
    catalog = credential_catalog(
        "cw-pilot",
        (
            CredentialProfile(name="reader", objects=("team/model",)),
            CredentialProfile(name="writer", objects=("team/api",)),
        ),
    )
    controller_paths = json.loads(catalog["policies"]["iris-controller"])["path"]
    assert set(controller_paths) == {
        "auth/token/create/reader",
        "auth/token/create/writer",
        "auth/token/lookup-self",
        "auth/token/revoke-accessor",
    }
    assert "auth/token/create" not in controller_paths
    for profile, role in catalog["token_roles"].items():
        gate = controller_paths[f"auth/token/create/{profile}"]
        assert gate["required_parameters"] == ["entity_alias"]
        assert gate["allowed_parameters"]["entity_alias"] == role["allowed_entity_aliases"]
        assert not any("*" in alias for alias in role["allowed_entity_aliases"])
        assert not role["orphan"]
        assert not role["renewable"]
        assert role["token_no_default_policy"]
        assert role["token_type"] == "service"
        assert role["token_explicit_max_ttl"] <= 600
    reader = json.loads(catalog["policies"]["workload-reader"])["path"]
    assert reader == {"secret/data/team/model": {"capabilities": ["read"]}}
    assert "secret/data/team/api" not in reader
    # A fresh login must pass the broker's parent-life gate before issuing any child.
    login_role = catalog["controller_jwt_role_parameters"]
    child_limit = max(role["token_explicit_max_ttl"] for role in catalog["token_roles"].values())
    assert login_role["token_ttl"] > child_limit + login_role["clock_skew_leeway"]


def test_openbao_dev_and_main_catalogs_have_disjoint_cluster_authority():
    root = REPOSITORY_ROOT / "infra/openbao"
    main = yaml.safe_load((root / "catalog.main.yaml").read_text())
    dev = yaml.safe_load((root / "catalog.dev.yaml").read_text())
    assert set(main["clusters"]).isdisjoint(dev["clusters"])
    # Dev and main share a profile name while their vault data and trust are independent.
    for source in (main, dev):
        profiles = tuple(CredentialProfile.model_validate(profile) for profile in source["profiles"])
        for cluster in source["clusters"]:
            catalog = credential_catalog(cluster, profiles)
            assert all(f":{cluster}:" in alias["name"] for alias in catalog["required_precreated_aliases"])
