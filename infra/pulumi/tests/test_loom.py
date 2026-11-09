# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pulumi
import pulumi_kubernetes as k8s
import pytest
from iac.auth.loom import Loom, LoomArgs, loom_deployment_manifest


def restricted_profile() -> dict:
    return {
        "profile": {
            "name": "reviewed",
            "agent_kind": "codex",
            "model": "operator-reviewed-model",
            "protocol": "acp",
            "class": "automation",
            "strict": True,
            "env_clear": True,
            "restricted": True,
            "mcp_access": {"mode": "none", "groups": []},
        },
        "env": [],
    }


@pytest.mark.parametrize("exposure", ["bootstrap", "enrolling", "serving"])
@pulumi.runtime.test
def test_loom_keeps_sessions_unprivileged_and_exposes_only_tls_after_enrollment(exposure, recording_mocks):
    mocks = recording_mocks
    pulumi.runtime.set_mocks(mocks, project="marin-loom-cw", stack="test", preview=False)
    declaration = loom_deployment_manifest(
        issuer="https://marin-auth.management.svc:8090",
        public_keys=["audited-public-key"],
        profiles=[restricted_profile()],
        profile_pins={
            "reviewed": {"cwd": "/home/app/workspace/repo", "revision": 1, "resolver_revision": "composition-sha256"}
        },
        exposure=exposure,
    )
    service = Loom(
        "loom",
        LoomArgs(
            namespace="sessions",
            management_namespace=pulumi.Output.from_input("management"),
            image="loom@sha256:" + "a" * 64,
            proxy_image="caddy@sha256:" + "b" * 64,
            exposure=exposure,
            storage_class="block",
            storage_size="10Gi",
            hostname="loom.example.org",
            ingress_tls_secret="edge-tls",
            listener_tls_secret="listener-tls",
            listener_tls_server_name="loom.sessions.svc",
            ingress_namespace="traefik",
            ingress_pod_labels={"app.kubernetes.io/name": "traefik"},
            dns_cidrs=("10.16.0.10/32",),
            https_egress_cidrs=("203.0.113.1/32",),
            deployment=declaration,
            env_secret_refs={"OPENAI_API_KEY": {"name": "model-credential", "key": "token"}},
            ca_secret="approved-trust-bundle",
        ),
        k8s_provider=k8s.Provider("test", kubeconfig="", context="test"),
    )

    def check(_: list) -> None:
        resources = {resource.typ: resource.inputs for resource in mocks.resources}
        assert "kubernetes:core/v1:Secret" not in resources
        assert not any("rbac" in resource.typ for resource in mocks.resources)
        assert (
            resources["kubernetes:core/v1:Namespace"]["metadata"]["labels"]["pod-security.kubernetes.io/enforce"]
            == "restricted"
        )
        stateful = resources["kubernetes:apps/v1:StatefulSet"]["spec"]
        assert stateful["replicas"] == 1
        assert stateful["persistentVolumeClaimRetentionPolicy"] == {"whenDeleted": "Retain", "whenScaled": "Retain"}
        pod = stateful["template"]["spec"]
        assert pod["automountServiceAccountToken"] is False
        assert not any("hostPath" in volume for volume in pod["volumes"])
        assert all(container["securityContext"]["allowPrivilegeEscalation"] is False for container in pod["containers"])
        loom, proxy = pod["containers"]
        assert "command" not in loom
        assert loom["args"][-1] == "127.0.0.1:7878"
        for kind in ("startupProbe", "readinessProbe", "livenessProbe"):
            probe = loom[kind]
            assert "httpGet" not in probe and "tcpSocket" not in probe
            command = probe["exec"]["command"]
            assert command[-1] == f"http://{loom['args'][-1]}/api/health"
        assert proxy["readinessProbe"]["httpGet"] == {"path": "/api/health", "port": "https", "scheme": "HTTPS"}
        assert {item["name"]: item.get("value") for item in loom["env"]}["LOOM_RUNNER"] == "local"
        assert proxy["ports"] == [{"name": "https", "containerPort": 8443}]
        assert resources["kubernetes:core/v1:Service"]["spec"]["ports"][0]["port"] == 8443
        rendered = json.loads(resources["kubernetes:core/v1:ConfigMap"]["data"]["deployment.json"])
        assert rendered["settings"]["auth.trust_loopback"] is False
        if exposure != "bootstrap":
            trust = json.loads(rendered["settings"]["auth.shared_grants"])
            assert trust["authorize_url"] == "https://marin-auth.management.svc:8090/v1/authorize"
            assert trust["credentials_url"] == "https://marin-auth.management.svc:8090/v1/loom/credentials"
            assert trust["ca_file"] == "/etc/loom-ca/ca.crt"
            if exposure == "serving":
                assert resources["kubernetes:networking.k8s.io/v1:Ingress"]["spec"]["tls"]
                ingress = resources["kubernetes:networking.k8s.io/v1:NetworkPolicy"]["spec"]["ingress"]
                assert ingress == [
                    {
                        "from": [
                            {
                                "namespaceSelector": {"matchLabels": {"kubernetes.io/metadata.name": "traefik"}},
                                "podSelector": {"matchLabels": {"app.kubernetes.io/name": "traefik"}},
                            },
                            {
                                "namespaceSelector": {"matchLabels": {"kubernetes.io/metadata.name": "management"}},
                                "podSelector": {"matchLabels": {"app.kubernetes.io/name": "marin-auth"}},
                            },
                        ],
                        "ports": [{"port": 8443}],
                    }
                ]
            else:
                assert "kubernetes:networking.k8s.io/v1:Ingress" not in resources
                assert resources["kubernetes:networking.k8s.io/v1:NetworkPolicy"]["spec"]["ingress"] == []
        else:
            assert rendered["settings"]["auth.shared_grants"] == ""
            assert "kubernetes:networking.k8s.io/v1:Ingress" not in resources
            assert resources["kubernetes:networking.k8s.io/v1:NetworkPolicy"]["spec"]["ingress"] == []

    ready = [service.stateful_set.spec, service.network_policy.spec]
    if service.ingress is not None:
        ready.append(service.ingress.spec)
    return pulumi.Output.all(*ready).apply(check)


def test_loom_refuses_shell_authority_in_a_shared_grant_profile():
    profile = restricted_profile()
    profile["profile"]["runtime_permissions"] = ["Bash"]
    with pytest.raises(ValueError, match="no shell or custom MCP"):
        loom_deployment_manifest(
            issuer="https://auth.example.org",
            public_keys=["public-key"],
            profiles=[profile],
            profile_pins={},
            exposure="bootstrap",
        )
