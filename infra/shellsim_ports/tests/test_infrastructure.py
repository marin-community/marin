# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check provider wire manifests with Pulumi mocks; never contact a live cluster."""

import pulumi
from pulumi.runtime import MockCallArgs, MockResourceArgs, Mocks
from pulumi.runtime.mocks import MockMonitor

from infra.shellsim_ports.infrastructure import ProxyConfig, deploy


class RecordingMocks(Mocks):
    def __init__(self) -> None:
        self.resources: list[MockResourceArgs] = []

    def new_resource(self, args: MockResourceArgs):
        self.resources.append(args)
        outputs = dict(args.inputs)
        if args.typ == "docker-build:index:Image":
            outputs["ref"] = "ghcr.io/marin-community/iris-task@sha256:" + "a" * 64
        return args.name, outputs

    def call(self, args: MockCallArgs):
        raise AssertionError(f"Unexpected external lookup: {args.token}")


class RecordingMonitor(MockMonitor):
    def __init__(self, mocks: Mocks) -> None:
        super().__init__(mocks)
        self.dependencies: dict[str, list[str]] = {}
        self.urns: dict[str, str] = {}

    def RegisterResource(self, request):
        self.dependencies[request.type] = list(request.dependencies)
        response = super().RegisterResource(request)
        self.urns[request.type] = response.urn
        return response


@pulumi.runtime.test
def test_proxy_manifests_keep_credentials_external_and_limit_network(monkeypatch):
    monkeypatch.setenv("KUBECONFIG", "/not-read-by-mocks")
    mocks = RecordingMocks()
    monitor = RecordingMonitor(mocks)
    pulumi.runtime.set_mocks(mocks, project="marin-shellsim-ports", stack="test", preview=True, monitor=monitor)
    url = deploy(
        ProxyConfig(
            cluster="cw-us-east-02a",
            hostname="ports.oa.dev",
            source_revision="d714557abfc9a09ac9205294ab5448f7e7e4e63e",
            prefix="s3://marin-us-east-02a/marin/shellsim/ports",
        )
    )

    def check(value):
        assert value == "https://ports.oa.dev"
        resources = {resource.typ: resource.inputs for resource in mocks.resources}
        deployment = resources["kubernetes:apps/v1:Deployment"]["spec"]["template"]["spec"]
        container = deployment["containers"][0]
        assert container["envFrom"] == [{"secretRef": {"name": "shellsim-ports-readonly"}}]
        assert container["args"] == ["--prefix", "s3://marin-us-east-02a/marin/shellsim/ports"]
        assert deployment["automountServiceAccountToken"] is False
        assert not any(resource.typ == "kubernetes:core/v1:Secret" for resource in mocks.resources)
        assert resources["pulumi:providers:kubernetes"]["kubeconfig"] == ""
        assert resources["pulumi:providers:kubernetes"]["context"] == "marin-gpu_US-EAST-02A"
        assert "apiToken" not in resources["pulumi:providers:cloudflare"]
        policy_type = "kubernetes:cilium.io/v2:CiliumNetworkPolicy"
        assert monitor.urns[policy_type] in monitor.dependencies["kubernetes:apps/v1:Deployment"]
        policy = resources[policy_type]["spec"]
        dns_rule = next(rule for rule in policy["egress"] if "toEndpoints" in rule)
        assert dns_rule["toEndpoints"] == [
            {
                "matchLabels": {
                    "k8s:io.kubernetes.pod.namespace": "kube-system",
                    "k8s:k8s-app": "node-local-dns",
                }
            }
        ]
        fqdn = [rule for rule in policy["egress"] if "toFQDNs" in rule]
        assert fqdn == [
            {
                "toFQDNs": [{"matchName": "marin-us-east-02a.cwobject.com"}],
                "toPorts": [{"ports": [{"port": "443", "protocol": "TCP"}]}],
            }
        ]
        ingress = resources["kubernetes:networking.k8s.io/v1:Ingress"]["spec"]
        assert ingress["tls"][0]["hosts"] == ["ports.oa.dev"]
        assert ingress["rules"][0]["http"]["paths"][0]["backend"]["service"]["name"] == "shellsim-ports"
        assert (
            resources["cloudflare:index/dnsRecord:DnsRecord"]["content"]
            == "iris-cw-us-east-02a.208261-marin-gpu.coreweave.app"
        )
        assert resources["docker-build:index:Image"]["buildOnPreview"] is False

    return url.apply(check)
