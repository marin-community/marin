# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import pulumi
import pulumi_gcp as gcp
from iac.gcp.gclb import (
    ControllerIngress,
    FinelogIngress,
    GcpGclbIap,
    GcpGclbIapArgs,
)
from iac.imports import ImportCatalog

PROJECT = "example"
PROJECT_NUMBER = "123456789"


def _args() -> GcpGclbIapArgs:
    return GcpGclbIapArgs(
        project=PROJECT,
        project_number=PROJECT_NUMBER,
        frontend_name="marin",
        network="default",
        subnetwork="default",
        controllers=(
            ControllerIngress(
                cluster="marin",
                domain="iris.example.com",
                zone="us-central1-a",
                instance="iris-controller-marin",
                ip_address="10.0.0.2",
                backend_service_id="111",
                network_tag="iris-marin-controller",
                programmatic_clients=("desktop-client.apps.googleusercontent.com",),
                iap_members=("serviceAccount:infra-probes@example.iam.gserviceaccount.com",),
            ),
            ControllerIngress(
                cluster="marin-dev",
                domain="iris-dev.example.com",
                zone="us-central1-a",
                instance="iris-controller-marin-dev",
                ip_address="10.0.0.3",
                backend_service_id="222",
                network_tag="iris-marin-dev-controller",
                programmatic_clients=("dev-client.apps.googleusercontent.com",),
            ),
        ),
        finelogs=(
            FinelogIngress(
                cluster="marin",
                domain="finelog.example.com",
                zone="us-central1-a",
                instance="finelog-marin",
                ip_address="10.0.0.4",
                port=10001,
                network_tag="finelog-marin-lb",
                sender_source_ranges=("192.0.2.0/24", "198.51.100.0/24"),
            ),
        ),
    )


@dataclass(frozen=True)
class RecordedGclb:
    inputs_by_name: dict[str, dict]
    options_by_name: dict[str, pulumi.ResourceOptions]
    imports: ImportCatalog


def _record_gclb(monkeypatch, args: GcpGclbIapArgs | None = None) -> RecordedGclb:
    if args is None:
        args = _args()
    imports = ImportCatalog()
    options_by_name: dict[str, pulumi.ResourceOptions] = {}
    inputs_by_name: dict[str, dict] = {}

    def resource_recorder(resource_type: str):
        def record(logical_name: str, **kwargs) -> pulumi.Resource:
            options_by_name[logical_name] = kwargs.pop("opts")
            inputs_by_name[logical_name] = kwargs
            resource = object.__new__(pulumi.Resource)
            resource._type = resource_type
            resource._name = logical_name
            for output in ("id", "name", "self_link", "generated_id", "address"):
                setattr(resource, output, logical_name)
            return resource

        return record

    constructors = (
        (gcp.compute, "GlobalAddress", "gcp:compute/globalAddress:GlobalAddress"),
        (gcp.compute, "ManagedSslCertificate", "gcp:compute/managedSslCertificate:ManagedSslCertificate"),
        (gcp.compute, "NetworkEndpointGroup", "gcp:compute/networkEndpointGroup:NetworkEndpointGroup"),
        (gcp.compute, "NetworkEndpoint", "gcp:compute/networkEndpoint:NetworkEndpoint"),
        (gcp.compute, "HealthCheck", "gcp:compute/healthCheck:HealthCheck"),
        (gcp.compute, "BackendService", "gcp:compute/backendService:BackendService"),
        (gcp.compute, "Firewall", "gcp:compute/firewall:Firewall"),
        (gcp.compute, "SecurityPolicy", "gcp:compute/securityPolicy:SecurityPolicy"),
        (gcp.compute, "URLMap", "gcp:compute/uRLMap:URLMap"),
        (gcp.compute, "TargetHttpsProxy", "gcp:compute/targetHttpsProxy:TargetHttpsProxy"),
        (gcp.compute, "GlobalForwardingRule", "gcp:compute/globalForwardingRule:GlobalForwardingRule"),
        (gcp.iap, "Settings", "gcp:iap/settings:Settings"),
        (gcp.iap, "WebBackendServiceIamMember", "gcp:iap/webBackendServiceIamMember:WebBackendServiceIamMember"),
    )
    for namespace, name, resource_type in constructors:
        monkeypatch.setattr(namespace, name, resource_recorder(resource_type))

    def initialize_component(resource, resource_type, name, *_args, **_kwargs):
        resource._type = resource_type
        resource._name = name

    monkeypatch.setattr(pulumi.ComponentResource, "__init__", initialize_component)
    monkeypatch.setattr(GcpGclbIap, "register_outputs", lambda *_args, **_kwargs: None)

    GcpGclbIap("gclb", args, gcp_provider=None, imports=imports)
    return RecordedGclb(inputs_by_name=inputs_by_name, options_by_name=options_by_name, imports=imports)


def test_gclb_preserves_ingress_security_boundaries(monkeypatch) -> None:
    recording = _record_gclb(monkeypatch)
    inputs_by_name = recording.inputs_by_name
    controller = inputs_by_name["marin-backend"]
    controller_proxy = inputs_by_name["marin-token-proxy-backend"]
    finelog = inputs_by_name["finelog-marin-backend"]
    assert controller["iap"].enabled is True
    assert controller_proxy.get("iap") is None
    assert finelog.get("iap") is None

    settings = inputs_by_name["marin-iap-settings"]
    oauth = settings["access_settings"].oauth_settings
    assert oauth.programmatic_clients == ["desktop-client.apps.googleusercontent.com"]
    assert oauth.client_id is None
    assert oauth.client_secret is None
    assert oauth.client_secret_sha256 is None
    assert recording.options_by_name["marin-iap-settings"].ignore_changes == [
        "accessSettings.oauthSettings.clientId",
        "accessSettings.oauthSettings.clientSecret",
        "accessSettings.oauthSettings.clientSecretSha256",
    ]
    url_map = inputs_by_name["url-map"]
    matchers = {matcher.name: matcher for matcher in url_map["path_matchers"]}
    assert set(matchers) == {"marin", "marin-dev", "finelog-marin"}
    for cluster in ("marin", "marin-dev"):
        assert matchers[cluster].path_rules == [
            gcp.compute.URLMapPathMatcherPathRuleArgs(
                paths=["/proxy/t", "/proxy/t/*"],
                service=f"{cluster}-token-proxy-backend",
            )
        ]
    assert matchers["finelog-marin"].path_rules == []

    controller_allow = inputs_by_name["marin-allow-lb"]
    assert controller_allow["source_ranges"] == ["130.211.0.0/22", "35.191.0.0/16"]
    assert controller_allow["target_tags"] == ["iris-marin-controller"]
    assert "marin-deny-public-10000" not in inputs_by_name

    finelog_allow = inputs_by_name["finelog-marin-allow-lb"]
    assert finelog_allow["source_ranges"] == ["10.0.0.0/8", "130.211.0.0/22", "35.191.0.0/16"]
    finelog_deny = inputs_by_name["finelog-marin-deny-public-10001"]
    assert finelog_deny["denies"] == [gcp.compute.FirewallDenyArgs(protocol="tcp", ports=["10001"])]
    assert finelog_deny["source_ranges"] == ["0.0.0.0/0"]
    assert finelog_allow["priority"] < finelog_deny["priority"]

    armor = inputs_by_name["finelog-marin-armor"]
    rules = {rule.priority: rule for rule in armor["rules"]}
    assert rules[1000].action == "allow"
    assert rules[1000].match.config.src_ip_ranges == ["192.0.2.0/24", "198.51.100.0/24"]
    assert rules[2147483647].action == "deny(403)"


def test_gclb_catalogs_every_existing_leaf(monkeypatch) -> None:
    recording = _record_gclb(monkeypatch)
    cataloged = {spec.identity.logical_name for spec in recording.imports.specs}

    assert cataloged == set(recording.inputs_by_name)
