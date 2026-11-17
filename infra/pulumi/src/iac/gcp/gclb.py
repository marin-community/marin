# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GCE VM ingress through a shared external HTTPS load balancer and IAP."""

from dataclasses import dataclass

import pulumi
import pulumi_gcp as gcp
from iris.cluster.controller.native_proxy import PROXY_RELAY_TIMEOUT_SECONDS
from iris.cluster.platforms.vm_lifecycle import DEFAULT_CONTROLLER_PORT

from iac.gcp.firewall import FirewallPort, GcpFirewallRuleArgs, create_firewall_rule
from iac.imports import NO_IMPORTS, ImportRegistrar

FINELOG_BACKEND_TIMEOUT = 120
GOOGLE_LB_RANGES = ("130.211.0.0/22", "35.191.0.0/16")
VPC_PRIVATE_RANGE = "10.0.0.0/8"
PUBLIC_IPV4_RANGE = "0.0.0.0/0"
TOKEN_PROXY_PATHS = ("/proxy/t", "/proxy/t/*")
FIREWALL_ALLOW_PRIORITY = 900
FIREWALL_DENY_PRIORITY = 1000
ARMOR_ALLOW_PRIORITY = 1000
ARMOR_DEFAULT_PRIORITY = 2147483647
LOAD_BALANCING_SCHEME = "EXTERNAL_MANAGED"
BACKEND_BALANCING_MODE = "RATE"
BACKEND_MAX_RATE_PER_ENDPOINT = 1000
BACKEND_CONNECTION_DRAINING_TIMEOUT = 0
HEALTH_CHECK_PORT_SPECIFICATION = "USE_FIXED_PORT"


@dataclass(frozen=True)
class ControllerIngress:
    cluster: str
    domain: str
    zone: str
    instance: str
    ip_address: str
    backend_service_id: str
    network_tag: str
    programmatic_clients: tuple[str, ...]
    port: int = DEFAULT_CONTROLLER_PORT
    token_proxy: bool = True
    deny_public: bool = False


@dataclass(frozen=True)
class FinelogIngress:
    cluster: str
    domain: str
    zone: str
    instance: str
    ip_address: str
    port: int
    network_tag: str
    sender_source_ranges: tuple[str, ...]


@dataclass(frozen=True)
class GcpGclbIapArgs:
    project: str
    project_number: str
    frontend_name: str
    controllers: tuple[ControllerIngress, ...]
    network: str
    subnetwork: str
    finelogs: tuple[FinelogIngress, ...] = ()


@dataclass(frozen=True)
class _BackendResources:
    service: gcp.compute.BackendService
    certificate: gcp.compute.ManagedSslCertificate
    matcher_name: str
    domain: str
    path_rules: tuple[gcp.compute.URLMapPathMatcherPathRuleArgs, ...] = ()


@dataclass(frozen=True)
class _NegBackendResources:
    neg: gcp.compute.NetworkEndpointGroup
    health_check: gcp.compute.HealthCheck
    service: gcp.compute.BackendService


@dataclass(frozen=True)
class _GcpResourceContext:
    args: GcpGclbIapArgs
    parent: pulumi.Resource
    provider: pulumi.ProviderResource
    imports: ImportRegistrar

    def options(
        self,
        *,
        depends_on: tuple[pulumi.Resource, ...] = (),
        protect: bool = False,
        retain_on_delete: bool = False,
        ignore_changes: tuple[str, ...] = (),
    ) -> pulumi.ResourceOptions:
        return pulumi.ResourceOptions(
            parent=self.parent,
            provider=self.provider,
            depends_on=list(depends_on),
            protect=protect,
            retain_on_delete=retain_on_delete,
            ignore_changes=list(ignore_changes),
        )

    def register(self, resource: pulumi.Resource, provider_id: str) -> None:
        self.imports.register(resource, parent=self.parent, provider_id=provider_id)


def _certificate_name(domain: str) -> str:
    return f"{domain.replace('.', '-')}-cert"


def _managed_certificate(
    *,
    logical_name: str,
    domain: str,
    context: _GcpResourceContext,
) -> gcp.compute.ManagedSslCertificate:
    name = _certificate_name(domain)
    certificate = gcp.compute.ManagedSslCertificate(
        logical_name,
        project=context.args.project,
        name=name,
        managed=gcp.compute.ManagedSslCertificateManagedArgs(domains=[domain]),
        opts=context.options(),
    )
    context.register(
        certificate,
        f"projects/{context.args.project}/global/sslCertificates/{name}",
    )
    return certificate


def _neg_backend(
    *,
    logical_prefix: str,
    physical_prefix: str,
    zone: str,
    instance: str,
    ip_address: str,
    port: int,
    timeout: int,
    context: _GcpResourceContext,
    iap: bool = False,
    security_policy: pulumi.Input[str] | None = None,
) -> _NegBackendResources:
    args = context.args
    neg_name = f"{physical_prefix}-neg"
    neg = gcp.compute.NetworkEndpointGroup(
        f"{logical_prefix}-neg",
        project=args.project,
        zone=zone,
        name=neg_name,
        network=args.network,
        subnetwork=args.subnetwork,
        network_endpoint_type="GCE_VM_IP_PORT",
        default_port=port,
        opts=context.options(),
    )
    context.register(
        neg,
        f"projects/{args.project}/zones/{zone}/networkEndpointGroups/{neg_name}",
    )
    endpoint = gcp.compute.NetworkEndpoint(
        f"{logical_prefix}-endpoint",
        project=args.project,
        zone=zone,
        network_endpoint_group=neg.name,
        instance=instance,
        ip_address=ip_address,
        port=port,
        opts=context.options(depends_on=(neg,)),
    )
    context.register(
        endpoint,
        f"projects/{args.project}/zones/{zone}/networkEndpointGroups/{neg_name}/" f"{instance}/{ip_address}/{port}",
    )

    health_check_name = f"{physical_prefix}-hc"
    health_check = gcp.compute.HealthCheck(
        f"{logical_prefix}-health-check",
        project=args.project,
        name=health_check_name,
        check_interval_sec=10,
        timeout_sec=5,
        healthy_threshold=2,
        unhealthy_threshold=3,
        http_health_check=gcp.compute.HealthCheckHttpHealthCheckArgs(
            port=port,
            port_specification=HEALTH_CHECK_PORT_SPECIFICATION,
            request_path="/health",
        ),
        opts=context.options(),
    )
    context.register(
        health_check,
        f"projects/{args.project}/global/healthChecks/{health_check_name}",
    )

    service_name = f"{physical_prefix}-be"
    service = gcp.compute.BackendService(
        f"{logical_prefix}-backend",
        project=args.project,
        name=service_name,
        protocol="HTTP",
        port_name="http",
        timeout_sec=timeout,
        connection_draining_timeout_sec=BACKEND_CONNECTION_DRAINING_TIMEOUT,
        load_balancing_scheme=LOAD_BALANCING_SCHEME,
        health_checks=health_check.id,
        backends=[
            gcp.compute.BackendServiceBackendArgs(
                group=neg.id,
                balancing_mode=BACKEND_BALANCING_MODE,
                max_rate_per_endpoint=BACKEND_MAX_RATE_PER_ENDPOINT,
            )
        ],
        iap=gcp.compute.BackendServiceIapArgs(enabled=True) if iap else None,
        security_policy=security_policy,
        opts=context.options(depends_on=(endpoint,)),
    )
    context.register(
        service,
        f"projects/{args.project}/global/backendServices/{service_name}",
    )
    return _NegBackendResources(neg=neg, health_check=health_check, service=service)


def _controller_backend(
    controller: ControllerIngress,
    *,
    context: _GcpResourceContext,
    relay_timeout: int,
) -> _BackendResources:
    args = context.args
    physical_prefix = f"iris-{controller.cluster}"
    backend = _neg_backend(
        logical_prefix=controller.cluster,
        physical_prefix=physical_prefix,
        zone=controller.zone,
        instance=controller.instance,
        ip_address=controller.ip_address,
        port=controller.port,
        timeout=relay_timeout,
        context=context,
        iap=True,
    )
    create_firewall_rule(
        f"{controller.cluster}-allow-lb",
        GcpFirewallRuleArgs(
            project=args.project,
            name=f"{physical_prefix}-allow-lb",
            network=args.network,
            priority=FIREWALL_ALLOW_PRIORITY,
            source_ranges=GOOGLE_LB_RANGES,
            target_tags=(controller.network_tag,),
            allows=(FirewallPort(protocol="tcp", ports=(str(controller.port),)),),
        ),
        gcp_provider=context.provider,
        parent=context.parent,
        imports=context.imports,
    )
    if controller.deny_public:
        create_firewall_rule(
            f"{controller.cluster}-deny-public-{controller.port}",
            GcpFirewallRuleArgs(
                project=args.project,
                name=f"{physical_prefix}-deny-public-{controller.port}",
                network=args.network,
                priority=FIREWALL_DENY_PRIORITY,
                source_ranges=(PUBLIC_IPV4_RANGE,),
                target_tags=(controller.network_tag,),
                denies=(FirewallPort(protocol="tcp", ports=(str(controller.port),)),),
            ),
            gcp_provider=context.provider,
            parent=context.parent,
            imports=context.imports,
        )

    settings_name = f"projects/{args.project_number}/iap_web/compute/services/{controller.backend_service_id}"
    settings = gcp.iap.Settings(
        f"{controller.cluster}-iap-settings",
        name=settings_name,
        access_settings=gcp.iap.SettingsAccessSettingsArgs(
            oauth_settings=gcp.iap.SettingsAccessSettingsOauthSettingsArgs(
                programmatic_clients=list(controller.programmatic_clients)
            )
        ),
        # The legacy Web OAuth anchor remains external because its secret must not
        # enter Pulumi state. Pulumi owns the non-secret programmatic client list.
        opts=context.options(
            depends_on=(backend.service,),
            ignore_changes=(
                "accessSettings.oauthSettings.clientId",
                "accessSettings.oauthSettings.clientSecret",
                "accessSettings.oauthSettings.clientSecretSha256",
            ),
        ),
    )
    context.register(settings, f"{settings_name}/iapSettings")

    path_rules: tuple[gcp.compute.URLMapPathMatcherPathRuleArgs, ...] = ()
    if controller.token_proxy:
        proxy_name = f"{physical_prefix}-proxy-be"
        proxy = gcp.compute.BackendService(
            f"{controller.cluster}-token-proxy-backend",
            project=args.project,
            name=proxy_name,
            protocol="HTTP",
            port_name="http",
            timeout_sec=relay_timeout,
            connection_draining_timeout_sec=BACKEND_CONNECTION_DRAINING_TIMEOUT,
            load_balancing_scheme=LOAD_BALANCING_SCHEME,
            health_checks=backend.health_check.id,
            backends=[
                gcp.compute.BackendServiceBackendArgs(
                    group=backend.neg.id,
                    balancing_mode=BACKEND_BALANCING_MODE,
                    max_rate_per_endpoint=BACKEND_MAX_RATE_PER_ENDPOINT,
                )
            ],
            opts=context.options(),
        )
        context.register(proxy, f"projects/{args.project}/global/backendServices/{proxy_name}")
        path_rules = (gcp.compute.URLMapPathMatcherPathRuleArgs(paths=list(TOKEN_PROXY_PATHS), service=proxy.id),)

    certificate = _managed_certificate(
        logical_name=f"{controller.cluster}-certificate",
        domain=controller.domain,
        context=context,
    )
    return _BackendResources(
        service=backend.service,
        certificate=certificate,
        matcher_name=controller.cluster,
        domain=controller.domain,
        path_rules=path_rules,
    )


def _finelog_backend(
    finelog: FinelogIngress,
    *,
    context: _GcpResourceContext,
) -> _BackendResources:
    args = context.args
    logical_prefix = f"finelog-{finelog.cluster}"
    physical_prefix = logical_prefix
    armor_name = f"{physical_prefix}-armor"
    armor = gcp.compute.SecurityPolicy(
        f"{logical_prefix}-armor",
        project=args.project,
        name=armor_name,
        description=f"relay sources admitted to {finelog.instance}",
        rules=[
            gcp.compute.SecurityPolicyRuleArgs(
                action="allow",
                description="",
                preview=False,
                priority=ARMOR_ALLOW_PRIORITY,
                match=gcp.compute.SecurityPolicyRuleMatchArgs(
                    versioned_expr="SRC_IPS_V1",
                    config=gcp.compute.SecurityPolicyRuleMatchConfigArgs(
                        src_ip_ranges=sorted(finelog.sender_source_ranges)
                    ),
                ),
            ),
            gcp.compute.SecurityPolicyRuleArgs(
                action="deny(403)",
                description="default rule",
                preview=False,
                priority=ARMOR_DEFAULT_PRIORITY,
                match=gcp.compute.SecurityPolicyRuleMatchArgs(
                    versioned_expr="SRC_IPS_V1",
                    config=gcp.compute.SecurityPolicyRuleMatchConfigArgs(src_ip_ranges=["*"]),
                ),
            ),
        ],
        opts=context.options(),
    )
    context.register(armor, f"projects/{args.project}/global/securityPolicies/{armor_name}")

    backend = _neg_backend(
        logical_prefix=logical_prefix,
        physical_prefix=physical_prefix,
        zone=finelog.zone,
        instance=finelog.instance,
        ip_address=finelog.ip_address,
        port=finelog.port,
        timeout=FINELOG_BACKEND_TIMEOUT,
        context=context,
        security_policy=armor.id,
    )
    create_firewall_rule(
        f"{logical_prefix}-allow-lb",
        GcpFirewallRuleArgs(
            project=args.project,
            name=f"{physical_prefix}-allow-lb",
            network=args.network,
            priority=FIREWALL_ALLOW_PRIORITY,
            source_ranges=(VPC_PRIVATE_RANGE, *GOOGLE_LB_RANGES),
            target_tags=(finelog.network_tag,),
            allows=(FirewallPort(protocol="tcp", ports=(str(finelog.port),)),),
        ),
        gcp_provider=context.provider,
        parent=context.parent,
        imports=context.imports,
    )
    create_firewall_rule(
        f"{logical_prefix}-deny-public-{finelog.port}",
        GcpFirewallRuleArgs(
            project=args.project,
            name=f"{physical_prefix}-deny-public-{finelog.port}",
            network=args.network,
            priority=FIREWALL_DENY_PRIORITY,
            source_ranges=(PUBLIC_IPV4_RANGE,),
            target_tags=(finelog.network_tag,),
            denies=(FirewallPort(protocol="tcp", ports=(str(finelog.port),)),),
        ),
        gcp_provider=context.provider,
        parent=context.parent,
        imports=context.imports,
    )
    certificate = _managed_certificate(
        logical_name=f"{logical_prefix}-certificate",
        domain=finelog.domain,
        context=context,
    )
    return _BackendResources(
        service=backend.service,
        certificate=certificate,
        matcher_name=logical_prefix,
        domain=finelog.domain,
    )


def _validate_args(args: GcpGclbIapArgs) -> None:
    if not args.controllers:
        raise ValueError("GCLB requires at least one controller backend")
    if args.frontend_name not in {controller.cluster for controller in args.controllers}:
        raise ValueError("GCLB frontend_name must identify one controller backend")
    domains = [backend.domain for backend in (*args.controllers, *args.finelogs)]
    if len(domains) != len(set(domains)):
        raise ValueError("GCLB backend domains must be unique")
    matchers = [backend.cluster for backend in args.controllers] + [
        f"finelog-{backend.cluster}" for backend in args.finelogs
    ]
    if len(matchers) != len(set(matchers)):
        raise ValueError("GCLB path matcher names must be unique")
    for finelog in args.finelogs:
        if not finelog.sender_source_ranges:
            raise ValueError(f"finelog {finelog.cluster!r} requires at least one sender source range")


class GcpGclbIap(pulumi.ComponentResource):
    """Own the shared GCLB, controller IAP backends, and finelog sender ingress."""

    ip_address: pulumi.Output[str]

    def __init__(
        self,
        name: str,
        args: GcpGclbIapArgs,
        *,
        gcp_provider: pulumi.ProviderResource,
        imports: ImportRegistrar = NO_IMPORTS,
        opts: pulumi.ResourceOptions | None = None,
    ) -> None:
        super().__init__("marin:gcp:GcpGclbIap", name, None, opts)
        _validate_args(args)
        context = _GcpResourceContext(
            args=args,
            parent=self,
            provider=gcp_provider,
            imports=imports,
        )

        controller_resources = [
            _controller_backend(
                controller,
                context=context,
                relay_timeout=PROXY_RELAY_TIMEOUT_SECONDS,
            )
            for controller in args.controllers
        ]
        finelog_resources = [_finelog_backend(finelog, context=context) for finelog in args.finelogs]
        backends = [*controller_resources, *finelog_resources]
        # Route order is semantically irrelevant to GCP but positional in provider state.
        # Reverse matcher-name order preserves the imported map and stays deterministic.
        route_backends = sorted(backends, key=lambda backend: backend.matcher_name, reverse=True)
        primary = next(
            backend
            for controller, backend in zip(args.controllers, controller_resources, strict=True)
            if controller.cluster == args.frontend_name
        )

        address_name = f"iris-{args.frontend_name}-ip"
        address = gcp.compute.GlobalAddress(
            "global-address",
            project=args.project,
            name=address_name,
            address_type="EXTERNAL",
            opts=context.options(protect=True, retain_on_delete=True),
        )
        context.register(address, f"projects/{args.project}/global/addresses/{address_name}")

        url_map_name = f"iris-{args.frontend_name}-urlmap"
        url_map = gcp.compute.URLMap(
            "url-map",
            project=args.project,
            name=url_map_name,
            default_service=primary.service.id,
            host_rules=[
                gcp.compute.URLMapHostRuleArgs(hosts=[backend.domain], path_matcher=backend.matcher_name)
                for backend in route_backends
            ],
            path_matchers=[
                gcp.compute.URLMapPathMatcherArgs(
                    name=backend.matcher_name,
                    default_service=backend.service.id,
                    path_rules=list(backend.path_rules),
                )
                for backend in route_backends
            ],
            opts=context.options(protect=True),
        )
        context.register(url_map, f"projects/{args.project}/global/urlMaps/{url_map_name}")

        proxy_name = f"iris-{args.frontend_name}-https-proxy"
        proxy = gcp.compute.TargetHttpsProxy(
            "https-proxy",
            project=args.project,
            name=proxy_name,
            url_map=url_map.id,
            ssl_certificates=[backend.certificate.id for backend in backends],
            opts=context.options(protect=True),
        )
        context.register(proxy, f"projects/{args.project}/global/targetHttpsProxies/{proxy_name}")

        forwarding_rule_name = f"iris-{args.frontend_name}-fr"
        forwarding_rule = gcp.compute.GlobalForwardingRule(
            "forwarding-rule",
            project=args.project,
            name=forwarding_rule_name,
            ip_address=address.address,
            ip_protocol="TCP",
            port_range="443",
            target=proxy.id,
            load_balancing_scheme=LOAD_BALANCING_SCHEME,
            opts=context.options(protect=True),
        )
        context.register(
            forwarding_rule,
            f"projects/{args.project}/global/forwardingRules/{forwarding_rule_name}",
        )

        self.ip_address = address.address
        self.register_outputs({"ip_address": self.ip_address})
