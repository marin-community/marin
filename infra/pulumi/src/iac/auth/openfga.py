# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Internal OpenFGA deployment with an operator-only write API."""

import hashlib
import re
from dataclasses import dataclass

import pulumi
import pulumi_kubernetes as k8s

from iac.auth.broker import BROKER_LABELS
from iac.auth.openbao import dns_egress

FGA_LABELS = {"app.kubernetes.io/name": "openfga"}


@dataclass(frozen=True)
class OpenFGAArgs:
    namespace: str
    image: str
    proxy_image: str
    database_secret: str
    database_ca_secret: str
    token_secret: str
    tls_secret: str
    store_id: str
    dns_cidrs: tuple[str, ...]
    database_cidrs: tuple[str, ...]
    database_namespace: str | None = None


class OpenFGA(pulumi.ComponentResource):
    """Persist relationships in external PostgreSQL; expose only Check to workloads.

    The full HTTP and gRPC APIs bind to loopback. Operators use Kubernetes pod
    port-forwarding to publish models and reconcile tuples. The broker has no
    Kubernetes credentials and can reach only the TLS Check gateway.
    """

    def __init__(
        self,
        name: str,
        args: OpenFGAArgs,
        *,
        k8s_provider: pulumi.ProviderResource,
        opts: pulumi.ResourceOptions | None = None,
    ):
        super().__init__("marin:auth:OpenFGA", name, None, opts)
        if any("@sha256:" not in image for image in (args.image, args.proxy_image)):
            raise ValueError("OpenFGA and its gateway require digest-pinned images")
        if not re.fullmatch(r"[0-9A-HJKMNP-TV-Z]{26}", args.store_id):
            raise ValueError("OpenFGA store_id must be an explicit ULID")
        if not args.database_cidrs and not args.database_namespace:
            raise ValueError("OpenFGA requires explicit PostgreSQL destination CIDRs")
        child = pulumi.ResourceOptions(parent=self, provider=k8s_provider)
        gateway = f"""pid /tmp/nginx.pid;
error_log /dev/stderr warn;
events {{}}
http {{
  access_log off;
  client_body_temp_path /tmp/client;
  proxy_temp_path /tmp/proxy;
  fastcgi_temp_path /tmp/fastcgi;
  uwsgi_temp_path /tmp/uwsgi;
  scgi_temp_path /tmp/scgi;
  server {{
    listen 8443 ssl;
    ssl_certificate /tls/tls.crt;
    ssl_certificate_key /tls/tls.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    location = /healthz {{ proxy_pass http://127.0.0.1:8080; }}
    location = /stores/{args.store_id}/check {{
      limit_except POST {{ deny all; }}
      proxy_pass http://127.0.0.1:8080;
      proxy_read_timeout 10s;
    }}
    location / {{ return 403; }}
  }}
}}
"""
        config = k8s.core.v1.ConfigMap(
            f"{name}-gateway",
            metadata={"namespace": args.namespace, "name": "openfga-gateway"},
            data={"nginx.conf": gateway},
            opts=child,
        )
        datastore = [
            {"name": "OPENFGA_DATASTORE_ENGINE", "value": "postgres"},
            {
                "name": "OPENFGA_DATASTORE_URI",
                "value_from": {"secret_key_ref": {"name": args.database_secret, "key": "uri"}},
            },
        ]
        database_ca_mount = {"name": "database-ca", "mount_path": "/var/run/postgres/ca", "read_only": True}
        security = {
            "allow_privilege_escalation": False,
            "read_only_root_filesystem": True,
            "capabilities": {"drop": ["ALL"]},
        }
        self.deployment = k8s.apps.v1.Deployment(
            name,
            metadata={"namespace": args.namespace, "name": "openfga"},
            spec={
                "replicas": 2,
                "selector": {"match_labels": FGA_LABELS},
                "template": {
                    "metadata": {
                        "labels": FGA_LABELS,
                        "annotations": {"marin.dev/gateway-sha256": hashlib.sha256(gateway.encode()).hexdigest()},
                    },
                    "spec": {
                        "automount_service_account_token": False,
                        "security_context": {
                            "run_as_non_root": True,
                            "run_as_user": 1000,
                            "run_as_group": 1000,
                            "fs_group": 1000,
                            "seccomp_profile": {"type": "RuntimeDefault"},
                        },
                        "init_containers": [
                            {
                                "name": "migrate",
                                "image": args.image,
                                "args": ["migrate"],
                                "env": datastore,
                                "volume_mounts": [database_ca_mount],
                                "security_context": security,
                            }
                        ],
                        "containers": [
                            {
                                "name": "openfga",
                                "image": args.image,
                                "args": [
                                    "run",
                                    "--http-addr=127.0.0.1:8080",
                                    "--grpc-addr=127.0.0.1:8081",
                                    "--authn-method=preshared",
                                    "--playground-enabled=false",
                                    "--metrics-enabled=false",
                                ],
                                "env": [
                                    *datastore,
                                    {
                                        "name": "OPENFGA_AUTHN_PRESHARED_KEYS",
                                        "value_from": {"secret_key_ref": {"name": args.token_secret, "key": "token"}},
                                    },
                                ],
                                "security_context": security,
                                "volume_mounts": [database_ca_mount],
                                "resources": {
                                    "requests": {"cpu": "100m", "memory": "128Mi"},
                                    "limits": {"cpu": "1", "memory": "512Mi"},
                                },
                            },
                            {
                                "name": "gateway",
                                "image": args.proxy_image,
                                "command": ["nginx", "-g", "daemon off;"],
                                "ports": [{"name": "https", "container_port": 8443}],
                                "readiness_probe": {
                                    "http_get": {"path": "/healthz", "port": "https", "scheme": "HTTPS"}
                                },
                                "liveness_probe": {
                                    "http_get": {"path": "/healthz", "port": "https", "scheme": "HTTPS"},
                                    "initial_delay_seconds": 30,
                                },
                                "security_context": security,
                                "resources": {
                                    "requests": {"cpu": "25m", "memory": "32Mi"},
                                    "limits": {"cpu": "250m", "memory": "128Mi"},
                                },
                                "volume_mounts": [
                                    {
                                        "name": "gateway",
                                        "mount_path": "/etc/nginx/nginx.conf",
                                        "sub_path": "nginx.conf",
                                        "read_only": True,
                                    },
                                    {"name": "tls", "mount_path": "/tls", "read_only": True},
                                    {"name": "tmp", "mount_path": "/tmp"},
                                ],
                            },
                        ],
                        "volumes": [
                            {
                                "name": "database-ca",
                                "secret": {
                                    "secret_name": args.database_ca_secret,
                                    "items": [{"key": "ca.crt", "path": "ca.crt"}],
                                },
                            },
                            {"name": "gateway", "config_map": {"name": config.metadata["name"]}},
                            {"name": "tls", "secret": {"secret_name": args.tls_secret, "default_mode": 0o440}},
                            {"name": "tmp", "empty_dir": {}},
                        ],
                    },
                },
            },
            opts=child,
        )
        service = k8s.core.v1.Service(
            f"{name}-checks",
            metadata={"namespace": args.namespace, "name": "openfga"},
            spec={"selector": FGA_LABELS, "ports": [{"name": "https", "port": 8443, "target_port": "https"}]},
            opts=child,
        )
        self.network_policy = k8s.networking.v1.NetworkPolicy(
            f"{name}-network",
            metadata={"namespace": args.namespace},
            spec={
                "pod_selector": {"match_labels": FGA_LABELS},
                "policy_types": ["Ingress", "Egress"],
                "ingress": [{"from": [{"pod_selector": {"match_labels": BROKER_LABELS}}], "ports": [{"port": 8443}]}],
                "egress": [
                    dns_egress(args.dns_cidrs),
                    {
                        "to": [
                            *({"ip_block": {"cidr": cidr}} for cidr in args.database_cidrs),
                            *(
                                [
                                    {
                                        "namespace_selector": {
                                            "match_labels": {"kubernetes.io/metadata.name": args.database_namespace}
                                        },
                                        "pod_selector": {"match_labels": {"cnpg.io/podRole": "instance"}},
                                    }
                                ]
                                if args.database_namespace
                                else []
                            ),
                        ],
                        "ports": [{"port": 5432}],
                    },
                ],
            },
            opts=child,
        )
        self.address = service.metadata.apply(lambda value: f"https://{value['name']}.{args.namespace}.svc:8443")
        self.register_outputs({"address": self.address})
