# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deploy OpenBao independently of training clusters and runtime grants."""

import pulumi
import pulumi_kubernetes as k8s
from iac.auth.broker import (
    BROKER_LABELS,
    TOKEN_DIRECTORY,
    GrantBroker,
    GrantBrokerArgs,
    coreweave_identity_metadata,
)
from iac.auth.openbao import OpenBao, OpenBaoArgs
from iac.auth.openfga import OpenFGA, OpenFGAArgs
from iac.auth.staging import staging_permissions


def main() -> None:
    config = pulumi.Config("marin-auth")
    if (
        config.require("namespace") == "marin-auth-staging"
        and config.require("kubernetes_context") != "marin_US-WEST-04A"
    ):
        raise ValueError("Synthetic auth staging is restricted to the approved CW cluster")
    provider = k8s.Provider("management", kubeconfig="", context=config.require("kubernetes_context"))
    namespace = config.require("namespace")
    service = OpenBao(
        "openbao",
        OpenBaoArgs(
            namespace=namespace,
            image=config.require("image"),
            public_hostname=config.require("public_hostname"),
            storage_class=config.require("raft_storage_class"),
            storage_size=config.require("storage_size"),
            listener_tls_secret=config.require("listener_tls_secret"),
            listener_tls_server_name=config.require("listener_tls_server_name"),
            ingress_tls_secret=config.require("ingress_tls_secret"),
            ingress_namespace=config.require("ingress_namespace"),
            ingress_pod_labels=config.require_object("ingress_pod_labels"),
            client_namespace=namespace,
            client_pod_labels=BROKER_LABELS,
            operator_groups=tuple(config.require_object("operator_groups")),
            dns_cidrs=tuple(config.require_object("dns_cidrs")),
            https_egress_cidrs=tuple(config.require_object("https_egress_cidrs")),
            replicas=config.require_int("replicas"),
            anti_affinity=config.get("anti_affinity") or "required",
            discovery_mode=config.get("discovery_mode") or "ingress",
        ),
        k8s_provider=provider,
    )
    pulumi.export("issuer", service.issuer)
    pulumi.export("private_api_address", service.api_address)
    pulumi.export("namespace", config.require("namespace"))
    if namespace == "marin-auth-staging":
        staging_permissions(
            provider,
            dns_cidrs=tuple(config.require_object("dns_cidrs")),
            api_cidrs=tuple(config.require_object("kubernetes_api_cidrs")),
            namespace_dependency=service,
        )
    fga = OpenFGA(
        "openfga",
        OpenFGAArgs(
            namespace=namespace,
            image=config.require("openfga_image"),
            proxy_image=config.require("openfga_proxy_image"),
            database_secret=config.require("openfga_database_secret"),
            database_ca_secret=config.require("openfga_database_ca_secret"),
            token_secret=config.require("openfga_token_secret"),
            tls_secret=config.require("openfga_tls_secret"),
            store_id=config.require("openfga_store_id"),
            dns_cidrs=tuple(config.require_object("dns_cidrs")),
            database_cidrs=tuple(config.require_object("openfga_database_cidrs")),
            database_namespace=config.get("database_namespace"),
        ),
        k8s_provider=provider,
        opts=pulumi.ResourceOptions(depends_on=[service]),
    )
    pulumi.export("authorization_api_address", fga.address)
    broker_image = config.get("broker_image")
    if broker_image:
        policy = config.require_object("broker_policy")
        expected_fields = {
            "broker_token_file",
            "subjects",
            "openfga",
            "storage",
            "max_token_ttl",
            "loom",
            "signing_key_version",
            "database",
        }
        if not expected_fields <= set(policy) or set(policy) - expected_fields - {"secrets", "iris"}:
            raise ValueError(f"broker_policy requires {sorted(expected_fields)} and accepts optional secrets and iris")
        pulumi.export("coreweave_identities", coreweave_identity_metadata(policy["storage"]))
        if policy["openfga"]["store_id"] != config.require("openfga_store_id"):
            raise ValueError("The broker and OpenFGA gateway must use the same store")
        policy["openfga"]["api_url"] = fga.address
        policy.update(
            bao_address=service.api_address,
            ca_bundle=f"{TOKEN_DIRECTORY}/ca/ca.crt",
            bao_issuer=service.issuer,
            issuer=f"https://marin-auth.{namespace}.svc:8090",
        )
        broker = GrantBroker(
            "grant-broker",
            GrantBrokerArgs(
                namespace=namespace,
                image=broker_image,
                storage_class=config.require("broker_storage_class"),
                storage_size=config.require("broker_storage_size"),
                policy=policy,
                token_secret_names=tuple(config.require_object("broker_token_secrets")),
                listener_tls_secret=config.require("broker_tls_secret"),
                ca_secret=config.require("broker_ca_secret"),
                client_namespace=config.require("broker_client_namespace"),
                client_pod_labels=config.require_object("broker_client_pod_labels"),
                dns_cidrs=tuple(config.require_object("dns_cidrs")),
                https_egress_cidrs=tuple(config.require_object("broker_https_egress_cidrs")),
                database_secret=config.get("broker_database_secret"),
                database_cidrs=tuple(config.get_object("broker_database_cidrs") or []),
                replicas=config.get_int("broker_replicas") or 1,
                database_namespace=config.get("database_namespace"),
            ),
            k8s_provider=provider,
            opts=pulumi.ResourceOptions(depends_on=[service, fga]),
        )
        pulumi.export("grant_api_address", broker.address)
    public_keys = config.get_object("grant_public_keys")
    if public_keys is not None:
        if not public_keys or any(not key.startswith("-----BEGIN PUBLIC KEY-----") for key in public_keys):
            raise ValueError("grant_public_keys must contain audited public PEM keys from OpenBao Transit")
        pulumi.export("grant_public_keys", public_keys)


main()
