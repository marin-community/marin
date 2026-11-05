# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deploy OpenBao independently of training clusters and runtime grants."""

import pulumi
import pulumi_kubernetes as k8s
from iac.auth.broker import (
    BROKER_LABELS,
    DATABASE_PATH,
    TOKEN_DIRECTORY,
    GrantBroker,
    GrantBrokerArgs,
    coreweave_identity_metadata,
)
from iac.auth.openbao import OpenBao, OpenBaoArgs


def main() -> None:
    config = pulumi.Config("marin-auth")
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
        ),
        k8s_provider=provider,
    )
    pulumi.export("issuer", service.issuer)
    pulumi.export("private_api_address", service.api_address)
    pulumi.export("namespace", config.require("namespace"))
    broker_image = config.get("broker_image")
    if broker_image:
        policy = config.require_object("broker_policy")
        expected_fields = {
            "broker_token_file",
            "subjects",
            "cedar",
            "storage",
            "max_token_ttl",
            "loom",
            "signing_key_version",
        }
        if set(policy) != expected_fields:
            raise ValueError(f"broker_policy must contain exactly {sorted(expected_fields)}")
        pulumi.export("coreweave_identities", coreweave_identity_metadata(policy["storage"]))
        policy.update(
            bao_address=service.api_address,
            ca_bundle=f"{TOKEN_DIRECTORY}/ca/ca.crt",
            bao_issuer=service.issuer,
            issuer=f"https://marin-auth.{namespace}.svc:8090",
            database=DATABASE_PATH,
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
            ),
            k8s_provider=provider,
            opts=pulumi.ResourceOptions(depends_on=[service]),
        )
        pulumi.export("grant_api_address", broker.address)
    public_keys = config.get_object("grant_public_keys")
    if public_keys is not None:
        if not public_keys or any(not key.startswith("-----BEGIN PUBLIC KEY-----") for key in public_keys):
            raise ValueError("grant_public_keys must contain audited public PEM keys from OpenBao Transit")
        pulumi.export("grant_public_keys", public_keys)


main()
