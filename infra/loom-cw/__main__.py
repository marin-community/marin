# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Declare the independent CoreWeave Loom staging service."""

import pulumi
import pulumi_kubernetes as k8s
from iac.auth.loom import Loom, LoomArgs, loom_deployment_manifest


def main() -> None:
    config = pulumi.Config("marin-loom-cw")
    exposure = config.require("exposure")
    if exposure not in ("bootstrap", "enrolling", "serving"):
        raise ValueError("Loom exposure must be bootstrap, enrolling or serving")
    auth = pulumi.StackReference(config.require("auth_stack"))
    metadata = pulumi.Output.all(
        issuer=auth.require_output("grant_api_address"),
        public_keys=auth.require_output("grant_public_keys"),
    )
    deployment = metadata.apply(
        lambda values: loom_deployment_manifest(
            issuer=values["issuer"],
            public_keys=values["public_keys"],
            profiles=config.require_object("profiles"),
            profile_pins=config.require_object("profile_pins"),
            exposure=exposure,
        )
    )
    provider = k8s.Provider("coreweave", kubeconfig="", context=config.require("kubernetes_context"))
    service = Loom(
        "loom",
        LoomArgs(
            namespace=config.require("namespace"),
            management_namespace=auth.require_output("namespace"),
            image=config.require("image"),
            proxy_image=config.require("proxy_image"),
            exposure=exposure,
            storage_class=config.require("storage_class"),
            storage_size=config.require("storage_size"),
            hostname=config.require("hostname"),
            ingress_tls_secret=config.require("ingress_tls_secret"),
            listener_tls_secret=config.require("listener_tls_secret"),
            listener_tls_server_name=config.require("listener_tls_server_name"),
            ingress_namespace=config.require("ingress_namespace"),
            ingress_pod_labels=config.require_object("ingress_pod_labels"),
            dns_cidrs=tuple(config.require_object("dns_cidrs")),
            https_egress_cidrs=tuple(config.require_object("https_egress_cidrs")),
            deployment=deployment,
            env_secret_refs=config.require_object("env_secret_refs"),
            ca_secret=config.require("ca_secret"),
        ),
        k8s_provider=provider,
    )
    pulumi.export("namespace", config.require("namespace"))
    pulumi.export("exposure", exposure)
    pulumi.export("deployment_config", service.config.metadata["name"])
    if exposure == "serving":
        pulumi.export("url", f"https://{config.require('hostname')}")


main()
