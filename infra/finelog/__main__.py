# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pulumi entry point for a per-cluster Finelog server."""

import os
from pathlib import Path

import pulumi
import pulumi_kubernetes as k8s
from finelog.deploy.config import k8s_env_secret_name, load_finelog_config
from iac.kubernetes.finelog import FinelogServer, FinelogServerArgs
from rigging.provenance import Provenance

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = "lib/finelog/deploy/Dockerfile"
CACHE_IMAGE = "ghcr.io/marin-community/finelog-cache:latest"


def main() -> None:
    stack_config = pulumi.Config("finelog")
    cluster = stack_config.require("cluster")
    config = load_finelog_config(cluster)
    if config.deployment.k8s is None:
        raise ValueError(f"finelog config {cluster!r} is not a Kubernetes deployment")
    deployment = config.deployment.k8s
    ambient_kubeconfig = os.environ.get("KUBECONFIG")
    if not ambient_kubeconfig:
        raise ValueError(
            f"finelog stack {cluster!r} requires KUBECONFIG; "
            "Pulumi reads Kubernetes credentials from the execution environment"
        )
    if deployment.kubeconfig:
        configured_kubeconfig = Path(deployment.kubeconfig).expanduser().resolve()
        if Path(ambient_kubeconfig).expanduser().resolve() != configured_kubeconfig:
            raise ValueError(
                f"finelog stack {cluster!r} requires KUBECONFIG={configured_kubeconfig}; "
                "the Pulumi provider and Finelog health check must target the same credentials"
            )

    adopt = stack_config.get_bool("import") or False
    # Keep credentials and machine-local paths out of state. The provider reads
    # KUBECONFIG from the environment and binds it to the committed context.
    provider = k8s.Provider(
        "k8s",
        kubeconfig="",
        context=deployment.kube_context or None,
        enable_patch_force=True,
    )
    server = FinelogServer(
        "server",
        FinelogServerArgs(
            config=config,
            build_context=str(REPOSITORY_ROOT),
            dockerfile=DOCKERFILE,
            cargo_profile="release",
            cache_image=CACHE_IMAGE,
            config_name=cluster,
            env_secret_name=k8s_env_secret_name(config),
            source_revision=Provenance.from_git(REPOSITORY_ROOT),
            adopt=adopt,
        ),
        k8s_provider=provider,
    )
    pulumi.export("image", server.image_ref)
    pulumi.export("namespace", server.namespace)
    pulumi.export("service", server.service_name)


main()
