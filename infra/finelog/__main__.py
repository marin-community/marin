# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pulumi entry point for a per-cluster Finelog server."""

import os
from pathlib import Path

import pulumi
import pulumi_kubernetes as k8s
from finelog.deploy.config import load_finelog_config
from iac.kubernetes.finelog import FinelogServer, FinelogServerArgs

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = "lib/finelog/deploy/Dockerfile"
CACHE_IMAGE = "ghcr.io/marin-community/finelog-cache:latest"
ENV_SECRET_SUFFIX = "-env"


def main() -> None:
    stack_config = pulumi.Config("finelog")
    cluster = stack_config.require("cluster")
    config = load_finelog_config(cluster)
    if config.deployment.k8s is None:
        raise ValueError(f"finelog config {cluster!r} is not a Kubernetes deployment")
    if not os.environ.get("KUBECONFIG"):
        raise ValueError(
            f"finelog stack {cluster!r} requires KUBECONFIG; "
            "Pulumi reads Kubernetes credentials from the execution environment"
        )

    deployment = config.deployment.k8s
    # Keep credentials and machine-local paths out of state. The provider reads
    # KUBECONFIG from the environment and binds it to the committed context.
    # Patch force transfers legacy kubectl-managed fields during the import pass.
    provider = k8s.Provider(
        "k8s",
        context=deployment.kube_context or None,
        enable_patch_force=True,
    )
    needs_env_secret = config.remote_log_dir.startswith("s3://") or config.forwarding is not None
    server = FinelogServer(
        "server",
        FinelogServerArgs(
            config=config,
            build_context=str(REPOSITORY_ROOT),
            dockerfile=DOCKERFILE,
            cargo_profile="release",
            cache_image=CACHE_IMAGE,
            env_secret_name=f"{config.name}{ENV_SECRET_SUFFIX}" if needs_env_secret else None,
            deploy_generation=stack_config.get_int("deploy_generation") or 0,
            adopt=stack_config.get_bool("import") or False,
        ),
        k8s_provider=provider,
    )
    pulumi.export("image", server.image_ref)
    pulumi.export("namespace", server.namespace)
    pulumi.export("service", server.service_name)


main()
