# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Finelog Kubernetes resource contract."""

from finelog.deploy.config import (
    CidrAuthLayer,
    Deployment,
    FinelogConfig,
    ForwardingConfig,
    K8sDeployment,
)
from iac.kubernetes.finelog import FinelogServerArgs, finelog_resource_args


def _args() -> FinelogServerArgs:
    config = FinelogConfig(
        name="finelog-cw",
        port=10001,
        image="ghcr.io/marin-community/finelog:latest",
        remote_log_dir="s3://logs/finelog/cw",
        deployment=Deployment(
            k8s=K8sDeployment(
                namespace="iris",
                storage_gb=250,
                priority_class_name="iris-system",
            )
        ),
        auth=(CidrAuthLayer(cidrs=("10.0.0.0/8",)),),
        forwarding=ForwardingConfig(
            target="https://finelog.oa.dev",
            cluster="cw",
            signing_key=("gcp-secret://projects/1/secrets/finelog-cw-signing-key/versions/1",),
        ),
    )
    return FinelogServerArgs(
        config=config,
        build_context="/repo",
        dockerfile="lib/finelog/deploy/Dockerfile",
        cargo_profile="release",
        cache_image="ghcr.io/marin-community/finelog-cache:latest",
        env_secret_name="finelog-cw-env",
        deploy_generation=0,
    )


def test_finelog_resource_args_preserve_stateful_single_writer_contract() -> None:
    resources = finelog_resource_args(_args(), "ghcr.io/marin-community/finelog@sha256:digest")
    assert resources.pvc.spec is not None
    assert resources.pvc.spec.resources is not None
    assert resources.deployment.spec is not None
    assert resources.deployment.spec.template.spec is not None
    pod_spec = resources.deployment.spec.template.spec
    container = pod_spec.containers[0]

    assert resources.pvc.spec.resources.requests == {"storage": "250Gi"}
    assert resources.deployment.spec.replicas == 1
    assert resources.deployment.spec.strategy is not None
    assert resources.deployment.spec.strategy.type == "Recreate"
    assert pod_spec.node_selector == {"kubernetes.io/arch": "amd64"}
    assert pod_spec.priority_class_name == "iris-system"
    assert resources.deployment.spec.template.metadata is not None
    assert resources.deployment.spec.template.metadata.annotations == {"finelog.marin/deploy-generation": "0"}
    assert container.image == "ghcr.io/marin-community/finelog@sha256:digest"
    assert container.readiness_probe is not None
    assert container.readiness_probe.http_get is not None
    assert container.readiness_probe.http_get.path == "/health"
    assert container.readiness_probe.http_get.port == 10001
    assert container.volume_mounts is not None
    assert [(mount.name, mount.mount_path) for mount in container.volume_mounts] == [("cache", "/var/cache/finelog")]


def test_finelog_resource_args_reference_secret_without_secret_values() -> None:
    resources = finelog_resource_args(_args(), "image@sha256:digest")
    assert resources.deployment.spec is not None
    assert resources.deployment.spec.template.spec is not None
    container = resources.deployment.spec.template.spec.containers[0]
    assert container.env_from is not None
    assert container.env_from[0].secret_ref is not None

    assert container.env_from[0].secret_ref.name == "finelog-cw-env"
    assert container.env is not None
    assert {entry.name for entry in container.env} == {
        "FINELOG_AUTH_POLICY",
        "FINELOG_FORWARDING",
        "FINELOG_PORT",
        "FINELOG_REMOTE_DIR",
    }
    assert "gcp-secret://" not in str(resources.deployment)
