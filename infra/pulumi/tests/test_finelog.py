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
from iac.kubernetes.finelog import FinelogServerArgs, finelog_manifests


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


def test_finelog_manifests_preserve_stateful_single_writer_contract() -> None:
    manifests = finelog_manifests(_args(), "ghcr.io/marin-community/finelog@sha256:digest")
    pod_spec = manifests.deployment["spec"]["template"]["spec"]
    container = pod_spec["containers"][0]

    assert manifests.pvc["spec"]["resources"]["requests"]["storage"] == "250Gi"
    assert manifests.deployment["spec"]["replicas"] == 1
    assert manifests.deployment["spec"]["strategy"] == {"type": "Recreate"}
    assert pod_spec["nodeSelector"] == {"kubernetes.io/arch": "amd64"}
    assert pod_spec["priorityClassName"] == "iris-system"
    assert manifests.deployment["spec"]["template"]["metadata"]["annotations"] == {
        "finelog.marin/deploy-generation": "0"
    }
    assert container["image"] == "ghcr.io/marin-community/finelog@sha256:digest"
    assert container["readinessProbe"]["httpGet"] == {"path": "/health", "port": 10001}
    assert container["volumeMounts"] == [{"name": "cache", "mountPath": "/var/cache/finelog"}]


def test_finelog_manifests_reference_secret_without_secret_values() -> None:
    manifests = finelog_manifests(_args(), "image@sha256:digest")
    container = manifests.deployment["spec"]["template"]["spec"]["containers"][0]

    assert container["envFrom"] == [{"secretRef": {"name": "finelog-cw-env"}}]
    assert {entry["name"] for entry in container["env"]} == {
        "FINELOG_AUTH_POLICY",
        "FINELOG_FORWARDING",
        "FINELOG_PORT",
        "FINELOG_REMOTE_DIR",
    }
    assert "PRIVATE KEY" not in str(manifests.deployment)
