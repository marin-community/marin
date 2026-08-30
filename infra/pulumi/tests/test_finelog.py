# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Finelog Kubernetes safety contracts."""

from finelog.deploy.config import (
    CidrAuthLayer,
    Deployment,
    FinelogConfig,
    ForwardingConfig,
    K8sDeployment,
)
from iac.kubernetes.finelog import FinelogServerArgs, finelog_resource_args
from rigging.provenance import Provenance


def _args(cache_pvc_name: str | None = None) -> FinelogServerArgs:
    config = FinelogConfig(
        name="finelog-cw",
        port=10001,
        image="ghcr.io/marin-community/finelog:latest",
        remote_log_dir="s3://logs/finelog/cw",
        deployment=Deployment(k8s=K8sDeployment(namespace="iris", cache_pvc_name=cache_pvc_name)),
        auth=(CidrAuthLayer(cidrs=("10.0.0.0/8",)),),
        forwarding=ForwardingConfig(
            target="https://finelog.oa.dev",
            cluster="cw",
            signing_key=("gcp-secret://projects/1/secrets/finelog-cw-signing-key/versions/1",),
        ),
        query_index_cache_mb=512,
    )
    return FinelogServerArgs(
        config=config,
        build_context="/repo",
        dockerfile="lib/finelog/deploy/Dockerfile",
        cargo_profile="release",
        cache_image="ghcr.io/marin-community/finelog-cache:latest",
        config_name="cw",
        env_secret_name="finelog-cw-env",
        source_revision=Provenance(
            tree_hash="0123456",
            base_commit="89abcde",
            dirty=False,
            branch="main",
            built_by="finelog-test",
        ),
    )


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
        "FINELOG_INDEX_CACHE_MB",
        "FINELOG_PORT",
        "FINELOG_REMOTE_DIR",
    }
    assert "gcp-secret://" not in str(resources.deployment)


def test_finelog_retains_deployment_history_for_rollback() -> None:
    resources = finelog_resource_args(_args(), "image@sha256:digest")
    assert resources.deployment.spec is not None

    assert resources.deployment.spec.revision_history_limit == 10


def test_finelog_can_mount_a_recovery_pvc_without_replacing_the_managed_claim() -> None:
    resources = finelog_resource_args(_args(cache_pvc_name="finelog-cw-cache-recovery"), "image@sha256:digest")
    assert resources.pvc.metadata is not None
    assert resources.pvc.metadata.name == "finelog-cw-cache"
    assert resources.deployment.spec is not None
    assert resources.deployment.spec.template.spec is not None
    volume = resources.deployment.spec.template.spec.volumes[0]
    assert volume.persistent_volume_claim is not None
    assert volume.persistent_volume_claim.claim_name == "finelog-cw-cache-recovery"
