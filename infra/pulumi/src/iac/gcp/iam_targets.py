# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compose deploy-target IAM declarations into the global IAM stack."""

from iac.gcp import echo, evaldash, grafana, loom
from iac.gcp.iam import GcpEncryptedMember, GcpIamArgs, merge_iam_grant_sets
from iac.gcp.iam_config import GcpIamConfig


def global_iam_args(project: str, config: GcpIamConfig) -> GcpIamArgs:
    """Return the complete global IAM graph, including each deploy target.

    Shared grants stay machine-editable in YAML, while target declarations remain beside their
    service-specific topology and are never serialized back into the shared file.
    """
    principals = {principal.principal_id: GcpEncryptedMember(principal.ciphertext) for principal in config.principals}
    args = GcpIamArgs(
        project=project,
        kms_location=config.kms_location,
        kms_key_ring=config.kms_key_ring,
        kms_key=config.kms_key,
        custom_roles=config.custom_roles,
        owned_service_accounts=config.owned_service_accounts,
        project_grants=config.project_grants,
        kms_grants=config.kms_grants,
        secrets=config.secrets,
        buckets=config.buckets,
        artifact_repositories=config.artifact_repositories,
        service_accounts=config.service_accounts,
        cloud_run_iap=(),
    )
    return merge_iam_grant_sets(
        args,
        (
            echo.iam_grants(project),
            evaldash.iam_grants(project, principals),
            grafana.iam_grants(project, principals),
            loom.iam_grants(project),
        ),
    )
