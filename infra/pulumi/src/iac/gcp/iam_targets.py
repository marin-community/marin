# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compose deploy-target IAM declarations into the global IAM stack."""

from iac.gcp import grafana, iris, loom, marina
from iac.gcp.iam import GcpEncryptedMember, GcpIamArgs, merge_iam_grant_sets
from iac.gcp.iam_config import GcpIamConfig


def _project_role_members(project: str, config: GcpIamConfig, role_id: str) -> tuple[str | GcpEncryptedMember, ...]:
    role = f"projects/{project}/roles/{role_id}"
    matching_grants = tuple(grant for grant in config.project_grants if grant.role == role)
    if len(matching_grants) != 1:
        raise ValueError(f"expected exactly one project grant for {role}, found {len(matching_grants)}")
    return matching_grants[0].members


def global_iam_args(project: str, config: GcpIamConfig) -> GcpIamArgs:
    """Return the complete global IAM graph, including each deploy target."""
    principals = {principal.principal_id: GcpEncryptedMember(principal.ciphertext) for principal in config.principals}
    marin_dev_members = _project_role_members(project, config, "marindev")
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
        backend_service_iap=(),
        cloud_run_iap=(),
    )
    return merge_iam_grant_sets(
        args,
        (
            iris.iam_grants(project, principals, marin_dev_members),
            grafana.iam_grants(project, principals),
            loom.iam_grants(project),
            marina.iam_grants(project, principals),
        ),
    )
