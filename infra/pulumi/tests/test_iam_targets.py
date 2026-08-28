# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from iac.gcp.iam import GcpIamArgs
from iac.gcp.iam_config import load_iam_config
from iac.gcp.iam_targets import global_iam_args

PROJECT = "hai-gcp-models"


def _grant_keys(args: GcpIamArgs) -> list[tuple[object, ...]]:
    keys = [
        ("project", grant.role, member, grant.condition) for grant in args.project_grants for member in grant.members
    ]
    keys.extend(("kms", grant.role, member, grant.condition) for grant in args.kms_grants for member in grant.members)
    keys.extend(
        ("secret", secret.secret, grant.role, member, grant.condition)
        for secret in args.secrets
        for grant in secret.grants
        for member in grant.members
    )
    keys.extend(
        ("bucket", bucket.bucket, grant.role, member, grant.condition)
        for bucket in args.buckets
        for grant in bucket.grants
        for member in grant.members
    )
    keys.extend(
        ("artifact", repository.location, repository.repository, grant.role, member, grant.condition)
        for repository in args.artifact_repositories
        for grant in repository.grants
        for member in grant.members
    )
    keys.extend(
        ("service-account", account.email, grant.role, member, grant.condition)
        for account in args.service_accounts
        for grant in account.grants
        for member in grant.members
    )
    keys.extend(
        ("backend-service", service.service, grant.role, member, grant.condition)
        for service in args.backend_service_iap
        for grant in service.iap_grants
        for member in grant.members
    )
    keys.extend(
        ("cloud-run", service.location, service.service, grant.role, member, grant.condition)
        for service in args.cloud_run_iap
        for grant in service.iap_grants
        for member in grant.members
    )
    return keys


def test_global_iam_composes_deploy_target_grants_without_duplicate_resources() -> None:
    config = load_iam_config()
    args = global_iam_args(PROJECT, config)
    grant_keys = _grant_keys(args)

    assert len(grant_keys) == len(set(grant_keys))
