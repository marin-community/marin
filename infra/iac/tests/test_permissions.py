# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest
from iac.gcp.permissions import GcpDeployPermissionsArgs, deploy_permission_sets

PROJECT = "hai-gcp-models"
DEPLOY_ACCOUNT = "iris-ci-smoke@hai-gcp-models.iam.gserviceaccount.com"


def _permissions_args() -> GcpDeployPermissionsArgs:
    return GcpDeployPermissionsArgs(
        project=PROJECT,
        project_number="748532799086",
        workload_identity_pool="github-pool",
        github_subject="repo:marin-community/marin:ref:refs/heads/main",
        state_bucket="marin-iac-state",
        kms_location="us-central1",
        kms_key_ring="marin-iac-keyring",
        kms_key="marin-iac-key",
        service_accounts=(DEPLOY_ACCOUNT,),
    )


def test_deploy_permission_sets_rejects_id_token_account_without_deploy_access():
    args = replace(
        _permissions_args(),
        id_token_service_accounts=frozenset({"undeclared@hai-gcp-models.iam.gserviceaccount.com"}),
    )

    with pytest.raises(ValueError):
        deploy_permission_sets(args)


def test_deploy_permission_sets_rejects_service_account_from_another_project():
    args = replace(
        _permissions_args(),
        service_accounts=("deploy@another-project.iam.gserviceaccount.com",),
    )

    with pytest.raises(ValueError):
        deploy_permission_sets(args)
