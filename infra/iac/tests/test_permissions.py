# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import yaml
from iac.gcp.permissions import GcpDeployPermissionsArgs, deploy_permission_sets

REPO_ROOT = Path(__file__).parents[3]
PROJECT = "hai-gcp-models"
PROJECT_NUMBER = "748532799086"
POOL = "github-pool"
GITHUB_SUBJECT = "repo:marin-community/marin:ref:refs/heads/main"
SERVICE_ACCOUNTS = (
    "iris-ci-smoke@hai-gcp-models.iam.gserviceaccount.com",
    "marin-cd-cloud-run-deploy@hai-gcp-models.iam.gserviceaccount.com",
)
WIF_PROVIDER = "projects/748532799086/locations/global/workloadIdentityPools/github-pool/providers/github-oidc"
KMS_PROVIDER = (
    "gcpkms://projects/hai-gcp-models/locations/us-central1/keyRings/" "marin-iac-keyring/cryptoKeys/marin-iac-key"
)


def test_deploy_permission_sets_resolve_scoped_members_and_resources():
    permission_sets = deploy_permission_sets(
        GcpDeployPermissionsArgs(
            project=PROJECT,
            project_number=PROJECT_NUMBER,
            workload_identity_pool=POOL,
            github_subject=GITHUB_SUBJECT,
            state_bucket="marin-iac-state",
            kms_location="us-central1",
            kms_key_ring="marin-iac-keyring",
            kms_key="marin-iac-key",
            service_accounts=SERVICE_ACCOUNTS,
            id_token_service_accounts=frozenset({SERVICE_ACCOUNTS[0]}),
        )
    )
    assert {permission_set.account_id for permission_set in permission_sets} == {
        "iris-ci-smoke",
        "marin-cd-cloud-run-deploy",
    }
    for permission_set, email in zip(permission_sets, SERVICE_ACCOUNTS, strict=True):
        assert permission_set.service_account_id == f"projects/{PROJECT}/serviceAccounts/{email}"
        assert permission_set.service_account_member == f"serviceAccount:{email}"
        assert permission_set.github_principal == (
            "principal://iam.googleapis.com/projects/748532799086/locations/global/"
            f"workloadIdentityPools/github-pool/subject/{GITHUB_SUBJECT}"
        )
        assert permission_set.state_bucket == "marin-iac-state"
        assert permission_set.crypto_key_id == (
            "projects/hai-gcp-models/locations/us-central1/keyRings/" "marin-iac-keyring/cryptoKeys/marin-iac-key"
        )
    assert [permission_set.mint_id_tokens for permission_set in permission_sets] == [True, False]


def _workflow(name: str) -> dict:
    return yaml.safe_load((REPO_ROOT / ".github" / "workflows" / name).read_text())


def test_deploy_workflows_use_main_branch_wif_service_accounts():
    expected = {
        "ops-ducky.yaml": (
            "github.ref == 'refs/heads/main'",
            "iris-ci-smoke@hai-gcp-models.iam.gserviceaccount.com",
        ),
        "ops-grafana.yaml": (
            "github.ref == 'refs/heads/main' && (github.event_name == 'push' || "
            "github.event_name == 'workflow_dispatch')",
            "marin-cd-cloud-run-deploy@hai-gcp-models.iam.gserviceaccount.com",
        ),
    }
    for workflow_name, (condition, service_account) in expected.items():
        deploy = _workflow(workflow_name)["jobs"]["deploy"]
        assert deploy["permissions"] == {"contents": "read", "id-token": "write"}
        assert deploy["if"] == condition
        step = next(step for step in deploy["steps"] if step.get("uses") == "./.github/actions/pulumi-deploy")
        assert step["with"]["service-account"] == service_account


def test_pulumi_deploy_action_uses_wif_without_passphrase_or_key_json():
    action = yaml.safe_load((REPO_ROOT / ".github" / "actions" / "pulumi-deploy" / "action.yaml").read_text())
    assert "gcp-credentials" not in action["inputs"]
    auth = next(step for step in action["runs"]["steps"] if step.get("uses") == "google-github-actions/auth@v3")
    assert auth["with"]["workload_identity_provider"] == "${{ inputs.workload-identity-provider }}"
    assert auth["with"]["service_account"] == "${{ inputs.service-account }}"
    assert "credentials_json" not in auth["with"]
    registry_auth = next(step for step in action["runs"]["steps"] if step.get("name") == "Configure registry auth")
    assert registry_auth["run"] == "gcloud auth configure-docker us-central1-docker.pkg.dev --quiet"


def test_service_stacks_use_shared_kms_provider():
    for stack_file in (
        REPO_ROOT / "infra" / "cloudsql" / "Pulumi.marin-cloudsql.yaml",
        REPO_ROOT / "infra" / "ducky" / "Pulumi.ducky-marin.yaml",
        REPO_ROOT / "infra" / "grafana" / "Pulumi.marin-grafana.yaml",
    ):
        stack = yaml.safe_load(stack_file.read_text())
        assert stack["secretsprovider"] == KMS_PROVIDER
        assert stack["encryptedkey"]
        assert "encryptionsalt" not in stack
