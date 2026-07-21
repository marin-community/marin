# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pulumi deployment for the split ops ingest and IAP dashboard services."""

import os

import pulumi
import pulumi_gcp as gcp
from iac.gcp.cloud_run import CloudRunAccess, CloudRunService, CloudRunServiceArgs, SecretEnv

PROJECT = "hai-gcp-models"
REGION = "us-central1"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CLOUDSQL_STACK = "organization/marin-cloudsql/marin-cloudsql"
WEBHOOK_SECRET = "marin-ops-grafana-webhook-hmac"


def main() -> None:
    config = pulumi.Config()
    viewers = config.require_object("viewers")
    loom_api_url = config.require("loom_api_url")
    loom_repo_root = config.require("loom_repo_root")
    loom_base = config.require("loom_base")
    repo_revision = config.require("repo_revision")
    skill_revision = config.require("skill_revision")
    provider = gcp.Provider("gcp", project=PROJECT)

    cloudsql = pulumi.StackReference(CLOUDSQL_STACK)
    connection_name = cloudsql.get_output("connection_name")

    # The value is populated out of band. Pulumi owns only the secret shell and
    # grants each runtime account access through SecretEnv.
    webhook_secret = gcp.secretmanager.Secret(
        "grafana-webhook-hmac",
        secret_id=WEBHOOK_SECRET,
        project=PROJECT,
        replication=gcp.secretmanager.SecretReplicationArgs(auto=gcp.secretmanager.SecretReplicationAutoArgs()),
        opts=pulumi.ResourceOptions(provider=provider),
    )

    common_env = {
        "PGHOST": connection_name.apply(lambda name: f"/cloudsql/{name}"),
        "PGDATABASE": "ops",
        "OPS_REPO_REVISION": repo_revision,
        "OPS_SKILL_REVISION": skill_revision,
    }
    ingest = CloudRunService(
        "ops-ingest",
        CloudRunServiceArgs(
            project=PROJECT,
            region=REGION,
            service_name="marin-ops-ingest",
            build_context=ROOT,
            dockerfile="infra/ops/Dockerfile",
            access=CloudRunAccess.PUBLIC,
            env={**common_env, "PGUSER": "ops_ingest", "OPS_SURFACE": "ingest"},
            secrets=(
                SecretEnv(name="PGPASSWORD", secret="cloudsql-ops-ingest-password"),
                SecretEnv(name="OPS_ALERT_WEBHOOK_SECRET", secret=WEBHOOK_SECRET),
            ),
            cloudsql_instances=(connection_name,),
            min_instances=0,
            max_instances=3,
            cpu="1",
            memory="512Mi",
        ),
        gcp_provider=provider,
        opts=pulumi.ResourceOptions(depends_on=[webhook_secret]),
    )
    ui = CloudRunService(
        "ops-ui",
        CloudRunServiceArgs(
            project=PROJECT,
            region=REGION,
            service_name="marin-ops-ui",
            build_context=ROOT,
            dockerfile="infra/ops/Dockerfile",
            env={
                **common_env,
                "PGUSER": "ops_app",
                "OPS_SURFACE": "ui",
                "LOOM_API_URL": loom_api_url,
                "LOOM_REPO_ROOT": loom_repo_root,
                "LOOM_BASE": loom_base,
            },
            secrets=(
                SecretEnv(name="PGPASSWORD", secret="cloudsql-ops-app-password"),
                SecretEnv(name="LOOM_TOKEN", secret="marin-ops-loom-token"),
            ),
            cloudsql_instances=(connection_name,),
            iap_members=tuple(str(viewer) for viewer in viewers),
            cpu_always_allocated=True,
        ),
        gcp_provider=provider,
    )
    pulumi.export("ingest_url", ingest.uri.apply(lambda uri: f"{uri}/api/ingest/grafana"))
    pulumi.export("ui_url", ui.uri)


main()
