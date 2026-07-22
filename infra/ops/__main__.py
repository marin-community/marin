# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pulumi deployment for the IAP dashboard and Grafana polling worker."""

import os

import pulumi
import pulumi_cloudflare as cloudflare
import pulumi_gcp as gcp
from iac.gcp.cloud_run import CloudRunService, CloudRunServiceArgs, SecretEnv

PROJECT = "hai-gcp-models"
REGION = "us-central1"
SERVICE = "marin-ops-ui"
CLOUD_RUN_FRONTEND = "ghs.googlehosted.com"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CLOUDSQL_STACK = "organization/marin-cloudsql/marin-cloudsql"
GRAFANA_STACK = "organization/marin-grafana/marin-grafana"
GRAFANA_SERVICE = "marin-grafana"
OPS_SERVICE_ACCOUNT_EMAIL = f"{SERVICE}@{PROJECT}.iam.gserviceaccount.com"
SLACK_WEBHOOK_SECRET = "marin-grafana-slack-webhook"


def main() -> None:
    config = pulumi.Config()
    viewers = config.require_object("viewers")
    agent_mode = config.require("agent_mode")
    if agent_mode not in ("stub", "loom"):
        raise ValueError("agent_mode must be stub or loom")
    repo_revision = config.require("repo_revision")
    skill_revision = config.require("skill_revision")
    custom_domain = config.get("custom_domain")
    public_url = f"https://{custom_domain}" if custom_domain else config.require("public_url")
    provider = gcp.Provider("gcp", project=PROJECT)

    cloudsql = pulumi.StackReference(CLOUDSQL_STACK)
    connection_name = cloudsql.get_output("connection_name")
    password_generation = cloudsql.get_output("ops_password_generation")
    grafana = pulumi.StackReference(GRAFANA_STACK)
    env = {
        "PGHOST": connection_name.apply(lambda name: f"/cloudsql/{name}"),
        "PGDATABASE": "ops",
        "PGUSER": "ops_app",
        "GRAFANA_API_URL": grafana.get_output("url"),
        "GRAFANA_POLL_INTERVAL": "60",
        "OPS_AGENT_MODE": agent_mode,
        "OPS_REPO_REVISION": repo_revision,
        "OPS_SKILL_REVISION": skill_revision,
        "OPS_PUBLIC_URL": public_url,
        "OPS_PASSWORD_GENERATION": password_generation,
        "OPS_SERVICE_ACCOUNT_EMAIL": OPS_SERVICE_ACCOUNT_EMAIL,
    }
    secrets = [
        SecretEnv(name="PGPASSWORD", secret="cloudsql-ops-app-password"),
        SecretEnv(name="OPS_SLACK_WEBHOOK", secret=SLACK_WEBHOOK_SECRET),
    ]
    if agent_mode == "loom":
        env.update(
            {
                "LOOM_API_URL": config.require("loom_api_url"),
                "LOOM_REPO_ROOT": config.require("loom_repo_root"),
                "LOOM_BASE": config.require("loom_base"),
            }
        )
        secrets.append(SecretEnv(name="LOOM_TOKEN", secret="marin-ops-loom-token"))
    ui = CloudRunService(
        "ops-ui",
        CloudRunServiceArgs(
            project=PROJECT,
            region=REGION,
            service_name=SERVICE,
            build_context=ROOT,
            dockerfile="infra/ops/Dockerfile",
            env=env,
            secrets=tuple(secrets),
            cloudsql_instances=(connection_name,),
            iap_members=tuple(str(viewer) for viewer in viewers),
            cpu_always_allocated=True,
        ),
        gcp_provider=provider,
    )
    pulumi.export("ui_url", ui.uri)
    pulumi.export("image", ui.image_ref)

    ops_service_account_id = ui.service_account_email.apply(lambda email: f"projects/{PROJECT}/serviceAccounts/{email}")
    ops_service_account_member = ui.service_account_email.apply(lambda email: f"serviceAccount:{email}")
    gcp.serviceaccount.IAMMember(
        "ops-self-token-creator",
        service_account_id=ops_service_account_id,
        role="roles/iam.serviceAccountTokenCreator",
        member=ops_service_account_member,
        opts=pulumi.ResourceOptions(provider=provider),
    )
    gcp.iap.WebCloudRunServiceIamMember(
        "ops-grafana-iap-access",
        project=PROJECT,
        location=REGION,
        cloud_run_service_name=GRAFANA_SERVICE,
        role="roles/iap.httpsResourceAccessor",
        member=ops_service_account_member,
        opts=pulumi.ResourceOptions(provider=provider),
    )

    if custom_domain:
        dns_zone_id = config.require("dns_zone_id")
        domain_mapping = gcp.cloudrun.DomainMapping(
            "ops-domain",
            name=custom_domain,
            location=REGION,
            metadata=gcp.cloudrun.DomainMappingMetadataArgs(namespace=PROJECT),
            spec=gcp.cloudrun.DomainMappingSpecArgs(route_name=SERVICE),
            opts=pulumi.ResourceOptions(
                provider=provider,
                depends_on=[ui],
                ignore_changes=["metadata", "spec", "statuses"],
            ),
        )
        cloudflare.DnsRecord(
            "ops-dns",
            zone_id=dns_zone_id,
            name=custom_domain,
            type="CNAME",
            content=CLOUD_RUN_FRONTEND,
            ttl=1,
            proxied=False,
            opts=pulumi.ResourceOptions(depends_on=[domain_mapping]),
        )
        pulumi.export("custom_domain", f"https://{custom_domain}")


main()
