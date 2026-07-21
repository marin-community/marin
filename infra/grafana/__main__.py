# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pulumi entry point for the Grafana Cloud Run service.

Deploys this directory (Grafana + the finelog bridge) as an IAP-gated Cloud Run service
through the reusable `iac.gcp.cloud_run.CloudRunService` component. Grafana's fixed shape
— project, region, one warm instance — lives here; the list of people admitted through
IAP is stack config (`marin-grafana:viewers`).

Runs on the shared repo venv (plain `python` runtime), which is where `iac` and the Pulumi
GCP/Docker providers live; `uv sync --all-packages` first. See README.md.
"""

import os

import pulumi
import pulumi_cloudflare as cloudflare
import pulumi_gcp as gcp
from iac.gcp.cloud_run import CloudRunService, CloudRunServiceArgs, SecretEnv

# Cloud Run serves every custom domain from this fixed Google frontend; the mapping resource
# routes the host to the service, and the DNS CNAME points the host at the frontend.
CLOUD_RUN_FRONTEND = "ghs.googlehosted.com"

# Kept in lockstep with the bridge's config.py, which pins the same project for the finelog
# VM lookup — deploying elsewhere while still reading hai-gcp-models would silently break.
PROJECT = "hai-gcp-models"
REGION = "us-central1"
SERVICE = "marin-grafana"

# The cloudsql stack (infra/cloudsql) publishes the marin-metadata connection name that backs
# Grafana's state. On the self-managed GCS backend a stack reference is
# "organization/<project>/<stack>" (the literal "organization"), so this names the
# marin-cloudsql stack of the marin-cloudsql project.
CLOUDSQL_STACK = "organization/marin-cloudsql/marin-cloudsql"

# This file sits beside the Dockerfile, dashboards, and bridge source; the whole directory
# is the image build context.
BUILD_CONTEXT = os.path.dirname(os.path.abspath(__file__))

# Email delivery is optional: the deploy wires Grafana's SMTP password only when this
# secret exists, so a project without it still deploys — critical alerts then reach
# Slack only.
SMTP_SECRET = "marin-grafana-smtp-credentials"


def smtp_secret_exists(provider: gcp.Provider) -> bool:
    found = gcp.secretmanager.get_secrets(
        project=PROJECT,
        filter=f"name:{SMTP_SECRET}",
        opts=pulumi.InvokeOptions(provider=provider),
    )
    return any(secret.secret_id == SMTP_SECRET for secret in found.secrets)


def main() -> None:
    config = pulumi.Config()
    # IAM members admitted through IAP, e.g. group:marin@…; set with
    #   pulumi config set --path 'viewers[0]' group:someone@example.com
    viewers = config.get_object("viewers") or []

    provider = gcp.Provider("gcp", project=PROJECT)

    # Grafana's state lives in the shared marin-metadata Postgres (infra/cloudsql), reached
    # through the Cloud SQL socket the service mounts under /cloudsql. The socket directory
    # travels in DATABASE_SOCKET_DIR (not GF_DATABASE_HOST: Grafana host:port parsing rejects
    # the colons in a connection name); entrypoint.sh composes GF_DATABASE_URL from it.
    cloudsql = pulumi.StackReference(CLOUDSQL_STACK)
    connection_name = cloudsql.get_output("connection_name")
    database_socket_dir = connection_name.apply(lambda name: f"/cloudsql/{name}")

    # Values stay in Secret Manager; the component only grants the runtime service
    # account access. GITHUB_TOKEN feeds the ferry/build panels; CW_READ_TOKEN is the
    # CoreWeave read-role token behind the k8s source; GF_DATABASE_PASSWORD is the
    # grafana Postgres user's password; SLACK_ALERTS_WEBHOOK and GF_SMTP_PASSWORD feed
    # the provisioned alerting contact points.
    secrets = [
        SecretEnv(name="GITHUB_TOKEN", secret="marin-status-page-github-token"),
        SecretEnv(name="GF_DATABASE_PASSWORD", secret="cloudsql-grafana-password"),
        SecretEnv(name="CW_READ_TOKEN", secret="marin-grafana-cw-read-token"),
        SecretEnv(name="SLACK_ALERTS_WEBHOOK", secret="marin-grafana-slack-webhook"),
    ]
    env = {
        "DATABASE_SOCKET_DIR": database_socket_dir,
        "GF_DATABASE_NAME": "grafana",
        "GF_DATABASE_USER": "grafana",
    }
    if smtp_secret_exists(provider):
        secrets.append(SecretEnv(name="GF_SMTP_PASSWORD", secret=SMTP_SECRET))
        env["GF_SMTP_ENABLED"] = "true"

    service = CloudRunService(
        "grafana",
        CloudRunServiceArgs(
            project=PROJECT,
            region=REGION,
            service_name=SERVICE,
            build_context=BUILD_CONTEXT,
            # Grafana 13's apiserver and search indexers run between requests and need CPU
            # while idle; the dashboards list hangs on them otherwise.
            cpu_always_allocated=True,
            # The bridge lists finelog and controller VM internal IPs through the Compute API.
            service_account_roles=("roles/compute.viewer",),
            env=env,
            secrets=tuple(secrets),
            cloudsql_instances=(connection_name,),
            iap_members=tuple(viewers),
        ),
        gcp_provider=provider,
    )
    pulumi.export("url", service.uri)
    pulumi.export("image", service.image_ref)

    # Optional vanity domain: custom_domain is the host, dns_zone_id its Cloudflare zone.
    # Cloud Run terminates TLS and IAP, so the CNAME is DNS-only; a Cloudflare proxy blocks
    # managed-cert issuance.
    custom_domain = config.get("custom_domain")
    if custom_domain:
        dns_zone_id = config.require("dns_zone_id")
        gcp.cloudrun.DomainMapping(
            "grafana-domain",
            name=custom_domain,
            location=REGION,
            metadata=gcp.cloudrun.DomainMappingMetadataArgs(namespace=PROJECT),
            spec=gcp.cloudrun.DomainMappingSpecArgs(route_name=SERVICE),
            # Domain mappings are immutable in the provider, and adoption fills server-set
            # metadata/status the program does not declare; without ignoring them every up
            # plans an unsupported update.
            opts=pulumi.ResourceOptions(
                provider=provider,
                ignore_changes=["metadata", "spec", "statuses"],
            ),
        )
        cloudflare.DnsRecord(
            "grafana-dns",
            zone_id=dns_zone_id,
            name=custom_domain,
            type="CNAME",
            content=CLOUD_RUN_FRONTEND,
            ttl=1,  # 1 = automatic
            proxied=False,
        )
        pulumi.export("custom_domain", f"https://{custom_domain}")


main()
