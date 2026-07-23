# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pulumi entry point for the eval-results dashboard Cloud Run service.

Deploys this directory as an IAP-gated Cloud Run service through the reusable
``iac.gcp.cloud_run.CloudRunService`` component. The service's fixed shape — project,
region, one warm instance for the background ingest loop, the CloudSQL connection, and
the record bucket — lives here; who is admitted through IAP is stack config
(``marin-evaldash:viewers``).

The image build context is the repo root (the runtime image copies the eval record/DB
modules from ``lib/marin``), so ``build_context`` points there and ``dockerfile`` is the
repo-root-relative path to this directory's Dockerfile.

Runs on the shared repo venv (plain ``python`` runtime), where ``iac`` and the Pulumi
providers live; ``uv sync --all-packages`` first. See README.md.
"""

import os

import pulumi
import pulumi_cloudflare as cloudflare
import pulumi_gcp as gcp
from iac.gcp.cloud_run import CloudRunService, CloudRunServiceArgs, SecretEnv

PROJECT = "hai-gcp-models"
REGION = "us-central1"
SERVICE = "marin-evaldash"

# Google's shared frontend for Cloud Run domain mappings; the vanity CNAME points here.
CLOUD_RUN_FRONTEND = "ghs.googlehosted.com"

# CloudSQL Postgres holding the indexed eval rows. The connector attaches over the CloudSQL
# admin API + the runtime SA's roles/cloudsql.client grant, so no VPC path is needed.
CLOUDSQL_INSTANCE = "hai-gcp-models:us-central1:marin-metadata"

# Canonical per-run records the ingest loop lists and upserts.
# Both record stores the ingest loop scans: the GCS default plus the CoreWeave object store
# that CW GPU runs (whose workers have no GCP credentials) record to.
RECORDS_PREFIXES = "gs://marin-eval-metadata/runs,s3://marin-us-east-02a/marin/eval-metadata/runs"
EVAL_DB_NAME = "evals"
EVAL_DB_USER = "evals"

# The runtime image copies lib/marin modules, so the whole repo is the build context; the
# Dockerfile is addressed relative to it.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOCKERFILE = "infra/evaldash/Dockerfile"


def main() -> None:
    config = pulumi.Config()
    # IAM members admitted through IAP, e.g. group:marin@…; set with
    #   pulumi config set --path 'viewers[0]' group:someone@example.com
    viewers = config.get_object("viewers") or []

    provider = gcp.Provider("gcp", project=PROJECT)
    service = CloudRunService(
        "evaldash",
        CloudRunServiceArgs(
            project=PROJECT,
            region=REGION,
            service_name=SERVICE,
            build_context=REPO_ROOT,
            dockerfile=DOCKERFILE,
            # The ingest loop lists GCS and upserts to Postgres between requests, so CPU must
            # stay allocated while idle.
            cpu_always_allocated=True,
            env={
                "RECORDS_PREFIXES": RECORDS_PREFIXES,
                "EVAL_DB_INSTANCE": CLOUDSQL_INSTANCE,
                "EVAL_DB_NAME": EVAL_DB_NAME,
                "EVAL_DB_USER": EVAL_DB_USER,
            },
            # EVAL_DB_PASSWORD: the CloudSQL password (the connector uses IAM to reach the
            # instance but still authenticates the DB user). CW_KEY_ID/CW_KEY_SECRET: CoreWeave
            # object-storage keys for the s3:// records prefix. Values stay in Secret Manager.
            secrets=(
                SecretEnv(name="EVAL_DB_PASSWORD", secret="cloudsql-evals-password"),
                SecretEnv(name="CW_KEY_ID", secret="cw-object-storage-key-id"),
                SecretEnv(name="CW_KEY_SECRET", secret="cw-object-storage-key-secret"),
            ),
            # Read access to the eval record bucket. roles/storage.objectViewer is
            # project-wide (the component grants project roles, not per-bucket); scope it to
            # the bucket with a bucket IAM binding if project-wide read is later judged too
            # broad. roles/compute.viewer lets the run-detail jobs/logs endpoints resolve the
            # iris controller and finelog VM internal IPs through the Compute API for Direct VPC
            # egress reads. roles/cloudsql.client comes with cloudsql_instances below.
            service_account_roles=("roles/storage.objectViewer", "roles/compute.viewer"),
            # Attaches the CloudSQL instance to the service so the connector can dial it.
            cloudsql_instances=(CLOUDSQL_INSTANCE,),
            iap_members=tuple(viewers),
        ),
        gcp_provider=provider,
    )
    pulumi.export("url", service.uri)
    pulumi.export("image", service.image_ref)

    # Optional vanity domain: custom_domain is the host, dns_zone_id its Cloudflare zone.
    # Cloud Run terminates TLS and IAP, so the CNAME is DNS-only; a Cloudflare proxy blocks
    # managed-cert issuance. Domain-mapping creation requires a verified oa.dev owner, so the
    # mapping is created out-of-band (gcloud beta run domain-mappings create) and adopted
    # into the stack with `pulumi import` before the first up that enables it.
    custom_domain = config.get("custom_domain")
    if custom_domain:
        dns_zone_id = config.require("dns_zone_id")
        gcp.cloudrun.DomainMapping(
            "evaldash-domain",
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
            "evaldash-dns",
            zone_id=dns_zone_id,
            name=custom_domain,
            type="CNAME",
            content=CLOUD_RUN_FRONTEND,
            ttl=1,  # 1 = automatic
            proxied=False,
        )
        pulumi.export("custom_domain", f"https://{custom_domain}")


main()
