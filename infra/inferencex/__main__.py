# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Package pinned InferenceX-app as an internal, read-only Cloud Run dashboard."""

import hashlib
import json
from pathlib import Path

import pulumi
import pulumi_cloudflare as cloudflare
import pulumi_gcp as gcp
from iac.gcp.cloud_run import CloudRunService, CloudRunServiceArgs, SecretEnv

PROJECT = "hai-gcp-models"
REGION = "us-central1"
SERVICE = "marin-inferencex"
ROOT = Path(__file__).resolve().parent


def main() -> CloudRunService:
    provider = gcp.Provider("gcp", project=PROJECT)
    cloudsql = pulumi.StackReference("organization/marin-cloudsql/marin-cloudsql")
    connection_name = cloudsql.require_output("connection_name")
    settings = json.loads((ROOT / "settings.json").read_text())
    config = pulumi.Config()
    domain = config.require("custom_domain")
    if settings["site_url"] != f"https://{domain}":
        raise ValueError("Build origin must match the configured Cloud Run domain")
    invalidate = gcp.secretmanager.Secret(
        "invalidate-secret",
        project=PROJECT,
        secret_id="marin-inferencex-invalidate-secret",
        replication=gcp.secretmanager.SecretReplicationArgs(auto=gcp.secretmanager.SecretReplicationAutoArgs()),
        opts=pulumi.ResourceOptions(provider=provider),
    )
    service = CloudRunService(
        "inferencex",
        CloudRunServiceArgs(
            project=PROJECT,
            region=REGION,
            service_name=SERVICE,
            build_context=str(ROOT),
            cpu="1",
            memory="2Gi",
            min_instances=0,
            max_instances=1,
            startup_cpu_boost=True,
            cloudsql_instances=(connection_name,),
            env={
                "DATABASE_USER": "inferencex_reader",
                "DATABASE_NAME": "inferencex",
                "DATABASE_SOCKET_DIR": connection_name.apply(lambda name: f"/cloudsql/{name}"),
                "DATABASE_DRIVER": "postgres",
                "DATABASE_SSL": "false",
                "INFERENCEX_SOURCE_REPOS": "marin-community/marin",
                "CACHE_NAMESPACE": hashlib.sha256((ROOT / "app.patch").read_bytes()).hexdigest(),
                "NEXT_PUBLIC_SITE_URL": settings["site_url"],
            },
            secrets=(
                SecretEnv(name="DATABASE_PASSWORD", secret="cloudsql-inferencex-reader-password"),
                SecretEnv(name="INVALIDATE_SECRET", secret="marin-inferencex-invalidate-secret", wait_for=(invalidate,)),
            ),
        ),
        gcp_provider=provider,
    )
    pulumi.export("url", service.uri)
    pulumi.export("image", service.image_ref)
    pulumi.export("connection_name", connection_name)
    pulumi.export("site_url", settings["site_url"])
    gcp.cloudrun.DomainMapping(
        "inferencex-domain",
        name=domain,
        location=REGION,
        metadata=gcp.cloudrun.DomainMappingMetadataArgs(namespace=PROJECT),
        spec=gcp.cloudrun.DomainMappingSpecArgs(route_name=SERVICE),
        opts=pulumi.ResourceOptions(provider=provider, ignore_changes=["metadata", "spec", "statuses"]),
    )
    cloudflare.DnsRecord(
        "inferencex-dns",
        zone_id=config.require("dns_zone_id"),
        name=domain,
        type="CNAME",
        content="ghs.googlehosted.com",
        ttl=1,
        proxied=False,
    )
    return service


if __name__ == "__main__":
    main()
