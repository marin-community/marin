# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pulumi entry point for shared data buckets and lifecycle policies."""

from pathlib import Path

import pulumi
import pulumi_cloudflare as cloudflare
import pulumi_command as command
import pulumi_coreweave as coreweave
import pulumi_gcp as gcp
from iac.buckets import DataBuckets
from iac.buckets.r2 import R2PublicBucket
from iac.config import CLOUDFLARE_OA_DEV_ZONE_ID
from iac.coreweave.identity import CoreweaveIdentity, CoreweaveIdentityPolicies
from rigging.filesystem.cluster_config import StoreType, load_cluster_config_from_dirs
from rigging.grants import Permission

GCP_PROJECT = "hai-gcp-models"
GCP_LOG_BUCKET = "marin-usage-logs"
CLOUDFLARE_ACCOUNT_ID = "74981a43be0de7712369306c7b19133d"
R2_BUCKET_NAME = "marin-na"
R2_PUBLIC_HOSTNAME = "public.oa.dev"
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIRS = (REPO_ROOT / "config",)


def identity_from_metadata(metadata: dict) -> CoreweaveIdentity:
    return CoreweaveIdentity(
        subject=metadata["subject"],
        permissions=tuple(
            Permission(
                service=permission["service"],
                actions=frozenset(permission["actions"]),
                resource=permission["resource"],
            )
            for permission in metadata["permissions"]
        ),
    )


def main() -> None:
    gcp_data_config = load_cluster_config_from_dirs("marin", CONFIG_DIRS)
    public_data_config = load_cluster_config_from_dirs("r2", CONFIG_DIRS)
    s3_data_config = load_cluster_config_from_dirs("coreweave", CONFIG_DIRS)
    gcp_provider = gcp.Provider("gcp", project=GCP_PROJECT)
    coreweave_provider = coreweave.Provider(
        "coreweave-object-storage",
        s3_endpoint=s3_data_config.stores[StoreType.COREWEAVE].endpoint,
    )
    config = pulumi.Config("marin-buckets")
    identity_names = config.get_object("auth_identity_names")
    if identity_names:
        if len(set(identity_names)) != len(identity_names):
            raise ValueError("auth_identity_names must select each provider identity once")
        auth = pulumi.StackReference(config.require("auth_stack"))
        metadata = auth.require_output("coreweave_identities")
        identities = {
            name: metadata.apply(lambda values, identity_name=name: identity_from_metadata(values[identity_name]))
            for name in identity_names
        }
        CoreweaveIdentityPolicies(
            "auth-identities",
            issuer=auth.require_output("issuer"),
            identities=identities,
            coreweave_provider=coreweave_provider,
        )
    cloudflare_provider = cloudflare.Provider("cloudflare-r2")

    DataBuckets(
        "data-buckets",
        project=GCP_PROJECT,
        gcp_data_config=gcp_data_config,
        s3_data_config=s3_data_config,
        log_bucket=GCP_LOG_BUCKET,
        gcp_provider=gcp_provider,
        coreweave_provider=coreweave_provider,
        cloudflare_provider=cloudflare_provider,
        r2_account_id=CLOUDFLARE_ACCOUNT_ID,
        r2_bucket_name=R2_BUCKET_NAME,
    )

    public = R2PublicBucket(
        "public-r2",
        account_id=CLOUDFLARE_ACCOUNT_ID,
        bucket_name=public_data_config.region_buckets["na"].name,
        hostname=R2_PUBLIC_HOSTNAME,
        zone_id=CLOUDFLARE_OA_DEV_ZONE_ID,
        cloudflare_provider=cloudflare_provider,
    )
    pulumi.export("publicUrl", public.url)
    pulumi.export("publicPrefix", public.prefix)

    command.local.Command(
        "hf-bucket-open-athena-marin",
        create=".venv/bin/hf buckets create open-athena/marin --private",
        dir="../..",
        opts=pulumi.ResourceOptions(protect=True, retain_on_delete=True),
    )
    pulumi.export("huggingface_bucket", "hf://buckets/open-athena/marin")


main()
