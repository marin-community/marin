# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pulumi entry point for shared data buckets and lifecycle policies."""

from pathlib import Path

import pulumi_cloudflare as cloudflare
import pulumi_coreweave as coreweave
import pulumi_gcp as gcp
from iac.buckets import DataBuckets
from rigging.filesystem.cluster_config import StoreType, load_cluster_config_from_dirs

GCP_PROJECT = "hai-gcp-models"
GCP_LOG_BUCKET = "marin-usage-logs"
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIRS = (REPO_ROOT / "config",)


def main() -> None:
    gcp_data_config = load_cluster_config_from_dirs("marin", CONFIG_DIRS)
    s3_data_config = load_cluster_config_from_dirs("coreweave", CONFIG_DIRS)
    gcp_provider = gcp.Provider("gcp", project=GCP_PROJECT)
    coreweave_provider = coreweave.Provider(
        "coreweave-object-storage",
        s3_endpoint=s3_data_config.stores[StoreType.COREWEAVE].endpoint,
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
    )


main()
