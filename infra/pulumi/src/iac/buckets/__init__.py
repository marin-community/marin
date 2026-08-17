# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared data-bucket ownership across GCS, CoreWeave, and Cloudflare R2."""

import pulumi
from rigging.filesystem.cluster_config import DataConfig

from iac.buckets.coreweave import CoreweaveDataBuckets
from iac.buckets.gcp import GcpDataBuckets
from iac.buckets.r2 import R2DataBuckets


class DataBuckets(pulumi.ComponentResource):
    """Own every shared Marin data bucket from one Pulumi stack."""

    def __init__(
        self,
        name: str,
        *,
        project: str,
        gcp_data_config: DataConfig,
        s3_data_config: DataConfig,
        log_bucket: str,
        gcp_provider: pulumi.ProviderResource,
        coreweave_provider: pulumi.ProviderResource,
        cloudflare_provider: pulumi.ProviderResource,
        opts: pulumi.ResourceOptions | None = None,
    ) -> None:
        super().__init__("marin:buckets:DataBuckets", name, None, opts)

        GcpDataBuckets(
            "gcp",
            project=project,
            data_config=gcp_data_config,
            log_bucket=log_bucket,
            gcp_provider=gcp_provider,
            opts=pulumi.ResourceOptions(parent=self),
        )

        CoreweaveDataBuckets(
            "coreweave",
            data_config=s3_data_config,
            lifecycle_config=gcp_data_config,
            coreweave_provider=coreweave_provider,
            opts=pulumi.ResourceOptions(parent=self),
        )

        R2DataBuckets(
            "r2",
            data_config=s3_data_config,
            lifecycle_config=gcp_data_config,
            cloudflare_provider=cloudflare_provider,
            opts=pulumi.ResourceOptions(parent=self),
        )
        self.register_outputs({})
