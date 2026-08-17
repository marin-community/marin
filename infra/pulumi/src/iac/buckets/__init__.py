# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared data-bucket ownership across GCS, CoreWeave, and Cloudflare R2."""

import pulumi
import pulumi_cloudflare as cloudflare
import pulumi_coreweave as coreweave_sdk
from rigging.filesystem.cluster_config import DataConfig, StoreType

from iac.buckets.coreweave import CoreweaveDataBuckets
from iac.buckets.gcp import GcpDataBuckets
from iac.buckets.r2 import R2DataBuckets
from iac.imports import NO_IMPORTS, ImportRegistrar


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
        imports: ImportRegistrar = NO_IMPORTS,
        opts: pulumi.ResourceOptions | None = None,
    ) -> None:
        super().__init__("marin:buckets:DataBuckets", name, None, opts)

        GcpDataBuckets(
            "gcp",
            project=project,
            data_config=gcp_data_config,
            log_bucket=log_bucket,
            gcp_provider=gcp_provider,
            imports=imports,
            opts=pulumi.ResourceOptions(parent=self),
        )

        coreweave_store = s3_data_config.stores[StoreType.COREWEAVE]
        coreweave_provider = coreweave_sdk.Provider(
            "coreweave-object-storage",
            s3_endpoint=coreweave_store.endpoint,
            opts=pulumi.ResourceOptions(parent=self),
        )
        CoreweaveDataBuckets(
            "coreweave",
            data_config=s3_data_config,
            lifecycle_config=gcp_data_config,
            coreweave_provider=coreweave_provider,
            imports=imports,
            opts=pulumi.ResourceOptions(parent=self),
        )

        cloudflare_provider = cloudflare.Provider(
            "cloudflare-r2",
            opts=pulumi.ResourceOptions(parent=self),
        )
        R2DataBuckets(
            "r2",
            data_config=s3_data_config,
            lifecycle_config=gcp_data_config,
            cloudflare_provider=cloudflare_provider,
            imports=imports,
            opts=pulumi.ResourceOptions(parent=self),
        )
        self.register_outputs({})
