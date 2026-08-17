# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regional GCS data buckets."""

from dataclasses import dataclass

import pulumi
import pulumi_gcp as gcp
from rigging.filesystem.cluster_config import DataConfig, StoreType

from iac.buckets.lifecycle import expiration_rules
from iac.imports import NO_IMPORTS, ImportRegistrar


@dataclass(frozen=True)
class GcpDataBucketsArgs:
    project: str
    data_config: DataConfig
    log_bucket: str


def _lifecycle_rules(data_config: DataConfig) -> list[gcp.storage.BucketLifecycleRuleArgs]:
    return [
        gcp.storage.BucketLifecycleRuleArgs(
            action=gcp.storage.BucketLifecycleRuleActionArgs(type="Delete"),
            condition=gcp.storage.BucketLifecycleRuleConditionArgs(
                age=rule.days,
                matches_prefixes=[rule.prefix],
            ),
        )
        for rule in expiration_rules(data_config)
    ]


class GcpDataBuckets(pulumi.ComponentResource):
    """Create the region-local GCS buckets declared by the Marin data config."""

    def __init__(
        self,
        name: str,
        args: GcpDataBucketsArgs,
        *,
        gcp_provider: pulumi.ProviderResource,
        imports: ImportRegistrar = NO_IMPORTS,
        opts: pulumi.ResourceOptions | None = None,
    ) -> None:
        super().__init__("marin:buckets:GcpDataBuckets", name, None, opts)
        lifecycle_rules = _lifecycle_rules(args.data_config)
        for region, bucket in sorted(args.data_config.region_buckets.items()):
            if bucket.store is not StoreType.GCS:
                continue
            resource = gcp.storage.Bucket(
                f"bucket-{region}",
                project=args.project,
                name=bucket.name,
                location=region,
                storage_class="STANDARD",
                autoclass=gcp.storage.BucketAutoclassArgs(enabled=True),
                force_destroy=False,
                uniform_bucket_level_access=True,
                public_access_prevention="enforced",
                soft_delete_policy=gcp.storage.BucketSoftDeletePolicyArgs(retention_duration_seconds=0),
                logging=gcp.storage.BucketLoggingArgs(
                    log_bucket=args.log_bucket,
                    log_object_prefix=f"usage/{bucket.name}",
                ),
                lifecycle_rules=lifecycle_rules,
                opts=pulumi.ResourceOptions(
                    parent=self,
                    provider=gcp_provider,
                    protect=True,
                    retain_on_delete=True,
                ),
            )
            imports.register(resource, parent=self, provider_id=f"{args.project}/{bucket.name}")
        self.register_outputs({})
