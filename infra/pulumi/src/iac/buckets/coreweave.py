# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CoreWeave AI Object Storage data buckets."""

import pulumi
import pulumi_coreweave as coreweave
from rigging.filesystem.cluster_config import DataConfig, StoreType

from iac.buckets.lifecycle import ExpirationRule, expiration_rules
from iac.imports import NO_IMPORTS, ImportRegistrar

COREWEAVE_EXTRA_EXPIRATION_RULES: dict[str, tuple[ExpirationRule, ...]] = {
    "marin-us-east-02a": (
        ExpirationRule(
            id="bench-probe-ttl",
            prefix="tmp/benchmark/scratch/ttl-probe/",
            days=1,
        ),
    ),
}


class CoreweaveDataBuckets(pulumi.ComponentResource):
    """Create CoreWeave buckets and their complete lifecycle configurations."""

    def __init__(
        self,
        name: str,
        *,
        data_config: DataConfig,
        lifecycle_config: DataConfig,
        coreweave_provider: pulumi.ProviderResource,
        imports: ImportRegistrar = NO_IMPORTS,
        opts: pulumi.ResourceOptions | None = None,
    ) -> None:
        super().__init__("marin:buckets:CoreweaveDataBuckets", name, None, opts)
        for bucket in sorted(data_config.region_buckets.values(), key=lambda spec: spec.name):
            if bucket.store is not StoreType.COREWEAVE:
                continue
            if bucket.signing_region is None:
                raise ValueError(f"CoreWeave bucket {bucket.name!r} has no signing region")

            resource = coreweave.ObjectStorageBucket(
                f"bucket-{bucket.name}",
                name=bucket.name,
                zone=bucket.signing_region,
                opts=pulumi.ResourceOptions(
                    parent=self,
                    provider=coreweave_provider,
                    protect=True,
                    retain_on_delete=True,
                ),
            )
            imports.register(resource, parent=self, provider_id=bucket.name)

            lifecycle = coreweave.ObjectStorageBucketLifecycleConfiguration(
                f"lifecycle-{bucket.name}",
                bucket=resource.name,
                rules=[
                    coreweave.ObjectStorageBucketLifecycleConfigurationRuleArgs(
                        id=rule.id,
                        status="Enabled",
                        filter=coreweave.ObjectStorageBucketLifecycleConfigurationRuleFilterArgs(prefix=rule.prefix),
                        expiration=coreweave.ObjectStorageBucketLifecycleConfigurationRuleExpirationArgs(days=rule.days),
                    )
                    for rule in (
                        *COREWEAVE_EXTRA_EXPIRATION_RULES.get(bucket.name, ()),
                        *expiration_rules(lifecycle_config),
                    )
                ],
                opts=pulumi.ResourceOptions(
                    parent=self,
                    provider=coreweave_provider,
                    protect=True,
                    retain_on_delete=True,
                ),
            )
            imports.register(lifecycle, parent=self, provider_id=bucket.name)
        self.register_outputs({})
