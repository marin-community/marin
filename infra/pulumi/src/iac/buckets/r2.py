# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cloudflare R2 data buckets."""

from dataclasses import dataclass
from urllib.parse import urlparse

import pulumi
import pulumi_cloudflare as cloudflare
from rigging.filesystem.cluster_config import DataConfig, StoreType

from iac.imports import NO_IMPORTS, ImportRegistrar

R2_ENDPOINT_SUFFIX = ".r2.cloudflarestorage.com"
R2_DEFAULT_JURISDICTION = "default"
R2_DEFAULT_LOCATION = "wnam"
R2_DEFAULT_STORAGE_CLASS = "Standard"
R2_DEFAULT_MULTIPART_EXPIRATION = 7
SECONDS_PER_DAY = 86_400


@dataclass(frozen=True)
class R2DataBucketsArgs:
    data_config: DataConfig
    lifecycle_config: DataConfig


def _account_id(data_config: DataConfig) -> str:
    endpoint = data_config.stores[StoreType.R2].endpoint
    hostname = urlparse(endpoint).hostname
    if hostname is None or not hostname.endswith(R2_ENDPOINT_SUFFIX):
        raise ValueError(f"R2 endpoint has no Cloudflare account hostname: {endpoint!r}")
    return hostname.removesuffix(R2_ENDPOINT_SUFFIX)


def _delete_rule(rule_id: str, prefix: str, days: int) -> cloudflare.R2BucketLifecycleRuleArgs:
    return cloudflare.R2BucketLifecycleRuleArgs(
        id=rule_id,
        enabled=True,
        conditions=cloudflare.R2BucketLifecycleRuleConditionsArgs(prefix=prefix),
        delete_objects_transition=cloudflare.R2BucketLifecycleRuleDeleteObjectsTransitionArgs(
            condition=cloudflare.R2BucketLifecycleRuleDeleteObjectsTransitionConditionArgs(
                type="Age",
                max_age=days * SECONDS_PER_DAY,
            )
        ),
    )


def _lifecycle_rules(data_config: DataConfig) -> list[cloudflare.R2BucketLifecycleRuleArgs]:
    rules = [
        cloudflare.R2BucketLifecycleRuleArgs(
            id="abort-incomplete-multipart-uploads",
            enabled=True,
            conditions=cloudflare.R2BucketLifecycleRuleConditionsArgs(prefix=""),
            abort_multipart_uploads_transition=cloudflare.R2BucketLifecycleRuleAbortMultipartUploadsTransitionArgs(
                condition=cloudflare.R2BucketLifecycleRuleAbortMultipartUploadsTransitionConditionArgs(
                    type="Age",
                    max_age=R2_DEFAULT_MULTIPART_EXPIRATION * SECONDS_PER_DAY,
                )
            ),
        )
    ]
    rules.extend(
        _delete_rule(
            f"marin-ttl-{ttl_days}d",
            f"{data_config.temp_path}/ttl={ttl_days}d/",
            ttl_days,
        )
        for ttl_days in data_config.ttl_days
    )
    return rules


class R2DataBuckets(pulumi.ComponentResource):
    """Create R2 buckets and their complete lifecycle configurations."""

    def __init__(
        self,
        name: str,
        args: R2DataBucketsArgs,
        *,
        cloudflare_provider: pulumi.ProviderResource,
        imports: ImportRegistrar = NO_IMPORTS,
        opts: pulumi.ResourceOptions | None = None,
    ) -> None:
        super().__init__("marin:buckets:R2DataBuckets", name, None, opts)
        account_id = _account_id(args.data_config)
        for bucket in sorted(args.data_config.region_buckets.values(), key=lambda spec: spec.name):
            if bucket.store is not StoreType.R2:
                continue

            resource = cloudflare.R2Bucket(
                f"bucket-{bucket.name}",
                account_id=account_id,
                name=bucket.name,
                jurisdiction=R2_DEFAULT_JURISDICTION,
                location=R2_DEFAULT_LOCATION,
                storage_class=R2_DEFAULT_STORAGE_CLASS,
                opts=pulumi.ResourceOptions(
                    parent=self,
                    provider=cloudflare_provider,
                    protect=True,
                    retain_on_delete=True,
                ),
            )
            imports.register(
                resource,
                parent=self,
                provider_id=f"{account_id}/{bucket.name}/{R2_DEFAULT_JURISDICTION}",
            )

            # Cloudflare's provider cannot import R2 lifecycle resources. The first
            # update adopts the existing whole-bucket policy by writing this equivalent
            # declaration after the bucket itself has been imported.
            cloudflare.R2BucketLifecycle(
                f"lifecycle-{bucket.name}",
                account_id=account_id,
                bucket_name=resource.name,
                jurisdiction=R2_DEFAULT_JURISDICTION,
                rules=_lifecycle_rules(args.lifecycle_config),
                opts=pulumi.ResourceOptions(
                    parent=self,
                    provider=cloudflare_provider,
                    protect=True,
                    retain_on_delete=True,
                ),
            )
        self.register_outputs({})
