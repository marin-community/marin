# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cloudflare R2 data buckets."""

import pulumi
import pulumi_cloudflare as cloudflare
from rigging.filesystem.cluster_config import DataConfig

from iac.buckets.lifecycle import expiration_rules

R2_DEFAULT_JURISDICTION = "default"
R2_DEFAULT_LOCATION = "wnam"
R2_DEFAULT_STORAGE_CLASS = "Standard"
R2_DEFAULT_MULTIPART_EXPIRATION = 7
R2_AGE_CONDITION_TYPE = "Age"
SECONDS_PER_DAY = 86_400


class R2DataBuckets(pulumi.ComponentResource):
    """Own the existing R2 bucket and its complete lifecycle configuration."""

    def __init__(
        self,
        name: str,
        *,
        account_id: str,
        bucket_name: str,
        lifecycle_config: DataConfig,
        cloudflare_provider: pulumi.ProviderResource,
        opts: pulumi.ResourceOptions | None = None,
    ) -> None:
        super().__init__("marin:buckets:R2DataBuckets", name, None, opts)
        lifecycle_rules = [
            cloudflare.R2BucketLifecycleRuleArgs(
                id="abort-incomplete-multipart-uploads",
                enabled=True,
                conditions=cloudflare.R2BucketLifecycleRuleConditionsArgs(prefix=""),
                abort_multipart_uploads_transition=cloudflare.R2BucketLifecycleRuleAbortMultipartUploadsTransitionArgs(
                    condition=cloudflare.R2BucketLifecycleRuleAbortMultipartUploadsTransitionConditionArgs(
                        type=R2_AGE_CONDITION_TYPE,
                        max_age=R2_DEFAULT_MULTIPART_EXPIRATION * SECONDS_PER_DAY,
                    )
                ),
            )
        ]
        for rule in sorted(expiration_rules(lifecycle_config), key=lambda rule: rule.id):
            lifecycle_rules.append(
                cloudflare.R2BucketLifecycleRuleArgs(
                    id=rule.id,
                    enabled=True,
                    conditions=cloudflare.R2BucketLifecycleRuleConditionsArgs(prefix=rule.prefix),
                    delete_objects_transition=cloudflare.R2BucketLifecycleRuleDeleteObjectsTransitionArgs(
                        condition=cloudflare.R2BucketLifecycleRuleDeleteObjectsTransitionConditionArgs(
                            type=R2_AGE_CONDITION_TYPE,
                            max_age=rule.days * SECONDS_PER_DAY,
                        )
                    ),
                )
            )

        resource = cloudflare.R2Bucket(
            f"bucket-{bucket_name}",
            account_id=account_id,
            name=bucket_name,
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

        # Cloudflare's provider cannot import R2 lifecycle resources. After the
        # bucket import, the first update writes this complete declared policy.
        cloudflare.R2BucketLifecycle(
            f"lifecycle-{bucket_name}",
            account_id=account_id,
            bucket_name=resource.name,
            jurisdiction=R2_DEFAULT_JURISDICTION,
            rules=lifecycle_rules,
            opts=pulumi.ResourceOptions(
                parent=self,
                provider=cloudflare_provider,
                protect=True,
                retain_on_delete=True,
            ),
        )
        self.register_outputs({})


class R2PublicBucket(pulumi.ComponentResource):
    """Expose a separate publication bucket through Cloudflare's managed HTTPS.

    Uploaders authenticate through S3; readers use the custom domain without
    credentials. The cache respects each object's publication metadata.
    """

    def __init__(
        self,
        name: str,
        *,
        account_id: str,
        bucket_name: str,
        hostname: str,
        zone_id: str,
        cloudflare_provider: pulumi.ProviderResource,
        opts: pulumi.ResourceOptions | None = None,
    ) -> None:
        super().__init__("marin:buckets:R2PublicBucket", name, None, opts)
        options = pulumi.ResourceOptions(parent=self, provider=cloudflare_provider)
        bucket = cloudflare.R2Bucket(
            f"bucket-{bucket_name}",
            account_id=account_id,
            name=bucket_name,
            location=R2_DEFAULT_LOCATION,
            storage_class=R2_DEFAULT_STORAGE_CLASS,
            opts=pulumi.ResourceOptions.merge(options, pulumi.ResourceOptions(protect=True, retain_on_delete=True)),
        )
        cloudflare.R2ManagedDomain(
            f"managed-domain-{bucket_name}",
            account_id=account_id,
            bucket_name=bucket.name,
            enabled=False,
            opts=options,
        )
        domain = cloudflare.R2CustomDomain(
            f"domain-{bucket_name}",
            account_id=account_id,
            bucket_name=bucket.name,
            domain=hostname,
            zone_id=zone_id,
            enabled=True,
            opts=options,
        )
        client_access = cloudflare.Ruleset(
            "public-bucket-client-access",
            zone_id=zone_id,
            name="Public object storage clients",
            kind="zone",
            phase="http_config_settings",
            rules=[
                cloudflare.RulesetRuleArgs(
                    ref="public_object_storage_clients",
                    action="set_config",
                    expression=f'(http.host eq "{hostname}")',
                    action_parameters=cloudflare.RulesetRuleActionParametersArgs(bic=False),
                ),
            ],
            opts=options,
        )
        cache = cloudflare.Ruleset(
            "public-bucket-cache",
            zone_id=zone_id,
            name="Public object storage cache",
            kind="zone",
            phase="http_request_cache_settings",
            rules=[
                cloudflare.RulesetRuleArgs(
                    ref="public_object_storage",
                    action="set_cache_settings",
                    expression=f'(http.host eq "{hostname}")',
                    action_parameters=cloudflare.RulesetRuleActionParametersArgs(
                        cache=True,
                        browser_ttl=cloudflare.RulesetRuleActionParametersBrowserTtlArgs(mode="respect_origin"),
                        origin_cache_control=True,
                        respect_strong_etags=True,
                        edge_ttl=cloudflare.RulesetRuleActionParametersEdgeTtlArgs(mode="respect_origin"),
                    ),
                ),
            ],
            opts=options,
        )
        self.url = pulumi.Output.all(domain.domain, cache.urn, client_access.urn).apply(
            lambda values: f"https://{values[0]}"
        )
        self.prefix = bucket.name.apply(lambda value: f"s3://{value}")
        self.register_outputs({"url": self.url, "prefix": self.prefix})
