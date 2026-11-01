# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CoreWeave AI Object Storage data buckets."""

import json

import pulumi
import pulumi_coreweave as coreweave
from rigging.filesystem.cluster_config import DataConfig, StoreType

from iac.buckets.lifecycle import expiration_rules

HERO_CHECKPOINT_PREFIX = "marin/grug/hero-"
PROTECTED_HERO_BUCKET = "marin-us-east-02a"
CHECKPOINT_DELETE_ACTIONS = ("s3:DeleteObject", "s3:DeleteObjectVersion")
COREWEAVE_ORGANIZATION_ID = "208261"
COREWEAVE_MULTIPART_ABORT_RULE_ID = "marin-abort-incomplete-mpu"
COREWEAVE_MULTIPART_EXPIRATION = 7

# Marin Belay's dedicated CoreWeave user. Iris attaches the submitter's keys to every CoreWeave job, so this
# principal's grants bound what Belay runs (including agent-written code) can read and write. A request needs both
# the bucket policy (which admits every principal in the organization) and an organization access policy; the user is
# in no groups, so the organization access policy below is its only grant.
BELAY_PRINCIPAL = "coreweave/unAfPLe6TaiRmJsewaUog"
BELAY_BUCKET = "marin-us-east-02a"
BELAY_RUN_ROOT = "marin/tmp/ttl=30d/belay"
BELAY_RUN_PREFIX = f"{BELAY_BUCKET}/{BELAY_RUN_ROOT}/*"
BELAY_READ_PREFIXES = (
    f"{BELAY_BUCKET}/marin/datakit/hero_tok/*",
    f"{BELAY_BUCKET}/marin/paloma/*",
    f"{BELAY_BUCKET}/marin/uncheatable_eval/*",
)


def coreweave_bucket_policy(bucket_name: str) -> str:
    """Return organization access, protecting hero checkpoints in the 02a bucket."""
    statements = [
        {
            "Sid": "AllowOrganizationS3Access",
            "Effect": "Allow",
            "Principal": {"CW": "*"},
            "Action": ["s3:*"],
            "Resource": [
                f"arn:aws:s3:::{bucket_name}",
                f"arn:aws:s3:::{bucket_name}/*",
            ],
            "Condition": {
                "StringEquals": {
                    "cw:PrincipalOrgID": [COREWEAVE_ORGANIZATION_ID],
                }
            },
        }
    ]
    if bucket_name == PROTECTED_HERO_BUCKET:
        statements.append(
            {
                "Sid": "DenyHeroCheckpointDeletion",
                "Effect": "Deny",
                "Principal": {"CW": "*"},
                "Action": list(CHECKPOINT_DELETE_ACTIONS),
                "Resource": f"arn:aws:s3:::{bucket_name}/{HERO_CHECKPOINT_PREFIX}*",
            }
        )
    return json.dumps(
        {
            "Version": "2012-10-17",
            "Statement": statements,
        },
        sort_keys=True,
    )


def coreweave_temp_bucket(data_config: DataConfig) -> str:
    store = data_config.stores.get(StoreType.COREWEAVE)
    if store is None or store.temp_bucket is None:
        raise ValueError("the CoreWeave data config has no temp_bucket")
    return store.temp_bucket.name


def belay_temp_checkpoint_prefixes(temp_bucket: str) -> list[str]:
    """Temporary checkpoints of Belay runs: the backend's scratch bucket, plus the run's own bucket for Belay
    code branches that predate the shared CoreWeave scratch bucket (#10221)."""
    return [
        f"{bucket}/tmp/ttl=14d/checkpoints-temp/{BELAY_BUCKET}/{BELAY_RUN_ROOT}/*"
        for bucket in dict.fromkeys((temp_bucket, BELAY_BUCKET))
    ]


def belay_policy_statements(temp_bucket: str) -> list[coreweave.ObjectStorageOrganizationAccessPolicyStatementArgs]:
    """Belay may list its buckets, read training and eval data and profiles, and write only its own runs."""
    return [
        coreweave.ObjectStorageOrganizationAccessPolicyStatementArgs(
            name="belay-list-buckets",
            effect="Allow",
            actions=["s3:ListBucket"],
            resources=list(dict.fromkeys((BELAY_BUCKET, temp_bucket))),
            principals=[BELAY_PRINCIPAL],
        ),
        coreweave.ObjectStorageOrganizationAccessPolicyStatementArgs(
            name="belay-read-data",
            effect="Allow",
            actions=["s3:GetObject"],
            resources=[*BELAY_READ_PREFIXES, f"{temp_bucket}/tmp/ttl=30d/xprof/*"],
            principals=[BELAY_PRINCIPAL],
        ),
        coreweave.ObjectStorageOrganizationAccessPolicyStatementArgs(
            name="belay-write-runs",
            effect="Allow",
            actions=[
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:AbortMultipartUpload",
                "s3:ListMultipartUploadParts",
            ],
            resources=[BELAY_RUN_PREFIX, *belay_temp_checkpoint_prefixes(temp_bucket)],
            principals=[BELAY_PRINCIPAL],
        ),
    ]


class CoreweaveDataBuckets(pulumi.ComponentResource):
    """Create CoreWeave buckets and their complete lifecycle configurations."""

    def __init__(
        self,
        name: str,
        *,
        data_config: DataConfig,
        lifecycle_config: DataConfig,
        coreweave_provider: pulumi.ProviderResource,
        opts: pulumi.ResourceOptions | None = None,
    ) -> None:
        super().__init__("marin:buckets:CoreweaveDataBuckets", name, None, opts)
        lifecycle_rules = [
            coreweave.ObjectStorageBucketLifecycleConfigurationRuleArgs(
                id=COREWEAVE_MULTIPART_ABORT_RULE_ID,
                status="Enabled",
                filter=coreweave.ObjectStorageBucketLifecycleConfigurationRuleFilterArgs(prefix=""),
                abort_incomplete_multipart_upload=(
                    coreweave.ObjectStorageBucketLifecycleConfigurationRuleAbortIncompleteMultipartUploadArgs(
                        days_after_initiation=COREWEAVE_MULTIPART_EXPIRATION
                    )
                ),
            )
        ]
        lifecycle_rules.extend(
            coreweave.ObjectStorageBucketLifecycleConfigurationRuleArgs(
                id=rule.id,
                status="Enabled",
                filter=coreweave.ObjectStorageBucketLifecycleConfigurationRuleFilterArgs(prefix=rule.prefix),
                expiration=coreweave.ObjectStorageBucketLifecycleConfigurationRuleExpirationArgs(days=rule.days),
            )
            for rule in expiration_rules(lifecycle_config)
        )
        for bucket in sorted(data_config.buckets.values(), key=lambda spec: spec.name):
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

            coreweave.ObjectStorageBucketLifecycleConfiguration(
                f"lifecycle-{bucket.name}",
                bucket=resource.name,
                rules=lifecycle_rules,
                opts=pulumi.ResourceOptions(
                    parent=self,
                    provider=coreweave_provider,
                    protect=True,
                    retain_on_delete=True,
                ),
            )

            coreweave.ObjectStorageBucketPolicy(
                f"policy-{bucket.name}",
                bucket=resource.name,
                policy=resource.name.apply(coreweave_bucket_policy),
                opts=pulumi.ResourceOptions(
                    parent=self,
                    provider=coreweave_provider,
                    protect=True,
                    retain_on_delete=True,
                ),
            )
        coreweave.ObjectStorageOrganizationAccessPolicy(
            "policy-belay",
            name="marin-belay",
            statements=belay_policy_statements(coreweave_temp_bucket(data_config)),
            opts=pulumi.ResourceOptions(parent=self, provider=coreweave_provider),
        )
        self.register_outputs({})
