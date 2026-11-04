# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CoreWeave AI Object Storage data buckets."""

import json
from collections.abc import Iterable
from dataclasses import dataclass

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
# principal's access bounds what Belay runs (including agent-written code) can read and write. The organization
# access policy below grants that access, but other organization grants also reach the user, so the bounds are
# enforced by explicit denies in every bucket policy; an explicit deny overrides any allow.
BELAY_USER_ID = "unAfPLe6TaiRmJsewaUog"
BELAY_PRINCIPAL = f"coreweave/{BELAY_USER_ID}"
BELAY_BUCKET_POLICY_PRINCIPAL = f"arn:aws:iam::{COREWEAVE_ORGANIZATION_ID}:coreweave/{BELAY_USER_ID}"
BELAY_BUCKET = "marin-us-east-02a"
BELAY_RUN_ROOT = "marin/tmp/ttl=30d/belay"
BELAY_RUN_PREFIX = f"{BELAY_BUCKET}/{BELAY_RUN_ROOT}/*"
BELAY_WRITE_ACTIONS = (
    "s3:PutObject",
    "s3:DeleteObject",
    "s3:AbortMultipartUpload",
    "s3:ListMultipartUploadParts",
)
BELAY_READ_ACTION = "s3:GetObject"
BELAY_LIST_ACTION = "s3:ListBucket"
BELAY_OBJECT_ACTIONS = (BELAY_READ_ACTION, *BELAY_WRITE_ACTIONS)
BELAY_READ_PREFIXES = (
    f"{BELAY_BUCKET}/marin/datakit/hero_tok/*",
    f"{BELAY_BUCKET}/marin/paloma/*",
    f"{BELAY_BUCKET}/marin/uncheatable_eval/*",
)


def coreweave_bucket_policy(bucket_name: str, temp_bucket: str) -> str:
    """Return organization access, protecting hero checkpoints in the 02a bucket and bounding Belay."""
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
    statements.extend(belay_bucket_denies(bucket_name, temp_bucket))
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


@dataclass(frozen=True)
class BelayGrants:
    """Everything Belay may do, as bucket names and `bucket/prefix*` object patterns."""

    list_buckets: tuple[str, ...]
    read_prefixes: tuple[str, ...]
    write_prefixes: tuple[str, ...]


def belay_grants(temp_bucket: str) -> BelayGrants:
    """Belay may list its buckets, read training and eval data and profiles, and write only its own runs."""
    return BelayGrants(
        list_buckets=tuple(dict.fromkeys((BELAY_BUCKET, temp_bucket))),
        read_prefixes=(*BELAY_READ_PREFIXES, f"{temp_bucket}/tmp/ttl=30d/xprof/*"),
        write_prefixes=(BELAY_RUN_PREFIX, *belay_temp_checkpoint_prefixes(temp_bucket)),
    )


def belay_policy_statements(temp_bucket: str) -> list[coreweave.ObjectStorageOrganizationAccessPolicyStatementArgs]:
    grants = belay_grants(temp_bucket)
    return [
        coreweave.ObjectStorageOrganizationAccessPolicyStatementArgs(
            name="belay-list-buckets",
            effect="Allow",
            actions=[BELAY_LIST_ACTION],
            resources=list(grants.list_buckets),
            principals=[BELAY_PRINCIPAL],
        ),
        coreweave.ObjectStorageOrganizationAccessPolicyStatementArgs(
            name="belay-read-data",
            effect="Allow",
            actions=[BELAY_READ_ACTION],
            resources=list(grants.read_prefixes),
            principals=[BELAY_PRINCIPAL],
        ),
        coreweave.ObjectStorageOrganizationAccessPolicyStatementArgs(
            name="belay-write-runs",
            effect="Allow",
            actions=list(BELAY_OBJECT_ACTIONS),
            resources=list(grants.write_prefixes),
            principals=[BELAY_PRINCIPAL],
        ),
    ]


def _object_arns_in(bucket_name: str, prefixes: Iterable[str]) -> list[str]:
    return [f"arn:aws:s3:::{p}" for p in prefixes if p.split("/", 1)[0] == bucket_name]


def _belay_deny(sid: str, **fields: list[str] | str) -> dict:
    return {"Sid": sid, "Effect": "Deny", "Principal": {"CW": [BELAY_BUCKET_POLICY_PRINCIPAL]}, **fields}


def belay_bucket_denies(bucket_name: str, temp_bucket: str) -> list[dict]:
    """Deny Belay everything in `bucket_name` beyond its grants: other actions, writes outside its runs, reads outside
    its data, and listing buckets it has no reason to list."""
    grants = belay_grants(temp_bucket)
    bucket_arn = f"arn:aws:s3:::{bucket_name}"
    objects_arn = f"{bucket_arn}/*"
    writable = _object_arns_in(bucket_name, grants.write_prefixes)
    readable = _object_arns_in(bucket_name, (*grants.read_prefixes, *grants.write_prefixes))
    allowed_actions = [BELAY_LIST_ACTION] if bucket_name in grants.list_buckets else []
    allowed_actions.extend(BELAY_OBJECT_ACTIONS)
    return [
        _belay_deny(
            "BelayDenyOtherActions",
            NotAction=allowed_actions,
            Resource=[bucket_arn, objects_arn],
        ),
        _belay_deny(
            "BelayDenyWritesOutsideRuns",
            Action=list(BELAY_WRITE_ACTIONS),
            **({"NotResource": writable} if writable else {"Resource": objects_arn}),
        ),
        _belay_deny(
            "BelayDenyReadsOutsideData",
            Action=[BELAY_READ_ACTION],
            **({"NotResource": readable} if readable else {"Resource": objects_arn}),
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
        temp_bucket = coreweave_temp_bucket(data_config)
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
                policy=resource.name.apply(lambda name: coreweave_bucket_policy(name, temp_bucket)),
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
            statements=belay_policy_statements(temp_bucket),
            opts=pulumi.ResourceOptions(parent=self, provider=coreweave_provider),
        )
        self.register_outputs({})
