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

# Temporary carve-out for one reviewed deletion of 17 hero checkpoint step directories: the losing 262K finalist
# (all seven permanents) and hero-ctx65k-step372k's merge inputs 385500 through 390000. While these constants are set,
# DenyHeroCheckpointDeletion lists protected paths instead of the ``marin/grug/hero-*`` wildcard: every hero-* run that
# existed on 2026-11-09 stays whole, except that in the two thinned runs only the kept steps and the executor and
# tracker files stay. hero-* runs created while this is applied are not protected. Owner: mcwitt; revert by 2026-11-13.
HERO_RUN_ROOT = "marin/grug"
HERO_THINNING_VERSION = "2026.08.19.2"
HERO_THINNING_KEPT_STEPS: dict[str, tuple[int, ...]] = {
    "hero-ctx65k-step372k": (378000, 384000, 390251),
    "hero-lce-global184-step390k": (),
}
HERO_THINNING_WHOLE_RUNS = (
    "hero-12d8b6f0-actlog-42k",
    "hero-12d8b6f0-actlog-42k-b",
    "hero-12d8b6f0-actlog-lr-42k",
    "hero-12d8b6f0-attngate-42k",
    "hero-12d8b6f0-attngatescaleeval-48k",
    "hero-12d8b6f0-attnko-42k",
    "hero-12d8b6f0-attnko-42k-b",
    "hero-12d8b6f0-attnko-42k-c",
    "hero-12d8b6f0-attnko-c4-42k",
    "hero-12d8b6f0-attnrangeeval-42k",
    "hero-12d8b6f0-dee637",
    "hero-12d8b6f0-embed1k-42k",
    "hero-12d8b6f0-embedfull-42k",
    "hero-12d8b6f0-expert-norms",
    "hero-12d8b6f0-gateweights-42k",
    "hero-12d8b6f0-norms-18k",
    "hero-12d8b6f0-norms-24k",
    "hero-12d8b6f0-norms-42k",
    "hero-12d8b6f0-norms-6k",
    "hero-12d8b6f0-rmsgain-42k",
    "hero-12d8b6f0-routerscale-42k",
    "hero-12d8b6f0-samplegen-42k",
    "hero-12d8b6f0-samplegen-42k-t0-s0",
    "hero-12d8b6f0-samplemulti-42k-t0.2-s0",
    "hero-12d8b6f0-samplemulti-step102000-t0.2-s0",
    "hero-12d8b6f0-samplemulti-step102000-t0.2-s1",
    "hero-12d8b6f0-samplemulti-step24000-t0.2-s0",
    "hero-12d8b6f0-samplemulti-step24000-t0.2-s1",
    "hero-12d8b6f0-samplemulti-step42000-t0.2-s1",
    "hero-12d8b6f0-samplemulti-step6000-t0.2-s0",
    "hero-12d8b6f0-samplemulti-step6000-t0.2-s1",
    "hero-12d8b6f0-samplemulti-step78000-t0.2-s0",
    "hero-12d8b6f0-samplemulti-step78000-t0.2-s1",
    "hero-12d8b6f0-samplemulti-step78000-t0.2-s2",
    "hero-12d8b6f0-samplemulti-step78000-t0.2-s3",
    "hero-12d8b6f0-samplemulti-step78000-t1-s0",
    "hero-12d8b6f0-sconvnorms-42k",
    "hero-12d8b6f0-skipeval-42k",
    "hero-20260819",
    "hero-20260820",
    "hero-4rack-bs4096-hist",
    "hero-4rack-bs4096-hist-10k",
    "hero-4rack-bs4096-hist-minmax",
    "hero-4rack-bs4096-int64fix",
    "hero-84579c80-072503",
    "hero-ctx16k-step257k",
    "hero-ctx65k-merge-hb-avg11",
    "hero-fa4sm100-nomask-step146k",
    "hero-format-sft",
    "hero-gradclip-step356k",
    "hero-lce-log65536sq-step390k",
    "hero-main-step121638",
    "hero-mfu30-step231k",
    "hero-mix-996f4891-step108k",
    "hero-nopdl-step108k",
    "hero-ragged_a2a-ep-step54k",
    "hero-ragged_a2a-ep-step81k",
    "hero-ragged_a2a-nccl2307-ep-step81k",
    "hero-wd-1rack-test-48k",
    "hero-wd-1rack-test-48k-b",
    "hero-wd-1rack-test-48k-c",
    "hero-wd-1rack-test-48k-d",
    "hero-wd-gate-router-p02-step58k",
    "hero-wd-twitter-pertoken-72k",
)


def hero_thinning_protected_paths() -> list[str]:
    """Return object globs that stay delete-protected during the hero checkpoint thinning."""
    paths = [f"{HERO_RUN_ROOT}/{run}/*" for run in HERO_THINNING_WHOLE_RUNS]
    for run, steps in HERO_THINNING_KEPT_STEPS.items():
        root = f"{HERO_RUN_ROOT}/{run}/{HERO_THINNING_VERSION}"
        paths.append(f"{root}/.*")
        paths.append(f"{root}/tracker_metrics.jsonl*")
        paths.extend(f"{root}/checkpoints/step-{step}/*" for step in steps)
    return paths


# Marin Belay's dedicated CoreWeave user. Iris attaches the submitter's keys to every CoreWeave job, so this
# principal's access bounds what Belay runs (including agent-written code) can read and write. The organization
# access policy below grants that access, but `admin-use-bucket-go-fast` grants every principal far more, so the bounds
# are explicit denies (which override any allow): per-bucket data denies in every bucket policy, and a management deny
# in the Belay organization policy. Rotating the Belay key needs that management deny lifted temporarily.
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
# Account and bucket management actions that the organization-wide `admin-use-bucket-go-fast` access policy (managed
# in the console, not here) grants every principal. Bucket policies cannot deny the `cwobject:` control-plane actions
# (access keys, organization access policies, audit logging), and CoreWeave evaluates bucket-policy changes against
# organization policies only, so the Belay organization policy denies them all.
BELAY_DENIED_MANAGEMENT_ACTIONS = (
    "s3:CreateBucket",
    "s3:DeleteBucket",
    "s3:DeleteBucketPolicy",
    "s3:DeleteBucketTagging",
    "s3:DeleteLifecycleConfiguration",
    "s3:DeleteObjectTagging",
    "s3:DeleteObjectVersion",
    "s3:PutBucketCORS",
    "s3:PutBucketPolicy",
    "s3:PutBucketTagging",
    "s3:PutBucketVersioning",
    "s3:PutLifecycleConfiguration",
    "s3:PutObjectTagging",
    "cwobject:CreateAccessKey",
    "cwobject:CreateAccessKeyOIDC",
    "cwobject:CreateAccessKeySAML",
    "cwobject:DeleteAccessPolicy",
    "cwobject:DisableBucketAuditLogging",
    "cwobject:DisableBucketAuditLoggingDefault",
    "cwobject:DisableControlPlaneAuditLogging",
    "cwobject:EnableBucketAuditLogging",
    "cwobject:EnableBucketAuditLoggingDefault",
    "cwobject:EnableControlPlaneAuditLogging",
    "cwobject:EnsureAccessPolicy",
    "cwobject:RevokeAccessKeyByAccessKey",
    "cwobject:RevokeAccessKeysByPrincipal",
    "cwobject:UpdateAccessKeyStatus",
)
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
                "Resource": [f"arn:aws:s3:::{bucket_name}/{path}" for path in hero_thinning_protected_paths()],
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
        coreweave.ObjectStorageOrganizationAccessPolicyStatementArgs(
            name="belay-deny-management",
            effect="Deny",
            actions=list(BELAY_DENIED_MANAGEMENT_ACTIONS),
            resources=["*"],
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
