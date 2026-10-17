# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CoreWeave AI Object Storage data buckets."""

import json

import pulumi
import pulumi_coreweave as coreweave
from rigging.filesystem.cluster_config import DataConfig, StoreType

from iac.buckets.lifecycle import expiration_rules

HERO_RUN_ROOT = "marin/grug"
PROTECTED_HERO_BUCKET = "marin-us-east-02a"
CHECKPOINT_DELETE_ACTIONS = ("s3:DeleteObject", "s3:DeleteObjectVersion")
COREWEAVE_ORGANIZATION_ID = "208261"
COREWEAVE_MULTIPART_ABORT_RULE_ID = "marin-abort-incomplete-mpu"
COREWEAVE_MULTIPART_EXPIRATION = 7

# Temporary carve-out for one reviewed thinning of the CoreWeave hero's permanent checkpoints.
# While these constants are set, the hero deletion deny names every protected path explicitly
# instead of the ``marin/grug/hero-*`` wildcard. Under the six thinned runs, anything outside
# ``analysis/``, ``.executor*`` and the kept step directories is deletable; when this was written
# that was exactly the 17 superseded step directories in the reviewed manifest. Hero runs created
# while this is applied are not protected. Owner: ClassicLarry; revert by 2026-10-23.
HERO_THINNING_VERSION = "2026.08.19.2"
HERO_THINNING_KEPT_STEPS: dict[str, tuple[int, ...]] = {
    "hero-12d8b6f0-dee637": (6000, 12000, 24000, 36000, 48000, 54000),
    "hero-wd-gate-router-p02-step58k": (60000, 72000, 78000),
    "hero-ragged_a2a-nccl2307-ep-step81k": (84000, 96000, 108000),
    "hero-main-step121638": (132000, 144000),
    "hero-fa4sm100-nomask-step146k": (156000, 168000, 180000, 192000, 204000, 216000, 228000),
    "hero-mfu30-step231k": (240000, 252000, 257553),
}
HERO_THINNING_UNTOUCHED_RUNS = (
    "hero-12d8b6f0-actlog-42k-b",
    "hero-12d8b6f0-actlog-42k",
    "hero-12d8b6f0-actlog-lr-42k",
    "hero-12d8b6f0-attngate-42k",
    "hero-12d8b6f0-attngatescaleeval-48k",
    "hero-12d8b6f0-attnko-42k-b",
    "hero-12d8b6f0-attnko-42k-c",
    "hero-12d8b6f0-attnko-42k",
    "hero-12d8b6f0-attnko-c4-42k",
    "hero-12d8b6f0-attnrangeeval-42k",
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
    "hero-12d8b6f0-samplegen-42k-t0-s0",
    "hero-12d8b6f0-samplegen-42k",
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
    "hero-4rack-bs4096-hist-10k",
    "hero-4rack-bs4096-hist-minmax",
    "hero-4rack-bs4096-hist",
    "hero-4rack-bs4096-int64fix",
    "hero-84579c80-072503",
    "hero-ctx16k-step257k",
    "hero-mix-996f4891-step108k",
    "hero-nopdl-step108k",
    "hero-ragged_a2a-ep-step54k",
    "hero-ragged_a2a-ep-step81k",
    "hero-wd-1rack-test-48k-b",
    "hero-wd-1rack-test-48k-c",
    "hero-wd-1rack-test-48k-d",
    "hero-wd-1rack-test-48k",
    "hero-wd-twitter-pertoken-72k",
)


def hero_thinning_protected_paths() -> list[str]:
    """Return object globs that stay delete-protected during the hero checkpoint thinning."""
    paths = [f"{HERO_RUN_ROOT}/{run}/*" for run in HERO_THINNING_UNTOUCHED_RUNS]
    for run, steps in HERO_THINNING_KEPT_STEPS.items():
        paths.append(f"{HERO_RUN_ROOT}/{run}/analysis/*")
        paths.append(f"{HERO_RUN_ROOT}/{run}/{HERO_THINNING_VERSION}/.executor*")
        paths.extend(f"{HERO_RUN_ROOT}/{run}/{HERO_THINNING_VERSION}/checkpoints/step-{step}/*" for step in steps)
    return paths


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
                "Resource": [f"arn:aws:s3:::{bucket_name}/{path}" for path in hero_thinning_protected_paths()],
            }
        )
    return json.dumps(
        {
            "Version": "2012-10-17",
            "Statement": statements,
        },
        sort_keys=True,
    )


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
        self.register_outputs({})
