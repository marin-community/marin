# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from fnmatch import fnmatch

from iac.buckets.coreweave import (
    BELAY_BUCKET,
    BELAY_BUCKET_POLICY_PRINCIPAL,
    BELAY_RUN_ROOT,
    HERO_CHECKPOINT_PREFIX,
    belay_policy_statements,
    coreweave_bucket_policy,
    coreweave_temp_bucket,
)
from rigging.filesystem.cluster_config import load_cluster_config, marin_temp_bucket, use_data_config

RUN = f"{BELAY_BUCKET}/{BELAY_RUN_ROOT}/runs/belay-X0001-control"


def allowed(statements, action: str, path: str) -> bool:
    return any(
        s.effect == "Allow" and action in s.actions and any(fnmatch(path, r) for r in s.resources) for s in statements
    )


def test_belay_can_write_where_training_actually_puts_its_temp_checkpoints() -> None:
    config = load_cluster_config("coreweave")
    statements = belay_policy_statements(coreweave_temp_bucket(config))
    with use_data_config(config):
        temp = marin_temp_bucket(14, f"checkpoints-temp/{RUN}/checkpoints", source_prefix=f"s3://{RUN}")
    assert allowed(statements, "s3:PutObject", temp.removeprefix("s3://") + "/step-100/manifest.ocdbt")
    assert allowed(statements, "s3:PutObject", f"{RUN}/final_eval_done.json")
    assert allowed(statements, "s3:GetObject", f"{BELAY_BUCKET}/marin/paloma/c4_en-hero-bpe-v16384/validation/shard")


def test_belay_cannot_touch_hero_checkpoints_or_other_runs() -> None:
    statements = belay_policy_statements(coreweave_temp_bucket(load_cluster_config("coreweave")))
    hero = f"{BELAY_BUCKET}/{HERO_CHECKPOINT_PREFIX}gradclip-step356k/checkpoints/step-366000/metadata.json"
    other_run = f"{BELAY_BUCKET}/marin/tmp/ttl=30d/someone-else/run/x"
    for action in ("s3:GetObject", "s3:PutObject", "s3:DeleteObject"):
        assert not allowed(statements, action, hero)
        assert not allowed(statements, action, other_run)
    assert not allowed(statements, "s3:PutObject", f"{BELAY_BUCKET}/marin/paloma/c4_en-hero-bpe-v16384/validation/shard")


def denied(bucket_policy: str, action: str, arn: str) -> bool:
    """Whether a Deny statement for the Belay principal matches; `arn` is a bucket or object ARN."""

    def matches(patterns, value):
        return any(fnmatch(value, p) for p in ([patterns] if isinstance(patterns, str) else patterns))

    for s in json.loads(bucket_policy)["Statement"]:
        if s["Effect"] != "Deny" or s["Principal"] != {"CW": [BELAY_BUCKET_POLICY_PRINCIPAL]}:
            continue
        action_hit = matches(s["Action"], action) if "Action" in s else not matches(s["NotAction"], action)
        resource_hit = matches(s["Resource"], arn) if "Resource" in s else not matches(s["NotResource"], arn)
        if action_hit and resource_hit:
            return True
    return False


def test_bucket_policies_deny_belay_everything_its_grants_do_not_cover() -> None:
    config = load_cluster_config("coreweave")
    temp_bucket = coreweave_temp_bucket(config)
    with use_data_config(config):
        temp = marin_temp_bucket(14, f"checkpoints-temp/{RUN}/checkpoints", source_prefix=f"s3://{RUN}")
    temp_object = temp.removeprefix("s3://") + "/step-100/manifest.ocdbt"

    def policy_for(path: str) -> str:
        return coreweave_bucket_policy(path.split("/", 1)[0], temp_bucket)

    def object_arn(path: str) -> str:
        return f"arn:aws:s3:::{path}"

    for path in (f"{RUN}/final_eval_done.json", temp_object):
        for action in ("s3:GetObject", "s3:PutObject", "s3:DeleteObject"):
            assert not denied(policy_for(path), action, object_arn(path))
    paloma = f"{BELAY_BUCKET}/marin/paloma/c4_en-hero-bpe-v16384/validation/shard"
    assert not denied(policy_for(paloma), "s3:GetObject", object_arn(paloma))
    assert denied(policy_for(paloma), "s3:PutObject", object_arn(paloma))
    assert not denied(policy_for(BELAY_BUCKET), "s3:ListBucket", f"arn:aws:s3:::{BELAY_BUCKET}")

    hero = f"{BELAY_BUCKET}/{HERO_CHECKPOINT_PREFIX}ctx65k-step372k/checkpoints/step-385500/metadata.json"
    other_run = f"{BELAY_BUCKET}/marin/tmp/ttl=1d/someone-else/run/x"
    old_hero_object = "hero-checkpoints/marin/grug/hero-main-step121638/checkpoints/step-1/metadata.json"
    for path in (hero, other_run, old_hero_object):
        for action in ("s3:GetObject", "s3:PutObject", "s3:DeleteObject"):
            assert denied(policy_for(path), action, object_arn(path))
    assert denied(policy_for(old_hero_object), "s3:ListBucket", "arn:aws:s3:::hero-checkpoints")
    assert denied(policy_for(BELAY_BUCKET), "s3:DeleteObjectVersion", object_arn(f"{RUN}/x"))


def test_belay_organization_policy_denies_account_and_policy_management() -> None:
    statements = belay_policy_statements(coreweave_temp_bucket(load_cluster_config("coreweave")))
    (deny,) = [s for s in statements if s.effect == "Deny"]
    assert deny.resources == ["*"]
    # CoreWeave evaluates bucket-policy changes against organization policies only, so only this deny can stop Belay
    # from rewriting a bucket policy to drop its bucket-level denies.
    for action in (
        "s3:PutBucketPolicy",
        "s3:DeleteBucketPolicy",
        "cwobject:EnsureAccessPolicy",
        "cwobject:DeleteAccessPolicy",
        "cwobject:CreateAccessKey",
    ):
        assert action in deny.actions
    granted = {a for s in statements if s.effect == "Allow" for a in s.actions}
    assert not granted & set(deny.actions)
