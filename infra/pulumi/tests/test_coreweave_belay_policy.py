# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from fnmatch import fnmatch

from iac.buckets.coreweave import (
    BELAY_BUCKET,
    BELAY_RUN_ROOT,
    HERO_CHECKPOINT_PREFIX,
    belay_policy_statements,
    coreweave_temp_bucket,
)
from rigging.filesystem.cluster_config import load_cluster_config, marin_temp_bucket, use_data_config

RUN = f"{BELAY_BUCKET}/{BELAY_RUN_ROOT}/runs/belay-X0001-control"


def allowed(statements, action: str, path: str) -> bool:
    return any(action in s.actions and any(fnmatch(path, r) for r in s.resources) for s in statements)


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
