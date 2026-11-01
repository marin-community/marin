# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from fnmatch import fnmatch

from iac.buckets.coreweave import BELAY_BUCKET, HERO_CHECKPOINT_PREFIX, belay_policy_statements


def allowed(action: str, path: str) -> bool:
    return any(action in s.actions and any(fnmatch(path, r) for r in s.resources) for s in belay_policy_statements())


def test_belay_can_write_its_runs_and_temp_checkpoints() -> None:
    run = f"{BELAY_BUCKET}/marin/tmp/ttl=30d/belay/runs/belay-X0001-control"
    assert allowed("s3:PutObject", f"{run}/final_eval_done.json")
    assert allowed("s3:PutObject", f"{BELAY_BUCKET}/tmp/ttl=14d/checkpoints-temp/{run}/checkpoints/step-100/x")
    assert allowed("s3:GetObject", f"{BELAY_BUCKET}/marin/paloma/c4_en-hero-bpe-v16384/validation/shard")


def test_belay_cannot_touch_hero_checkpoints_or_other_runs() -> None:
    hero = f"{BELAY_BUCKET}/{HERO_CHECKPOINT_PREFIX}gradclip-step356k/checkpoints/step-366000/metadata.json"
    other_run = f"{BELAY_BUCKET}/marin/tmp/ttl=30d/someone-else/run/x"
    for action in ("s3:GetObject", "s3:PutObject", "s3:DeleteObject"):
        assert not allowed(action, hero)
        assert not allowed(action, other_run)
    assert not allowed("s3:PutObject", f"{BELAY_BUCKET}/marin/paloma/c4_en-hero-bpe-v16384/validation/shard")
