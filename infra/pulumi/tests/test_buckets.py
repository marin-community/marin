# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

from iac.buckets.coreweave import hero_checkpoint_delete_policy


def test_hero_checkpoint_delete_policy_denies_delete_actions_for_prefix() -> None:
    assert json.loads(hero_checkpoint_delete_policy("marin-east")) == {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "DenyHeroCheckpointDeletion",
                "Effect": "Deny",
                "Principal": {"CW": "*"},
                "Action": ["s3:DeleteObject", "s3:DeleteObjectVersion"],
                "Resource": "arn:aws:s3:::marin-east/marin/grug/hero-*",
            }
        ],
    }
