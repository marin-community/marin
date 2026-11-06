# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.grug.moe_hero_ep.checkpoints import _parent_run_and_step


def test_merged_handoff_links_to_the_fork_parent():
    merged = "s3://bucket/marin/grug/hero-merge/checkpoints/step-390251"
    config = {
        "hero_handoff_checkpoint": merged,
        "hero_wandb_fork_from": "hero-parent?_step=390250",
        "trainer": {"trainer": {"load_checkpoint_path": ["s3://bucket/own", merged]}},
    }

    assert _parent_run_and_step(config, "hero-child") == ("hero-parent", 390251)
