# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import datetime

import jax.numpy as jnp

from experiments.amd.june_67b_a2b.checkpoint_policy import snowball_checkpointer


def _on_disk(path):
    return sorted(int(p.name.removeprefix("step-")) for p in path.glob("step-*") if (p / "metadata.json").exists())


def test_keeps_the_last_temporaries_and_every_milestone(tmp_path):
    """With a temporary save every step: after each step, the newest temporaries since the last milestone and every
    milestone are on disk, and nothing else."""
    now = [datetime.datetime(2026, 11, 25)]
    config = snowball_checkpointer(
        str(tmp_path), save_interval=datetime.timedelta(minutes=30), keep_last_temporary=2, milestone_every=5
    )
    checkpointer = config.create("run")
    checkpointer._dt_now_injection = lambda: now[0]
    tree = {"w": jnp.arange(4.0)}

    seen = {}
    for step in range(13):
        now[0] += datetime.timedelta(minutes=31)
        checkpointer.on_step(tree=tree, step=step)
        checkpointer.wait_until_finished()
        seen[step] = (_on_disk(tmp_path / "checkpoints-temp"), _on_disk(tmp_path / "checkpoints"))

    assert seen[4] == ([3, 4], [])
    assert seen[5] == ([], [5])
    assert seen[6] == ([6], [5])
    assert seen[9] == ([8, 9], [5])
    assert seen[10] == ([], [5, 10])
    assert seen[12] == ([11, 12], [5, 10])
