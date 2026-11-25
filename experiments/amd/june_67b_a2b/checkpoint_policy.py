# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint retention for the Snowball run on the Vultr 2x8 MI355X cluster.

Two kinds of checkpoint share ``/work`` (8 TB pooled quota, with about 3.8 TB of it holding the data cache):

- temporary checkpoints, saved every ``save_interval`` (set from the measured memory-corruption rate), under
  ``checkpoints-temp``. Levanter keeps the newest ``KEEP_LAST_TEMPORARY`` and deletes them all once a milestone
  commits, since the milestone is newer.
- milestones, every ``MILESTONE_EVERY`` steps, under ``checkpoints``. Levanter never deletes them;
  ``vultr/offload_milestones.py`` copies each to CoreWeave S3 and deletes the local copy once a newer checkpoint is
  on ``/work``.

At most ``KEEP_LAST_TEMPORARY`` + 2 checkpoints are on ``/work`` at once: the kept temporaries, the one being saved
and the newest milestone. A 67B-A2B checkpoint with MuonH state is 502 GiB (0.54 TB), so that is 2.7 TB, leaving
about 1.1 TB of the quota spare for a milestone whose upload has not finished.
"""

from datetime import timedelta

from levanter.checkpoint import CheckpointerConfig

KEEP_LAST_TEMPORARY = 3
MILESTONE_EVERY = 1000


def snowball_checkpointer(
    checkpoint_dir: str, *, save_interval: timedelta, keep_last_temporary: int, milestone_every: int
) -> CheckpointerConfig:
    """Temporary checkpoints every ``save_interval`` under ``<checkpoint_dir>/checkpoints-temp``, of which the newest
    ``keep_last_temporary`` are kept, and a permanent milestone every ``milestone_every`` steps under
    ``<checkpoint_dir>/checkpoints``."""
    return CheckpointerConfig(
        base_path=f"{checkpoint_dir}/checkpoints",
        temporary_base_path=f"{checkpoint_dir}/checkpoints-temp",
        append_run_id_to_base_path=False,
        save_interval=save_interval,
        keep=[{"every": milestone_every}],
        keep_last_temporary_checkpoints=keep_last_temporary,
    )
