# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Exp3 long-loop RL curriculum: the task mix the .55 restart enforced.

Run ``users/benfeuer/checkpoints/mochi-exp3-rl-ib-2026.11.17.55`` (W&B
``marin-community/marin-long-loop/f7m70xhc``) trained 20 updates on two pinned
parquet releases. The recipe's ``data.sampling`` stayed at its default
(``kind: null``): the run routed groups through the rollout buffer's consumed
route quotas, admitting exactly 32 agentic and 32 nonagentic groups per
64-group batch, and drew uniformly within each route pool by seeded passes over
the merged manifests.

This module records that enforced mix as a weighted curriculum keyed by the
routes the buffer admitted: 0.5 agentic and 0.5 nonagentic, which compose every
64-group window into the exact 32/32 split the run ran. Per-source composition is
provenance, not sampler policy — the run never enforced per-source weights — so
the adopted releases pin the per-source row counts (``verify_release`` checks
them) and the PR description carries the per-source shares, sampled counts, and
rewards.

Sampling contract: the trainer's domain-weighted sampler resolves pool rows by
their ``teacher_route`` column and rejects unknown routes. The recorded .55
parquets carry ``data_source`` provenance and no ``teacher_route`` column, so a
launch that samples this curriculum needs a route-stamped pool; against the
recorded pool the curriculum is the mix's record, not an executable sampler
config.

Seed: the run's trainer seed, 20261020.
"""

from types import MappingProxyType

from marin.rl.curriculum import RlCurriculum

from experiments.post_training.task_curation.curriculum import weighted_curriculum
from experiments.post_training.task_curation.datasets.long_loop_rl import AGENTIC_RELEASE, NONAGENTIC_RELEASE

CURRICULUM_NAME = "long-loop/exp3-rl"
CURRICULUM_VERSION = "2026.11.17.55"
RUN_ID = "users/benfeuer/checkpoints/mochi-exp3-rl-ib-2026.11.17.55"
WANDB_RUN = "marin-community/marin-long-loop/f7m70xhc"

SEED = 20261020
BATCH_GROUPS = 64
ROUTE_QUOTAS = MappingProxyType({"agentic": 32, "nonagentic": 32})

WEIGHTS = MappingProxyType({"agentic": 0.5, "nonagentic": 0.5})


def exp3_rl_curriculum() -> RlCurriculum:
    """The .55 run's enforced mix: exact 32/32 route quotas per 64-group window."""
    return weighted_curriculum(
        CURRICULUM_NAME,
        CURRICULUM_VERSION,
        weights=dict(WEIGHTS),
        seed=SEED,
        catalog={"agentic": AGENTIC_RELEASE, "nonagentic": NONAGENTIC_RELEASE},
    )
