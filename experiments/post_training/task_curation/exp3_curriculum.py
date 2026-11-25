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

Validation holdout: the curriculum owns a random 200-prompt holdout drawn from
these route pools at the 0.5/0.5 mixture, recorded in its metadata and removed
from the pools the run trains on. The .55 run predates curriculum-owned
holdouts — its launch consumed the release's own 120-row validation.parquet,
which the launch path no longer adopts — so this holdout is the record's split
under the current policy, not a description of what the historical run
evaluated.

Sampling contract: the trainer's domain-weighted sampler resolves pool rows by
their ``teacher_route`` column and rejects unknown routes. The recorded .55
parquets carry ``data_source`` provenance and no ``teacher_route`` column, so a
launch that samples this curriculum needs a route-stamped pool; against the
recorded pool the curriculum is the mix's record, not an executable sampler
config.

Seed: the run's trainer seed, 20261020.
"""

from collections.abc import Mapping, Sequence
from types import MappingProxyType

from marin.rl.curriculum import CurriculumValidation, PoolRow, RlCurriculum, draw_validation_split

from experiments.post_training.task_curation.curriculum import curriculum_with_validation
from experiments.post_training.task_curation.datasets.long_loop_rl import (
    AGENTIC_RELEASE,
    AGENTIC_RELEASE_URL,
    AGENTIC_ROUTE,
    AGENTIC_TRAIN_FILE,
    NONAGENTIC_RELEASE,
    NONAGENTIC_RELEASE_URL,
    NONAGENTIC_ROUTE,
    NONAGENTIC_TRAIN_FILE,
    release_pool_rows,
)

CURRICULUM_NAME = "long-loop/exp3-rl"
CURRICULUM_VERSION = "2026.11.17.55"
RUN_ID = "users/benfeuer/checkpoints/mochi-exp3-rl-ib-2026.11.17.55"
WANDB_RUN = "marin-community/marin-long-loop/f7m70xhc"

SEED = 20261020
BATCH_GROUPS = 64
ROUTE_QUOTAS = MappingProxyType({"agentic": 32, "nonagentic": 32})

WEIGHTS = MappingProxyType({AGENTIC_ROUTE: 0.5, NONAGENTIC_ROUTE: 0.5})
VALIDATION_SIZE = 200


def exp3_pool_rows() -> dict[str, tuple[PoolRow, ...]]:
    """The route pools' row identities, streamed from the pinned release files."""
    combined: dict[str, tuple[PoolRow, ...]] = {}
    for url, member in (
        (AGENTIC_RELEASE_URL, AGENTIC_TRAIN_FILE),
        (NONAGENTIC_RELEASE_URL, NONAGENTIC_TRAIN_FILE),
    ):
        for route, rows in release_pool_rows(url, member).items():
            combined[route] = combined.get(route, ()) + rows
    return combined


def exp3_rl_curriculum(pool_rows: Mapping[str, Sequence[PoolRow]]) -> RlCurriculum:
    """The .55 run's enforced mix, with its holdout drawn from the given rows.

    ``pool_rows`` carries each route pool's row identities (``exp3_pool_rows``
    streams them from the pinned releases). The holdout is drawn here, so every
    construction of this curriculum from the same pools records the same split
    and the same identity."""
    validation = draw_validation_split(dict(WEIGHTS), pool_rows, size=VALIDATION_SIZE, seed=SEED).validation
    return exp3_curriculum_from_validation(validation)


def exp3_curriculum_from_validation(validation: CurriculumValidation) -> RlCurriculum:
    """Bind the Exp3 mix to its published, curriculum-owned holdout."""
    return curriculum_with_validation(
        CURRICULUM_NAME,
        CURRICULUM_VERSION,
        weights=dict(WEIGHTS),
        seed=SEED,
        validation=validation,
        catalog={"agentic": AGENTIC_RELEASE, "nonagentic": NONAGENTIC_RELEASE},
    )
