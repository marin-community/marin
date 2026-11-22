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
64-group window into the exact 32/32 split the run ran. The curriculum's
entries pin the two route pools the run trained on — the releases' train
parquets, one per route — by artifact identity. Per-source composition is
provenance, not sampler policy — the run never enforced per-source weights —
so the adopted releases pin the per-source row counts (``verify_release``
checks them).

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

Seed: the run's trainer seed, 20261020 — it seeds both the sampler projection
and the holdout draw.
"""

from collections.abc import Mapping, Sequence
from types import MappingProxyType

from marin.execution.lazy import ArtifactStep
from marin.rl.curriculum import PoolRow, RlCurriculum
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join
from taskcompendium.identity import canonical_sha256

from experiments.post_training.task_curation.curriculum import weighted_curriculum
from experiments.post_training.task_curation.datasets.long_loop_rl import (
    AGENTIC_RELEASE_URL,
    AGENTIC_ROUTE,
    AGENTIC_TRAIN_FILE,
    AGENTIC_TRAIN_SHA256,
    NONAGENTIC_RELEASE_URL,
    NONAGENTIC_ROUTE,
    NONAGENTIC_TRAIN_FILE,
    NONAGENTIC_TRAIN_SHA256,
    RELEASE_SOURCE_ROUTES,
    stream_task_specs,
)

CURRICULUM_NAME = "long-loop/exp3-rl"
CURRICULUM_VERSION = "2026.11.17.55"
RUN_ID = "users/benfeuer/checkpoints/mochi-exp3-rl-ib-2026.11.17.55"
WANDB_RUN = "marin-community/marin-long-loop/f7m70xhc"

SEED = 20261020
BATCH_GROUPS = 64
ROUTE_QUOTAS = MappingProxyType({AGENTIC_ROUTE: 32, NONAGENTIC_ROUTE: 32})

WEIGHTS = MappingProxyType({AGENTIC_ROUTE: 0.5, NONAGENTIC_ROUTE: 0.5})
VALIDATION_SIZE = 200


def exp3_pool_steps(version: str = CURRICULUM_VERSION) -> dict[str, ArtifactStep]:
    """The route pools the .55 run trained on: the two releases' train parquets.

    Each step's name is the trainer's sampling key (the ``teacher_route`` value
    a route-stamped pool carries), and its identity pins the release file the
    pool is drawn from."""
    return {
        AGENTIC_ROUTE: ArtifactStep.adopt(
            AGENTIC_ROUTE,
            version,
            source=prefix_join(AGENTIC_RELEASE_URL, AGENTIC_TRAIN_FILE),
            config={"sha256": AGENTIC_TRAIN_SHA256},
        ),
        NONAGENTIC_ROUTE: ArtifactStep.adopt(
            NONAGENTIC_ROUTE,
            version,
            source=prefix_join(NONAGENTIC_RELEASE_URL, NONAGENTIC_TRAIN_FILE),
            config={"sha256": NONAGENTIC_TRAIN_SHA256},
        ),
    }


def release_pool_rows(url: str, member: str) -> dict[str, tuple[PoolRow, ...]]:
    """Stream one pinned release parquet and return its row identities by route.

    Every task spec's ``source.dataset`` is the trainer's ``data_source`` and
    must resolve to an admitted route. Row identity is the task spec's ``id``
    plus the canonical SHA-256 of its task document — the same identity the
    holdout records and extraction verifies."""
    filesystem, base = filesystem_for(url)
    path = prefix_join(base, member)
    rows_by_route: dict[str, list[PoolRow]] = {}
    for task in stream_task_specs(filesystem, path):
        dataset = task.get("source", {}).get("dataset")
        route = RELEASE_SOURCE_ROUTES.get(dataset)
        if route is None:
            raise ValueError(f"{path}: task source {dataset!r} names no admitted route")
        rows_by_route.setdefault(route, []).append(PoolRow(id=task["id"], sha256=canonical_sha256(task)))
    return {route: tuple(rows) for route, rows in rows_by_route.items()}


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
    pools = exp3_pool_steps()
    return weighted_curriculum(
        CURRICULUM_NAME,
        CURRICULUM_VERSION,
        train={pools[AGENTIC_ROUTE]: WEIGHTS[AGENTIC_ROUTE], pools[NONAGENTIC_ROUTE]: WEIGHTS[NONAGENTIC_ROUTE]},
        seed=SEED,
        validation_size=VALIDATION_SIZE,
        pool_rows=pool_rows,
    )
