# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Exp3 long-loop RL curriculum: the exact task mix the .55 restart trained on.

Run ``users/benfeuer/checkpoints/mochi-exp3-rl-ib-2026.11.17.55`` (W&B
``marin-community/marin-long-loop/f7m70xhc``) trained 20 updates on two pinned
parquet releases with exact consumed route quotas: every 64-group training batch
admitted 32 agentic groups and 32 nonagentic groups, drawn uniformly within each
route pool by seeded passes over the merged manifests. The recipe's
``data.sampling`` stayed at its default (``kind: null``), so the mix came from the
rollout buffer's consumed route quotas, not from ``weighted_quotas``.

This module records that mix as the equivalent weighted curriculum over the seven
``data_source`` populations the run's telemetry resolves reward domains by. Each
source's weight is its route's batch share times its row share inside the route
pool, so composing every training window with these weights reproduces the run's
per-window composition exactly at the route level and in expectation within each
route. The seed is the run's trainer seed. Resolving the curriculum against the
catalog reproduces what .55 consumed, source by source; the per-source task
counts come from the adopted releases' pinned provenance, so the curriculum and
the release verification cannot drift apart.
"""

import csv
from pathlib import Path
from types import MappingProxyType

from marin.rl.curriculum import RlCurriculum, SamplingMode

from experiments.post_training.task_curation.curriculum import curriculum_entry
from experiments.post_training.task_curation.datasets.long_loop_rl import (
    AGENTIC_RELEASE_NAME,
    AGENTIC_TRAIN_SOURCE,
    NONAGENTIC_RELEASE_NAME,
    REASONING_GYM_SOURCE,
    agentic_release,
    nonagentic_release,
)
from experiments.post_training.task_curation.source import RlDataSource

CURRICULUM_NAME = "long-loop/exp3-rl"
CURRICULUM_VERSION = "2026.11.17.55"
RUN_ID = "users/benfeuer/checkpoints/mochi-exp3-rl-ib-2026.11.17.55"
WANDB_RUN = "marin-community/marin-long-loop/f7m70xhc"

SEED = 20261020
BATCH_GROUPS = 64
ROUTE_QUOTAS = MappingProxyType({"agentic": 32, "nonagentic": 32})

AGENTIC_RELEASE = agentic_release()
NONAGENTIC_RELEASE = nonagentic_release()

SOURCE_ROUTES = MappingProxyType(
    {
        AGENTIC_TRAIN_SOURCE: "agentic",
        "PRIME-RL/Eurus-2-RL-Data": "nonagentic",
        "nvidia/Nemotron-RL-Ultra-Training-Blends": "nonagentic",
        "nvidia/OpenScience": "nonagentic",
        REASONING_GYM_SOURCE: "nonagentic",
        "open-athena/Snowball-Smartie-RL-2026.10.15.32-Data": "nonagentic",
        "open-thoughts/TaskTrove": "nonagentic",
    }
)

SOURCE_RELEASES = MappingProxyType(
    {
        AGENTIC_TRAIN_SOURCE: AGENTIC_RELEASE_NAME,
        "PRIME-RL/Eurus-2-RL-Data": NONAGENTIC_RELEASE_NAME,
        "nvidia/Nemotron-RL-Ultra-Training-Blends": NONAGENTIC_RELEASE_NAME,
        "nvidia/OpenScience": NONAGENTIC_RELEASE_NAME,
        REASONING_GYM_SOURCE: NONAGENTIC_RELEASE_NAME,
        "open-athena/Snowball-Smartie-RL-2026.10.15.32-Data": NONAGENTIC_RELEASE_NAME,
        "open-thoughts/TaskTrove": NONAGENTIC_RELEASE_NAME,
    }
)


def _source_counts() -> dict[str, int]:
    """The per-source task counts, read from the adopted releases' pinned provenance."""
    counts: dict[str, int] = {}
    for release in (AGENTIC_RELEASE, NONAGENTIC_RELEASE):
        config = release.config
        if config is None:
            raise ValueError(f"adopted release {release.name} has no configuration")
        for member in config.members:
            if member.source_counts:
                counts.update(member.source_counts)
    return counts


SOURCE_ROWS = MappingProxyType(_source_counts())


def release_catalog() -> dict[str, RlDataSource]:
    """The two adopted releases, keyed by catalog name."""
    return {AGENTIC_RELEASE_NAME: AGENTIC_RELEASE, NONAGENTIC_RELEASE_NAME: NONAGENTIC_RELEASE}


def weights() -> dict[str, float]:
    """Per-source expected group share under the run's consumed route quotas.

    Each route's batch share is its consumed quota over the 64-group batch; within
    a route, seeded passes draw every pool row uniformly, so a source's share of
    that route's groups is its row share of the route pool.
    """
    route_rows = {
        route: sum(SOURCE_ROWS[label] for label in SOURCE_ROWS if SOURCE_ROUTES[label] == route)
        for route in ROUTE_QUOTAS
    }
    return {
        label: (
            (ROUTE_QUOTAS[SOURCE_ROUTES[label]] / BATCH_GROUPS) * (SOURCE_ROWS[label] / route_rows[SOURCE_ROUTES[label]])
        )
        for label in SOURCE_ROWS
    }


def exp3_rl_curriculum() -> RlCurriculum:
    """The .55 run's task mix as a versioned weighted curriculum over the seven sources."""
    catalog = release_catalog()
    return RlCurriculum(
        name=CURRICULUM_NAME,
        version=CURRICULUM_VERSION,
        mode=SamplingMode.WEIGHTED,
        entries=tuple(
            curriculum_entry(label, catalog[SOURCE_RELEASES[label]], weight) for label, weight in weights().items()
        ),
        stages=(),
        seed=SEED,
    )


CSV_COLUMNS = (
    "label",
    "route",
    "release",
    "pin_revision",
    "release_url",
    "task_count",
    "curriculum_share",
    "sampled_groups",
    "sampled_share",
    "reward_initial",
    "reward_initial_update",
    "reward_peak",
    "reward_peak_update",
)

# The run report. Sampled groups are the exact per-source group counts the 20
# updates trained on, counted from the durable training-order dumps under the
# run's export path (one uid per rollout row; 8 rollouts per group). Rewards are
# each domain's mean raw (unshaped) reward per update from the W&B history
# (f7m70xhc), the first update that sampled the domain and its maximum.
RUN_SAMPLED_GROUPS = MappingProxyType(
    {
        AGENTIC_TRAIN_SOURCE: 640,
        "open-thoughts/TaskTrove": 170,
        REASONING_GYM_SOURCE: 139,
        "nvidia/OpenScience": 133,
        "nvidia/Nemotron-RL-Ultra-Training-Blends": 76,
        "open-athena/Snowball-Smartie-RL-2026.10.15.32-Data": 63,
        "PRIME-RL/Eurus-2-RL-Data": 59,
    }
)
RUN_UPDATES = 20
RUN_REWARDS = MappingProxyType(
    {
        AGENTIC_TRAIN_SOURCE: ((0.244140625, 1), (0.479296875, 20)),
        "open-thoughts/TaskTrove": ((0.58203125, 1), (0.9285714285714286, 13)),
        REASONING_GYM_SOURCE: ((0.29407934030009325, 1), (0.868154761904762, 7)),
        "nvidia/OpenScience": ((0.8333333333333334, 1), (0.975, 19)),
        "open-athena/Snowball-Smartie-RL-2026.10.15.32-Data": ((0.0, 1), (0.5833333333333334, 12)),
        "nvidia/Nemotron-RL-Ultra-Training-Blends": ((0.40625, 1), (0.8035714285714286, 3)),
        "PRIME-RL/Eurus-2-RL-Data": ((0.4375, 2), (1.0, 20)),
    }
)
assert sum(RUN_SAMPLED_GROUPS.values()) == BATCH_GROUPS * RUN_UPDATES


def write_composition_csv(path: Path) -> Path:
    """Write the per-source composition and run-report table to ``path``."""
    curriculum = exp3_rl_curriculum()
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for entry in curriculum.entries:
            (initial, initial_update), (peak, peak_update) = RUN_REWARDS[entry.label]
            writer.writerow(
                [
                    entry.label,
                    SOURCE_ROUTES[entry.label],
                    entry.source,
                    entry.pin.revision,
                    entry.pin.url,
                    SOURCE_ROWS[entry.label],
                    f"{entry.weight:.6f}",
                    RUN_SAMPLED_GROUPS[entry.label],
                    f"{RUN_SAMPLED_GROUPS[entry.label] / (BATCH_GROUPS * RUN_UPDATES):.6f}",
                    initial,
                    initial_update,
                    peak,
                    peak_update,
                ]
            )
    return path


if __name__ == "__main__":
    print(write_composition_csv(Path(__file__).parent / "exp3_curriculum.csv"))
