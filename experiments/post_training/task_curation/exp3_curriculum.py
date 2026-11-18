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
catalog reproduces what .55 consumed, source by source.
"""

import csv
import math
from collections.abc import Mapping
from pathlib import Path

from marin.rl.curriculum import CurriculumEntry, RlCurriculum, SamplingMode, SourcePin

from experiments.post_training.task_curation.curriculum import recipe_identity
from experiments.post_training.task_curation.datasets.long_loop_rl import (
    AGENTIC_TRAIN_ROWS,
    EURUS_ROWS,
    NEMOTRON_BLENDS_ROWS,
    NONAGENTIC_TRAIN_ROWS,
    OPEN_SCIENCE_ROWS,
    REASONING_GYM_SOURCE,
    REASONING_GYM_ROWS,
    SNOWBALL_SMARTIE_ROWS,
    TASKTROVE_ROWS,
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
ROUTE_QUOTAS = {"agentic": 32, "nonagentic": 32}

SOURCE_ROUTES: dict[str, str] = {
    "long-loop-smartie-swesmith": "agentic",
    "PRIME-RL/Eurus-2-RL-Data": "nonagentic",
    "nvidia/Nemotron-RL-Ultra-Training-Blends": "nonagentic",
    "nvidia/OpenScience": "nonagentic",
    REASONING_GYM_SOURCE: "nonagentic",
    "open-athena/Snowball-Smartie-RL-2026.10.15.32-Data": "nonagentic",
    "open-thoughts/TaskTrove": "nonagentic",
}

SOURCE_RELEASES: dict[str, str] = {
    "long-loop-smartie-swesmith": "long-loop-rl-agentic-train",
    "PRIME-RL/Eurus-2-RL-Data": "long-loop-rl-nonagentic-train",
    "nvidia/Nemotron-RL-Ultra-Training-Blends": "long-loop-rl-nonagentic-train",
    "nvidia/OpenScience": "long-loop-rl-nonagentic-train",
    REASONING_GYM_SOURCE: "long-loop-rl-nonagentic-train",
    "open-athena/Snowball-Smartie-RL-2026.10.15.32-Data": "long-loop-rl-nonagentic-train",
    "open-thoughts/TaskTrove": "long-loop-rl-nonagentic-train",
}

SOURCE_ROWS: dict[str, int] = {
    "long-loop-smartie-swesmith": AGENTIC_TRAIN_ROWS,
    "PRIME-RL/Eurus-2-RL-Data": EURUS_ROWS,
    "nvidia/Nemotron-RL-Ultra-Training-Blends": NEMOTRON_BLENDS_ROWS,
    "nvidia/OpenScience": OPEN_SCIENCE_ROWS,
    REASONING_GYM_SOURCE: REASONING_GYM_ROWS,
    "open-athena/Snowball-Smartie-RL-2026.10.15.32-Data": SNOWBALL_SMARTIE_ROWS,
    "open-thoughts/TaskTrove": TASKTROVE_ROWS,
}
assert sum(SOURCE_ROWS.values()) == AGENTIC_TRAIN_ROWS + NONAGENTIC_TRAIN_ROWS


def release_catalog() -> dict[str, RlDataSource]:
    """The two adopted releases, keyed by catalog name."""
    sources = (agentic_release(), nonagentic_release())
    return {source.name: source for source in sources}


def weights(rows: Mapping[str, int] = SOURCE_ROWS) -> dict[str, float]:
    """Per-source expected group share under the run's consumed route quotas.

    Each route's batch share is its consumed quota over the 64-group batch; within
    a route, seeded passes draw every pool row uniformly, so a source's share of
    that route's groups is its row share of the route pool.
    """
    route_rows = {
        route: sum(count for label, count in rows.items() if SOURCE_ROUTES[label] == route) for route in ROUTE_QUOTAS
    }
    return {
        label: (ROUTE_QUOTAS[SOURCE_ROUTES[label]] / BATCH_GROUPS) * (rows[label] / route_rows[SOURCE_ROUTES[label]])
        for label in rows
    }


def exp3_rl_curriculum() -> RlCurriculum:
    """The .55 run's task mix as a versioned weighted curriculum over the seven sources."""
    catalog = release_catalog()
    entries = []
    for label, weight in weights().items():
        release = catalog[SOURCE_RELEASES[label]]
        dataset = release.dataset
        if dataset is None or release.pipeline is None:
            raise ValueError(f"release {release.name} is not runnable")
        entries.append(
            CurriculumEntry(
                label=label,
                source=release.name,
                version=release.version,
                pin=SourcePin(dataset.name, dataset.revision, dataset.url),
                recipe=recipe_identity(release.pipeline),
                weight=weight,
            )
        )
    return RlCurriculum(
        name=CURRICULUM_NAME,
        version=CURRICULUM_VERSION,
        mode=SamplingMode.WEIGHTED,
        entries=tuple(entries),
        stages=(),
        seed=SEED,
    )


def percentile(values: list[float], percent: float) -> float:
    """The linear-interpolation percentile of a small sample."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile of an empty sample")
    position = percent / 100 * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution_summary(shares: Mapping[str, float]) -> dict:
    """The across-source distribution summary the report quotes."""
    values = list(shares.values())
    descending = sorted(values, reverse=True)
    cumulative = []
    total = 0.0
    for share in descending:
        total += share
        cumulative.append(total)
    sources_for = {
        f"{int(threshold * 100)}%": next(index + 1 for index, running in enumerate(cumulative) if running >= threshold)
        for threshold in (0.5, 0.9)
    }
    return {
        "sources": len(values),
        "p10": percentile(values, 10),
        "p25": percentile(values, 25),
        "p50": percentile(values, 50),
        "p75": percentile(values, 75),
        "p90": percentile(values, 90),
        "max": max(values),
        **sources_for,
    }


CSV_COLUMNS = (
    "label",
    "route",
    "release",
    "pin_revision",
    "release_url",
    "task_count",
    "curriculum_share",
)


def write_composition_csv(path: Path) -> Path:
    """Write the per-source composition table next to the curriculum definition."""
    catalog = release_catalog()
    curriculum = exp3_rl_curriculum()
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for entry in curriculum.entries:
            release = catalog[entry.source]
            writer.writerow(
                [
                    entry.label,
                    SOURCE_ROUTES[entry.label],
                    entry.source,
                    entry.pin.revision,
                    entry.pin.url,
                    SOURCE_ROWS[entry.label],
                    f"{entry.weight:.6f}",
                ]
            )
    return path


if __name__ == "__main__":
    print(write_composition_csv(Path(__file__).parent / "exp3_curriculum.csv"))
