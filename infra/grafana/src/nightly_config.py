# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Nightly-lane schedule configuration for the regression matrix.

Each lane names one scheduled GitHub Actions workflow the matrix tracks: which
repo/workflow/branch to poll, when it is expected to run (UTC weekday + time of
day), how long past that a missing run counts as overdue, and — where known — the
duration range a healthy run falls in.
"""

import dataclasses
from collections.abc import Sequence

# UTC weekdays, matching datetime's ISO weekday mapped to 0=Sunday..6=Saturday.
_ALL_DAYS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
_MONDAY: tuple[int, ...] = (1,)


@dataclasses.dataclass(frozen=True)
class NightlyLane:
    """One tracked nightly/canary workflow.

    weekdays uses 0=Sunday..6=Saturday. active_from/active_until (inclusive,
    "YYYY-MM-DD") bound the lane's lifecycle; None means unbounded on that side.
    expected_min_seconds/expected_max_seconds are None until a duration baseline
    has been established for the lane.
    """

    id: str
    label: str
    short_label: str
    group: str
    subgroup: str
    repository: str
    workflow_file: str
    branch: str
    weekdays: tuple[int, ...]
    hour: int
    minute: int
    active_from: str | None
    active_until: str | None
    overdue_grace_minutes: int
    expected_min_seconds: int | None
    expected_max_seconds: int | None


NIGHTLY_LANES: tuple[NightlyLane, ...] = (
    NightlyLane(
        id="tpu-ferry",
        label="TPU ferry",
        short_label="TPU ferry",
        group="marin",
        subgroup="training",
        repository="marin-community/marin",
        workflow_file="marin-canary-ferry.yaml",
        branch="main",
        weekdays=_ALL_DAYS,
        hour=6,
        minute=0,
        active_from=None,
        active_until=None,
        overdue_grace_minutes=300,
        expected_min_seconds=60 * 60,
        expected_max_seconds=195 * 60,
    ),
    NightlyLane(
        id="cw-gpu-ferry",
        label="CoreWeave GPU ferry",
        short_label="CW ferry",
        group="marin",
        subgroup="training",
        repository="marin-community/marin",
        workflow_file="marin-canary-ferry-coreweave.yaml",
        branch="main",
        weekdays=_ALL_DAYS,
        hour=10,
        minute=0,
        active_from=None,
        active_until=None,
        overdue_grace_minutes=300,
        expected_min_seconds=15 * 60,
        expected_max_seconds=40 * 60,
    ),
    NightlyLane(
        id="grug-multislice",
        label="Grug multislice",
        short_label="Grug",
        group="marin",
        subgroup="training",
        repository="marin-community/marin",
        workflow_file="marin-canary-grug-multislice.yaml",
        branch="main",
        weekdays=_ALL_DAYS,
        hour=10,
        minute=30,
        active_from=None,
        active_until=None,
        overdue_grace_minutes=240,
        expected_min_seconds=None,
        expected_max_seconds=None,
    ),
    NightlyLane(
        id="datakit-t1",
        label="Datakit tier 1",
        short_label="Data T1",
        group="marin",
        subgroup="data",
        repository="marin-community/marin",
        workflow_file="marin-canary-datakit-tier1.yaml",
        branch="main",
        weekdays=_ALL_DAYS,
        hour=6,
        minute=30,
        active_from=None,
        active_until=None,
        overdue_grace_minutes=360,
        expected_min_seconds=65 * 60,
        expected_max_seconds=85 * 60,
    ),
    NightlyLane(
        id="datakit-t2",
        label="Datakit tier 2",
        short_label="Data T2",
        group="marin",
        subgroup="data",
        repository="marin-community/marin",
        workflow_file="marin-canary-datakit-tier2.yaml",
        branch="main",
        weekdays=_ALL_DAYS,
        hour=7,
        minute=0,
        active_from=None,
        active_until=None,
        overdue_grace_minutes=360,
        expected_min_seconds=65 * 60,
        expected_max_seconds=85 * 60,
    ),
    NightlyLane(
        id="datakit-t3",
        label="Datakit tier 3, Mondays",
        short_label="Data T3 · Mon",
        group="marin",
        subgroup="data",
        repository="marin-community/marin",
        workflow_file="marin-canary-datakit-tier3.yaml",
        branch="main",
        weekdays=_MONDAY,
        hour=1,
        minute=0,
        active_from=None,
        active_until=None,
        overdue_grace_minutes=480,
        expected_min_seconds=70 * 60,
        expected_max_seconds=180 * 60,
    ),
    NightlyLane(
        id="cluster-smoke",
        label="Cluster smoke",
        short_label="Cluster",
        group="marin",
        subgroup="cluster",
        repository="marin-community/marin",
        workflow_file="marin-cluster-smoke.yaml",
        branch="main",
        weekdays=_ALL_DAYS,
        hour=7,
        minute=30,
        active_from="2026-07-17",
        active_until=None,
        overdue_grace_minutes=300,
        expected_min_seconds=None,
        expected_max_seconds=None,
    ),
    NightlyLane(
        id="evalchemy",
        label="Evalchemy",
        short_label="Evalchemy",
        group="forks",
        subgroup="evaluation",
        repository="marin-community/evalchemy",
        workflow_file="e2e-nightly.yaml",
        branch="main",
        weekdays=_ALL_DAYS,
        hour=7,
        minute=0,
        active_from="2026-07-14",
        active_until=None,
        overdue_grace_minutes=240,
        expected_min_seconds=14 * 60,
        expected_max_seconds=20 * 60,
    ),
    NightlyLane(
        id="harbor",
        label="Harbor",
        short_label="Harbor",
        group="forks",
        subgroup="evaluation",
        repository="marin-community/harbor",
        workflow_file="marin-nightly.yaml",
        branch="main",
        weekdays=_ALL_DAYS,
        hour=8,
        minute=0,
        active_from="2026-07-15",
        active_until=None,
        overdue_grace_minutes=240,
        expected_min_seconds=6 * 60,
        expected_max_seconds=12 * 60,
    ),
    NightlyLane(
        id="marinskyrl",
        label="MarinSkyRL",
        short_label="SkyRL",
        group="forks",
        subgroup="rl",
        repository="marin-community/MarinSkyRL",
        workflow_file="marin-nightly.yaml",
        branch="main",
        weekdays=_ALL_DAYS,
        hour=9,
        minute=0,
        active_from="2026-07-15",
        active_until=None,
        overdue_grace_minutes=240,
        expected_min_seconds=None,
        expected_max_seconds=None,
    ),
    NightlyLane(
        id="vllm-gpu",
        label="vLLM GPU",
        short_label="vLLM GPU",
        group="forks",
        subgroup="inference",
        repository="marin-community/vllm",
        workflow_file="marin-nightly.yaml",
        branch="main",
        weekdays=_ALL_DAYS,
        hour=10,
        minute=0,
        active_from="2026-07-15",
        active_until=None,
        overdue_grace_minutes=240,
        expected_min_seconds=6 * 60,
        expected_max_seconds=15 * 60,
    ),
    NightlyLane(
        id="tpu-inference",
        label="TPU inference",
        short_label="TPU infer",
        group="forks",
        subgroup="inference",
        repository="marin-community/tpu-inference",
        workflow_file="marin-e2e-nightly.yaml",
        branch="main",
        weekdays=_ALL_DAYS,
        hour=11,
        minute=0,
        active_from="2026-07-15",
        active_until=None,
        overdue_grace_minutes=240,
        expected_min_seconds=5 * 60,
        expected_max_seconds=10 * 60,
    ),
)


def validate_nightly_lanes(lanes: Sequence[NightlyLane]) -> None:
    """Raise if lanes contain a duplicate id/workflow or an invalid schedule.

    Checks: duplicate lane id, duplicate (repository, workflow_file) pair, empty
    weekdays, out-of-range hour/minute, negative grace, an active_until before
    active_from, and an inverted expected-duration range.
    """
    ids: set[str] = set()
    workflows: set[tuple[str, str]] = set()
    for lane in lanes:
        if lane.id in ids:
            raise ValueError(f"duplicate nightly lane id: {lane.id}")
        ids.add(lane.id)

        workflow_key = (lane.repository, lane.workflow_file)
        if workflow_key in workflows:
            raise ValueError(f"duplicate nightly workflow: {lane.repository}/{lane.workflow_file}")
        workflows.add(workflow_key)

        if not lane.weekdays:
            raise ValueError(f"{lane.id}: schedule must include at least one weekday")
        if not 0 <= lane.hour <= 23:
            raise ValueError(f"{lane.id}: invalid UTC hour")
        if not 0 <= lane.minute <= 59:
            raise ValueError(f"{lane.id}: invalid UTC minute")
        if lane.overdue_grace_minutes < 0:
            raise ValueError(f"{lane.id}: grace must be non-negative")
        if lane.active_from and lane.active_until and lane.active_from > lane.active_until:
            raise ValueError(f"{lane.id}: active_until precedes active_from")
        if (
            lane.expected_min_seconds is not None
            and lane.expected_max_seconds is not None
            and (lane.expected_min_seconds < 0 or lane.expected_min_seconds > lane.expected_max_seconds)
        ):
            raise ValueError(f"{lane.id}: invalid expected duration range")


validate_nightly_lanes(NIGHTLY_LANES)
