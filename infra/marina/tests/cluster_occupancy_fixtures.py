# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A mirror's worth of Iris controller rows for the Cluster Occupancy app.

The dataclasses carry only the columns the app queries; values are the columns
the real ``iris.controller_db__*`` namespaces hold. Every user is synthetic.
The embedded Finelog server these rows land in stands in for the ``marin`` hub,
and ``live_rows()`` stands in for the live controller reads.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import ClassVar

from infra.marina.apps.cluster_occupancy.backend.app import JOB_CONFIG, JOBS, TASK_ATTEMPTS

CLUSTER_A = "cw-us-east-08a"
CLUSTER_B = "cw-us-east-02a"
CLUSTER_UNMIRRORED = "cw-rno2a"

# The window under test is [T0, T0 + 1h).
T0 = datetime(2026, 11, 23, 6, 0, tzinfo=UTC)
WINDOW_START_MS = int(T0.timestamp() * 1000)

RACK_122 = "DH1-122-US-EAST-08A"
RACK_124 = "DH1-124-US-EAST-08A"

ALICE = "alice"
BOB = "bob"
CAROL = "carol"


def _at(minutes: float) -> datetime:
    return T0 + timedelta(minutes=minutes)


@dataclass
class JobRow:
    key_column: ClassVar[str] = "job_id"
    cluster: str
    job_id: str
    root_job_id: str | None
    user_id: str | None
    name: str | None
    num_tasks: int | None
    started_at: datetime | None
    finished_at: datetime | None
    ts: datetime


@dataclass
class ConfigRow:
    key_column: ClassVar[str] = "job_id"
    cluster: str
    job_id: str
    priority_band: str | None
    accelerator_type: str | None
    accelerators_per_task: int | None
    ts: datetime


@dataclass
class AttemptRow:
    key_column: ClassVar[str] = "task_id"
    cluster: str
    job_id: str
    task_id: str
    node_name: str | None
    started_at: datetime | None
    finished_at: datetime | None
    topology_json: str
    ts: datetime


def rack(domain: str) -> str:
    return f'{{"ds.coreweave.com/nvlink.domain": "{domain}"}}'


def seed_occupancy_fixtures(client) -> None:
    """Write the whole fixture scenario: three clusters, one unmirrored."""
    jobs = client.get_table(JOBS, JobRow)
    configs = client.get_table(JOB_CONFIG, ConfigRow)
    attempts = client.get_table(TASK_ATTEMPTS, AttemptRow)

    jobs.write(
        [
            # The root job behind alice's training worker, running through the window.
            JobRow(CLUSTER_A, "root-1", "root-1", ALICE, "/alice/train-coordinator", 1, _at(-180), None, _at(-180)),
            # alice's worker: a stale mirror snapshot first, then the current one.
            JobRow(
                CLUSTER_A, "job-1", "root-1", ALICE, "/alice/train-coordinator/worker-a", 99, _at(-180), None, _at(-10)
            ),
            JobRow(
                CLUSTER_A, "job-1", "root-1", ALICE, "/alice/train-coordinator/worker-a", 4, _at(-180), None, _at(-5)
            ),
            # bob's batch eval finished 20 minutes into the window; its root finished
            # before the window, so the root-name lookup must ignore window membership.
            JobRow(CLUSTER_A, "root-4", "root-4", BOB, "/bob/eval-coordinator", 1, _at(-300), _at(-120), _at(-120)),
            JobRow(CLUSTER_A, "job-2", "root-4", BOB, "/bob/eval-coordinator/worker", 2, _at(-30), _at(20), _at(-25)),
            # carol's old interactive job, entirely before the window.
            JobRow(CLUSTER_A, "job-3", "job-3", CAROL, "/carol/scratch", 1, _at(-120), _at(-90), _at(-120)),
            # bob's job finished exactly when the window starts: it held nothing inside.
            JobRow(CLUSTER_A, "job-6", "job-6", BOB, "/bob/boundary-end", 1, _at(-60), T0, _at(-30)),
            # carol's job starts exactly when the window ends: outside the window.
            JobRow(
                CLUSTER_A,
                "job-7",
                "job-7",
                CAROL,
                "/carol/boundary-start",
                1,
                T0 + timedelta(hours=1),
                None,
                T0 + timedelta(hours=1),
            ),
            # carol's CPU job runs on a GPU node but holds no GPUs.
            JobRow(CLUSTER_A, "job-8", "job-8", CAROL, "/carol/cpu-job", 1, _at(-20), None, _at(-20)),
            # 02a: a running coordinator, bob's interactive job, carol's short production job.
            JobRow(CLUSTER_B, "root-2a", "root-2a", BOB, "/bob/eval-east-coordinator", 1, _at(-360), None, _at(-40)),
            JobRow(CLUSTER_B, "job-4", "root-2a", BOB, "/bob/eval-east-coordinator/worker", 1, _at(-10), None, _at(-10)),
            JobRow(CLUSTER_B, "job-5", "job-5", CAROL, "/carol/prod-east", 1, _at(-40), _at(10), _at(-35)),
        ]
    )
    configs.write(
        [
            ConfigRow(CLUSTER_A, "root-1", "PRIORITY_BAND_PRODUCTION", None, 0, _at(-180)),
            ConfigRow(CLUSTER_A, "job-1", "PRIORITY_BAND_PRODUCTION", "GB200", 1, _at(-10)),
            ConfigRow(CLUSTER_A, "job-1", "PRIORITY_BAND_PRODUCTION", "GB200", 4, _at(-5)),
            ConfigRow(CLUSTER_A, "root-4", "PRIORITY_BAND_BATCH", None, 0, _at(-300)),
            ConfigRow(CLUSTER_A, "job-2", "PRIORITY_BAND_BATCH", "GB200", 4, _at(-25)),
            ConfigRow(CLUSTER_A, "job-3", "PRIORITY_BAND_INTERACTIVE", "H100", 1, _at(-120)),
            ConfigRow(CLUSTER_A, "job-6", "PRIORITY_BAND_BATCH", "GB200", 4, _at(-30)),
            ConfigRow(CLUSTER_A, "job-7", "PRIORITY_BAND_INTERACTIVE", "H100", 8, T0 + timedelta(hours=1)),
            ConfigRow(CLUSTER_A, "job-8", "PRIORITY_BAND_INTERACTIVE", None, 0, _at(-20)),
            ConfigRow(CLUSTER_B, "root-2a", "PRIORITY_BAND_BATCH", None, 0, _at(-360)),
            ConfigRow(CLUSTER_B, "job-4", "PRIORITY_BAND_INTERACTIVE", "H100", 8, _at(-10)),
            ConfigRow(CLUSTER_B, "job-5", "PRIORITY_BAND_PRODUCTION", "H100", 8, _at(-35)),
        ]
    )
    attempts.write(
        [
            AttemptRow(CLUSTER_A, "job-1", "t1", "n1", _at(-180), None, rack(RACK_122), _at(-180)),
            AttemptRow(CLUSTER_A, "job-1", "t2", "n2", _at(-180), None, rack(RACK_122), _at(-180)),
            AttemptRow(CLUSTER_A, "job-2", "t1", "n1", _at(-30), _at(20), rack(RACK_122), _at(-30)),
            AttemptRow(CLUSTER_A, "job-2", "t2", "n3", _at(-30), _at(20), rack(RACK_122), _at(-30)),
            AttemptRow(CLUSTER_A, "job-3", "t1", "n9", _at(-120), _at(-90), "{}", _at(-120)),
            AttemptRow(CLUSTER_A, "job-6", "t1", "n5", _at(-60), T0, rack(RACK_124), _at(-60)),
            AttemptRow(CLUSTER_A, "job-7", "t1", "n6", T0 + timedelta(hours=1), None, "{}", T0 + timedelta(hours=1)),
            AttemptRow(CLUSTER_A, "job-8", "t1", "n9", _at(-20), None, "{}", _at(-20)),
            AttemptRow(CLUSTER_B, "job-4", "t1", "g1", _at(-10), None, "{}", _at(-10)),
            AttemptRow(CLUSTER_B, "job-5", "t1", "g1", _at(-40), _at(10), "{}", _at(-40)),
        ]
    )
    for table in (jobs, configs, attempts):
        assert table.flush(timeout=60) is not None


def _gpu_row(
    job: str,
    root: str,
    user: str,
    node: str,
    count: int,
    *,
    band: int = 1,
    variant: str = "H100",
    started_ms: int = 1000,
    rack: str = "",
) -> dict:
    """One live running-task row, as the controller's raw query returns it."""
    return {
        "root_job_id": root,
        "user_id": user,
        "job_id": job,
        "priority_band": band,
        "res_device_json": "{}" if count == 0 else f'{{"gpu": {{"variant": "{variant}", "count": {count}}}}}',
        "node_name": node,
        "started_at_ms": started_ms,
        "topology_json": f'{{"ds.coreweave.com/nvlink.domain": "{rack}"}}' if rack else "{}",
    }


def live_rows() -> dict[str, list[dict]]:
    """The current-moment scenario: who holds which node on which cluster."""
    started_a = int(_at(-180).timestamp() * 1000)
    started_b = int(_at(-30).timestamp() * 1000)
    started_c = int(_at(-40).timestamp() * 1000)
    started_d = int(_at(-90).timestamp() * 1000)
    return {
        CLUSTER_A: [
            _gpu_row(
                "job-1", "/alice/train-coordinator", ALICE, "n1", 4, variant="GB200", started_ms=started_a, rack=RACK_122
            ),
            _gpu_row(
                "job-1", "/alice/train-coordinator", ALICE, "n1", 4, variant="GB200", started_ms=started_a, rack=RACK_122
            ),
            _gpu_row(
                "job-2",
                "/bob/eval-coordinator",
                BOB,
                "n2",
                4,
                band=3,
                variant="GB200",
                started_ms=started_b,
                rack=RACK_122,
            ),
        ],
        CLUSTER_B: [
            _gpu_row("job-5", "/carol/prod-east", CAROL, "g1", 8, started_ms=started_c),
        ],
        CLUSTER_UNMIRRORED: [
            _gpu_row("/dave/train", "/dave/train", "dave", "r1", 8, started_ms=started_d),
        ],
    }
