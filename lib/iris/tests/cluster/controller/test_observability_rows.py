# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The controller's durable job, job config, task and attempt rows in Finelog."""

import dataclasses
import json
import logging
import sqlite3
import threading
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import pytest
from finelog.deploy.config import load_finelog_config
from iris.cluster.config import load_config
from iris.cluster.controller import observability_rows, ops, write_lock_cost, writes
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.federation_store import ControllerFederationStore
from iris.cluster.controller.observability_rows import ObservabilityRowTables
from iris.cluster.controller.ops.task import Assignment
from iris.cluster.controller.reconcile.snapshot import TaskUpdate
from iris.cluster.controller.write_lock_cost import PendingCost
from iris.cluster.federation.store import HandoffAdmission, HandoffSpec
from iris.cluster.stats.tables import IrisJob, IrisJobConfig, IrisTask, IrisTaskAttempt
from iris.cluster.types import JobName
from iris.rpc import job_pb2
from iris.test_util import FakeStatsTable
from iris.testing import observability_rows_check
from iris.testing.controller import (
    fail_worker,
    make_controller_state,
    make_direct_job_request,
    make_job_request,
    make_worker_metadata,
    query_task,
    register_worker,
    submit_job,
    transition_task,
)
from iris.testing.controller_state import ControllerTestState
from iris.testing.transitions import WorkerTaskUpdates, apply_task_observations
from rigging.timing import RateLimiter, Timestamp


@dataclass
class RecordedRows:
    tables: ObservabilityRowTables

    def _rows(self, table) -> list:
        assert isinstance(table, FakeStatsTable)
        return [row for write in table.writes for row in write]

    @property
    def jobs(self) -> list[IrisJob]:
        return self._rows(self.tables.job_table)

    @property
    def job_configs(self) -> list[IrisJobConfig]:
        return self._rows(self.tables.job_config_table)

    @property
    def tasks(self) -> list[IrisTask]:
        return self._rows(self.tables.task_table)

    @property
    def attempts(self) -> list[IrisTaskAttempt]:
        return self._rows(self.tables.task_attempt_table)


def _attach(state: ControllerTestState, cluster: str) -> RecordedRows:
    tables = ObservabilityRowTables(
        cluster=cluster,
        job_table=FakeStatsTable(),
        job_config_table=FakeStatsTable(),
        task_table=FakeStatsTable(),
        task_attempt_table=FakeStatsTable(),
    )
    state._db.attach_observability_rows(tables)
    return RecordedRows(tables)


@pytest.fixture
def state():
    with make_controller_state() as controller_state:
        yield controller_state


def _run_on_node(state: ControllerTestState, task, worker_id, node_name: str) -> None:
    with state._db.transaction() as cur:
        ops.task.assign(cur, [Assignment(task_id=task.task_id, worker_id=worker_id)], health=state._health)
    with state._db.transaction() as cur:
        apply_task_observations(
            cur,
            [
                WorkerTaskUpdates(
                    worker_id=worker_id,
                    updates=[
                        TaskUpdate(
                            task_id=task.task_id,
                            attempt_id=query_task(state, task.task_id).current_attempt_id,
                            new_state=job_pb2.TASK_STATE_RUNNING,
                            pod_name="pod-a",
                            node_name=node_name,
                            topology={"rack": "rack-3"},
                        )
                    ],
                )
            ],
            health=state._health,
            now=Timestamp.now(),
        )


def test_submit_writes_one_job_row_and_one_config_row(state):
    rows = _attach(state, "marin")

    submit_job(state, "train", make_job_request("train", cpu=4, memory_bytes=8 * 1024**3))

    assert len(rows.jobs) == 1
    job = rows.jobs[0]
    assert job.cluster == "marin"
    assert job.job_id == "/test-user/train"
    assert job.root_job_id == "/test-user/train"
    assert job.state == "JOB_STATE_PENDING"
    assert job.placement == "local"
    assert len(rows.job_configs) == 1
    config = rows.job_configs[0]
    assert (config.root_job_id, config.root_submitted_at_ms) == (job.root_job_id, job.root_submitted_at_ms)
    assert config.res_cpu_millicores == 4000
    assert config.res_memory_bytes == 8 * 1024**3
    assert config.priority_band == "PRIORITY_BAND_INTERACTIVE"
    assert rows.attempts == []


def test_attempt_rows_record_the_node_and_join_their_job(state):
    worker = register_worker(state, "w1", "host1:8080", make_worker_metadata())
    tasks = submit_job(state, "train", make_job_request("train"))
    rows = _attach(state, "cw-rno2a")

    _run_on_node(state, tasks[0], worker, "gpu-node-7")
    transition_task(state, tasks[0].task_id, job_pb2.TASK_STATE_SUCCEEDED, exit_code=0)

    running = [a for a in rows.attempts if a.state == "TASK_STATE_RUNNING"]
    assert running and running[-1].node_name == "gpu-node-7"
    assert json.loads(running[-1].topology_json) == {"rack": "rack-3"}
    assert running[-1].worker_id == "w1"
    assert running[-1].started_at is not None
    final = rows.attempts[-1]
    assert final.state == "TASK_STATE_SUCCEEDED"
    assert final.exit_code == 0
    assert final.finished_at is not None
    assert final.node_name == "gpu-node-7"
    assert len({a.attempt_uid for a in rows.attempts}) == 1

    job = rows.jobs[-1]
    assert job.state == "JOB_STATE_SUCCEEDED"
    assert {(a.cluster, a.root_job_id, a.root_submitted_at_ms, a.job_id, a.submitted_at_ms) for a in rows.attempts} == {
        (job.cluster, job.root_job_id, job.root_submitted_at_ms, job.job_id, job.submitted_at_ms)
    }


def test_mark_jobs_running_writes_a_running_job_row(state):
    submit_job(state, "train", make_job_request("train"))
    rows = _attach(state, "marin")
    job_id = JobName.root("test-user", "train")

    with state._db.transaction() as cur:
        writes.mark_jobs_running(cur, (j for j in [job_id]), Timestamp.now().epoch_ms())

    assert [(j.job_id, j.state) for j in rows.jobs] == [(job_id.to_wire(), "JOB_STATE_RUNNING")]
    assert rows.jobs[0].started_at is not None


def test_mark_jobs_running_on_a_running_job_writes_no_row(state):
    submit_job(state, "train", make_job_request("train"))
    job_id = JobName.root("test-user", "train")
    with state._db.transaction() as cur:
        writes.mark_jobs_running(cur, [job_id], Timestamp.now().epoch_ms())
    rows = _attach(state, "marin")

    with state._db.transaction() as cur:
        writes.mark_jobs_running(cur, [job_id], Timestamp.now().epoch_ms())

    assert rows.jobs == []


def test_cancel_writes_killed_job_and_attempt_rows(state):
    worker = register_worker(state, "w1", "host1:8080", make_worker_metadata())
    tasks = submit_job(state, "train", make_job_request("train"))
    _run_on_node(state, tasks[0], worker, "node-a")
    rows = _attach(state, "marin")

    transition_task(state, tasks[0].task_id, job_pb2.TASK_STATE_KILLED, error="user cancelled")

    assert rows.jobs[-1].state == "JOB_STATE_KILLED"
    assert rows.jobs[-1].finished_at is not None
    assert rows.attempts[-1].state == "TASK_STATE_KILLED"


def test_rolled_back_transaction_writes_no_rows(state):
    rows = _attach(state, "marin")
    request = make_job_request("train")

    with pytest.raises(RuntimeError, match="abort"):
        with state._db.transaction() as cur:
            ops.job.submit(
                cur,
                job_id=JobName.root("test-user", "train"),
                request=request,
                ts=Timestamp.now(),
                priority_band=job_pb2.PRIORITY_BAND_INTERACTIVE,
            )
            raise RuntimeError("abort")

    assert (rows.jobs, rows.job_configs, rows.attempts) == ([], [], [])


def test_failed_snapshot_drops_rows_but_commits_the_controller_change(state, monkeypatch):
    observability_rows_check.uninstall(state._db)  # this test drops rows on purpose
    rows = _attach(state, "marin")

    def broken_snapshot(cur, tables, marked):
        marked.job_rows.append(object())  # a partial snapshot must be dropped too
        raise RuntimeError("snapshot bug")

    monkeypatch.setattr(observability_rows, "_snapshot", broken_snapshot)
    tasks = submit_job(state, "train", make_job_request("train"))

    assert len(tasks) == 1
    assert (rows.jobs, rows.job_configs, rows.attempts) == ([], [], [])


class _ClosedTable(FakeStatsTable):
    def write(self, rows) -> None:
        raise RuntimeError("Table(iris.controller_db__jobs) is closed")


def test_failed_write_commits_and_runs_later_post_commit_hooks(state):
    observability_rows_check.uninstall(state._db)  # this test drops rows on purpose
    state._db.attach_observability_rows(
        ObservabilityRowTables(
            cluster="marin",
            job_table=_ClosedTable(),
            job_config_table=FakeStatsTable(),
            task_table=FakeStatsTable(),
            task_attempt_table=FakeStatsTable(),
        )
    )
    later_hook_ran: list[bool] = []
    job_id = JobName.root("test-user", "train")

    with state._db.transaction() as cur:
        ops.job.submit(
            cur,
            job_id=job_id,
            request=make_job_request("train"),
            ts=Timestamp.now(),
            priority_band=job_pb2.PRIORITY_BAND_INTERACTIVE,
        )
        cur.register(lambda: later_hook_ran.append(True))

    assert later_hook_ran == [True]


def test_handed_off_job_is_written_by_the_executing_peer_only():
    """The parent parks and later kills its handle for a handed-off job; the peer runs it."""
    with make_controller_state() as parent, make_controller_state() as peer:
        parent_rows = _attach(parent, "marin")
        peer_rows = _attach(peer, "cw-rno2a")
        job_id = JobName.root("test-user", "fed-job")

        spec = HandoffSpec(
            local_job_id=job_id,
            peer_id="cw-rno2a",
            owner_principal="test-user",
            submitting_user="test-user",
            request=make_direct_job_request("fed-job", replicas=1, priority_band=job_pb2.PRIORITY_BAND_INTERACTIVE),
        )
        assert ControllerFederationStore(parent._db).admit_and_persist_queued(spec) is HandoffAdmission.ADMITTED
        with parent._db.transaction() as cur:
            writes.promote_queued_handoff(cur, job_id, "cw-rno2a")
            writes.mark_federated_job_killed(cur, job_id, now_ms=Timestamp.now().epoch_ms(), error="cancelled")

        received = make_direct_job_request("fed-job", replicas=1)
        received.federation.requester_id = "marin"
        received.federation.owner_principal = "test-user"
        submit_job(peer, "fed-job", received)

        assert (parent_rows.jobs, parent_rows.job_configs, parent_rows.attempts) == ([], [], [])
        assert [(j.cluster, j.job_id) for j in peer_rows.jobs] == [("cw-rno2a", job_id.to_wire())]
        assert len(peer_rows.job_configs) == 1


def test_failed_worker_attempt_keeps_its_worker_id(state):
    # ops.worker.fail finishes the worker's attempts (commit_effects) and then
    # remove_worker clears task_attempts.worker_id, all in one transaction.
    worker = register_worker(state, "w1", "host1:8080", make_worker_metadata())
    tasks = submit_job(state, "train", make_job_request("train"))
    rows = _attach(state, "marin")
    _run_on_node(state, tasks[0], worker, "")

    fail_worker(state, worker, "Connection lost")

    final = rows.attempts[-1]
    assert final.state == "TASK_STATE_WORKER_FAILED"
    assert final.worker_id == "w1"


def test_a_replaced_child_gets_its_own_job_key(state):
    # A live parent resubmits a finished child under the same name. The child
    # inherits the parent's root key, so only its own submit time tells the runs apart.
    worker = register_worker(state, "w1", "host1:8080", make_worker_metadata())
    submit_job(state, "train", make_job_request("train"), timestamp_ms=1_000)
    child = "/test-user/train/eval"
    first = submit_job(state, child, make_job_request(child), timestamp_ms=2_000)
    _run_on_node(state, first[0], worker, "")
    transition_task(state, first[0].task_id, job_pb2.TASK_STATE_SUCCEEDED, exit_code=0)
    rows = _attach(state, "marin")

    with state._db.transaction() as cur:
        assert ops.job.remove_finished(cur, JobName.from_string(child))
    submit_job(state, child, make_job_request(child), timestamp_ms=3_000)

    (config,) = rows.job_configs
    assert (config.job_id, config.root_submitted_at_ms, config.submitted_at_ms) == (child, 1_000, 3_000)
    assert {(j.job_id, j.submitted_at_ms) for j in rows.jobs} == {(child, 3_000)}


_CLUSTER_CONFIGS = sorted((Path(__file__).parents[3] / "config").glob("*.yaml"))


@pytest.mark.parametrize("config_path", _CLUSTER_CONFIGS, ids=lambda p: p.stem)
def test_every_cluster_config_forwards_under_its_own_name_or_not_at_all(config_path):
    # A forwarder ships only rows named for its forwarding.cluster, and the controller
    # names its rows with the cluster config name, so a mismatch would drop them.
    config = load_config(config_path)
    if not config.finelog.config:
        return
    forwarding = load_finelog_config(config.finelog.config).forwarding
    assert forwarding is None or forwarding.cluster == config.name


@pytest.mark.parametrize(
    "device, expected",
    [
        (job_pb2.DeviceConfig(gpu=job_pb2.GpuDevice(variant="H100", count=8)), ("gpu", "H100", 8)),
        (job_pb2.DeviceConfig(tpu=job_pb2.TpuDevice(variant="v5p-64", count=4)), ("tpu", "v5p-64", 4)),
        (None, (None, None, 0)),
    ],
    ids=["gpu", "tpu", "cpu-only"],
)
def test_job_config_flattens_the_device_request(state, device, expected):
    rows = _attach(state, "marin")
    request = make_job_request("train")
    if device is not None:
        request.resources.device.CopyFrom(device)

    submit_job(state, "train", request)

    config = rows.job_configs[0]
    assert (config.accelerator_kind, config.accelerator_type, config.accelerators_per_task) == expected


def test_job_config_stores_no_environment_command_lines_or_other_user_text(state):
    rows = _attach(state, "marin")
    request = make_job_request("train")
    fake = {name: f"made-up-{name.lower()}" for name in ("HF_TOKEN", "WANDB_API_KEY", "CW_KEY_ID", "CW_KEY_SECRET")}
    request.environment.secrets.update(fake)
    request.environment.env_vars.update({"SERVICE_URL": "made-up-env-value"})
    request.environment.setup_scripts.append("export HF_TOKEN=made-up-script-token && pip install made-up-package")
    request.entrypoint.setup_commands.append("huggingface-cli login --token made-up-login-token")
    request.entrypoint.run_command.argv[:] = ["python", "made-up-script.py", "--hf-token", "made-up-cli-token"]
    request.submit_argv[:] = ["iris", "job", "run", "-e", "SERVICE_URL", "made-up-argv-value", "--", "python"]

    submit_job(state, "train", request)

    written = repr([*rows.jobs, *rows.job_configs, *rows.tasks, *rows.attempts])
    for value in [
        *fake,
        *fake.values(),
        "SERVICE_URL",
        "made-up-env-value",
        "made-up-script",
        "made-up-package",
        "made-up-login-token",
        "made-up-cli-token",
        "made-up-argv-value",
    ]:
        assert value not in written
    assert rows.job_configs[0].provenance_base_commit is None


_MADE_UP_TOKEN = "ghp_madeUpToken0123456789abcdefghijkl"


def _provenance_json(**overrides: object) -> str:
    fields: dict[str, object] = {
        "tree_hash": "1a2b3c4d5e",
        "base_commit": "6f7a8b9c0d",
        "dirty": True,
        "branch": "made-up-branch",
        "built_by": "made-up-user",
        "git_remote": "https://made-up-user:made-up-password@github.com/made-up-org/repo.git",
        "created_at": "2026-01-02T03:04:05+00:00",
        "command_line": ["python", "train.py", "--api-key", "made-up-key", "--password=made-up-pw", "--lr", "0.1"],
    }
    fields.update(overrides)
    return json.dumps(fields)


def test_job_config_keeps_only_the_launch_git_identity(state):
    rows = _attach(state, "marin")
    request = make_job_request("train")
    request.environment.env_vars.update({"MARIN_PROVENANCE": _provenance_json(), "SERVICE_URL": "made-up-env-value"})
    request.environment.secrets.update({"HF_TOKEN": "made-up-secret"})

    submit_job(state, "train", request)

    config = rows.job_configs[0]
    assert (
        config.provenance_base_commit,
        config.provenance_tree_hash,
        config.provenance_dirty,
        config.provenance_branch,
    ) == ("6f7a8b9c0d", "1a2b3c4d5e", True, "made-up-branch")
    written = repr(config)
    for value in [
        "made-up-user",
        "made-up-password",
        "made-up-org",
        "made-up-key",
        "made-up-pw",
        "train.py",
        "2026-01-02",
    ]:
        assert value not in written


def test_job_config_redacts_a_token_in_the_launch_branch(state):
    rows = _attach(state, "marin")
    request = make_job_request("train")
    request.environment.env_vars["MARIN_PROVENANCE"] = _provenance_json(branch=f"fix-{_MADE_UP_TOKEN}")

    submit_job(state, "train", request)

    branch = rows.job_configs[0].provenance_branch
    assert _MADE_UP_TOKEN not in branch
    assert "[REDACTED]" in branch


@pytest.mark.parametrize(
    "raw",
    ["made-up-not-json", "[" * 100_000, _provenance_json(dirty="yes")],
    ids=["not-json", "deeply-nested", "string-dirty"],
)
def test_job_config_drops_a_malformed_launch_provenance_and_keeps_the_rows(state, raw, monkeypatch, caplog):
    monkeypatch.setattr(observability_rows, "_malformed_provenance_log", RateLimiter(interval_seconds=0))
    rows = _attach(state, "marin")
    request = make_job_request("train")
    request.environment.env_vars["MARIN_PROVENANCE"] = raw

    with caplog.at_level(logging.WARNING, logger=observability_rows.__name__):
        submit_job(state, "train", request)

    config = rows.job_configs[0]
    assert (config.provenance_base_commit, config.provenance_tree_hash, config.provenance_dirty) == (None, None, None)
    assert len(rows.jobs) == 1
    assert rows.tasks
    (warning,) = [r.getMessage() for r in caplog.records if "malformed MARIN_PROVENANCE" in r.getMessage()]
    assert "/test-user/train" in warning
    assert raw[:20] not in warning


@dataclass(frozen=True)
class _Classified:
    """One controller DB table's columns, classified by the rule in
    docs/observability-rows.md ("Which columns are copied")."""

    row_class: type
    # DB column -> the Finelog field it is copied to.
    copied: dict[str, str]
    # DB column -> why it is not copied.
    excluded: dict[str, str]
    # Finelog fields not copied from a DB column: the keys from the job, the
    # writing cluster, the snapshot time and values derived in the copy.
    derived: frozenset[str]


_JOB_KEY = frozenset({"cluster", "ts", "root_job_id", "root_submitted_at_ms", "submitted_at_ms"})
_COLUMNS: dict[str, _Classified] = {
    "jobs": _Classified(
        IrisJob,
        {
            "job_id": "job_id",
            "user_id": "user_id",
            "submitting_user": "submitting_user",
            "parent_job_id": "parent_job_id",
            "root_job_id": "root_job_id",
            "depth": "depth",
            "state": "state",
            "submitted_at_ms": "submitted_at_ms",
            "root_submitted_at_ms": "root_submitted_at_ms",
            "started_at_ms": "started_at",
            "finished_at_ms": "finished_at",
            "scheduling_deadline_epoch_ms": "scheduling_deadline_at",
            "error": "error",
            "exit_code": "exit_code",
            "num_tasks": "num_tasks",
            "name": "name",
            "backend_id": "backend_id",
            "cluster": "placement",
        },
        {},
        frozenset({"cluster", "ts"}),
    ),
    "job_config": _Classified(
        IrisJobConfig,
        {
            "job_id": "job_id",
            "res_cpu_millicores": "res_cpu_millicores",
            "res_memory_bytes": "res_memory_bytes",
            "res_disk_bytes": "res_disk_bytes",
            "res_device_json": "res_device_json",
            "constraints_json": "constraints_json",
            "has_coscheduling": "has_coscheduling",
            "coscheduling_group_by": "coscheduling_group_by",
            "coscheduling_group_sizes": "coscheduling_group_sizes",
            "scheduling_timeout_ms": "scheduling_timeout_ms",
            "max_task_failures": "max_task_failures",
            "max_retries_failure": "max_retries_failure",
            "max_retries_preemption": "max_retries_preemption",
            "timeout_ms": "timeout_ms",
            "termination_grace_ms": "termination_grace_ms",
            "preemption_policy": "preemption_policy",
            "existing_job_policy": "existing_job_policy",
            "priority_band": "priority_band",
            "task_image": "task_image",
            "fail_if_exists": "fail_if_exists",
            "container_profile": "container_profile",
            "egress_policy": "egress_policy",
        },
        {
            "name": "duplicate",
            # Only MARIN_PROVENANCE's git identity is read from it, into provenance_*.
            "environment_json": "user-text",
            "entrypoint_json": "user-text",
            "submit_argv_json": "user-text",
            "bundle_id": "user-text",
            "ports_json": "user-text",
            "egress_allow_json": "user-text",
        },
        _JOB_KEY
        | {"accelerator_kind", "accelerator_type", "accelerators_per_task"}
        | {"provenance_base_commit", "provenance_tree_hash", "provenance_dirty", "provenance_branch"},
    ),
    "tasks": _Classified(
        IrisTask,
        {
            "task_id": "task_id",
            "job_id": "job_id",
            "task_index": "task_index",
            "state": "state",
            "error": "error",
            "exit_code": "exit_code",
            "submitted_at_ms": "task_submitted_at",
            "started_at_ms": "started_at",
            "finished_at_ms": "finished_at",
            "max_retries_failure": "max_retries_failure",
            "max_retries_preemption": "max_retries_preemption",
            "current_attempt_id": "current_attempt_id",
            "priority_root_submitted_ms": "priority_root_submitted_ms",
            "priority_band": "priority_band",
            "container_id": "container_id",
            "status_message": "status_message",
            "current_worker_id": "current_worker_id",
            "backend_id": "backend_id",
            "cluster": "placement",
        },
        {
            "priority_neg_depth": "scheduler-queue-order",
            "priority_insertion": "scheduler-queue-order",
            "current_worker_address": "in-iris.worker",
        },
        _JOB_KEY,
    ),
    "task_attempts": _Classified(
        IrisTaskAttempt,
        {
            "task_id": "task_id",
            "attempt_id": "attempt_id",
            "worker_id": "worker_id",
            "state": "state",
            "created_at_ms": "created_at",
            "started_at_ms": "started_at",
            "finished_at_ms": "finished_at",
            "exit_code": "exit_code",
            "error": "error",
            "attempt_uid": "attempt_uid",
            "backend_id": "backend_id",
            "pod_name": "pod_name",
            "pod_uid": "pod_uid",
            "node_name": "node_name",
            "topology_json": "topology_json",
            "terminal_reason": "terminal_reason",
        },
        {},
        _JOB_KEY | {"job_id"},
    ),
}
_THIS_TEST = "lib/iris/tests/cluster/controller/test_observability_rows.py"
_RULE = 'lib/iris/docs/observability-rows.md, "Which columns are copied"'


@pytest.mark.parametrize("table", sorted(_COLUMNS))
def test_every_controller_db_column_is_classified(tmp_path, table):
    classified = _COLUMNS[table]
    copied, excluded = classified.copied, classified.excluded
    ControllerDB(db_dir=tmp_path).close()
    with closing(sqlite3.connect(tmp_path / ControllerDB.DB_FILENAME)) as conn:
        db_columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}

    unclassified = sorted(db_columns - copied.keys() - excluded.keys())
    assert not unclassified, (
        f"controller DB table {table!r} has unclassified column(s) {unclassified}: classify each as "
        f"COPIED or EXCLUDED in {_THIS_TEST}; see {_RULE}"
    )
    gone = sorted((copied.keys() | excluded.keys()) - db_columns)
    assert not gone, (
        f"controller DB table {table!r} no longer has column(s) {gone}: drop an EXCLUDED one; a COPIED one "
        f"keeps its Finelog field, made `| None = None`, so classify that field as retired in {_THIS_TEST}; "
        f"see {_RULE}"
    )
    fields = {field.name: field for field in dataclasses.fields(classified.row_class)}
    sources = set(copied.values()) | classified.derived
    # Exact, so an EXCLUDED column copied anyway, or under a new name, fails too.
    assert fields.keys() == sources, (
        f"{classified.row_class.__name__} fields without a classified source: {sorted(fields.keys() - sources)}; "
        f"classified sources without a field: {sorted(sources - fields.keys())}; see {_RULE}"
    )


def test_task_rows_follow_state_and_placement_but_not_the_pending_reason(state):
    rows = _attach(state, "marin")
    worker_id = register_worker(state, "w1", "w1:1", make_worker_metadata())
    task = submit_job(state, "train", make_job_request("train", replicas=2))[0]
    assert sorted(t.task_index for t in rows.tasks) == [0, 1]
    assert {t.state for t in rows.tasks} == {"TASK_STATE_PENDING"}
    assert {t.root_job_id for t in rows.tasks} == {"/test-user/train"}

    _run_on_node(state, task, worker_id, "node-a")
    mine = [t for t in rows.tasks if t.task_id == task.task_id.to_wire()]
    assert [t.state for t in mine] == ["TASK_STATE_PENDING", "TASK_STATE_ASSIGNED", "TASK_STATE_RUNNING"]
    assert mine[-1].current_worker_id == str(worker_id)

    written = len(rows.tasks)
    with state._db.transaction() as cur:
        apply_task_observations(
            cur,
            [
                WorkerTaskUpdates(
                    worker_id=worker_id,
                    updates=[
                        TaskUpdate(
                            task_id=task.task_id,
                            attempt_id=query_task(state, task.task_id).current_attempt_id,
                            new_state=job_pb2.TASK_STATE_RUNNING,
                            status_message="made-up pending reason",
                        )
                    ],
                )
            ],
            health=state._health,
            now=Timestamp.now(),
        )
    assert len(rows.tasks) == written


def test_removing_a_worker_writes_the_task_rows_that_pointed_at_it(state):
    rows = _attach(state, "marin")
    worker_id = register_worker(state, "w1", "w1:1", make_worker_metadata())
    task = submit_job(state, "train", make_job_request("train"))[0]
    _run_on_node(state, task, worker_id, "node-a")
    written = len(rows.tasks)

    with state._db.transaction() as cur:
        writes.remove_worker(cur, worker_id, state._health)

    new = rows.tasks[written:]
    assert [(t.task_id, t.state, t.current_worker_id) for t in new] == [
        (task.task_id.to_wire(), "TASK_STATE_RUNNING", None)
    ]


def test_a_controller_with_observability_rows_off_attaches_no_row_tables(make_controller, tmp_path):
    on = make_controller(local_state_dir=tmp_path / "on")
    off = make_controller(observability_rows=False, local_state_dir=tmp_path / "off")

    with on._db.transaction() as tx:
        assert tx.observability_rows is not None
    with off._db.transaction() as tx:
        assert tx.observability_rows is None


def test_slow_rows_are_logged_and_counted_in_the_dbs_lock_cost(state, monkeypatch, caplog):
    monkeypatch.setattr(observability_rows, "SLOW_ROWS_SECONDS", 0.0)
    monkeypatch.setattr(observability_rows, "_slow_rows_log", RateLimiter(interval_seconds=0))
    _attach(state, "marin")

    with caplog.at_level(logging.WARNING, logger=observability_rows.__name__):
        submit_job(state, "train", make_job_request("train"))

    (warning,) = [r.getMessage() for r in caplog.records if "under the write lock" in r.getMessage()]
    assert "finelog.controller_rows: false" in warning


def _emitted(monkeypatch, state) -> list[PendingCost]:
    """Capture what each write transaction emits, checking the write lock is already released."""
    lock = state._db._lock
    emitted: list[PendingCost] = []

    def emit(pending: PendingCost) -> None:
        free: list[bool] = []

        def probe() -> None:
            acquired = lock.acquire(blocking=False)
            free.append(acquired)
            if acquired:
                lock.release()

        thread = threading.Thread(target=probe)
        thread.start()
        thread.join()
        assert free == [True], "emitted while the write lock was held"
        emitted.append(pending)

    monkeypatch.setattr(write_lock_cost, "emit", emit)
    return emitted


def test_a_submit_emits_one_row_record_after_releasing_the_lock(state, monkeypatch):
    emitted = _emitted(monkeypatch, state)
    _attach(state, "marin")

    submit_job(state, "train", make_job_request("train", replicas=3))

    (record,) = [r for pending in emitted for r in pending.row_transactions]
    assert (record.job_rows, record.job_config_rows, record.task_rows) == (1, 1, 3)
    assert record.snapshot_seconds > 0 and record.enqueue_seconds > 0


def test_a_rolled_back_transaction_emits_no_row_record(state, monkeypatch):
    submit_job(state, "train", make_job_request("train"))
    emitted = _emitted(monkeypatch, state)
    _attach(state, "marin")

    def fail_after_the_snapshot() -> None:
        raise RuntimeError("made-up failure after the snapshot")

    with pytest.raises(RuntimeError, match="made-up failure"):
        with state._db.transaction() as cur:
            observability_rows.record_job_changes(cur, [JobName.root("test-user", "train")])
            cur.register_pre_commit(fail_after_the_snapshot)

    assert [r for pending in emitted for r in pending.row_transactions] == []
    assert emitted, "the rolled-back transaction still reports its lock time"


def test_a_status_message_refresh_writes_no_attempt_row_and_no_row_event(state, monkeypatch):
    # A same-state observation with a new status message builds an attempt delta that
    # changes nothing on the attempt row, so it must not copy one.
    worker = register_worker(state, "w1", "host1:8080", make_worker_metadata())
    (task,) = submit_job(state, "train", make_job_request("train"))
    _run_on_node(state, task, worker, "gpu-node-7")
    rows = _attach(state, "marin")
    emitted = _emitted(monkeypatch, state)

    with state._db.transaction() as cur:
        apply_task_observations(
            cur,
            [
                WorkerTaskUpdates(
                    worker_id=worker,
                    updates=[
                        TaskUpdate(
                            task_id=task.task_id,
                            attempt_id=query_task(state, task.task_id).current_attempt_id,
                            new_state=job_pb2.TASK_STATE_RUNNING,
                            pod_name="pod-a",
                            node_name="gpu-node-7",
                            topology={"rack": "rack-3"},
                            status_message="made-up pending reason",
                        )
                    ],
                )
            ],
            health=state._health,
            now=Timestamp.now(),
        )

    assert rows.attempts == []
    assert rows.tasks == []
    assert [r for pending in emitted for r in pending.row_transactions] == []
