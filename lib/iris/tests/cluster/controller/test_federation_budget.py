# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared budget policy across independently submitted job trees."""

import asyncio
import importlib
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, closing

import httpx
import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from google.protobuf.json_format import MessageToDict, ParseDict
from iris.cluster.bundle import BundleStore
from iris.cluster.controller import federation_budget as budget_module
from iris.cluster.controller import writes
from iris.cluster.controller.auth import ControllerAuth
from iris.cluster.controller.dashboard import ControllerDashboard
from iris.cluster.controller.endpoint_service import EndpointServiceImpl
from iris.cluster.controller.federation_budget import FederationBudgets
from iris.cluster.controller.reconcile.dispatch import drain_for_dispatch
from iris.cluster.controller.scheduling.policy import (
    apply_scheduling_gates,
    build_scheduling_context,
    compute_scheduling_order,
)
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.types import JobName, UserBudgetDefaults
from iris.rpc import controller_pb2, job_pb2
from iris.test_util import wait_for_condition
from iris.testing.controller import MockController, make_controller_state, make_job_request, submit_job
from iris.testing.journeys.world import ManualClock
from rigging.server_auth import VerifiedIdentity, identity_scope
from rigging.timing import Duration
from starlette.testclient import TestClient

INTERACTIVE = job_pb2.PRIORITY_BAND_INTERACTIVE
BATCH = job_pb2.PRIORITY_BAND_BATCH


def exchange(parent, child, peer_id):
    request = controller_pb2.Controller.FederationSyncRequest(requester_id="hub")
    parent.prepare_sync(peer_id, request)
    report = child.receive_sync("hub", request)
    parent.record_report(peer_id, report)


def test_budget_migration_forces_full_priority_resync_and_preserves_reports():
    migrate = importlib.import_module("iris.cluster.controller.migrations.0053_federation_budget_reports").migrate
    with closing(sqlite3.connect(":memory:")) as conn:
        conn.execute("CREATE TABLE federation_sync_state (peer_id TEXT PRIMARY KEY, cursor TEXT NOT NULL)")
        conn.execute("INSERT INTO federation_sync_state VALUES ('peer', '123')")
        migrate(conn)
        assert conn.execute("SELECT peer_id FROM federation_sync_state").fetchall() == []
        conn.execute("INSERT INTO federation_budget_reports VALUES ('peer', '{\"alice\":12}', 1000, 'epoch', 1)")
        migrate(conn)
        assert conn.execute("SELECT user_spend FROM federation_budget_reports").fetchall() == [('{"alice":12}',)]


@pytest.fixture
def federation():
    with ExitStack() as stack:
        states = [stack.enter_context(make_controller_state()) for _ in range(3)]
        clock = ManualClock()
        defaults = UserBudgetDefaults(budget_limit=10, max_band=INTERACTIVE)
        parent = FederationBudgets(states[0]._db, defaults, cluster_id="hub", peer_ids=("a", "b"), now=clock.now)
        children = [
            FederationBudgets(state._db, UserBudgetDefaults(budget_limit=0), parent_id="hub", now=clock.now)
            for state in states[1:]
        ]
        for _ in range(2):
            for peer, child in zip(("a", "b"), children, strict=True):
                exchange(parent, child, peer)
        yield states, parent, children, clock


def context(budgets):
    with budgets.db.read_snapshot() as tx:
        return budgets.context(tx)


@pytest.fixture
def peer_service(federation, log_client, tmp_path):
    states, _, children, _ = federation
    return ControllerServiceImpl(
        controller=MockController(),
        bundle_store=BundleStore(storage_dir=str(tmp_path / "bundles")),
        log_client=log_client,
        db=states[1]._db,
        endpoint_service=EndpointServiceImpl(db=states[1]._db),
        auth=ControllerAuth(provider="static"),
        budgets=children[0],
    )


def activate(state, name, *, band=INTERACTIVE, replicas=1):
    submit_job(state, name, make_job_request(name=name, priority_band=band, replicas=replicas))
    with state._db.transaction() as tx:
        drain_for_dispatch(tx)


def test_direct_roots_and_children_share_budget_across_clusters(federation):
    states, parent, children, _ = federation
    activate(states[0], "/alice/local")
    activate(states[1], "/alice/direct")
    activate(states[1], "/alice/direct/child")
    activate(states[2], "/alice/other")
    activate(states[2], "/alice/batch", band=BATCH, replicas=10)
    for _ in range(2):
        for peer, child in zip(("a", "b"), children, strict=True):
            exchange(parent, child, peer)

    assert context(parent).user("alice").budget_spent == 24
    assert context(children[0]).user("alice").budget_spent == 24
    # The peer's unlimited local setting cannot override the hub's limit.
    assert context(children[0]).user("alice").budget_limit == 10
    submit_job(states[1], "/alice/later", make_job_request(name="/alice/later", priority_band=INTERACTIVE))
    with states[1]._db.transaction() as tx:
        batch = drain_for_dispatch(tx, budget_context=children[0].context(tx))
    later = [request for request in batch.tasks_to_run if request.task_id == "/alice/later/0"]
    assert [request.priority for request in later] == [BATCH]
    # Local changes since the report must count before another round trip.
    assert context(children[0]).user("alice").budget_spent == 30
    exchange(parent, children[0], "a")
    exchange(parent, children[0], "a")
    assert context(parent).user("alice").budget_spent == 30
    assert context(children[0]).user("alice").budget_spent == 30


def test_lost_peer_retains_spend_and_blocks_new_scheduling_until_sync(federation):
    states, parent, children, clock = federation
    activate(states[1], "/alice/active")
    exchange(parent, children[0], "a")
    clock.advance(31)
    assert not context(parent).ready
    assert not context(children[0]).ready
    assert context(parent).user("alice").budget_spent == 6
    submit_job(states[1], "/alice/wait", make_job_request(name="/alice/wait", priority_band=INTERACTIVE))
    with states[1]._db.transaction() as tx:
        batch = drain_for_dispatch(tx, budget_context=children[0].context(tx))
    assert "/alice/wait/0" not in {request.task_id for request in batch.tasks_to_run}
    for _ in range(2):
        for peer, child in zip(("a", "b"), children, strict=True):
            exchange(parent, child, peer)
    assert context(parent).ready
    assert context(children[0]).ready


def test_parent_restart_keeps_usage_but_requires_new_reports(federation):
    states, parent, children, clock = federation
    activate(states[1], "/alice/active")
    exchange(parent, children[0], "a")
    restarted = FederationBudgets(parent.db, parent.defaults, cluster_id="hub", peer_ids=("a", "b"), now=clock.now)
    assert context(restarted).user("alice").budget_spent == 6
    assert not context(restarted).ready
    exchange(restarted, children[0], "a")
    exchange(restarted, children[1], "b")
    assert context(restarted).ready
    with states[1]._db.transaction() as tx:
        writes.delete_job(tx, JobName.from_string("/alice/active"))
    exchange(restarted, children[0], "a")
    exchange(restarted, children[0], "a")
    assert context(restarted).user("alice").budget_spent == 0


def test_worker_scheduler_uses_usage_from_other_peers(federation):
    states, parent, children, _ = federation
    activate(states[2], "/alice/remote", replicas=2)
    exchange(parent, children[1], "b")
    exchange(parent, children[0], "a")
    submit_job(states[1], "/alice/local", make_job_request(name="/alice/local", priority_band=INTERACTIVE))
    with states[1]._db.read_snapshot() as tx:
        scheduling = build_scheduling_context(
            tx,
            states[1]._health,
            states[1]._worker_attrs,
            children[0].defaults,
            budget_context=children[0].context(tx),
        )
    gated = apply_scheduling_gates(scheduling, max_tasks_per_job_per_cycle=100)
    order = compute_scheduling_order(scheduling, gated)
    assert order.task_band_map[JobName.from_string("/alice/local").task(0)] == BATCH


def test_unrelated_federation_parent_cannot_replace_budget_policy(federation):
    _, parent, children, _ = federation
    request = controller_pb2.Controller.FederationSyncRequest(requester_id="other")
    parent.prepare_sync("a", request)
    request.budget_policy.default_limit = 0
    report = children[0].receive_sync("other", request)
    assert report.authority_id == "hub"
    assert not report.user_spend
    assert context(children[0]).defaults.budget_limit == 10


def test_secondary_routing_hub_does_not_charge_usage_owned_by_another_authority(federation):
    states, parent, children, clock = federation
    activate(states[1], "/alice/production", replicas=2)
    exchange(parent, children[0], "a")
    with make_controller_state() as secondary_state:
        secondary = FederationBudgets(
            secondary_state._db, UserBudgetDefaults(), cluster_id="dev", peer_ids=("a",), now=clock.now
        )
        request = controller_pb2.Controller.FederationSyncRequest(requester_id="dev")
        secondary.prepare_sync("a", request)
        report = children[0].receive_sync("dev", request)
        secondary.record_report("a", report)
        assert context(secondary).ready
        assert context(secondary).user("alice").budget_spent == 0
        assert context(parent).user("alice").budget_spent == 12


def test_direct_submission_waits_for_parent_and_obeys_its_updated_ceiling(federation, peer_service):
    states, parent, children, clock = federation
    service = peer_service
    with states[0]._db.transaction() as tx:
        writes.set_user_budget(tx, "alice", 10, BATCH, clock.now())

    def submit(band):
        with identity_scope(VerifiedIdentity(user_id="alice", role="user")):
            return service.launch_job(make_job_request(name="/alice/direct-check", priority_band=band), None)

    def sync_until_done(future):
        exchange(parent, children[0], "a")
        return future.done()

    with ThreadPoolExecutor(max_workers=2) as executor:
        refused = executor.submit(submit, INTERACTIVE)
        wait_for_condition(lambda: sync_until_done(refused))
        with pytest.raises(ConnectError, match="max band: batch") as caught:
            refused.result()
        assert caught.value.code == Code.PERMISSION_DENIED
        admitted = executor.submit(submit, BATCH)
        wait_for_condition(lambda: sync_until_done(admitted))
        assert admitted.result().job_id == "/alice/direct-check"

    with identity_scope(VerifiedIdentity(user_id="admin", role="admin")):
        with pytest.raises(ConnectError, match="Set budgets on parent"):
            service.set_user_budget(
                controller_pb2.Controller.SetUserBudgetRequest(user_id="alice", budget_limit=0), None
            )


def test_waiting_submission_leaves_rpc_capacity_for_parent_budget_sync(federation, peer_service, monkeypatch):
    _, parent, _, _ = federation
    monkeypatch.setattr(budget_module, "BUDGET_REFRESH_TIMEOUT", Duration.from_seconds(4))
    application = ControllerDashboard(peer_service).app

    async def scenario():
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application), base_url="http://controller"
        ) as client:
            launch = asyncio.create_task(
                client.post(
                    "/iris.cluster.ControllerService/LaunchJob",
                    json=MessageToDict(make_job_request(name="/alice/concurrent", priority_band=INTERACTIVE)),
                )
            )
            try:
                await asyncio.sleep(0)  # Let LaunchJob reach its parent-policy wait before syncing.
                async with asyncio.timeout(3):
                    while not launch.done():
                        request = controller_pb2.Controller.FederationSyncRequest(requester_id="hub")
                        parent.prepare_sync("a", request)
                        response = await asyncio.wait_for(
                            client.post("/iris.cluster.ControllerService/FederationSync", json=MessageToDict(request)), 2
                        )
                        assert response.status_code == 200
                        report = ParseDict(
                            response.json()["budgetReport"], controller_pb2.Controller.FederationBudgetReport()
                        )
                        parent.record_report("a", report)
                assert (await launch).status_code == 200
            finally:
                launch.cancel()
                await asyncio.gather(launch, return_exceptions=True)

    with TestClient(application):
        asyncio.run(scenario())
