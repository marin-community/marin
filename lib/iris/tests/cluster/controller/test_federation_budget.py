# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared budgets through the existing local-list and parent-get RPCs."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass

import httpx
import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from google.protobuf.json_format import MessageToDict
from iris.cluster.bundle import BundleStore
from iris.cluster.controller import writes
from iris.cluster.controller.auth import ControllerAuth
from iris.cluster.controller.dashboard import ControllerDashboard
from iris.cluster.controller.endpoint_service import EndpointServiceImpl
from iris.cluster.controller.federation_budget import FederationBudgets, read_budget_context
from iris.cluster.controller.reconcile.dispatch import drain_for_dispatch
from iris.cluster.controller.scheduling.policy import (
    apply_scheduling_gates,
    build_scheduling_context,
    compute_scheduling_order,
)
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.types import JobName, UserBudgetDefaults
from iris.rpc import controller_pb2, job_pb2
from iris.testing.budgets import InProcessBudgetClient
from iris.testing.controller import MockController, make_controller_state, make_job_request, submit_job
from iris.testing.journeys.world import ManualClock
from rigging.auth import IapCredentialsUnavailable
from rigging.server_auth import VerifiedIdentity, identity_scope
from starlette.testclient import TestClient

INTERACTIVE = job_pb2.PRIORITY_BAND_INTERACTIVE
BATCH = job_pb2.PRIORITY_BAND_BATCH


@dataclass
class BudgetFederation:
    states: list
    parent: FederationBudgets
    children: list[FederationBudgets]
    services: dict[int, ControllerServiceImpl]
    parent_clients: list[InProcessBudgetClient]
    peer_clients: dict[str, InProcessBudgetClient]
    clock: ManualClock

    def refresh(self):
        self.clock.advance(3)
        self.parent.refresh()
        for child in self.children:
            child.refresh()


@pytest.fixture
def federation(log_client, tmp_path):
    with ExitStack() as stack:
        states = [stack.enter_context(make_controller_state()) for _ in range(3)]
        services = {}
        clock = ManualClock()
        peer_clients = {
            peer: InProcessBudgetClient(lambda index=index: services[index])
            for index, peer in enumerate(("a", "b"), start=1)
        }
        parent = FederationBudgets(
            states[0]._db, UserBudgetDefaults(budget_limit=10), peer_clients=peer_clients, now=clock.now
        )
        parent_clients = [InProcessBudgetClient(lambda: services[0]) for _ in states[1:]]
        children = [
            FederationBudgets(
                state._db, UserBudgetDefaults(budget_limit=0), parent_id="hub", parent_client=client, now=clock.now
            )
            for state, client in zip(states[1:], parent_clients, strict=True)
        ]
        for index, (state, budgets) in enumerate(zip(states, [parent, *children], strict=True)):
            services[index] = ControllerServiceImpl(
                controller=MockController(),
                bundle_store=BundleStore(storage_dir=str(tmp_path / f"bundles-{index}")),
                log_client=log_client,
                db=state._db,
                endpoint_service=EndpointServiceImpl(db=state._db),
                auth=ControllerAuth(provider="static"),
                budgets=budgets,
            )
        parent.refresh()
        yield BudgetFederation(states, parent, children, services, parent_clients, peer_clients, clock)


def context(budgets):
    with budgets.db.read_snapshot() as tx:
        return budgets.context(tx)


def activate(state, name, *, band=INTERACTIVE, replicas=1):
    submit_job(state, name, make_job_request(name=name, priority_band=band, replicas=replicas))
    with state._db.transaction() as tx:
        drain_for_dispatch(tx, budget_context=read_budget_context(tx, UserBudgetDefaults()))


def test_direct_roots_and_children_share_budget_across_clusters(federation):
    states, parent, children = federation.states, federation.parent, federation.children
    activate(states[0], "/alice/local")
    activate(states[1], "/alice/direct")
    activate(states[1], "/alice/direct/child")
    activate(states[2], "/alice/other")
    activate(states[2], "/alice/batch", band=BATCH, replicas=10)
    federation.refresh()

    assert context(parent).user("alice").budget_spent == 24
    assert context(children[0]).user("alice").budget_spent == 24
    assert context(children[0]).user("alice").budget_limit == 10
    submit_job(states[1], "/alice/later", make_job_request(name="/alice/later", priority_band=INTERACTIVE))
    with states[1]._db.transaction() as tx:
        batch = drain_for_dispatch(tx, budget_context=children[0].context(tx))
    later = [request for request in batch.tasks_to_run if request.task_id == "/alice/later/0"]
    assert [request.priority for request in later] == [BATCH]
    federation.refresh()
    assert context(parent).user("alice").budget_spent == 30
    assert context(children[0]).user("alice").budget_spent == 30


def test_local_budget_lists_do_not_reexport_aggregate_spend(federation):
    activate(federation.states[0], "/alice/local")
    activate(federation.states[1], "/alice/remote")
    federation.refresh()
    local = federation.peer_clients["a"].list_user_budgets()
    assert [(budget.user_id, budget.budget_spent) for budget in local.users] == [("alice", 6)]
    assert federation.parent_clients[0].get_user_budget("alice").budget_spent == 12
    federation.refresh()
    assert federation.parent_clients[0].get_user_budget("alice").budget_spent == 12


def test_child_counts_new_local_work_before_the_next_parent_poll(federation):
    child = federation.children[0]
    child.check_submission("alice")
    activate(federation.states[1], "/alice/burst", replicas=2)
    submit_job(federation.states[1], "/alice/later", make_job_request(name="/alice/later", priority_band=INTERACTIVE))
    with federation.states[1]._db.transaction() as tx:
        batch = drain_for_dispatch(tx, budget_context=child.context(tx))
    later = [request for request in batch.tasks_to_run if request.task_id == "/alice/later/0"]
    assert [request.priority for request in later] == [BATCH]


def test_parent_unavailable_response_invalidates_a_recent_child_snapshot(federation):
    activate(federation.states[1], "/alice/active")
    federation.refresh()
    federation.peer_clients["a"].reachable = False
    federation.clock.advance(29)
    federation.children[0].check_submission("alice")
    assert context(federation.children[0]).ready
    federation.clock.advance(3)
    federation.children[0].refresh()
    assert not context(federation.children[0]).ready


def test_lost_peer_retains_spend_and_blocks_new_scheduling_until_recovery(federation):
    activate(federation.states[1], "/alice/active")
    federation.refresh()
    federation.peer_clients["a"].reachable = False
    federation.clock.advance(31)
    federation.parent.refresh()
    federation.children[0].refresh()
    assert not context(federation.parent).ready
    assert not context(federation.children[0]).ready
    assert context(federation.parent).user("alice").budget_spent == 6
    submit_job(federation.states[1], "/alice/wait", make_job_request(name="/alice/wait", priority_band=INTERACTIVE))
    with federation.states[1]._db.transaction() as tx:
        batch = drain_for_dispatch(tx, budget_context=federation.children[0].context(tx))
    assert "/alice/wait/0" not in {request.task_id for request in batch.tasks_to_run}
    federation.peer_clients["a"].reachable = True
    federation.refresh()
    assert context(federation.parent).ready
    assert context(federation.children[0]).ready


def test_parent_restart_requires_fresh_local_usage_reads(federation):
    activate(federation.states[1], "/alice/active")
    restarted = FederationBudgets(
        federation.parent.db, federation.parent.defaults, peer_clients=federation.peer_clients, now=federation.clock.now
    )
    assert not context(restarted).ready
    restarted.refresh()
    assert context(restarted).ready
    assert context(restarted).user("alice").budget_spent == 6


def test_worker_scheduler_uses_usage_from_other_peers(federation):
    activate(federation.states[2], "/alice/remote", replicas=2)
    submit_job(federation.states[1], "/alice/local", make_job_request(name="/alice/local", priority_band=INTERACTIVE))
    federation.refresh()
    state = federation.states[1]
    with state._db.read_snapshot() as tx:
        scheduling = build_scheduling_context(
            tx, state._health, state._worker_attrs, budget_context=federation.children[0].context(tx)
        )
    gated = apply_scheduling_gates(scheduling, max_tasks_per_job_per_cycle=100)
    assert (
        compute_scheduling_order(scheduling, gated).task_band_map[JobName.from_string("/alice/local").task(0)] == BATCH
    )


def test_direct_submission_reads_parent_ceiling_and_rejects_local_overrides(federation):
    with federation.states[0]._db.transaction() as tx:
        writes.set_user_budget(tx, "alice", 10, BATCH, federation.clock.now())
    service = federation.services[1]
    with identity_scope(VerifiedIdentity(user_id="alice", role="user")):
        with pytest.raises(ConnectError, match="max band: batch") as caught:
            service.launch_job(make_job_request(name="/alice/direct-check", priority_band=INTERACTIVE), None)
        assert caught.value.code == Code.PERMISSION_DENIED
        assert service.launch_job(make_job_request(name="/alice/direct-check", priority_band=BATCH), None).job_id
    with identity_scope(VerifiedIdentity(user_id="admin", role="admin")):
        with pytest.raises(ConnectError, match="Set budgets on parent"):
            service.set_user_budget(
                controller_pb2.Controller.SetUserBudgetRequest(user_id="alice", budget_limit=0), None
            )


def test_parent_lookup_returns_defaults_for_a_new_user(federation):
    budget = federation.parent_clients[0].get_user_budget("new-user")
    assert (budget.user_id, budget.budget_limit, budget.max_band, budget.budget_spent) == (
        "new-user",
        10,
        INTERACTIVE,
        0,
    )


def test_direct_parent_check_does_not_wait_for_another_users_cached_policy(federation):
    submit_job(federation.states[1], "/bob/pending", make_job_request(name="/bob/pending"))
    with identity_scope(VerifiedIdentity(user_id="alice", role="user")):
        response = federation.services[1].launch_job(make_job_request(name="/alice/direct"), None)
    assert response.job_id == "/alice/direct"


def test_handoff_from_another_routing_parent_obeys_the_budget_parent_ceiling(federation):
    with federation.states[0]._db.transaction() as tx:
        writes.set_user_budget(tx, "alice", 10, BATCH, federation.clock.now())
    request = make_job_request(name="/alice/foreign-handoff", priority_band=INTERACTIVE)
    request.federation.CopyFrom(
        controller_pb2.Controller.FederationHandoff(requester_id="other", submitting_user="alice")
    )
    with identity_scope(VerifiedIdentity(user_id="other", role="admin")):
        with pytest.raises(ConnectError, match="max band: batch") as caught:
            federation.services[1].launch_job(request, None)
    assert caught.value.code == Code.PERMISSION_DENIED


@pytest.mark.parametrize(
    "parent_error", [ConnectionError("unreachable"), IapCredentialsUnavailable("no machine credentials")]
)
def test_direct_submission_fails_when_parent_is_unreachable(federation, monkeypatch, parent_error):
    def unreachable(user_id):
        raise parent_error

    monkeypatch.setattr(federation.parent_clients[0], "get_user_budget", unreachable)
    with identity_scope(VerifiedIdentity(user_id="alice", role="user")):
        with pytest.raises(ConnectError) as caught:
            federation.services[1].launch_job(make_job_request(name="/alice/direct-check"), None)
    assert caught.value.code == Code.UNAVAILABLE


def test_waiting_submission_leaves_rpc_capacity_for_parent_usage_read(federation, monkeypatch):
    application = ControllerDashboard(federation.services[1]).app
    original = federation.parent_clients[0].get_user_budget

    async def scenario():
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application), base_url="http://controller"
        ) as client:

            def check_parent(user_id):
                # A real parent can call back to read this child's local usage.
                response = asyncio.run_coroutine_threadsafe(
                    client.post("/iris.cluster.ControllerService/ListUserBudgets", json={}), loop
                ).result(timeout=2)
                assert response.status_code == 200
                return original(user_id)

            monkeypatch.setattr(federation.parent_clients[0], "get_user_budget", check_parent)
            response = await asyncio.wait_for(
                client.post(
                    "/iris.cluster.ControllerService/LaunchJob",
                    json=MessageToDict(make_job_request(name="/alice/concurrent", priority_band=INTERACTIVE)),
                ),
                timeout=3,
            )
            assert response.status_code == 200

    with TestClient(application):
        asyncio.run(scenario())
