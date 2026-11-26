# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared budgets exchanged through federation sync."""

from contextlib import ExitStack
from dataclasses import dataclass

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.cluster.bundle import BundleStore
from iris.cluster.controller import writes
from iris.cluster.controller.auth import ControllerAuth
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
from iris.testing.controller import MockController, make_controller_state, make_job_request, submit_job
from rigging.server_auth import VerifiedIdentity, identity_scope
from rigging.timing import Timestamp

INTERACTIVE = job_pb2.PRIORITY_BAND_INTERACTIVE
BATCH = job_pb2.PRIORITY_BAND_BATCH


@dataclass
class BudgetFederation:
    states: list
    parent: FederationBudgets
    children: list[FederationBudgets]
    services: dict[int, ControllerServiceImpl]

    def sync(self):
        # One pass reports local usage; the next distributes the combined view.
        for _ in range(2):
            for index, peer in enumerate(("a", "b"), start=1):
                request = controller_pb2.Controller.FederationSyncRequest(
                    requester_id="hub", budget_policy=self.parent.budget_policy(peer)
                )
                with identity_scope(VerifiedIdentity(user_id="hub", role="federation-peer")):
                    response = self.services[index].federation_sync(request, None)
                self.parent.record_usage(peer, response.budget_usage)


@pytest.fixture
def unsynced_federation(log_client, tmp_path):
    with ExitStack() as stack:
        states = [stack.enter_context(make_controller_state()) for _ in range(3)]
        parent = FederationBudgets(
            states[0]._db, UserBudgetDefaults(budget_limit=10), cluster_id="hub", peers=("a", "b")
        )
        children = [
            FederationBudgets(state._db, UserBudgetDefaults(budget_limit=5), parent_id="hub") for state in states[1:]
        ]
        services = {
            index: ControllerServiceImpl(
                controller=MockController(),
                bundle_store=BundleStore(storage_dir=str(tmp_path / f"bundles-{index}")),
                log_client=log_client,
                db=state._db,
                endpoint_service=EndpointServiceImpl(db=state._db),
                auth=ControllerAuth(provider="static"),
                budgets=budgets,
            )
            for index, (state, budgets) in enumerate(zip(states, [parent, *children], strict=True))
        }
        federation = BudgetFederation(states, parent, children, services)
        yield federation


@pytest.fixture
def federation(unsynced_federation):
    unsynced_federation.sync()
    return unsynced_federation


def context(budgets):
    with budgets.db.read_snapshot() as tx:
        return budgets.context(tx)


def activate(state, name, *, band=INTERACTIVE, replicas=1):
    submit_job(state, name, make_job_request(name=name, priority_band=band, replicas=replicas))
    with state._db.transaction() as tx:
        drain_for_dispatch(tx)


def test_direct_roots_and_children_share_budget_across_clusters(federation):
    states, parent, children = federation.states, federation.parent, federation.children
    activate(states[0], "/alice/local")
    activate(states[1], "/alice/direct")
    activate(states[1], "/alice/direct/child")
    activate(states[2], "/alice/other")
    activate(states[2], "/alice/batch", band=BATCH, replicas=10)
    federation.sync()

    assert context(parent).user("alice").budget_spent == 24
    assert context(children[0]).user("alice").budget_spent == 24
    assert context(children[0]).user("alice").budget_limit == 10
    submit_job(states[1], "/alice/later", make_job_request(name="/alice/later", priority_band=INTERACTIVE))
    with states[1]._db.transaction() as tx:
        batch = drain_for_dispatch(tx, budget_context=children[0].context(tx))
    later = [request for request in batch.tasks_to_run if request.task_id == "/alice/later/0"]
    assert [request.priority for request in later] == [BATCH]
    federation.sync()
    assert context(parent).user("alice").budget_spent == 30
    assert context(children[0]).user("alice").budget_spent == 30


def test_child_counts_new_local_work_before_the_next_sync(federation):
    child = federation.children[0]
    child.snapshot()
    activate(federation.states[1], "/alice/burst", replicas=2)
    submit_job(federation.states[1], "/alice/later", make_job_request(name="/alice/later", priority_band=INTERACTIVE))
    with federation.states[1]._db.transaction() as tx:
        batch = drain_for_dispatch(tx, budget_context=child.context(tx))
    later = [request for request in batch.tasks_to_run if request.task_id == "/alice/later/0"]
    assert [request.priority for request in later] == [BATCH]


def test_sync_outage_keeps_policy_and_counts_new_local_work(federation):
    with federation.states[0]._db.transaction() as tx:
        writes.set_user_budget(tx, "alice", 10, BATCH, Timestamp.now())
    activate(federation.states[2], "/alice/remote")
    federation.sync()
    # No more syncs: remote usage and policy remain, local usage keeps changing.
    activate(federation.states[1], "/alice/local")
    assert context(federation.children[0]).user("alice").budget_spent == 12
    with identity_scope(VerifiedIdentity(user_id="alice", role="user")):
        service = federation.services[1]
        with pytest.raises(ConnectError) as caught:
            service.launch_job(make_job_request(name="/alice/high", priority_band=INTERACTIVE), None)
        assert caught.value.code == Code.PERMISSION_DENIED
        assert service.launch_job(make_job_request(name="/alice/batch", priority_band=BATCH), None).job_id
    # Policy changes arrive on the same exchange and replace the cached ceiling.
    with federation.states[0]._db.transaction() as tx:
        writes.set_user_budget(tx, "alice", 10, INTERACTIVE, Timestamp.now())
    federation.sync()
    with identity_scope(VerifiedIdentity(user_id="alice", role="user")):
        assert service.launch_job(make_job_request(name="/alice/recovered", priority_band=INTERACTIVE), None).job_id
    assert context(federation.children[0]).user("alice").budget_spent == 12


def test_child_admits_and_schedules_before_initial_sync_then_uses_parent_policy(unsynced_federation):
    federation = unsynced_federation
    state, child, service = federation.states[1], federation.children[0], federation.services[1]
    with identity_scope(VerifiedIdentity(user_id="alice", role="user")):
        service.launch_job(make_job_request(name="/alice/startup", priority_band=INTERACTIVE), None)
    with state._db.transaction() as tx:
        first = drain_for_dispatch(tx, budget_context=child.context(tx))
    assert [task.priority for task in first.tasks_to_run] == [INTERACTIVE]

    with identity_scope(VerifiedIdentity(user_id="alice", role="user")):
        service.launch_job(make_job_request(name="/alice/local-limit", priority_band=INTERACTIVE), None)
    with state._db.transaction() as tx:
        second = drain_for_dispatch(tx, budget_context=child.context(tx))
    assert [task.priority for task in second.tasks_to_run if task.task_id == "/alice/local-limit/0"] == [BATCH]

    with federation.states[0]._db.transaction() as tx:
        writes.set_user_budget(tx, "alice", 20, INTERACTIVE, Timestamp.now())
    federation.sync()
    with identity_scope(VerifiedIdentity(user_id="alice", role="user")):
        service.launch_job(make_job_request(name="/alice/parent-limit", priority_band=INTERACTIVE), None)
    with state._db.transaction() as tx:
        third = drain_for_dispatch(tx, budget_context=child.context(tx))
    assert [task.priority for task in third.tasks_to_run if task.task_id == "/alice/parent-limit/0"] == [INTERACTIVE]


def test_foreign_parent_cannot_replace_policy_or_contribute_usage(federation):
    foreign = controller_pb2.Controller.FederationSyncRequest(
        requester_id="other", budget_policy=controller_pb2.Controller.FederationBudgetPolicy(default_budget_limit=1)
    )
    with identity_scope(VerifiedIdentity(user_id="other", role="federation-peer")):
        with pytest.raises(ConnectError) as caught:
            federation.services[1].federation_sync(foreign, None)
    assert caught.value.code == Code.PERMISSION_DENIED
    federation.parent.record_usage(
        "a", controller_pb2.Controller.FederationBudgetUsage(budget_parent="other", spend={"alice": 100})
    )
    assert context(federation.parent).user("alice").budget_spent == 0
    assert context(federation.children[0]).user("alice").budget_limit == 10


def test_worker_scheduler_uses_usage_from_other_peers(federation):
    activate(federation.states[2], "/alice/remote", replicas=2)
    submit_job(federation.states[1], "/alice/local", make_job_request(name="/alice/local", priority_band=INTERACTIVE))
    federation.sync()
    state = federation.states[1]
    with state._db.read_snapshot() as tx:
        scheduling = build_scheduling_context(
            tx,
            state._health,
            state._worker_attrs,
            UserBudgetDefaults(),
            budget_context=federation.children[0].context(tx),
        )
    gated = apply_scheduling_gates(scheduling, max_tasks_per_job_per_cycle=100)
    assert (
        compute_scheduling_order(scheduling, gated).task_band_map[JobName.from_string("/alice/local").task(0)] == BATCH
    )


def test_direct_submission_reads_parent_ceiling_and_rejects_local_overrides(federation):
    with federation.states[0]._db.transaction() as tx:
        writes.set_user_budget(tx, "alice", 10, BATCH, Timestamp.now())
    federation.sync()
    service = federation.services[1]
    with identity_scope(VerifiedIdentity(user_id="alice", role="user")):
        with pytest.raises(ConnectError) as caught:
            service.launch_job(make_job_request(name="/alice/direct-check", priority_band=INTERACTIVE), None)
        assert caught.value.code == Code.PERMISSION_DENIED
        assert service.launch_job(make_job_request(name="/alice/direct-check", priority_band=BATCH), None).job_id
    with identity_scope(VerifiedIdentity(user_id="admin", role="admin")):
        with pytest.raises(ConnectError) as caught:
            service.set_user_budget(
                controller_pb2.Controller.SetUserBudgetRequest(user_id="alice", budget_limit=0), None
            )
        assert caught.value.code == Code.FAILED_PRECONDITION


def test_synced_policy_applies_defaults_to_a_new_user(federation):
    budget = federation.children[0].snapshot().user("new-user")
    assert (budget.user_id, budget.budget_limit, budget.max_band, budget.budget_spent) == (
        "new-user",
        10,
        INTERACTIVE,
        0,
    )


def test_synced_policy_admits_a_new_user_with_other_users_pending(federation):
    submit_job(federation.states[1], "/bob/pending", make_job_request(name="/bob/pending"))
    with identity_scope(VerifiedIdentity(user_id="alice", role="user")):
        response = federation.services[1].launch_job(make_job_request(name="/alice/direct"), None)
    assert response.job_id == "/alice/direct"


def test_handoff_from_another_routing_parent_obeys_the_budget_parent_ceiling(federation):
    with federation.states[0]._db.transaction() as tx:
        writes.set_user_budget(tx, "alice", 10, BATCH, Timestamp.now())
    federation.sync()
    request = make_job_request(name="/alice/foreign-handoff", priority_band=INTERACTIVE)
    request.federation.CopyFrom(
        controller_pb2.Controller.FederationHandoff(requester_id="other", submitting_user="alice")
    )
    with identity_scope(VerifiedIdentity(user_id="other", role="admin")):
        with pytest.raises(ConnectError) as caught:
            federation.services[1].launch_job(request, None)
    assert caught.value.code == Code.PERMISSION_DENIED
