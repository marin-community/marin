# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Parent-owned budget policy exchanged over the authenticated federation sync."""

import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from rigging.timing import Duration, Timestamp
from sqlalchemy import delete, select
from sqlalchemy.dialects.sqlite import insert

from iris.cluster.controller import reads
from iris.cluster.controller.budget import compute_local_user_spend, compute_user_spend
from iris.cluster.controller.db import ControllerDB, Tx
from iris.cluster.controller.schema import federation_budget_reports_table
from iris.cluster.types import UserBudgetDefaults
from iris.rpc import controller_pb2

BUDGET_POLICY_MAX_AGE = Duration.from_seconds(30)
BUDGET_REFRESH_TIMEOUT = Duration.from_seconds(15)


@dataclass(frozen=True)
class BudgetContext:
    users: dict[str, controller_pb2.Controller.GetUserBudgetResponse]
    defaults: UserBudgetDefaults
    ready: bool = True

    @property
    def spend(self) -> dict[str, int]:
        return {user: row.budget_spent for user, row in self.users.items()}

    @property
    def limits(self) -> dict[str, int]:
        return {user: row.budget_limit for user, row in self.users.items()}

    def user(self, user_id: str) -> controller_pb2.Controller.GetUserBudgetResponse:
        return self.users.get(user_id) or controller_pb2.Controller.GetUserBudgetResponse(
            user_id=user_id, budget_limit=self.defaults.budget_limit, max_band=self.defaults.max_band
        )


class FederationBudgets:
    """Serve local policy or require a fresh policy from a named parent.

    Usage reports cover every local task, including direct submissions and
    children. Mirrored tasks never contribute. Lost peers retain their last
    spend and stop new scheduling once their report expires.
    """

    def __init__(
        self,
        db: ControllerDB,
        defaults: UserBudgetDefaults,
        *,
        cluster_id: str = "",
        parent_id: str = "",
        peer_ids: tuple[str, ...] = (),
        now: Callable[[], Timestamp] | None = None,
    ):
        self.db = db
        self.defaults = defaults
        self.cluster_id = cluster_id
        self.parent_id = parent_id
        self.peer_ids = peer_ids
        self._now = now or Timestamp.now
        self._condition = threading.Condition()
        self._epoch = uuid.uuid4().hex
        self._generation = 0
        self._ack_generation = -1
        self._policy: BudgetContext | None = None
        self._received_at: Timestamp | None = None
        self._reported_spend: dict[str, int] = {}
        self._observed_peers: set[str] = set()
        with db.transaction() as tx:
            tx.execute(
                delete(federation_budget_reports_table).where(~federation_budget_reports_table.c.cluster.in_(peer_ids))
            )

    def context(self, tx: Tx) -> BudgetContext:
        """Read a scheduling snapshot without waiting while a DB snapshot is held."""
        if self.parent_id:
            with self._condition:
                fresh = (
                    self._received_at is not None
                    and self._now().epoch_ms() - self._received_at.epoch_ms() < BUDGET_POLICY_MAX_AGE.to_ms()
                )
                if self._policy is None:
                    return BudgetContext({}, self.defaults, ready=False)
                users = {}
                local_spend = compute_local_user_spend(tx)
                for user_id in self._policy.users.keys() | local_spend.keys():
                    row = self._policy.user(user_id)
                    users[user_id] = controller_pb2.Controller.GetUserBudgetResponse(
                        user_id=user_id,
                        budget_limit=row.budget_limit,
                        max_band=row.max_band,
                        budget_spent=row.budget_spent
                        - self._reported_spend.get(user_id, 0)
                        + local_spend.get(user_id, 0),
                    )
                return BudgetContext(users, self._policy.defaults, ready=fresh and self._policy.ready)
        budgets = reads.list_user_budgets(tx)
        spend = compute_user_spend(tx)
        users = {
            row.user_id: controller_pb2.Controller.GetUserBudgetResponse(
                user_id=row.user_id,
                budget_limit=row.budget_limit,
                max_band=row.max_band,
                budget_spent=spend.get(row.user_id, 0),
            )
            for row in budgets
        }
        for user_id, value in spend.items():
            if user_id not in users:
                users[user_id] = controller_pb2.Controller.GetUserBudgetResponse(
                    user_id=user_id,
                    budget_limit=self.defaults.budget_limit,
                    max_band=self.defaults.max_band,
                    budget_spent=value,
                )
        reports = tx.execute(select(federation_budget_reports_table)).all()
        fresh_peers = {
            row.cluster
            for row in reports
            if self._now().epoch_ms() - row.reported_at_ms.epoch_ms() < BUDGET_POLICY_MAX_AGE.to_ms()
        }
        with self._condition:
            ready = set(self.peer_ids) <= fresh_peers & self._observed_peers
        return BudgetContext(users, self.defaults, ready=ready)

    def check_submission(self) -> BudgetContext:
        """Wait for a parent round trip before admitting a direct or child submission."""
        if self.parent_id:
            with self._condition:
                self._generation += 1
                requested = self._generation
                refreshed = self._condition.wait_for(
                    lambda: self._ack_generation >= requested,
                    timeout=BUDGET_REFRESH_TIMEOUT.to_seconds(),
                )
                if not refreshed:
                    raise ConnectError(
                        Code.UNAVAILABLE, f"Budget parent {self.parent_id!r} did not answer the budget check"
                    )
        with self.db.read_snapshot() as tx:
            context = self.context(tx)
        if not context.ready:
            raise ConnectError(
                Code.UNAVAILABLE, "Federation budget usage is unavailable or stale; retry after sync recovers"
            )
        return context

    def prepare_sync(self, peer_id: str, request: controller_pb2.Controller.FederationSyncRequest) -> None:
        """Attach the authoritative context and acknowledge that peer's refresh request."""
        if self.parent_id:
            return
        with self.db.read_snapshot() as tx:
            context = self.context(tx)
            report = tx.execute(
                select(federation_budget_reports_table).where(federation_budget_reports_table.c.cluster == peer_id)
            ).first()
        request.budget_policy.CopyFrom(
            controller_pb2.Controller.FederationBudgetPolicy(
                users=list(context.users.values()),
                default_limit=context.defaults.budget_limit,
                default_max_band=context.defaults.max_band,
                refresh_epoch=report.refresh_epoch if report is not None else "",
                refresh_generation=report.refresh_generation if report is not None else 0,
                ready=context.ready,
                peer_spend=report.user_spend if report is not None else {},
            )
        )

    def receive_sync(
        self, requester_id: str, request: controller_pb2.Controller.FederationSyncRequest
    ) -> controller_pb2.Controller.FederationBudgetReport:
        """Accept policy only from the configured parent and return local usage."""
        if not self.parent_id:
            return controller_pb2.Controller.FederationBudgetReport()
        if requester_id != self.parent_id:
            # A second routing hub may observe this peer, but only its declared
            # budget parent owns policy and usage. Tell other hubs that the peer
            # is managed elsewhere so their readiness does not wait forever.
            return controller_pb2.Controller.FederationBudgetReport(authority_id=self.parent_id)
        with self._condition:
            policy = request.budget_policy
            if request.HasField("budget_policy") and policy.refresh_epoch == self._epoch:
                self._policy = BudgetContext(
                    {row.user_id: row for row in policy.users},
                    UserBudgetDefaults(budget_limit=policy.default_limit, max_band=policy.default_max_band),
                    ready=policy.ready,
                )
                self._received_at = self._now()
                self._reported_spend = dict(policy.peer_spend)
                self._ack_generation = max(self._ack_generation, policy.refresh_generation)
                self._condition.notify_all()
            epoch, generation = self._epoch, self._generation
        with self.db.read_snapshot() as tx:
            spend = compute_local_user_spend(tx)
        return controller_pb2.Controller.FederationBudgetReport(
            authority_id=self.parent_id,
            user_spend=spend,
            refresh_epoch=epoch,
            refresh_generation=generation,
        )

    def record_report(self, peer_id: str, report: controller_pb2.Controller.FederationBudgetReport) -> None:
        """Replace one peer's usage, so retries and mirrors cannot double-charge it."""
        if self.parent_id or peer_id not in self.peer_ids or not report.authority_id:
            return
        values = dict(
            cluster=peer_id,
            user_spend=dict(report.user_spend) if report.authority_id == self.cluster_id else {},
            reported_at_ms=self._now(),
            refresh_epoch=report.refresh_epoch,
            refresh_generation=report.refresh_generation,
        )
        with self.db.transaction() as tx:
            tx.execute(
                insert(federation_budget_reports_table)
                .values(**values)
                .on_conflict_do_update(index_elements=["cluster"], set_=values)
            )
        with self._condition:
            self._observed_peers.add(peer_id)
