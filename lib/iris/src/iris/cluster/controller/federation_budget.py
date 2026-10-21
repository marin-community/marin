# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Parent-owned budget policy exchanged over the authenticated federation sync."""

import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from rigging.timing import Duration, Timestamp
from sqlalchemy import delete, select
from sqlalchemy.dialects.sqlite import insert

from iris.cluster.controller import reads
from iris.cluster.controller.budget import compute_effective_band, compute_local_user_spend, compute_user_spend
from iris.cluster.controller.db import ControllerDB, Tx
from iris.cluster.controller.schema import federation_budget_reports_table
from iris.cluster.types import UserBudgetDefaults
from iris.rpc import controller_pb2

BUDGET_POLICY_MAX_AGE = Duration.from_seconds(30)
BUDGET_REFRESH_TIMEOUT = Duration.from_seconds(15)


@dataclass(frozen=True)
class UserBudgetSnapshot:
    """One user's resolved policy and usage, independent of mutable RPC messages."""

    user_id: str
    budget_limit: int
    max_band: int
    budget_spent: int = 0

    def to_proto(self) -> controller_pb2.Controller.GetUserBudgetResponse:
        return controller_pb2.Controller.GetUserBudgetResponse(
            user_id=self.user_id,
            budget_limit=self.budget_limit,
            max_band=self.max_band,
            budget_spent=self.budget_spent,
        )


@dataclass(frozen=True)
class BudgetContext:
    """Immutable policy and usage from one scheduling snapshot."""

    users: Mapping[str, UserBudgetSnapshot]
    defaults: UserBudgetDefaults
    ready: bool = True

    def __post_init__(self):
        object.__setattr__(self, "users", MappingProxyType(dict(self.users)))

    @property
    def spend(self) -> dict[str, int]:
        return {user: row.budget_spent for user, row in self.users.items()}

    @property
    def limits(self) -> dict[str, int]:
        return {user: row.budget_limit for user, row in self.users.items()}

    def user(self, user_id: str) -> UserBudgetSnapshot:
        return self.users.get(user_id) or UserBudgetSnapshot(
            user_id=user_id, budget_limit=self.defaults.budget_limit, max_band=self.defaults.max_band
        )

    def effective_band(self, requested_band: int, user_id: str) -> int:
        budget = self.user(user_id)
        return compute_effective_band(requested_band, budget.budget_spent, budget.budget_limit)


def read_budget_context(tx: Tx, defaults: UserBudgetDefaults) -> BudgetContext:
    """Read locally owned policy and combined usage from one database snapshot."""
    budgets = {row.user_id: row for row in reads.list_user_budgets(tx)}
    spend = compute_user_spend(tx)
    return BudgetContext(
        {
            user_id: UserBudgetSnapshot(
                user_id=user_id,
                budget_limit=budgets[user_id].budget_limit if user_id in budgets else defaults.budget_limit,
                max_band=budgets[user_id].max_band if user_id in budgets else defaults.max_band,
                budget_spent=spend.get(user_id, 0),
            )
            for user_id in budgets.keys() | spend.keys()
        },
        defaults,
    )


@dataclass(frozen=True)
class _ParentPolicy:
    context: BudgetContext
    received_at: Timestamp
    reported_spend: Mapping[str, int]


class FederationBudgets:
    """Serve local policy or require a fresh policy from a named parent.

    Usage reports cover every local task, including direct submissions and
    children. Mirrored tasks never contribute. Lost budget peers retain their last
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
        self._policy: _ParentPolicy | None = None
        self._observed_peers: set[str] = set()
        with db.transaction() as tx:
            tx.execute(
                delete(federation_budget_reports_table).where(~federation_budget_reports_table.c.cluster.in_(peer_ids))
            )

    def context(self, tx: Tx) -> BudgetContext:
        """Read a scheduling snapshot without waiting while a DB snapshot is held."""
        if self.parent_id:
            with self._condition:
                policy = self._policy
            if policy is None:
                return BudgetContext({}, self.defaults, ready=False)
            fresh = self._now().epoch_ms() - policy.received_at.epoch_ms() < BUDGET_POLICY_MAX_AGE.to_ms()
            local_spend = compute_local_user_spend(tx)
            return BudgetContext(
                {
                    user_id: replace(
                        policy.context.user(user_id),
                        budget_spent=policy.context.user(user_id).budget_spent
                        - policy.reported_spend.get(user_id, 0)
                        + local_spend.get(user_id, 0),
                    )
                    for user_id in policy.context.users.keys() | local_spend.keys()
                },
                policy.context.defaults,
                ready=fresh and policy.context.ready,
            )
        context = read_budget_context(tx, self.defaults)
        reports = tx.execute(select(federation_budget_reports_table)).all()
        known_peers = {row.cluster for row in reports}
        budget_peers = {row.cluster for row in reports if row.authority_id == self.cluster_id}
        fresh_peers = {
            row.cluster
            for row in reports
            if self._now().epoch_ms() - row.reported_at_ms.epoch_ms() < BUDGET_POLICY_MAX_AGE.to_ms()
        }
        with self._condition:
            ready = set(self.peer_ids) <= known_peers and budget_peers <= fresh_peers & self._observed_peers
        return replace(context, ready=ready)

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

    def policy_for_peer(self, peer_id: str) -> controller_pb2.Controller.FederationBudgetPolicy | None:
        """Return authoritative policy with the peer's refresh acknowledgement."""
        if self.parent_id:
            return None
        with self.db.read_snapshot() as tx:
            context = self.context(tx)
            report = tx.execute(
                select(federation_budget_reports_table).where(federation_budget_reports_table.c.cluster == peer_id)
            ).first()
        return controller_pb2.Controller.FederationBudgetPolicy(
            users=[budget.to_proto() for budget in context.users.values()],
            default_limit=context.defaults.budget_limit,
            default_max_band=context.defaults.max_band,
            refresh_epoch=report.refresh_epoch if report is not None else "",
            refresh_generation=report.refresh_generation if report is not None else 0,
            ready=context.ready,
            peer_spend=report.user_spend if report is not None else {},
        )

    def receive_policy(
        self, requester_id: str, policy: controller_pb2.Controller.FederationBudgetPolicy | None
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
            if policy is not None and policy.refresh_epoch == self._epoch:
                self._policy = _ParentPolicy(
                    context=BudgetContext(
                        {
                            row.user_id: UserBudgetSnapshot(
                                row.user_id, row.budget_limit, row.max_band, row.budget_spent
                            )
                            for row in policy.users
                        },
                        UserBudgetDefaults(budget_limit=policy.default_limit, max_band=policy.default_max_band),
                        ready=policy.ready,
                    ),
                    received_at=self._now(),
                    reported_spend=MappingProxyType(dict(policy.peer_spend)),
                )
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
            authority_id=report.authority_id,
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
