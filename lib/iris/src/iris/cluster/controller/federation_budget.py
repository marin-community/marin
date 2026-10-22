# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared budget enforcement through federation sync."""

import threading
from dataclasses import dataclass

from connectrpc.code import Code
from connectrpc.errors import ConnectError

from iris.cluster.controller import reads
from iris.cluster.controller.budget import compute_user_spend
from iris.cluster.controller.db import ControllerDB, Tx
from iris.cluster.types import UserBudgetDefaults
from iris.rpc import controller_pb2


@dataclass(frozen=True)
class BudgetContext:
    """Policy and usage from one scheduling snapshot."""

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


def read_budget_context(tx: Tx, defaults: UserBudgetDefaults) -> BudgetContext:
    """Read locally owned policy and local usage from one database snapshot."""
    budgets = {row.user_id: row for row in reads.list_user_budgets(tx)}
    spend = compute_user_spend(tx)
    return BudgetContext(
        {
            user_id: controller_pb2.Controller.GetUserBudgetResponse(
                user_id=user_id,
                budget_limit=budgets[user_id].budget_limit if user_id in budgets else defaults.budget_limit,
                max_band=budgets[user_id].max_band if user_id in budgets else defaults.max_band,
                budget_spent=spend.get(user_id, 0),
            )
            for user_id in budgets.keys() | spend.keys()
        },
        defaults,
    )


class FederationBudgets:
    """Share budgets through sync, retaining the last received values on outages.

    Children need their first policy after startup before admitting new work.
    """

    def __init__(
        self,
        db: ControllerDB,
        defaults: UserBudgetDefaults,
        *,
        cluster_id: str = "",
        parent_id: str = "",
        peers: tuple[str, ...] = (),
    ):
        self.db = db
        self.defaults = defaults
        self.parent_id = parent_id
        self._cluster_id = cluster_id
        self._peers = peers
        self._lock = threading.Lock()
        self._usage: dict[str, dict[str, int]] = {}
        self._policy: controller_pb2.Controller.FederationBudgetPolicy | None = None

    def _context(self, tx: Tx, exclude_peer: str = "") -> BudgetContext:
        with self._lock:
            policy = self._policy
            reports = dict(self._usage)
        if self.parent_id:
            if policy is None:
                return BudgetContext({}, self.defaults, ready=False)
            context = BudgetContext(
                {budget.user_id: budget for budget in policy.users},
                UserBudgetDefaults(policy.default_budget_limit, policy.default_max_band),
            )
            spend = context.spend
            for user, value in compute_user_spend(tx).items():
                spend[user] = spend.get(user, 0) + value
            reports = {}
        else:
            context = read_budget_context(tx, self.defaults)
            spend = context.spend
        for peer, report in reports.items():
            if peer != exclude_peer:
                for user, value in report.items():
                    spend[user] = spend.get(user, 0) + value
        return BudgetContext(
            {
                user: controller_pb2.Controller.GetUserBudgetResponse(
                    user_id=user,
                    budget_limit=context.user(user).budget_limit,
                    max_band=context.user(user).max_band,
                    budget_spent=spend.get(user, 0),
                )
                for user in context.users.keys() | spend.keys()
            },
            context.defaults,
        )

    def context(self, tx: Tx) -> BudgetContext:
        return self._context(tx)

    def budget_policy(self, peer_id: str) -> controller_pb2.Controller.FederationBudgetPolicy | None:
        """Build child policy with usage from every other cluster."""
        if peer_id not in self._peers:
            return None
        with self.db.read_snapshot() as tx:
            context = self._context(tx, exclude_peer=peer_id)
        return controller_pb2.Controller.FederationBudgetPolicy(
            users=list(context.users.values()),
            default_budget_limit=context.defaults.budget_limit,
            default_max_band=context.defaults.max_band,
        )

    def record_usage(self, peer_id: str, usage: controller_pb2.Controller.FederationBudgetUsage) -> None:
        if peer_id not in self._peers:
            return
        with self._lock:
            self._usage[peer_id] = dict(usage.spend) if usage.budget_parent == self._cluster_id else {}

    def receive_policy(self, requester_id: str, policy: controller_pb2.Controller.FederationBudgetPolicy) -> None:
        if not self.parent_id or requester_id != self.parent_id:
            raise ConnectError(Code.PERMISSION_DENIED, "Only the configured budget parent may send policy")
        snapshot = controller_pb2.Controller.FederationBudgetPolicy()
        snapshot.CopyFrom(policy)
        with self._lock:
            self._policy = snapshot

    def local_usage(self, tx: Tx) -> controller_pb2.Controller.FederationBudgetUsage:
        return controller_pb2.Controller.FederationBudgetUsage(
            budget_parent=self.parent_id, spend=compute_user_spend(tx)
        )

    def check_submission(self) -> BudgetContext:
        with self.db.read_snapshot() as tx:
            context = self.context(tx)
        if not context.ready:
            raise ConnectError(Code.UNAVAILABLE, "Waiting for the budget parent's first sync")
        return context
