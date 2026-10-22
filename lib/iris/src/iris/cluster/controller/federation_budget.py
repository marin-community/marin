# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared budget enforcement through the existing budget read RPCs."""

import logging
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Protocol

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from rigging.auth import IapCredentialsUnavailable
from rigging.timing import Duration, Timestamp

from iris.cluster.controller import reads
from iris.cluster.controller.budget import compute_effective_band, compute_user_spend
from iris.cluster.controller.db import ControllerDB, Tx
from iris.cluster.types import UserBudgetDefaults
from iris.rpc import controller_pb2

logger = logging.getLogger(__name__)
BUDGET_POLICY_MAX_AGE = Duration.from_seconds(30)
BUDGET_REFRESH_INTERVAL = Duration.from_seconds(3)


class BudgetClient(Protocol):
    def get_user_budget(self, user_id: str) -> controller_pb2.Controller.GetUserBudgetResponse: ...

    def list_user_budgets(self) -> controller_pb2.Controller.ListUserBudgetsResponse: ...

    def close(self) -> None: ...


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
    """Read locally owned policy and local usage from one database snapshot."""
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
class _BudgetRead:
    users: tuple[UserBudgetSnapshot, ...]
    received_at: Timestamp


def _snapshot(budget: controller_pb2.Controller.GetUserBudgetResponse) -> UserBudgetSnapshot:
    return UserBudgetSnapshot(budget.user_id, budget.budget_limit, budget.max_band, budget.budget_spent)


class FederationBudgets:
    """Combine peer-local usage at the parent and read parent decisions on children.

    ListUserBudgets always reports local usage, so parent reads cannot recurse.
    Remote reads happen outside database transactions. Expired reads pause new
    scheduling while existing attempts continue reconciling.
    """

    def __init__(
        self,
        db: ControllerDB,
        defaults: UserBudgetDefaults,
        *,
        parent_id: str = "",
        parent_client: BudgetClient | None = None,
        peer_clients: Mapping[str, BudgetClient] | None = None,
        now: Callable[[], Timestamp] | None = None,
    ):
        if bool(parent_id) != (parent_client is not None):
            raise ValueError("A budget parent requires an authenticated parent client")
        self.db = db
        self.defaults = defaults
        self.parent_id = parent_id
        self._parent = parent_client
        self._peers = dict(peer_clients or {})
        self._now = now or Timestamp.now
        self._lock = threading.Lock()
        self._reads: dict[str, _BudgetRead] = {}

    def _fresh(self, read: _BudgetRead, max_age: Duration) -> bool:
        return self._now().epoch_ms() - read.received_at.epoch_ms() < max_age.to_ms()

    def _required_users(self, tx: Tx) -> set[str]:
        return {row.job_id.user for row in reads.pending_tasks_with_jobs(tx)} | compute_user_spend(tx).keys()

    def context(self, tx: Tx) -> BudgetContext:
        """Read a scheduling snapshot without performing network I/O."""
        with self._lock:
            remote = dict(self._reads)
        if self._parent is not None:
            required = self._required_users(tx)
            local_spend = compute_user_spend(tx)
            return BudgetContext(
                {
                    budget.user_id: replace(
                        budget, budget_spent=max(budget.budget_spent, local_spend.get(budget.user_id, 0))
                    )
                    for read in remote.values()
                    for budget in read.users
                },
                self.defaults,
                ready=all(user in remote and self._fresh(remote[user], BUDGET_POLICY_MAX_AGE) for user in required),
            )
        local = read_budget_context(tx, self.defaults)
        spend = local.spend
        for read in remote.values():
            for budget in read.users:
                spend[budget.user_id] = spend.get(budget.user_id, 0) + budget.budget_spent
        return BudgetContext(
            {
                user: replace(local.user(user), budget_spent=spend.get(user, 0))
                for user in local.users.keys() | spend.keys()
            },
            self.defaults,
            ready=all(peer in remote and self._fresh(remote[peer], BUDGET_POLICY_MAX_AGE) for peer in self._peers),
        )

    def _read_parent(self, user_id: str) -> None:
        assert self._parent is not None
        try:
            budget = self._parent.get_user_budget(user_id)
        except ConnectError as error:
            if error.code == Code.UNAVAILABLE:
                with self._lock:
                    self._reads.pop(user_id, None)
            raise
        with self._lock:
            self._reads[user_id] = _BudgetRead((_snapshot(budget),), self._now())

    def _read_peer(self, peer_id: str) -> None:
        budgets = self._peers[peer_id].list_user_budgets()
        with self._lock:
            self._reads[peer_id] = _BudgetRead(tuple(_snapshot(budget) for budget in budgets.users), self._now())

    def refresh(self, stop_event: threading.Event | None = None) -> None:
        """Refresh scheduling inputs with bounded RPCs before opening a DB snapshot."""
        if self._parent is not None:
            with self.db.read_snapshot() as tx:
                keys = self._required_users(tx)
            read_remote = self._read_parent
        else:
            keys = self._peers.keys()
            read_remote = self._read_peer
        with self._lock:
            due = [
                key
                for key in keys
                if key not in self._reads or not self._fresh(self._reads[key], BUDGET_REFRESH_INTERVAL)
            ]
        if not due:
            return
        with ThreadPoolExecutor(max_workers=min(len(due), 8), thread_name_prefix="budget-read") as executor:
            futures = {executor.submit(read_remote, key): key for key in due}
            while futures:
                if stop_event is not None and stop_event.is_set():
                    for future in futures:
                        future.cancel()
                    return
                done, _ = wait(futures, timeout=1, return_when=FIRST_COMPLETED)
                for future in done:
                    key = futures.pop(future)
                    try:
                        future.result()
                    except (ConnectError, ConnectionError, OSError, IapCredentialsUnavailable):
                        # Never replace unreachable peers with zero spend.
                        logger.warning("Budget read from %s failed", key, exc_info=True)

    def check_submission(self, user_id: str) -> BudgetContext:
        """Read the parent's current decision before admitting a direct or child job."""
        if self._parent is not None:
            try:
                self._read_parent(user_id)
            except (ConnectionError, OSError, IapCredentialsUnavailable) as error:
                raise ConnectError(Code.UNAVAILABLE, f"Budget parent {self.parent_id!r} is unavailable") from error
        with self.db.read_snapshot() as tx:
            context = self.context(tx)
        if self._parent is not None:
            return BudgetContext({user_id: context.user(user_id)}, self.defaults)
        if not context.ready:
            raise ConnectError(Code.UNAVAILABLE, "Budget usage is unavailable or stale; retry after reads recover")
        return context

    def close(self) -> None:
        for client in self._peers.values():
            client.close()
        if self._parent is not None:
            self._parent.close()
