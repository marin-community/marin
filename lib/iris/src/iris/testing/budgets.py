# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""In-process budget transport over real controller service operations."""

from collections.abc import Callable

from rigging.server_auth import VerifiedIdentity, identity_scope

from iris.cluster.controller.service import ControllerServiceImpl
from iris.rpc import controller_pb2


class InProcessBudgetClient:
    def __init__(self, service: Callable[[], ControllerServiceImpl]):
        self._service = service
        self.reachable = True

    def _require_reachable(self) -> None:
        if not self.reachable:
            raise ConnectionError("Budget controller is unreachable")

    def get_user_budget(
        self, request: controller_pb2.Controller.GetUserBudgetRequest, *, timeout_ms: int
    ) -> controller_pb2.Controller.GetUserBudgetResponse:
        del timeout_ms
        self._require_reachable()
        with identity_scope(VerifiedIdentity(user_id="budget-reader", role="admin")):
            return self._service().get_user_budget(request, None)

    def list_user_budgets(
        self, request: controller_pb2.Controller.ListUserBudgetsRequest, *, timeout_ms: int
    ) -> controller_pb2.Controller.ListUserBudgetsResponse:
        del timeout_ms
        self._require_reachable()
        with identity_scope(VerifiedIdentity(user_id="budget-reader", role="admin")):
            return self._service().list_user_budgets(request, None)

    def close(self) -> None:
        pass
