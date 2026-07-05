# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris's RBAC policy over the generic ``rigging.server_auth`` framework.

The mechanism — verifying a token, binding the per-request identity, the
authenticator stack and the Connect interceptors — lives in
``rigging.server_auth``. This module holds only what is Iris-specific: the role
semantics. It names the privileged actions and the role each requires, the
read-only allowlist for the ``dashboard`` role, and the cookie the dashboard
authenticates browsers with, and it reads the identity ``rigging.server_auth``
bound for the request to enforce them.
"""

from enum import StrEnum

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from rigging.server_auth import VerifiedIdentity, require_identity

# Browser session cookie the dashboard sets; passed to rigging's auth
# interceptors as ``cookie_name`` so a cookie-bearing browser RPC authenticates.
SESSION_COOKIE = "iris_session"

# Read-only role granted to an IAP-authenticated caller whose email is not
# provisioned in the user store (see the controller's IAP role resolver). It may
# only call the read RPCs in DASHBOARD_READABLE_RPCS; see authorize_method. A
# provisioned admin/user behind IAP resolves to their real role instead.
DASHBOARD_ROLE = "dashboard"


class AuthzAction(StrEnum):
    """Actions requiring authorization. Add new actions here; policy is in POLICY."""

    ACT_AS_WORKER = "act_as_worker"
    MANAGE_BUDGETS = "manage_budgets"
    SET_CONTAINER_PROFILE = "set_container_profile"


# Action → frozenset of roles allowed. Admin is implicitly always allowed.
POLICY: dict[AuthzAction, frozenset[str]] = {
    AuthzAction.ACT_AS_WORKER: frozenset({"worker"}),
    AuthzAction.MANAGE_BUDGETS: frozenset(),  # admin only
    AuthzAction.SET_CONTAINER_PROFILE: frozenset(),  # admin only (elevated container profiles)
}


# RPC methods the read-only `dashboard` role may call. A default-deny allowlist:
# a dashboard caller (an IAP-authenticated browser whose email is not provisioned
# in the user store) may invoke only these read methods; everything else — job
# submit/terminate, worker registration, key/budget management, exec, profiling,
# raw queries — is denied. A newly added RPC is therefore denied to the dashboard
# role until it is explicitly listed here, which is the safe direction for a
# read-only tier.
DASHBOARD_READABLE_RPCS: frozenset[str] = frozenset(
    {
        # Jobs and tasks
        "GetJobStatus",
        "GetJobState",
        "ListJobs",
        "GetTaskStatus",
        "ListTasks",
        "GetProcessStatus",
        # Workers, endpoints, scheduler, autoscaler
        "ListWorkers",
        "GetWorkerStatus",
        "ListEndpoints",
        "GetAutoscalerStatus",
        "GetSchedulerState",
        "GetKubernetesClusterStatus",
        "ListBackends",
        # Federation (read-only peer observation)
        "ListPeers",
        # Identity, users, budgets (read)
        "GetCurrentUser",
        "ListUsers",
        "GetUserBudget",
        "ListUserBudgets",
        # RPC stats panel
        "GetRpcStats",
    }
)


def authorize_method(identity: VerifiedIdentity, method_name: str) -> None:
    """Enforce per-method access for restricted roles before dispatch.

    An endpoint-scoped token (``identity.audience`` set) has zero RPC authority:
    it may reach only its endpoint's /proxy path (enforced there), never the
    control RPC surface.

    The ``dashboard`` role is read-only: it may call only the methods in
    ``DASHBOARD_READABLE_RPCS``. Other roles are unrestricted here — their
    mutating actions remain gated inside the handlers by ``authorize`` /
    ``authorize_resource_owner``. Raises ``PERMISSION_DENIED`` for a dashboard
    caller invoking a non-readable method.
    """
    if identity.audience is not None:
        raise ConnectError(
            Code.PERMISSION_DENIED,
            "endpoint-scoped token cannot call control RPCs",
        )
    if identity.role == DASHBOARD_ROLE and method_name not in DASHBOARD_READABLE_RPCS:
        raise ConnectError(
            Code.PERMISSION_DENIED,
            f"Read-only dashboard access cannot call {method_name}; "
            "this identity is not provisioned for write access",
        )


def authorize(action: AuthzAction) -> VerifiedIdentity:
    """Require the current caller has permission for the given action.

    Admin role is always authorized. Other roles are checked against POLICY.
    """
    identity = require_identity()
    if identity.role == "admin":
        return identity
    allowed = POLICY.get(action, frozenset())
    if identity.role not in allowed:
        raise ConnectError(Code.PERMISSION_DENIED, f"{action} not allowed for role {identity.role}")
    return identity


def authorize_resource_owner(resource_owner: str) -> VerifiedIdentity:
    """Require the caller owns the resource or is admin."""
    identity = require_identity()
    if identity.role == "admin":
        return identity
    if identity.user_id != resource_owner:
        raise ConnectError(
            Code.PERMISSION_DENIED,
            f"User '{identity.user_id}' cannot access resources owned by '{resource_owner}'",
        )
    return identity
