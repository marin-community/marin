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

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from rigging.server_auth import VerifiedIdentity

from iris.cluster.authorization import (
    FEDERATION_PEER_ROLE,
    AuthzAction,
)
from iris.cluster.authorization import (
    authorize as authorize_resource_action,
)
from iris.cluster.authorization import (
    authorize_resource_owner as authorize_native_resource_owner,
)
from iris.resources.errors import ResourcePermissionDenied

# Read-only role granted to an IAP-authenticated caller whose email is not
# provisioned in the user store (see the controller's IAP role resolver). It may
# only call the read RPCs in DASHBOARD_READABLE_RPCS; see authorize_method. A
# provisioned admin/user behind IAP resolves to their real role instead.
DASHBOARD_ROLE = "dashboard"

# Role carried by a verified federation token — a trusted peer handing a job off or
# driving its sync/cancel/proxy. Method-scoped to FEDERATION_RPCS and the
# target-scoped FEDERATION_SCOPED_RPCS below, never a general identity: a federation
# bearer the composite verifier accepts cannot reach any other RPC even though it
# authenticated.
# RPCs a federation-peer identity may call unconditionally: whole-job handoff, routed
# cancel, delta-sync, and the capability heartbeat. A default-deny allowlist, like
# DASHBOARD_READABLE_RPCS.
FEDERATION_RPCS: frozenset[str] = frozenset({"LaunchJob", "TerminateJob", "FederationSync", "ListBackends"})

# On-demand debug proxies a federation-peer identity may call, but only after the
# handler confirms the target task belongs to a job that peer federated here (its
# RECEIVED handle) — a peer must not profile/exec/inspect the receiving cluster's own
# tasks or its controller. authorize_method admits these; the controller service's
# _authorize_federated_debug_target enforces the per-target ownership.
FEDERATION_SCOPED_RPCS: frozenset[str] = frozenset(
    {
        "ProfileTask",
        "ExecInContainer",
        "GetProcessStatus",
        "CancelJob",
        "RetryTask",
        "TerminateAttempt",
        "ExecAttempt",
        "ProfileAttempt",
    }
)


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
        "DescribeJob",
        "DescribeTask",
        "BatchDescribeTasks",
        "DescribeAttempt",
        "ListActivity",
        "FetchLogs",
        "GetActionReceipt",
        # Workers, endpoints, scheduler, autoscaler
        "ListWorkers",
        "GetWorkerStatus",
        "ListEndpoints",
        "GetAutoscalerStatus",
        "GetSchedulerState",
        "GetKubernetesClusterStatus",
        "ListBackends",
        "ListNodes",
        "DescribeNode",
        "ListSlices",
        "DescribeSlice",
        "DescribeEndpoint",
        "BatchDescribeEndpoints",
        # Federation (read-only peer observation)
        "ListPeers",
        # Identity, users, budgets (read)
        "GetCurrentUser",
        "ListUsers",
        "GetUserBudget",
        "ListUserBudgets",
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
            f"Read-only dashboard access cannot call {method_name}; this identity is not provisioned for write access",
        )
    if (
        identity.role == FEDERATION_PEER_ROLE
        and method_name not in FEDERATION_RPCS
        and method_name not in FEDERATION_SCOPED_RPCS
    ):
        raise ConnectError(
            Code.PERMISSION_DENIED,
            f"Federation-peer identity cannot call {method_name}; it is scoped to the federation RPC subset",
        )


def authorize(action: AuthzAction) -> VerifiedIdentity:
    """Map native Iris action authorization failures to Connect."""
    try:
        return authorize_resource_action(action)
    except ResourcePermissionDenied as exc:
        raise ConnectError(Code.PERMISSION_DENIED, str(exc)) from exc


def authorize_resource_owner(resource_owner: str) -> VerifiedIdentity:
    """Map native Iris resource-owner authorization failures to Connect."""
    try:
        return authorize_native_resource_owner(resource_owner)
    except ResourcePermissionDenied as exc:
        raise ConnectError(Code.PERMISSION_DENIED, str(exc)) from exc
