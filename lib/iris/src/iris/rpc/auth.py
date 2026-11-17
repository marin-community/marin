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
from rigging.server_auth import VerifiedIdentity, get_verified_identity, require_identity

# Browser session cookie the dashboard sets; passed to rigging's auth
# interceptors as ``cookie_name`` so a cookie-bearing browser RPC authenticates.
SESSION_COOKIE = "iris_session"

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
FEDERATION_PEER_ROLE = "federation-peer"

# RPCs a federation-peer identity may call unconditionally: whole-job handoff, routed
# cancel, delta-sync, and the capability heartbeat. A default-deny allowlist, like
# DASHBOARD_READABLE_RPCS.
FEDERATION_RPCS: frozenset[str] = frozenset({"LaunchJob", "TerminateJob", "FederationSync", "ListBackends"})

# Task operations a federation-peer identity may call, but only after the
# handler confirms the target task belongs to a job that peer federated here (its
# RECEIVED handle) — a peer must not profile/exec/inspect the receiving cluster's own
# tasks or its controller. authorize_method admits these; each operation
# validates ownership against the peer's received handoff.
FEDERATION_SCOPED_RPCS: frozenset[str] = frozenset({"ProfileTask", "ExecInContainer", "GetProcessStatus", "KickTasks"})

# JWT claim carrying a scoped token's job-name prefix. A token with this claim
# is a prefix-scoped identity: owner-gated RPCs additionally require the target
# job to sit under the prefix (authorize_job_prefix). Absent ⇒ unscoped.
JOB_PREFIX_CLAIM = "prefix"


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


def validate_job_prefix(prefix: str) -> str:
    """Validate a job-name prefix, returning it unchanged.

    A prefix is a wire job-name prefix (e.g. ``/benfeuer/exp2-``) that scopes a
    token: every owner-gated target must be a wire name that starts with it. It
    must be absolute, whitespace-free, and have no empty segment (no ``//``, no
    trailing ``/``). It must also span at least two segments — a user segment
    plus a job segment. A one-segment prefix (``/benfeuer``) would string-match
    other users' names that extend it (``/benfeuerarchy/...``), so it is
    rejected: a scoped token always names a job component under its user.

    Raises ``ValueError`` on a malformed prefix. Verify paths call this and
    propagate the error, so a token with a malformed claim is denied outright.
    """
    if not prefix:
        raise ValueError("prefix is empty")
    if not prefix.startswith("/"):
        raise ValueError(f"prefix {prefix!r} must be an absolute job-name prefix starting with '/'")
    if any(ch.isspace() for ch in prefix):
        raise ValueError(f"prefix {prefix!r} must not contain whitespace")
    segments = prefix.split("/")
    if any(not segment for segment in segments[1:]):  # segments[0] is "" (the leading '/')
        raise ValueError(f"prefix {prefix!r} must not contain an empty segment ('//' or a trailing '/')")
    if len(segments) < 3:  # "", user, job — at least two named segments
        raise ValueError(f"prefix {prefix!r} must name a job component after the user (e.g. '/{segments[1]}/exp-')")
    return prefix


def job_prefix_user(prefix: str) -> str:
    """Return the user segment of a validated job-name prefix."""
    return prefix.lstrip("/").split("/", 1)[0]


def authorize_job_prefix(job_wire_name: str) -> None:
    """Refuse the current identity when it is prefix-scoped and ``job_wire_name``
    falls outside its prefix.

    An identity with no prefix claim — unscoped user, admin, worker, task, or
    federation peer — passes unchanged. The prefix claim is validated at token
    verify time, so a scoped identity always carries a well-formed prefix.
    """
    identity = get_verified_identity()
    if identity is None or not identity.prefix:
        return
    if not job_wire_name.startswith(identity.prefix):
        raise ConnectError(
            Code.PERMISSION_DENIED,
            f"Token is scoped to jobs under '{identity.prefix}'; '{job_wire_name}' is outside that prefix",
        )
