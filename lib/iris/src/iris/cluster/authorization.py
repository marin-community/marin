# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resource authorization independent of RPC transport modules."""

from enum import StrEnum

from rigging.server_auth import VerifiedIdentity, require_identity

from iris.resources.errors import ResourcePermissionDenied

FEDERATION_PEER_ROLE = "federation-peer"
DASHBOARD_ROLE = "dashboard"


class AuthzAction(StrEnum):
    """Actions requiring authorization."""

    ACT_AS_WORKER = "act_as_worker"
    MANAGE_BUDGETS = "manage_budgets"
    SET_CONTAINER_PROFILE = "set_container_profile"


POLICY: dict[AuthzAction, frozenset[str]] = {
    AuthzAction.ACT_AS_WORKER: frozenset({"worker"}),
    AuthzAction.MANAGE_BUDGETS: frozenset(),
    AuthzAction.SET_CONTAINER_PROFILE: frozenset(),
}


def authorize(action: AuthzAction) -> VerifiedIdentity:
    """Require the current caller to have permission for an Iris action."""
    identity = require_identity()
    if identity.role == "admin":
        return identity
    if identity.role not in POLICY.get(action, frozenset()):
        raise ResourcePermissionDenied(f"{action} not allowed for role {identity.role}")
    return identity


def authorize_resource_owner(resource_owner: str) -> VerifiedIdentity:
    """Require the caller to own the resource or hold the admin role."""
    identity = require_identity()
    if identity.role == "admin":
        return identity
    if identity.user_id != resource_owner:
        raise ResourcePermissionDenied(f"User '{identity.user_id}' cannot access resources owned by '{resource_owner}'")
    return identity
