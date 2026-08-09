# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resource authorization independent of RPC transport modules."""

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from rigging.server_auth import VerifiedIdentity, require_identity


def authorize_resource_owner(resource_owner: str) -> VerifiedIdentity:
    """Require the caller to own the resource or hold the admin role."""
    identity = require_identity()
    if identity.role == "admin":
        return identity
    if identity.user_id != resource_owner:
        raise ConnectError(
            Code.PERMISSION_DENIED,
            f"User '{identity.user_id}' cannot access resources owned by '{resource_owner}'",
        )
    return identity
