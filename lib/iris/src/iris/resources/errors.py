# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Public failures raised by Iris resource operations."""


class ResourceError(Exception):
    """Base class for public resource operation failures."""


class InvalidResourceKey(ResourceError, ValueError):
    """A logical or exact resource identifier is malformed."""


class InvalidResourceRequest(ResourceError, ValueError):
    """A resource operation request is malformed."""


class InvalidPageToken(ResourceError, ValueError):
    """A page token is malformed or does not match its request."""


class ResourceNotFound(ResourceError):
    """The authorized logical or exact resource does not exist."""


class ResourceReplaced(ResourceError):
    """A logical resource now names a different exact incarnation."""


class ResourcePermissionDenied(ResourceError):
    """The caller is not authorized to perform the resource operation."""


class ResourcePreconditionFailed(ResourceError):
    """The resource state does not satisfy an operation precondition."""


class ResourceConflict(ResourceError):
    """The requested resource conflicts with retained state."""


class ResourceExhausted(ResourceError):
    """The resource operation exceeds an admitted limit."""


class BackendIdentityUnknown(ResourceError):
    """The exact execution backend cannot be resolved."""


class ActionPolicyRejected(ResourceError):
    """The target state or policy rejects an action."""


class ActionIdempotencyConflict(ResourceError):
    """An idempotency key was reused with a different action payload."""


class ResourceSourceUnavailable(ResourceError):
    """A required exact live source is unavailable."""


class UnsupportedResourceVerb(ResourceError):
    """The selected backend does not implement a resource operation."""


class UnsupportedResourceSchema(ResourceError):
    """The database schema is not an accepted resource schema."""


class AmbiguousResourceMigration(ResourceError):
    """Source rows have conflicting identities or coordinates."""
