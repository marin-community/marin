# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Who is calling: one policy for every route, resolved from IAP or loopback.

Marina adds no permission vocabulary. A signed-in person can reach every app. The
kernel only needs to know who the caller is, for the identity chip and for attributing
what an agent later changes. In production IAP fronts the service and signs an assertion
header per request; the policy verifies it against the service's audience. On a
developer machine the request arrives over loopback and is admitted as the anonymous
admin.
"""

from rigging.server_auth import (
    IapAssertionVerifier,
    RequestAuthPolicy,
    VerifiedIdentity,
    extract_bearer_token,
    scope_client_address,
    scope_headers,
)
from starlette.requests import Request

ADMIN_ROLE = "admin"


def build_policy(iap_audience: str | None) -> RequestAuthPolicy:
    """The enforcing chain: IAP assertion when an audience is configured, then loopback."""
    verifier = IapAssertionVerifier(iap_audience, lambda _email: ADMIN_ROLE) if iap_audience else None
    return RequestAuthPolicy.enforcing(iap_assertion_verifier=verifier)


def identity_for(request: Request, policy: RequestAuthPolicy) -> VerifiedIdentity:
    """Resolve the identity of a request the route middleware already admitted."""
    headers = scope_headers(request.scope)
    return policy.resolve(
        extract_bearer_token(headers),
        client_address=scope_client_address(request.scope),
        headers=headers,
    )
