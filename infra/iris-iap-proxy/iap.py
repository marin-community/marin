# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""IAP (Identity-Aware Proxy) JWT validation.

Validates the ``X-Goog-IAP-JWT-Assertion`` header that GCP IAP attaches to
every request it forwards.  This is defense-in-depth: IAP already validated
the token before the request reached GAE, but we re-verify to extract the
caller's email and to guard against misconfiguration where IAP is accidentally
disabled.
"""

import logging
import os

from google.auth.transport import requests as google_requests
from google.oauth2 import id_token

logger = logging.getLogger(__name__)

IAP_AUDIENCE = os.environ.get("IAP_AUDIENCE", "")

_IAP_HEADER = "x-goog-iap-jwt-assertion"


class IapValidationError(Exception):
    """Raised when IAP JWT validation fails."""


def validate_iap_jwt(headers: dict[str, str]) -> str:
    """Validate the IAP JWT and return the verified email.

    Args:
        headers: Request headers (lowercase keys).

    Returns:
        The email address of the authenticated user.

    Raises:
        IapValidationError: If the JWT is missing or invalid.
    """
    jwt_assertion = headers.get(_IAP_HEADER)
    if not jwt_assertion:
        raise IapValidationError("Missing IAP JWT assertion header")

    if not IAP_AUDIENCE:
        raise IapValidationError("IAP_AUDIENCE environment variable not configured")

    try:
        decoded = id_token.verify_token(
            jwt_assertion,
            google_requests.Request(),
            audience=IAP_AUDIENCE,
        )
    except Exception as exc:
        raise IapValidationError(f"IAP JWT verification failed: {exc}") from exc

    email = decoded.get("email")
    if not email:
        raise IapValidationError("IAP JWT does not contain an email claim")

    return email
