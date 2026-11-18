# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exchange broker-private identities for lease-bounded native S3 keys."""

import math
import time
from datetime import datetime

import requests
from rigging.grants import Grant, GrantDenied
from rigging.openbao import OpenBaoStorageIssuer

OIDC_EXCHANGE_URL = "https://api.coreweave.com/v1/cwobject/temporary-credentials/oidc"


def exchange_credentials(identity: OpenBaoStorageIssuer, grant: Grant, *, organization: str, max_ttl: int) -> dict:
    """Keep the provider identity private and reject keys exceeding the lease."""
    token = identity.issue(grant)["token"]
    started = time.time()
    deadline = min(grant.credential_expires_at, started + max_ttl)
    duration = math.floor(deadline - started) - 2
    if duration <= 0:
        raise GrantDenied("Storage grant has expired")
    response = requests.post(
        OIDC_EXCHANGE_URL,
        json={"durationSeconds": duration, "orgId": organization, "oidcToken": token},
        # The provider's edge rejects the default Python requests user agent.
        headers={"User-Agent": "marin-auth/1"},
        timeout=10,
        allow_redirects=False,
    )
    if response.status_code != 200:
        # Provider error bodies can contain the identity or credential material.
        raise GrantDenied("CoreWeave credential exchange failed")
    result = response.json()
    expiration = datetime.fromisoformat(result["expiry"].replace("Z", "+00:00"))
    if expiration.tzinfo is None or not time.time() < expiration.timestamp() <= deadline:
        raise GrantDenied("CoreWeave credential lifetime exceeds the active grant")
    if not result["accessKeyId"] or not result["secretKey"]:
        raise GrantDenied("CoreWeave returned incomplete credentials")
    return {
        "AccessKeyId": result["accessKeyId"],
        "SecretAccessKey": result["secretKey"],
        "Token": "",
        "Expiration": result["expiry"],
    }
