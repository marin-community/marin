# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CoreWeave temporary-credential exchanges. Provider requests never run under a DB lock."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from urllib.request import Request, urlopen

from rigging.timing import Timestamp
from rigging.token_authority import JwtSigner, signing_key_from_private_pem

from iris.cluster.config import StorageCredentialProfile

COREWEAVE_API = "https://api.coreweave.com/v1/cwobject"
OIDC_CREDENTIAL_LIFETIME = 1800
MINIMUM_PROVIDER_LIFETIME = 1740


@dataclass(frozen=True)
class StorageCredentials:
    access_key: str = field(repr=False)
    secret_key: str = field(repr=False)
    session_token: str = field(repr=False)
    expires_at: Timestamp


def _provider_request(path: str, *, token: str = "", body: dict | None = None) -> dict:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    request = Request(f"{COREWEAVE_API}/{path}", data=data, headers=headers)
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def exchange_credentials(profile: StorageCredentialProfile, principal: str) -> StorageCredentials:
    """Exchange a controller-only API token or OIDC assertion for expiring S3 keys.

    API-token mode retains the service principal's storage permissions. OIDC mode
    depends on separately provisioned issuer trust and bucket policies; neither
    exchange promises per-prefix downscoping.
    """
    if profile.api_token is not None:
        if not isinstance(profile.api_token, str):
            raise ValueError("CoreWeave API token must be resolved before exchanging credentials")
        response = _provider_request("temporary-credentials/api-token", token=profile.api_token)
        expiry = response["Expiration"]
        credentials = StorageCredentials(
            access_key=response["AccessKeyId"],
            secret_key=response["SecretAccessKey"],
            session_token=response["Token"],
            expires_at=_expiration(expiry),
        )
    elif profile.oidc is not None:
        oidc = profile.oidc
        if not isinstance(oidc.signing_key, str):
            raise ValueError("OIDC signing key must be resolved before exchanging credentials")
        signer = JwtSigner(signing_key_from_private_pem(oidc.signing_key), issuer=oidc.issuer)
        assertion = signer.mint({"sub": principal}, audience=oidc.audience, ttl_seconds=OIDC_CREDENTIAL_LIFETIME)
        response = _provider_request(
            "temporary-credentials/oidc",
            body={"durationSeconds": OIDC_CREDENTIAL_LIFETIME, "orgId": oidc.organization, "oidcToken": assertion},
        )
        credentials = StorageCredentials(
            access_key=response["accessKeyId"],
            secret_key=response["secretKey"],
            session_token="",
            expires_at=_expiration(response["expiry"]),
        )
    else:
        raise ValueError("A storage credential profile must select api_token or oidc")
    if credentials.expires_at.epoch_seconds() <= Timestamp.now().epoch_seconds() + MINIMUM_PROVIDER_LIFETIME:
        raise ValueError("CoreWeave credentials must last at least 30 minutes (with 60 seconds of exchange slack)")
    return credentials


def _expiration(value: str) -> Timestamp:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("CoreWeave credential expiry must include a timezone")
    return Timestamp.from_seconds(parsed.timestamp())
