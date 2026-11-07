# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Configure public OpenBao metadata and ACLs using an operator's ephemeral token."""

import argparse
import json
import os
import ssl
import urllib.error
import urllib.request
from urllib.parse import urlsplit, urlunsplit

GRANT_KEY = "marin-grants"
IDENTITY_KEY = "marin-coreweave"
IDENTITY_ROLES = {"marin-auth": "marin-auth", "marin-coreweave": "https://coreweave.com/iam"}
ACL_POLICIES = {
    "marin-auth-broker": (
        f"""path "transit/sign/{GRANT_KEY}" {{ capabilities = ["update"] }}
path "transit/keys/{GRANT_KEY}" {{ capabilities = ["read"] }}
"""
    ),
    "marin-auth-identity": (
        """path "identity/oidc/token/marin-auth" { capabilities = ["read"] }
"""
    ),
    "marin-coreweave-identity": (
        """path "identity/oidc/token/marin-coreweave" { capabilities = ["read"] }
"""
    ),
}


class RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class OperatorClient:
    """Authenticated API access; the token never becomes a Pulumi input or output."""

    def __init__(self, address: str, token: str, ca_file: str | None) -> None:
        parsed = urlsplit(address)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("The OpenBao operator API address must use HTTPS")
        self.address = address.rstrip("/")
        self.token = token
        self.ca_file = ca_file
        context = ssl.create_default_context(cafile=ca_file)
        self.opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=context), RejectRedirects())

    def request(self, method: str, path: str, payload: dict | None = None) -> dict:
        body = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            f"{self.address}/v1/{path}",
            data=body,
            method=method,
            headers={"X-Vault-Token": self.token, "Content-Type": "application/json"},
        )
        try:
            with self.opener.open(request, timeout=30) as response:
                data = response.read()
        except urllib.error.HTTPError as error:
            detail = error.read(4096).decode(errors="replace")
            error.add_note(f"OpenBao {method} {path} returned {error.code}: {detail}")
            raise
        return json.loads(data) if data else {}


def configure(client: OperatorClient, issuer: str) -> None:
    """Create non-exportable signing keys and separate broker and identity ACLs."""
    parsed = urlsplit(issuer)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path != "/v1/identity/oidc"
    ):
        raise ValueError("The identity issuer must be https://HOST/v1/identity/oidc")
    mounts = client.request("GET", "sys/mounts")["data"]
    if "transit/" not in mounts:
        client.request("POST", "sys/mounts/transit", {"type": "transit"})
    elif mounts["transit/"]["type"] != "transit":
        raise ValueError("The transit/ path is occupied by a different secrets engine")
    try:
        key = client.request("GET", f"transit/keys/{GRANT_KEY}")["data"]
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        client.request(
            "POST",
            f"transit/keys/{GRANT_KEY}",
            {"type": "ed25519", "exportable": False, "allow_plaintext_backup": False},
        )
    else:
        if key["type"] != "ed25519" or key["exportable"] or key["allow_plaintext_backup"]:
            raise ValueError("Existing grant signing key must be non-exportable ed25519 without plaintext backup")
    # OpenBao appends /v1/identity/oidc; its configuration API accepts only the origin.
    issuer_origin = urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
    client.request("POST", "identity/oidc/config", {"issuer": issuer_origin})
    client.request(
        "POST",
        f"identity/oidc/key/{IDENTITY_KEY}",
        {
            "algorithm": "RS256",
            "allowed_client_ids": list(IDENTITY_ROLES.values()),
            "rotation_period": "24h",
            "verification_ttl": "24h",
        },
    )
    for role, audience in IDENTITY_ROLES.items():
        client.request("POST", f"identity/oidc/role/{role}", {"key": IDENTITY_KEY, "client_id": audience, "ttl": "10m"})
    for name, policy in ACL_POLICIES.items():
        client.request("PUT", f"sys/policies/acl/{name}", {"policy": policy})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issuer", required=True)
    args = parser.parse_args()
    client = OperatorClient(os.environ["BAO_ADDR"], os.environ["BAO_TOKEN"], os.environ.get("BAO_CACERT"))
    configure(client, args.issuer)
    print("Configured signing keys, identity roles, and ACL policies; configure trusted auth methods separately.")


if __name__ == "__main__":
    main()
