# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Configure public OpenBao metadata and ACLs using an operator's ephemeral token."""

import argparse
import hashlib
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
GOOGLE_ISSUER = "https://accounts.google.com"
GOOGLE_MOUNT = "google"
SESSION_POLICY = 'path "auth/token/revoke-self" { capabilities = ["update"] }\n'


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
            error.add_note(f"OpenBao {method} {path} returned HTTP {error.code}")
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


def configure_google(
    client: OperatorClient,
    *,
    subject: str,
    audience: str,
    email: str,
    principal: str,
    hosted_domain: str | None = None,
    issuer: str = GOOGLE_ISSUER,
    issuer_ca_pem: str | None = None,
) -> dict:
    """Enroll an operator-reviewed Google subject with a stable Bao entity.

    The caller verifies the Google ID token before supplying these claims.
    Broker principal mapping and OpenFGA relationships are separate operator steps.
    """
    if not all((subject, audience, email, principal)):
        raise ValueError("Google enrollment requires subject, audience, email and operator-selected principal")
    mounts = client.request("GET", "sys/auth")["data"]
    if f"{GOOGLE_MOUNT}/" not in mounts:
        client.request("POST", f"sys/auth/{GOOGLE_MOUNT}", {"type": "jwt"})
        mounts = client.request("GET", "sys/auth")["data"]
    mount = mounts[f"{GOOGLE_MOUNT}/"]
    if mount["type"] != "jwt":
        raise ValueError("The google/ auth mount must use the JWT auth method")
    provider = {
        "oidc_discovery_url": issuer,
        "bound_issuer": issuer,
        "jwt_supported_algs": ["RS256"],
        "default_role": "",
    }
    if issuer_ca_pem is not None:
        provider["oidc_discovery_ca_pem"] = issuer_ca_pem
    client.request(
        "POST",
        f"auth/{GOOGLE_MOUNT}/config",
        provider,
    )
    entity_name = "google-" + hashlib.sha256(subject.encode()).hexdigest()
    try:
        entity = client.request("GET", f"identity/entity/name/{entity_name}")["data"]
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        entity = client.request("POST", "identity/entity", {"name": entity_name})["data"]
    if entity.get("policies") or entity.get("disabled", False):
        raise ValueError("The Google identity entity must be enabled and have no direct Bao policies")
    entity_id = entity["id"]
    # The alias includes the mount accessor, so recreating the mount preserves
    # the explicitly named entity and therefore its broker subject mapping.
    lookup = client.request(
        "POST", "identity/lookup/entity", {"alias_name": subject, "alias_mount_accessor": mount["accessor"]}
    )
    if lookup.get("data") is None:
        client.request(
            "POST",
            "identity/entity-alias",
            {"name": subject, "mount_accessor": mount["accessor"], "canonical_id": entity_id},
        )
    elif lookup["data"]["id"] != entity_id:
        raise ValueError("The Google subject alias already belongs to another Bao entity")
    role = "user-" + hashlib.sha256((audience + "\0" + subject).encode()).hexdigest()[:32]
    claims: dict = {"email": email, "email_verified": True}
    if hosted_domain is not None:
        claims["hd"] = hosted_domain
    client.request("PUT", "sys/policies/acl/marin-auth-session", {"policy": SESSION_POLICY})
    client.request(
        "POST",
        f"auth/{GOOGLE_MOUNT}/role/{role}",
        {
            "role_type": "jwt",
            "user_claim": "sub",
            "bound_subject": subject,
            "bound_audiences": [audience],
            "bound_claims": claims,
            "token_policies": ["marin-auth-identity", "marin-auth-session"],
            "token_no_default_policy": True,
            "token_ttl": "10m",
            "token_explicit_max_ttl": "10m",
            "expiration_leeway": -1,
            "clock_skew_leeway": -1,
            "not_before_leeway": -1,
        },
    )
    return {
        "role": role,
        "subject": subject,
        "audience": audience,
        "email": email,
        "hosted_domain": hosted_domain,
        "entity_id": entity_id,
        "principal": principal,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issuer", required=True)
    args = parser.parse_args()
    client = OperatorClient(os.environ["BAO_ADDR"], os.environ["BAO_TOKEN"], os.environ.get("BAO_CACERT"))
    configure(client, args.issuer)
    print("Configured signing keys, identity roles, and ACL policies; configure trusted auth methods separately.")


if __name__ == "__main__":
    main()
