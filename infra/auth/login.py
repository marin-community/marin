# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Discover, enroll and use a Google identity for the shared grant broker."""

import argparse
import json
import os
import stat
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import google.auth.exceptions
import google.auth.transport.requests
import google.oauth2.id_token
import jwt
import requests
from rigging.auth import MARIN_DESKTOP_OAUTH_CLIENT, OAuthClient, read_desktop_client, run_iap_desktop_login
from rigging.openbao import OpenBaoIdentityVerifier
from rigging.shared_grants import authority_origin

from infra.auth.bootstrap import GOOGLE_MOUNT, OperatorClient, configure_google


@dataclass(frozen=True)
class GoogleIdentity:
    subject: str
    audience: str
    email: str
    hosted_domain: str | None


def verify_google_identity(
    token: str, *, audience: str, expected_email: str, hosted_domain: str | None = None
) -> GoogleIdentity:
    """Verify Google's signature, issuer, audience and operator-selected account."""
    try:
        claims = google.oauth2.id_token.verify_oauth2_token(
            token, google.auth.transport.requests.Request(), audience=audience
        )
    except (ValueError, google.auth.exceptions.GoogleAuthError):
        raise ValueError("Google identity verification failed") from None
    if claims.get("email_verified") is not True or claims.get("email") != expected_email:
        raise ValueError("Google did not verify the expected email account")
    if hosted_domain is not None and claims.get("hd") != hosted_domain:
        raise ValueError("Google did not verify the expected Workspace domain")
    if not claims.get("sub"):
        raise ValueError("Google returned no subject identifier")
    return GoogleIdentity(claims["sub"], audience, claims["email"], claims.get("hd"))


def private_directory(directory: Path) -> Path:
    directory = directory.resolve()
    checkout = Path(__file__).resolve().parents[2]
    if directory.is_relative_to(checkout):
        raise ValueError("Login credentials require a private directory outside the checkout")
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if stat.S_IMODE(directory.stat().st_mode) != 0o700:
        raise ValueError("The login directory must have mode 0700")
    return directory


def read_private(path: Path) -> str:
    if path.is_symlink() or stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise ValueError(f"Private file permissions must be 0600 for {path.name}")
    return path.read_text().strip()


def write_private(path: Path, value: str) -> None:
    """Atomically replace a credential while preserving owner-only permissions."""
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(value)
            stream.flush()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def discover(directory: Path, *, client: OAuthClient, expected_email: str, hosted_domain: str | None, headless: bool):
    token, _refresh_token = run_iap_desktop_login(client.client_id, client.client_secret, headless=headless)
    identity = verify_google_identity(
        token, audience=client.client_id, expected_email=expected_email, hosted_domain=hosted_domain
    )
    write_private(directory / "google.jwt", token)
    write_private(directory / "google-identity.json", json.dumps(asdict(identity)))
    return identity


def discovered_identity(
    directory: Path, *, audience: str, expected_email: str, hosted_domain: str | None
) -> GoogleIdentity:
    return verify_google_identity(
        read_private(directory / "google.jwt"),
        audience=audience,
        expected_email=expected_email,
        hosted_domain=hosted_domain,
    )


def enroll(
    directory: Path,
    operator: OperatorClient,
    *,
    principal: str,
    audience: str,
    expected_email: str,
    hosted_domain: str | None,
) -> dict:
    """Bind a verified subject using an operator credential and selected principal."""
    identity = discovered_identity(
        directory, audience=audience, expected_email=expected_email, hosted_domain=hosted_domain
    )
    enrollment = configure_google(operator, **asdict(identity), principal=principal)
    write_private(directory / "enrollment.json", json.dumps(enrollment))
    return {"subjects": {enrollment["entity_id"]: principal}, "role": enrollment["role"]}


def login(directory: Path, *, address: str, issuer: str, ca_file: str) -> dict:
    if (directory / "bao.token").exists():
        raise ValueError("Close the Bao session in this directory before logging in again")
    enrollment = json.loads(read_private(directory / "enrollment.json"))
    identity = discovered_identity(
        directory,
        audience=enrollment["audience"],
        expected_email=enrollment["email"],
        hosted_domain=enrollment["hosted_domain"],
    )
    if (identity.subject, identity.audience) != (enrollment["subject"], enrollment["audience"]):
        raise ValueError("The Google account differs from the operator's enrollment")
    anonymous = OperatorClient(address, "", ca_file)
    authenticated = anonymous.request(
        "POST",
        f"auth/{GOOGLE_MOUNT}/login",
        {"role": enrollment["role"], "jwt": read_private(directory / "google.jwt")},
    )["auth"]
    # Retain the session before subsequent checks so every partial login can
    # be closed using the same cleanup command.
    write_private(directory / "bao.token", authenticated["client_token"])
    session = OperatorClient(address, authenticated["client_token"], ca_file)
    if set(authenticated["policies"]) != {"marin-auth-identity", "marin-auth-session"} or authenticated.get(
        "identity_policies"
    ):
        session.request("POST", "auth/token/revoke-self")
        (directory / "bao.token").unlink()
        raise ValueError("The Bao session has authority beyond the enrolled identity and session policies")
    if authenticated["entity_id"] != enrollment["entity_id"]:
        raise ValueError("Bao authenticated a different entity from the operator's enrollment")
    token = session.request("GET", "identity/oidc/token/marin-auth")["data"]["token"]
    principal = OpenBaoIdentityVerifier(
        issuer=issuer,
        audience="marin-auth",
        subjects={enrollment["entity_id"]: enrollment["principal"]},
        ca_file=ca_file,
    ).verify(token)
    write_private(directory / "identity.jwt", token)
    return {
        "principal": principal,
        "entity_id": authenticated["entity_id"],
        "policies": authenticated["policies"],
        "expires_at": jwt.decode(token, options={"verify_signature": False})["exp"],
    }


def close_session(directory: Path, *, address: str, ca_file: str, auth_url: str | None, grants: list[str]) -> None:
    if grants:
        if auth_url is None:
            raise ValueError("Revoking root grants requires --auth-url")
        identity = read_private(directory / "identity.jwt")
        for grant in grants:
            if not grant or any(character not in "0123456789abcdef" for character in grant):
                raise ValueError("Invalid root grant ID")
            response = requests.post(
                authority_origin(auth_url) + f"/v1/grants/{grant}/revoke",
                headers={"Authorization": f"Bearer {identity}"},
                timeout=10,
                verify=ca_file,
                allow_redirects=False,
            )
            response.raise_for_status()
            if response.is_redirect or response.json().get("revoked") is not True:
                raise ValueError("The grant broker did not confirm revocation")
    session = OperatorClient(address, read_private(directory / "bao.token"), ca_file)
    session.request("POST", "auth/token/revoke-self")
    for name in ("bao.token", "identity.jwt", "google.jwt"):
        (directory / name).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    discovery = commands.add_parser("discover")
    discovery.add_argument("--email", required=True)
    discovery.add_argument("--hosted-domain")
    discovery.add_argument("--client-file")
    discovery.add_argument("--headless", action="store_true")
    enrollment = commands.add_parser("enroll")
    enrollment.add_argument("--principal", required=True)
    enrollment.add_argument("--email", required=True)
    enrollment.add_argument("--hosted-domain")
    enrollment.add_argument("--client-file")
    authenticated = commands.add_parser("login")
    authenticated.add_argument("--bao-issuer", required=True)
    cleanup = commands.add_parser("close-session")
    cleanup.add_argument("--auth-url")
    cleanup.add_argument("--grant-id", action="append", default=[])
    args = parser.parse_args()
    directory = private_directory(args.directory)
    if args.command == "discover":
        client = read_desktop_client(args.client_file) if args.client_file else MARIN_DESKTOP_OAUTH_CLIENT
        discover(
            directory, client=client, expected_email=args.email, hosted_domain=args.hosted_domain, headless=args.headless
        )
        print("Verified Google identity saved in the private directory.")
    elif args.command == "enroll":
        operator = OperatorClient(os.environ["BAO_ADDR"], os.environ["BAO_TOKEN"], os.environ.get("BAO_CACERT"))
        client = read_desktop_client(args.client_file) if args.client_file else MARIN_DESKTOP_OAUTH_CLIENT
        print(
            json.dumps(
                enroll(
                    directory,
                    operator,
                    principal=args.principal,
                    audience=client.client_id,
                    expected_email=args.email,
                    hosted_domain=args.hosted_domain,
                )
            )
        )
    elif args.command == "login":
        print(
            json.dumps(
                login(
                    directory,
                    address=os.environ["BAO_ADDR"],
                    issuer=args.bao_issuer,
                    ca_file=os.environ["BAO_CACERT"],
                )
            )
        )
    else:
        close_session(
            directory,
            address=os.environ["BAO_ADDR"],
            ca_file=os.environ["BAO_CACERT"],
            auth_url=args.auth_url,
            grants=args.grant_id,
        )
        print("Requested root grants revoked and local Bao session closed.")


if __name__ == "__main__":
    main()
