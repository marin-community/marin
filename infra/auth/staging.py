# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Out-of-band synthetic staging credentials, initialization, and policy publication.

All private values stay in a mode-0700 directory outside the checkout and in
Kubernetes Secrets. This command never passes credentials to Pulumi.
"""

import argparse
import base64
import datetime
import json
import os
import secrets
import stat
import subprocess
import urllib.error
from pathlib import Path
from urllib.parse import quote

import requests
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from rigging.timing import Duration, ExponentialBackoff

from infra.auth.bootstrap import OperatorClient, configure
from infra.auth.fga_operator import OpenFGAOperator
from infra.auth.openfga import scope_object
from infra.auth.proof_fga import configure_proof

CONTEXT = "marin_US-WEST-04A"
NAMESPACE = "marin-auth-staging"
DATABASE_NAMESPACE = "marin-auth-db-staging"
DATABASE_HOST = f"auth-postgres-rw.{DATABASE_NAMESPACE}.svc"
ISSUER = f"https://openbao.{NAMESPACE}.svc:8200/v1/identity/oidc"
CHECKOUT = Path(__file__).resolve().parents[2]


def private_directory(directory: Path) -> Path:
    directory = directory.resolve()
    if directory.is_relative_to(CHECKOUT):
        raise ValueError("Staging credentials must live outside the repository")
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if stat.S_IMODE(directory.stat().st_mode) != 0o700:
        raise ValueError("The staging credential directory must have mode 0700")
    return directory


def private_file(path: Path, value: bytes) -> None:
    """Create once, so a repeated prepare preserves deployment identities."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(value)


def credential(directory: Path, name: str) -> str:
    path = directory / name
    if path.is_symlink() or stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise ValueError(f"Private credential file permissions are invalid: {name}")
    return path.read_text().strip()


def prepare(directory: Path) -> None:
    for name in (
        "launcher.password",
        "iris-controller.password",
        "openfga.token",
        "openfga.password",
        "grants.password",
    ):
        if not (directory / name).exists():
            private_file(directory / name, secrets.token_urlsafe(40).encode())
        credential(directory, name)
    now = datetime.datetime.now(datetime.UTC)
    if (directory / "ca.key").exists():
        ca_key = serialization.load_pem_private_key(credential(directory, "ca.key").encode(), password=None)
        assert isinstance(ca_key, rsa.RSAPrivateKey)
        ca = x509.load_pem_x509_certificate(credential(directory, "ca.crt").encode())
    else:
        ca_key = rsa.generate_private_key(public_exponent=65537, key_size=3072)
        ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Marin synthetic auth staging CA")])
        ca = (
            x509.CertificateBuilder()
            .subject_name(ca_name)
            .issuer_name(ca_name)
            .public_key(ca_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=5))
            .not_valid_after(now + datetime.timedelta(days=30))
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .sign(ca_key, hashes.SHA256())
        )
        private_file(
            directory / "ca.key",
            ca_key.private_bytes(
                serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
            ),
        )
        private_file(directory / "ca.crt", ca.public_bytes(serialization.Encoding.PEM))
    for service, names in (
        (
            "openbao",
            [
                f"openbao.{NAMESPACE}.svc",
                "localhost",
                *(f"openbao-{index}.openbao-internal.{NAMESPACE}.svc" for index in range(3)),
            ],
        ),
        ("openfga", [f"openfga.{NAMESPACE}.svc", "localhost"]),
        ("broker", [f"marin-auth.{NAMESPACE}.svc", "localhost"]),
    ):
        if (directory / f"{service}.key").exists():
            credential(directory, f"{service}.key")
            credential(directory, f"{service}.crt")
            continue
        leaf_certificate(directory, service, names, ca_key, ca, now)
    # Remove the superseded shared leaf only after all isolated leaves exist.
    for name in ("tls.key", "tls.crt"):
        (directory / name).unlink(missing_ok=True)


def leaf_certificate(
    directory: Path,
    service: str,
    names: list[str],
    ca_key: rsa.RSAPrivateKey,
    ca: x509.Certificate,
    now: datetime.datetime,
) -> None:
    key = rsa.generate_private_key(public_exponent=65537, key_size=3072)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, names[0])]))
        .issuer_name(ca.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=14))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName(name) for name in names]), critical=False)
        .sign(ca_key, hashes.SHA256())
    )
    private_file(
        directory / f"{service}.key",
        key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()),
    )
    private_file(directory / f"{service}.crt", certificate.public_bytes(serialization.Encoding.PEM))


def kubectl(arguments: list[str], payload: dict | None = None) -> bytes:
    result = subprocess.run(
        ["kubectl", "--context", CONTEXT, *arguments],
        input=None if payload is None else json.dumps(payload).encode(),
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"Kubernetes staging operation failed ({arguments[0]}); private output withheld")
    return result.stdout


def install_secret(namespace: str, name: str, data: dict[str, str]) -> None:
    kubectl(
        ["apply", "--server-side", "--field-manager=marin-auth-staging", "-f", "-"],
        {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {"name": name, "namespace": namespace},
            "type": "Opaque",
            "data": {key: base64.b64encode(value.encode()).decode() for key, value in data.items()},
        },
    )


def install_secrets(directory: Path) -> None:
    """Run after Pulumi has created both namespaces; repeat after DB is ready."""
    for service in ("openbao", "openfga", "broker"):
        install_secret(
            NAMESPACE,
            f"{service}-tls",
            {
                "tls.key": credential(directory, f"{service}.key") + "\n",
                "tls.crt": credential(directory, f"{service}.crt") + "\n",
                "ca.crt": credential(directory, "ca.crt") + "\n",
            },
        )
    install_database_secrets(directory)
    for name in ("openfga", "grants"):
        password = credential(directory, f"{name}.password")
        userinfo = f"{name}:{quote(password, safe='')}"
        if name == "openfga":
            install_secret(
                NAMESPACE,
                "openfga-database",
                {
                    "uri": (
                        f"postgresql://{userinfo}@{DATABASE_HOST}/openfga"
                        "?sslmode=verify-full&sslrootcert=/var/run/postgres/ca/ca.crt"
                    )
                },
            )
        else:
            install_secret(
                NAMESPACE, "grants-database", {"url": f"postgresql+pg8000://{userinfo}@{DATABASE_HOST}/grants"}
            )
    install_secret(NAMESPACE, "openfga-token", {"token": credential(directory, "openfga.token")})
    for username in ("launcher", "iris-controller"):
        install_secret(
            NAMESPACE,
            f"synthetic-{username}",
            {
                "username": username,
                "password": credential(directory, f"{username}.password"),
            },
        )
    if (directory / "broker.token").exists():
        install_secret(NAMESPACE, "broker-token", {"token": credential(directory, "broker.token")})


def install_database_secrets(directory: Path) -> None:
    for name in ("openfga", "grants"):
        install_secret(
            DATABASE_NAMESPACE,
            f"{name}-database",
            {
                "username": name,
                "password": credential(directory, f"{name}.password"),
            },
        )


def install_database_ca(directory: Path) -> None:
    secret = json.loads(kubectl(["-n", DATABASE_NAMESPACE, "get", "secret", "auth-postgres-ca", "-o", "json"]))
    database_ca = base64.b64decode(secret["data"]["ca.crt"]).decode()
    install_secret(NAMESPACE, "postgres-ca", {"ca.crt": database_ca})
    install_secret(NAMESPACE, "auth-ca", {"ca.crt": credential(directory, "ca.crt") + "\n" + database_ca})


def initialize(directory: Path, addresses: list[str]) -> dict:
    ca_file = str(directory / "ca.crt")
    initialization = directory / "initialization.json"
    leader = OperatorClient(addresses[0], "", ca_file)
    initialized = leader.request("GET", "sys/init")["initialized"]
    if not initialized:
        if initialization.exists():
            raise ValueError("Saved initialization exists but this OpenBao deployment is uninitialized")
        private_file(
            initialization,
            json.dumps(
                leader.request(
                    "POST",
                    "sys/init",
                    {
                        "secret_shares": 1,
                        "secret_threshold": 1,
                    },
                )
            ).encode(),
        )
    material = json.loads(credential(directory, "initialization.json"))
    for index, address in enumerate(addresses):
        client = OperatorClient(address, "", ca_file)
        ExponentialBackoff(initial=0.2, maximum=2).wait_until_or_raise(
            lambda client=client: unsealed_peer(client, material["keys_base64"][0]),
            Duration.from_seconds(90),
            "OpenBao did not complete its Raft join and unseal",
        )
        ExponentialBackoff(initial=0.2, maximum=2).wait_until_or_raise(
            lambda address=address, index=index: requests.get(
                address + "/v1/sys/health",
                params={} if not initialized and index == 0 else {"standbyok": "true"},
                verify=ca_file,
                timeout=3,
                allow_redirects=False,
            ).status_code
            == 200,
            Duration.from_seconds(90),
            "OpenBao did not become ready after unsealing",
        )
    operator = OperatorClient(addresses[0], material["root_token"], ca_file)
    configure(operator, ISSUER)
    mounts = operator.request("GET", "sys/auth")["data"]
    if "synthetic/" not in mounts:
        operator.request("POST", "sys/auth/synthetic", {"type": "userpass"})
    elif mounts["synthetic/"]["type"] != "userpass":
        raise ValueError("synthetic/ must contain only the staging userpass method")
    identities = {}
    for username, principal in (("launcher", "launcher"), ("iris-controller", "service:iris-staging")):
        operator.request(
            "POST",
            f"auth/synthetic/users/{username}",
            {
                "password": credential(directory, f"{username}.password"),
                "token_policies": ["marin-auth-identity"],
                "token_ttl": "1h",
                "token_max_ttl": "24h",
            },
        )
        login = operator.request(
            "POST",
            f"auth/synthetic/login/{username}",
            {
                "password": credential(directory, f"{username}.password"),
            },
        )["auth"]
        identities[username] = {"subject": login["entity_id"], "principal": principal, "issuer": ISSUER}
    if not (directory / "broker.token").exists():
        token = operator.request(
            "POST",
            "auth/token/create",
            {
                "policies": ["marin-auth-broker"],
                "no_default_policy": True,
                "ttl": "24h",
                "renewable": True,
            },
        )["auth"]["client_token"]
        private_file(directory / "broker.token", token.encode())
    identity = identities["launcher"]
    path = directory / "identity.json"
    if not path.exists():
        private_file(path, json.dumps(identity).encode())
    elif json.loads(credential(directory, "identity.json")) != identity:
        raise ValueError("The synthetic identity changed; inspect deployment recovery before proceeding")
    controller_path = directory / "iris-controller.json"
    if not controller_path.exists():
        private_file(controller_path, json.dumps(identities["iris-controller"]).encode())
    elif json.loads(credential(directory, "iris-controller.json")) != identities["iris-controller"]:
        raise ValueError("The synthetic controller identity changed; inspect deployment recovery before proceeding")
    return identities


def unsealed_peer(client: OperatorClient, key: str) -> bool:
    if not client.request("GET", "sys/seal-status")["sealed"]:
        return True
    try:
        return not client.request("POST", "sys/unseal", {"key": key})["sealed"]
    except urllib.error.HTTPError as error:
        # Shamir followers need the share to finish the Raft join challenge.
        # retry_join may still be finding the initialized leader on this pass.
        if error.code == 400 and not client.request("GET", "sys/init")["initialized"]:
            return False
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument(
        "command",
        choices=[
            "prepare",
            "install-secrets",
            "install-database-secrets",
            "install-database-ca",
            "initialize",
            "publish-policy",
        ],
    )
    parser.add_argument("--bao-address", action="append")
    parser.add_argument("--openfga-address", default="http://localhost:18080")
    args = parser.parse_args()
    directory = private_directory(args.directory)
    if args.command == "prepare":
        prepare(directory)
    elif args.command == "install-secrets":
        install_secrets(directory)
    elif args.command == "install-database-secrets":
        install_database_secrets(directory)
    elif args.command == "install-database-ca":
        install_database_ca(directory)
    elif args.command == "initialize":
        print(json.dumps(initialize(directory, args.bao_address or ["https://localhost:18200"])))
    else:
        path = directory / "openfga.json"
        operator = OpenFGAOperator(args.openfga_address, directory / "openfga.token")
        if not path.exists():
            policy = configure_proof(operator)
            policy["token_file"] = str(policy["token_file"])
            private_file(path, json.dumps(policy).encode())
        policy = json.loads(credential(directory, "openfga.json"))
        # Reconcile the dedicated store so repeating publication is idempotent.
        relationships = []
        continuation = ""
        while True:
            response = operator.request(
                f"stores/{policy['store_id']}/read",
                {
                    "page_size": 100,
                    "continuation_token": continuation,
                },
            )
            relationships.extend(row["key"] for row in response.get("tuples", []))
            continuation = response.get("continuation_token", "")
            if not continuation:
                break
        relationships.extend(
            {
                "user": "group:marin-research#member",
                "relation": "accessor",
                "object": scope_object(service, action, resource),
            }
            for service, action, resource in (
                ("loom", "session.create", "/task-proof/"),
                ("openbao", "secret.read", "/secrets/task-canary/"),
            )
        )
        operator.reconcile(policy["store_id"], policy["model_id"], relationships)
        print(json.dumps(policy))


if __name__ == "__main__":
    main()
