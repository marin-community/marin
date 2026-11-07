# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check logical PostgreSQL restore and isolated OpenBao Raft snapshot recovery.

These drills preserve the source deployment. The PostgreSQL destination must
be a dedicated *-recovery database on the synthetic staging cluster. Backups
remain private local artifacts; this does not establish off-cluster recovery.
"""

import argparse
import hashlib
import json
import os
import ssl
import subprocess
import tempfile
import urllib.request
from contextlib import closing
from pathlib import Path

import pg8000.dbapi
import requests
from rigging.timing import Duration, ExponentialBackoff
from sqlalchemy.engine import URL, make_url

from infra.auth.bootstrap import OperatorClient
from infra.auth.prove import isolated_openbao
from infra.auth.staging import DATABASE_HOST, credential, private_directory, private_file


def database_digest(url: URL, ca_file: Path) -> dict:
    with closing(
        pg8000.dbapi.connect(
            host=url.host,
            port=url.port or 5432,
            user=url.username,
            password=url.password,
            database=url.database,
            ssl_context=ssl.create_default_context(cafile=str(ca_file)),
            timeout=30,
        )
    ) as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT ssl FROM pg_stat_ssl WHERE pid=pg_backend_pid()")
        (connection_security,) = cursor.fetchall()
        assert connection_security[0] is True
        cursor.execute(
            "SELECT schemaname, tablename FROM pg_tables "
            "WHERE schemaname NOT IN ('pg_catalog','information_schema') ORDER BY 1,2"
        )
        tables = cursor.fetchall()
        digest = {}
        for schema, table in tables:
            identifier = ".".join('"' + name.replace('"', '""') + '"' for name in (schema, table))
            cursor.execute(f"SELECT row_to_json(t)::text FROM {identifier} t ORDER BY row_to_json(t)::text")
            rows = cursor.fetchall()
            value = hashlib.sha256()
            for row in rows:
                value.update(row[0].encode() + b"\n")
            digest[f"{schema}.{table}"] = {"rows": len(rows), "sha256": value.hexdigest()}
        return digest


def database_restore(directory: Path, source_file: Path, ca_file: Path, postgres_bin: Path) -> dict:
    """Compare a quiescent synthetic source to a restored logical backup."""
    source = make_url(source_file.read_text().strip())
    if source.host != DATABASE_HOST or source.database not in {"openfga", "grants"}:
        raise ValueError("Restore drills are restricted to the synthetic staging databases")
    target = source.set(database=f"{source.database}-recovery")
    before = database_digest(source, ca_file)
    if not before or not any(table["rows"] for table in before.values()):
        raise ValueError("Seed meaningful synthetic state before taking a database backup")
    backup = directory / f"{source.database}.dump"
    private_file(backup, b"")
    environment = {
        **os.environ,
        "PGHOST": source.host,
        "PGPORT": str(source.port or 5432),
        "PGUSER": source.username,
        "PGPASSWORD": source.password,
        "PGSSLMODE": "verify-full",
        "PGSSLROOTCERT": str(ca_file),
    }
    # Passwords stay in the child environment, never process arguments or output.
    for command in (
        [
            str(postgres_bin / "pg_dump"),
            "--format=custom",
            "--no-owner",
            "--no-acl",
            "--file",
            str(backup),
            source.database,
        ],
        [
            str(postgres_bin / "pg_restore"),
            "--exit-on-error",
            "--single-transaction",
            "--clean",
            "--if-exists",
            "--no-owner",
            "--no-acl",
            "--dbname",
            target.database,
            str(backup),
        ],
    ):
        result = subprocess.run(command, env=environment, capture_output=True, timeout=180)
        if result.returncode:
            raise RuntimeError(f"{Path(command[0]).name} failed; private subprocess output withheld")
    after = database_digest(target, ca_file)
    assert before == after, "Restored rows differ; keep synthetic source writes paused during the drill"
    return {
        "source": source.database,
        "recovery": target.database,
        "verified_tls": True,
        "tables": before,
        "backup_sha256": hashlib.sha256(backup.read_bytes()).hexdigest(),
        "limit": "quiescent logical same-cluster restore; no concurrent-write, off-cluster DR or PITR guarantee",
    }


def bao_snapshot(directory: Path, address: str) -> dict:
    initialized = json.loads(credential(directory, "initialization.json"))
    source = OperatorClient(address, initialized["root_token"], str(directory / "ca.crt"))
    identity = json.loads(credential(directory, "identity.json"))
    expected_key = source.request("GET", "transit/keys/marin-grants")["data"]["keys"]
    source.request("GET", f"identity/entity/id/{identity['subject']}")
    request = urllib.request.Request(
        source.address + "/v1/sys/storage/raft/snapshot", headers={"X-Vault-Token": source.token}
    )
    with source.opener.open(request, timeout=60) as response:
        snapshot = response.read()
    private_file(directory / "openbao.snap", snapshot)
    private_file(
        directory / "openbao-recovery.json",
        json.dumps(
            {
                "keys": expected_key,
                "subject": identity["subject"],
            }
        ).encode(),
    )
    return {"snapshot_sha256": hashlib.sha256(snapshot).hexdigest()}


def bao_restore(directory: Path, executable: Path) -> dict:
    # A restored Raft peer map must have no route to the source cluster.
    if {path.name for path in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise ValueError("Run the restore command inside Docker --network=none")
    initialized = json.loads(credential(directory, "initialization.json"))
    expected = json.loads(credential(directory, "openbao-recovery.json"))
    snapshot = (directory / "openbao.snap").read_bytes()
    with tempfile.TemporaryDirectory(prefix="restore-", dir=directory) as scratch:
        with isolated_openbao(executable, Path(scratch)) as (local, certificate):
            request = urllib.request.Request(
                local.address + "/v1/sys/storage/raft/snapshot-force",
                data=snapshot,
                method="POST",
                headers={"X-Vault-Token": local.token, "Content-Type": "application/octet-stream"},
            )
            with local.opener.open(request, timeout=60) as response:
                response.read()
            ExponentialBackoff(initial=0.1, maximum=1).wait_until_or_raise(
                lambda: local.request("GET", "sys/seal-status")["sealed"],
                Duration.from_seconds(30),
                "Restored OpenBao did not seal before recovery",
            )
            local.request("POST", "sys/unseal", {"key": initialized["keys_base64"][0]})
            restored = OperatorClient(local.address, initialized["root_token"], str(certificate))
            ExponentialBackoff(initial=0.1, maximum=1).wait_until_or_raise(
                lambda: requests.get(
                    restored.address + "/v1/sys/health", verify=str(certificate), timeout=3, allow_redirects=False
                ).status_code
                == 200,
                Duration.from_seconds(30),
                "Restored OpenBao did not elect a local leader",
            )
            assert restored.request("GET", "transit/keys/marin-grants")["data"]["keys"] == expected["keys"]
            assert (
                restored.request("GET", f"identity/entity/id/{expected['subject']}")["data"]["id"] == expected["subject"]
            )
            login = restored.request(
                "POST",
                "auth/synthetic/login/launcher",
                {
                    "password": credential(directory, "launcher.password"),
                },
            )["auth"]
            assert login["entity_id"] == expected["subject"]
    return {
        "snapshot_sha256": hashlib.sha256(snapshot).hexdigest(),
        "signing_keys_retained": True,
        "synthetic_identity_retained": True,
        "fixed_password_login": True,
        "limit": "isolated single-process restore; no production seal custody or off-cluster DR guarantee",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    database = subparsers.add_parser("database")
    database.add_argument("--url-file", type=Path, required=True)
    database.add_argument("--ca-file", type=Path, required=True)
    database.add_argument("--postgres-bin", type=Path, required=True)
    snapshot = subparsers.add_parser("openbao-snapshot")
    snapshot.add_argument("--address", default="https://localhost:18200")
    bao = subparsers.add_parser("openbao")
    bao.add_argument("--bao", type=Path, required=True)
    args = parser.parse_args()
    directory = private_directory(args.directory)
    if args.command == "database":
        evidence = database_restore(directory, args.url_file, args.ca_file, args.postgres_bin)
    elif args.command == "openbao-snapshot":
        evidence = bao_snapshot(directory, args.address)
    else:
        evidence = bao_restore(directory, args.bao)
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
