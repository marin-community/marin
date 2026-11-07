# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise PostgreSQL failover through real OpenFGA and OpenBao processes.

Run only in the disposable CloudNativePG trial namespace. The proof's Kubernetes
role can modify this cluster and delete its three database pods, but cannot read
Secrets or operate another namespace. No production identities are enrolled.
"""

import json
import logging
import ssl
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, closing
from pathlib import Path
from urllib.parse import quote

import httpx
import pg8000.dbapi
from kubernetes import client as k8s
from kubernetes import config as kubeconfig
from rigging.grant_auth import JWT_TOKEN_TYPE, TOKEN_EXCHANGE_GRANT
from rigging.grants import Permission
from rigging.timing import Duration, ExponentialBackoff

from infra.auth.bootstrap import configure
from infra.auth.proof_fga import configure_proof, free_port, isolated_openfga
from infra.auth.proof_http import serve
from infra.auth.prove import create_proof_identity, isolated_openbao, proof_identity_token
from infra.auth.service import ServiceConfig, create_service

NAMESPACE = "marin-pg-trial"
CLUSTER = "auth-postgres"
HOST = f"{CLUSTER}-rw.{NAMESPACE}.svc"
MEMBERSHIP = {"user": "group:marin-team#member", "relation": "member", "object": "group:marin-research"}
logger = logging.getLogger(__name__)


def cluster_status(api: k8s.CustomObjectsApi) -> dict:
    return api.get_namespaced_custom_object("postgresql.cnpg.io", "v1", NAMESPACE, "clusters", CLUSTER)["status"]


def wait_healthy(api: k8s.CustomObjectsApi) -> None:
    ExponentialBackoff(initial=0.5, maximum=3).wait_until_or_raise(
        lambda: cluster_status(api).get("readyInstances") == 3,
        Duration.from_seconds(180),
        "The three PostgreSQL instances did not recover",
    )


def fence(api: k8s.CustomObjectsApi, instances: list[str]) -> None:
    api.patch_namespaced_custom_object(
        "postgresql.cnpg.io",
        "v1",
        NAMESPACE,
        "clusters",
        CLUSTER,
        {"metadata": {"annotations": {"cnpg.io/fencedInstances": json.dumps(instances)}}},
    )


def prove() -> dict:
    kubeconfig.load_incluster_config()
    api = k8s.CustomObjectsApi()
    pods = k8s.CoreV1Api()
    credentials = Path("/var/run/postgres/credentials")
    username = (credentials / "username").read_text()
    password = (credentials / "password").read_text()
    ca = "/var/run/postgres/ca/ca.crt"
    uri = (
        f"postgresql://{quote(username, safe='')}:{quote(password, safe='')}@{HOST}/authproof"
        f"?sslmode=verify-full&sslrootcert={ca}"
    )
    tls = ssl.create_default_context(cafile=ca)

    def sql(statement: str) -> list:
        with closing(
            pg8000.dbapi.connect(
                host=HOST,
                database="authproof",
                user=username,
                password=password,
                ssl_context=tls,
                application_name="marin-pg-durability",
                timeout=60,
            )
        ) as connection:
            cursor = connection.cursor()
            cursor.execute(statement)
            rows = cursor.fetchall() if cursor.description else []
            connection.commit()
            return rows

    wait_healthy(api)
    assert sql("SELECT ssl FROM pg_stat_ssl WHERE pid=pg_backend_pid()")[0][0] is True
    assert sql("SHOW synchronous_commit")[0][0] == "on"
    assert sql("SELECT count(*) FROM pg_stat_replication")[0][0] == 2
    assert "ANY 1" in sql("SHOW synchronous_standby_names")[0][0]
    evidence = {"storage": "shared-vast; independent disk failure not established", "checks": []}
    with tempfile.TemporaryDirectory(prefix="marin-pg-proof-") as scratch, ExitStack() as resources:
        directory = Path(scratch)
        operator, certificate = resources.enter_context(
            isolated_openbao(Path("/usr/local/bin/bao"), directory, database_uri=uri)
        )
        fga = resources.enter_context(isolated_openfga(Path("/usr/local/bin/openfga"), directory, database_uri=uri))
        fga_config = configure_proof(fga)
        issuer = operator.address + "/v1/identity/oidc"
        configure(operator, issuer)
        operator.request("POST", "sys/auth/proof", {"type": "userpass"})
        subject, identity_file = create_proof_identity(operator, "launcher", "marin-auth-identity", directory)
        broker_token = operator.request(
            "POST",
            "auth/token/create",
            {"policies": ["marin-auth-broker"], "no_default_policy": True, "ttl": "30m"},
        )["auth"]["client_token"]
        token_file = directory / "broker.token"
        token_file.touch(mode=0o600)
        token_file.write_text(broker_token)
        port = free_port()
        origin = f"http://127.0.0.1:{port}"
        app, authority = create_service(
            ServiceConfig(
                bao_address=operator.address,
                ca_bundle=str(certificate),
                bao_issuer=issuer,
                issuer=origin,
                broker_token_file=token_file,
                signing_key_version=1,
                database={"backend": "sqlite", "path": directory / "grants.db"},
                subjects={subject: "launcher"},
                openfga=fga_config,
                storage=(),
                max_token_ttl=900,
                loom=None,
            )
        )
        permission = Permission("iris", frozenset({"job.submit"}), "/clusters/isolated-proof/jobs/proof/")
        resources.enter_context(serve(app, port))
        http = resources.enter_context(httpx.Client(base_url=origin, timeout=20))
        identity = proof_identity_token(operator, identity_file, "marin-auth", certificate)
        response = http.post(
            "/v1/grants",
            headers={"Authorization": f"Bearer {identity}"},
            json={
                "actor": "launcher",
                "permissions": [{"service": "iris", "actions": ["job.submit"], "resource": permission.resource}],
                "expires_at": time.time() + 1200,
            },
        )
        assert response.status_code == 201, response.text
        grant = authority.store.active(response.json()["id"])
        response = http.post(
            "/oauth/token",
            data={
                "grant_type": TOKEN_EXCHANGE_GRANT,
                "subject_token_type": JWT_TOKEN_TYPE,
                "subject_token": identity,
                "audience": "iris",
                "grant_id": grant.id,
            },
        )
        assert response.status_code == 200, response.text
        token = response.json()["access_token"]

        def decision() -> int:
            return http.post(
                "/v1/authorize",
                headers={"Authorization": f"Bearer {token}"},
                json={"audience": "iris", "action": "job.submit", "resource": permission.resource},
            ).status_code

        assert decision() == 200
        # A successful tuple deletion must remain effective after database failover.
        fga.write(fga_config["store_id"], fga_config["model_id"], writes=[], deletes=[MEMBERSHIP])
        assert decision() == 403
        before = cluster_status(api)["currentPrimary"]
        started = time.monotonic()
        pods.delete_namespaced_pod(before, NAMESPACE, grace_period_seconds=0)
        transitions = []

        def denied_on_new_primary() -> bool:
            status = decision()
            assert status in {403, 503}, f"Revoked access returned HTTP {status} during failover"
            if not transitions or transitions[-1]["http_status"] != status:
                transitions.append({"after": round(time.monotonic() - started, 3), "http_status": status})
            return status == 403 and cluster_status(api).get("currentPrimary") not in {None, before}

        ExponentialBackoff(initial=0.5, maximum=2).wait_until_or_raise(
            denied_on_new_primary,
            Duration.from_seconds(120),
            "Authorization did not recover on a different primary",
        )
        recovered_after = time.monotonic() - started
        wait_healthy(api)
        assert decision() == 403
        fga.write(fga_config["store_id"], fga_config["model_id"], writes=[MEMBERSHIP], deletes=[])
        assert decision() == 200
        renewed = authority.mint_for(grant, audience="iris")["access_token"]
        assert (
            authority.authorize(renewed, audience="iris", action="job.submit", resource=permission.resource).id
            == grant.id
        )
        evidence["failover"] = {
            "old_primary": before,
            "new_primary": cluster_status(api)["currentPrimary"],
            "authorization_recovered_after": round(recovered_after, 3),
            "healthy_after": round(time.monotonic() - started, 3),
            "authorization_transitions": transitions,
        }
        evidence["checks"].append("acknowledged OpenFGA group removal survives primary loss; OpenBao signing recovers")
        logger.info("Primary failover retained the revocation: %s", evidence["failover"])

        # Fencing both standbys tests strict write durability without racing replacement pods.
        primary = cluster_status(api)["currentPrimary"]
        standbys = [f"{CLUSTER}-{index}" for index in range(1, 4) if f"{CLUSTER}-{index}" != primary]
        with ThreadPoolExecutor(max_workers=1) as writes:
            try:
                fence(api, standbys)
                ExponentialBackoff(initial=0.5, maximum=2).wait_until_or_raise(
                    lambda: sql("SELECT count(*) FROM pg_stat_replication")[0][0] == 0,
                    Duration.from_seconds(60),
                    "Standbys did not stop streaming",
                )
                probe = "durability_" + directory.name.replace("-", "_")
                pending = writes.submit(sql, f"CREATE TABLE {probe} (id integer)")
                ExponentialBackoff(initial=0.1, maximum=0.5).wait_until_or_raise(
                    lambda: sql(
                        "SELECT count(*) FROM pg_stat_activity "
                        "WHERE application_name='marin-pg-durability' AND wait_event='SyncRep'"
                    )[0][0]
                    == 1,
                    Duration.from_seconds(10),
                    "The write did not reach PostgreSQL's synchronous-replication wait",
                )
                try:
                    pending.result(timeout=3)
                except TimeoutError:
                    assert not pending.done(), "The write failed instead of waiting for replication"
                else:
                    raise AssertionError("A write was acknowledged without a synchronous standby")
                assert decision() == 200
                evidence["checks"].append(
                    "loss of both standbys blocks write acknowledgment while primary authorization reads work"
                )
            finally:
                fence(api, [])
            pending.result(timeout=90)
        wait_healthy(api)
        logger.info("Synchronous write blocked without standbys and completed after recovery")

        try:
            fence(api, [f"{CLUSTER}-{index}" for index in range(1, 4)])
            ExponentialBackoff(initial=0.5, maximum=2).wait_until_or_raise(
                lambda: decision() == 503,
                Duration.from_seconds(90),
                "A database outage did not fail authorization closed",
            )
            evidence["checks"].append("complete database outage returns 503 rather than allowing existing tokens")
        finally:
            fence(api, [])
        recovery_started = time.monotonic()
        wait_healthy(api)

        def authorization_recovered() -> bool:
            status = decision()
            assert status in {200, 503}, f"Restored access returned unexpected HTTP {status}"
            return status == 200

        ExponentialBackoff(initial=0.5, maximum=2).wait_until_or_raise(
            authorization_recovered,
            Duration.from_seconds(90),
            "Authorization did not recover after the database outage",
        )
        evidence["outage_recovery_after"] = round(time.monotonic() - recovery_started, 3)
        evidence["checks"].append("database recovery restores the existing grant without replacing the signing key")
    return evidence


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(json.dumps(prove(), indent=2))
