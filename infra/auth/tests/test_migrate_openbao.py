# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import ssl
import stat
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import DatabaseError

from infra.auth import migrate_openbao
from infra.auth.login import private_directory
from infra.auth.prove import _certificate
from infra.auth.recover_staging import bao_verify
from infra.auth.staging import private_file


@pytest.fixture
def migration_source(tmp_path, monkeypatch):
    directory = private_directory(tmp_path / "credentials")
    certificate, key = _certificate(directory)
    private_file(directory / "ca.crt", certificate.read_bytes())
    private_file(directory / "initialization.json", b'{"root_token":"operator"}')
    private_file(directory / "identity.json", b'{"subject":"retained-entity"}')
    private_file(directory / "launcher.password", b"fixed-password")
    state = {"keys": {"1": {"public_key": "retained-public-key"}}, "entity": "retained-entity", "version": "2.7.1"}
    snapshot = b"encrypted-raft-snapshot"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            responses = {
                "/v1/sys/leader": {"is_self": True, "leader_cluster_address": "https://10.0.0.2:8201"},
                "/v1/sys/health": {"version": state["version"], "initialized": True, "sealed": False},
                "/v1/sys/seal-status": {"storage_type": "postgresql"},
                "/v1/transit/keys/marin-grants": {"data": {"keys": state["keys"]}},
                "/v1/identity/entity/id/retained-entity": {"data": {"id": state["entity"]}},
            }
            body = snapshot if self.path.endswith("/snapshot") else json.dumps(responses[self.path]).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert self.path == "/v1/auth/synthetic/login/launcher" and payload == {"password": "fixed-password"}
            body = json.dumps({"auth": {"entity_id": state["entity"]}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    resources = {
        "pod": {
            "status": {"podIP": "10.0.0.2"},
            "spec": {
                "containers": [
                    {
                        "name": "openbao",
                        "image": "ghcr.io/openbao/openbao@sha256:" + "a" * 64,
                        "volumeMounts": [
                            {"name": "data", "mountPath": "/openbao/data"},
                            {"name": "config", "mountPath": "/openbao/config"},
                        ],
                    }
                ],
                "volumes": [
                    {"name": "data", "persistentVolumeClaim": {"claimName": "retained-peer-two"}},
                    {"name": "config", "configMap": {"name": "openbao-config"}},
                ],
            },
        },
        "configmap": {"data": {"config.json": json.dumps({"storage": {"raft": {"path": "/openbao/data"}}})}},
        "pvc": {"metadata": {"uid": "original-volume"}, "status": {"phase": "Bound"}},
        "cluster": {"spec": {"imageName": "ghcr.io/cloudnative-pg/postgresql:17@sha256:" + "b" * 64}},
        "statefulset": {"spec": {"replicas": 0}},
        "pods": {"items": []},
        "job": {"status": {"conditions": [{"type": "Complete", "status": "True"}]}},
    }
    created = []

    def kubectl(arguments, payload=None):
        if payload is not None:
            created.append(payload)
            return b"{}"
        return json.dumps(resources[arguments[3]]).encode()

    monkeypatch.setattr(migrate_openbao, "kubectl", kubectl)
    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls.load_cert_chain(certificate, key)
        server.socket = tls.wrap_socket(server.socket, server_side=True)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            yield directory, f"https://127.0.0.1:{server.server_port}", resources, created, state, snapshot
        finally:
            server.shutdown()
            thread.join()


@pytest.mark.parametrize("version", ["2.7.1", "2.7.2"])
def test_migration_preserves_selected_leader_snapshot_and_discovers_retained_volume(migration_source, version):
    directory, address, _, created, state, snapshot = migration_source
    state["version"] = version
    source = migrate_openbao.prepare(directory, "openbao-2", address)
    assert source["pvc"] == "retained-peer-two"
    assert source["pvc_uid"] == "original-volume"
    assert source["snapshot_sha256"] == hashlib.sha256(snapshot).hexdigest()
    assert (directory / "openbao.snap").read_bytes() == snapshot
    assert stat.S_IMODE((directory / migrate_openbao.SOURCE_FILE).stat().st_mode) == 0o600
    result = migrate_openbao.run(directory)
    assert result["completed"] is True
    assert len(created) == 1


@pytest.mark.parametrize("replicas,pods", [(3, []), (0, [{"metadata": {"deletionTimestamp": "draining"}}])])
def test_migration_never_creates_copy_job_while_servers_can_write(migration_source, replicas, pods):
    directory, address, resources, created, _, _ = migration_source
    migrate_openbao.prepare(directory, "openbao-2", address)
    resources["statefulset"]["spec"]["replicas"] = replicas
    resources["pods"]["items"] = pods
    with pytest.raises(ValueError, match="Scale OpenBao"):
        migrate_openbao.run(directory)
    assert created == []


def test_replaced_source_volume_is_rejected_before_copy(migration_source):
    directory, address, resources, created, _, _ = migration_source
    migrate_openbao.prepare(directory, "openbao-2", address)
    resources["pvc"]["metadata"]["uid"] = "replacement-volume"
    with pytest.raises(ValueError, match="source PVC changed"):
        migrate_openbao.run(directory)
    assert created == []


@pytest.mark.parametrize("changed", ["keys", "entity"])
def test_postgres_recovery_verification_rejects_lost_signing_or_identity_state(migration_source, changed):
    directory, address, _, _, state, _ = migration_source
    migrate_openbao.prepare(directory, "openbao-2", address)
    assert bao_verify(directory, address)["storage"] == "postgresql"
    state[changed] = {} if changed == "keys" else "replacement-entity"
    with pytest.raises(AssertionError):
        bao_verify(directory, address)


@pytest.mark.integration
def test_destination_check_after_kubernetes_argument_expansion(postgres_database_url):
    # The Kubelet expands $$ in command/args even when the container's shell
    # receives the SQL through a quoted positional parameter.
    sql = migrate_openbao.EMPTY_DATABASE_SQL.replace("$$", "$")
    engine = create_engine(postgres_database_url)
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql("CREATE TABLE migration_guard (value text)")
            connection.exec_driver_sql(sql)
            connection.exec_driver_sql("INSERT INTO migration_guard VALUES ('retained')")
            with pytest.raises(DatabaseError, match="destination contains data"):
                with connection.begin_nested():
                    connection.exec_driver_sql(sql)
            assert connection.exec_driver_sql("SELECT value FROM migration_guard").scalar_one() == "retained"
    finally:
        engine.dispose()
