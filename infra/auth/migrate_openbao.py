# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare and run the one-time offline synthetic staging Raft-to-PostgreSQL copy."""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit

from rigging.timing import Duration, ExponentialBackoff

from infra.auth.bootstrap import OperatorClient
from infra.auth.proof_pod import NAMESPACE
from infra.auth.recover_staging import bao_snapshot
from infra.auth.staging import DATABASE_HOST, DATABASE_NAMESPACE, credential, kubectl, private_directory, private_file

JOB_NAME = "openbao-raft-to-postgres"
SOURCE_FILE = "openbao-migration.json"
LABELS = {"app.kubernetes.io/name": "openbao"}
# Kubernetes unescapes double dollar signs in args before the container starts.
EMPTY_DATABASE_SQL = """DO '
DECLARE item record; populated boolean;
BEGIN
  FOR item IN SELECT schemaname, tablename FROM pg_tables
    WHERE schemaname NOT IN (''pg_catalog'', ''information_schema'')
  LOOP
    EXECUTE format(''SELECT EXISTS (SELECT FROM %I.%I)'', item.schemaname, item.tablename) INTO populated;
    IF populated THEN RAISE EXCEPTION ''Migration destination contains data''; END IF;
  END LOOP;
END';"""


@dataclass(frozen=True)
class MigrationSource:
    leader_pod: str
    pvc: str
    pvc_uid: str
    raft_path: str
    image: str
    postgres_image: str
    snapshot_sha256: str


def prepare(directory: Path, leader_pod: str, address: str) -> dict:
    """Capture the selected active peer's retained volume and recovery snapshot."""
    initialized = json.loads(credential(directory, "initialization.json"))
    client = OperatorClient(address, initialized["root_token"], str(directory / "ca.crt"))
    pod = json.loads(kubectl(["-n", NAMESPACE, "get", "pod", leader_pod, "-o", "json"]))
    leader = client.request("GET", "sys/leader")
    if not leader["is_self"] or urlsplit(leader["leader_cluster_address"]).hostname != pod["status"]["podIP"]:
        raise ValueError("Select and port-forward the current active OpenBao leader")
    if client.request("GET", "sys/health")["version"] not in {"2.7.1", "2.7.2"}:
        raise ValueError("The staging migration requires a tested OpenBao 2.7.1 or 2.7.2 source")
    container = next(container for container in pod["spec"]["containers"] if container["name"] == "openbao")
    image = container["image"]
    if not image.startswith("ghcr.io/openbao/openbao@sha256:"):
        raise ValueError("The selected leader must use the pinned official OpenBao image")
    volumes = {volume["name"]: volume for volume in pod["spec"]["volumes"]}
    config_volume = next(mount for mount in container["volumeMounts"] if mount["mountPath"] == "/openbao/config")
    config_name = volumes[config_volume["name"]]["configMap"]["name"]
    config = json.loads(kubectl(["-n", NAMESPACE, "get", "configmap", config_name, "-o", "json"]))
    raft_path = json.loads(config["data"]["config.json"])["storage"]["raft"]["path"]
    mount = next(mount for mount in container["volumeMounts"] if mount["mountPath"] == raft_path)
    if "subPath" in mount or "subPathExpr" in mount:
        raise ValueError("The migration requires a dedicated Raft volume mount")
    pvc_name = volumes[mount["name"]]["persistentVolumeClaim"]["claimName"]
    pvc = json.loads(kubectl(["-n", NAMESPACE, "get", "pvc", pvc_name, "-o", "json"]))
    cluster = json.loads(kubectl(["-n", DATABASE_NAMESPACE, "get", "cluster", "auth-postgres", "-o", "json"]))
    postgres_image = cluster["spec"]["imageName"]
    if "@sha256:" not in postgres_image:
        raise ValueError("The database check requires the cluster's pinned PostgreSQL image")
    snapshot = bao_snapshot(directory, address)
    if not client.request("GET", "sys/leader")["is_self"]:
        raise ValueError("OpenBao leadership changed while capturing the migration source")
    source = MigrationSource(
        leader_pod, pvc_name, pvc["metadata"]["uid"], raft_path, image, postgres_image, snapshot["snapshot_sha256"]
    )
    private_file(directory / SOURCE_FILE, json.dumps(asdict(source)).encode())
    return asdict(source)


def migration_job(source: MigrationSource) -> dict:
    """Create a Job whose empty-destination check precedes the offline copy."""
    config = {
        "storage_source": {"raft": {"path": source.raft_path, "node_id": source.leader_pod}},
        "storage_destination": {"postgresql": {"ha_enabled": True}},
        "cluster_addr": "https://127.0.0.1:8201",
    }
    environment = [
        {"name": "PGHOST", "value": DATABASE_HOST},
        {"name": "PGDATABASE", "value": "openbao"},
        {"name": "PGSSLMODE", "value": "verify-full"},
        {"name": "PGSSLROOTCERT", "value": "/postgres-ca/ca.crt"},
        *(
            {"name": name, "valueFrom": {"secretKeyRef": {"name": "openbao-database", "key": key}}}
            for name, key in (("PGUSER", "username"), ("PGPASSWORD", "password"))
        ),
    ]
    mounts = [{"name": "work", "mountPath": "/work"}, {"name": "ca", "mountPath": "/postgres-ca", "readOnly": True}]
    security = {
        "allowPrivilegeEscalation": False,
        "readOnlyRootFilesystem": True,
        "capabilities": {"drop": ["ALL"]},
    }
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": JOB_NAME, "namespace": NAMESPACE},
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": 900,
            "template": {
                "metadata": {"labels": LABELS},
                "spec": {
                    "restartPolicy": "Never",
                    "serviceAccountName": "openbao",
                    "automountServiceAccountToken": False,
                    "securityContext": {
                        "runAsNonRoot": True,
                        "runAsUser": 1000,
                        "runAsGroup": 1000,
                        "fsGroup": 1000,
                        "seccompProfile": {"type": "RuntimeDefault"},
                    },
                    "initContainers": [
                        {
                            "name": "empty-destination",
                            "image": source.postgres_image,
                            "command": ["/bin/sh", "-ec"],
                            "args": [
                                'psql --no-psqlrc --set=ON_ERROR_STOP=1 --command "$1" > /work/check.log 2>&1 '
                                "|| { echo 'Destination check failed; private output withheld'; exit 1; }",
                                "empty-destination",
                                EMPTY_DATABASE_SQL,
                            ],
                            "env": environment,
                            "volumeMounts": mounts,
                            "securityContext": security,
                        }
                    ],
                    "containers": [
                        {
                            "name": "migrate",
                            "image": source.image,
                            "command": ["/bin/sh", "-ec"],
                            "args": [
                                "umask 077\ncat > /work/migrate.json <<'CONFIG'\n"
                                + json.dumps(config)
                                + "\nCONFIG\nbao operator migrate -config=/work/migrate.json > /work/migrate.log 2>&1 "
                                + "|| { echo 'Migration failed; private output withheld'; exit 1; }"
                            ],
                            "env": environment,
                            "volumeMounts": [*mounts, {"name": "source", "mountPath": source.raft_path}],
                            "securityContext": security,
                            "resources": {
                                "requests": {"cpu": "250m", "memory": "512Mi"},
                                "limits": {"cpu": "2", "memory": "2Gi"},
                            },
                        }
                    ],
                    "volumes": [
                        {"name": "source", "persistentVolumeClaim": {"claimName": source.pvc}},
                        {"name": "ca", "secret": {"secretName": "postgres-ca", "defaultMode": 0o440}},
                        {"name": "work", "emptyDir": {"medium": "Memory", "sizeLimit": "128Mi"}},
                    ],
                },
            },
        },
    }


def run(directory: Path) -> dict:
    """Copy the recorded source only after every OpenBao server has stopped."""
    source = MigrationSource(**json.loads(credential(directory, SOURCE_FILE)))
    stateful_set = json.loads(kubectl(["-n", NAMESPACE, "get", "statefulset", "openbao", "-o", "json"]))
    pods = json.loads(kubectl(["-n", NAMESPACE, "get", "pods", "-l", "app.kubernetes.io/name=openbao", "-o", "json"]))
    server_pods = [
        pod
        for pod in pods["items"]
        if not any(
            owner["kind"] == "Job" and owner["name"] == JOB_NAME for owner in pod["metadata"].get("ownerReferences", [])
        )
    ]
    if stateful_set["spec"]["replicas"] != 0 or server_pods:
        raise ValueError("Scale OpenBao to zero and wait for all server Pods to disappear before migration")
    pvc = json.loads(kubectl(["-n", NAMESPACE, "get", "pvc", source.pvc, "-o", "json"]))
    if pvc["metadata"]["uid"] != source.pvc_uid or pvc["status"]["phase"] != "Bound":
        raise ValueError("The recorded retained source PVC changed or is unbound")
    kubectl(["create", "-f", "-"], migration_job(source))

    def completed() -> bool:
        job = json.loads(kubectl(["-n", NAMESPACE, "get", "job", JOB_NAME, "-o", "json"]))
        conditions = {
            condition["type"]
            for condition in job.get("status", {}).get("conditions", [])
            if condition["status"] == "True"
        }
        if "Failed" in conditions:
            raise RuntimeError("OpenBao migration Job failed; inspect its retained private artifacts")
        return "Complete" in conditions

    ExponentialBackoff(initial=1, maximum=10).wait_until_or_raise(
        completed, Duration.from_seconds(930), "OpenBao migration Job did not complete"
    )
    return {"job": JOB_NAME, "source_pvc": source.pvc, "snapshot_sha256": source.snapshot_sha256, "completed": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("prepare")
    capture.add_argument("--leader-pod", required=True)
    capture.add_argument("--address", required=True)
    commands.add_parser("run")
    args = parser.parse_args()
    directory = private_directory(args.directory)
    result = prepare(directory, args.leader_pod, args.address) if args.command == "prepare" else run(directory)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
