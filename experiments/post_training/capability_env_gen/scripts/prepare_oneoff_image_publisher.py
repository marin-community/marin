#!/usr/bin/env python3
"""Prepare a reviewed, credential-free one-off Kubernetes publisher Job."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path

from capability_pipeline.image_publication_handoff import unpack_handoff

SCHEMA = "capability-oneoff-image-publisher-preparation-v1"
_NAME = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _source(root: Path, archive: Path) -> str:
    if archive.exists() or archive.is_symlink():
        raise ValueError("publisher source archive already exists")
    files = sorted((root / "capability_pipeline").glob("*.py")) + [
        root / "scripts/capture_task_images.py",
        root / "scripts/publish_generic_task_image.py",
        root / "scripts/image_publication_handoff.py",
        root / "scripts/run_oneoff_image_publisher.py",
    ]
    if any(path.is_symlink() or not path.is_file() for path in files):
        raise ValueError("trusted publisher source is incomplete or linked")
    with tarfile.open(archive, "x:gz") as output:
        for path in files:
            data = path.read_bytes()
            name = path.relative_to(root).as_posix()
            entry = tarfile.TarInfo(name)
            entry.size, entry.mode, entry.mtime = len(data), 0o444, 0
            output.addfile(entry, io.BytesIO(data))
        for name in ("scripts/__init__.py",):
            entry = tarfile.TarInfo(name)
            entry.size, entry.mode, entry.mtime = 0, 0o444, 0
            output.addfile(entry, io.BytesIO(b""))
    return _sha(archive)


def _s3(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://") or "/" not in uri[5:]:
        raise ValueError("publisher S3 prefix is invalid")
    bucket, key = uri[5:].split("/", 1)
    if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", bucket)
            or not re.fullmatch(r"[A-Za-z0-9_./-]+", key)
            or any(part in {"", ".", ".."} for part in key.split("/"))):
        raise ValueError("publisher S3 prefix is unsafe")
    return bucket, key


def _s3_env() -> list[dict]:
    return [
        {"name": "AWS_ACCESS_KEY_ID", "valueFrom": {"secretKeyRef": {"name": "cw-s3", "key": "accesskey"}}},
        {"name": "AWS_SECRET_ACCESS_KEY", "valueFrom": {"secretKeyRef": {"name": "cw-s3", "key": "secretkey"}}},
        {"name": "AWS_DEFAULT_REGION", "value": "auto"},
    ]


def prepare(packet_archive: Path, output: Path, source_root: Path, s3_prefix: str,
            job_name: str, registry_secret: str, max_rootfs_bytes: int) -> dict:
    """Write local artifacts only; operator uploads archives and applies the Job."""
    if output.exists() or output.is_symlink():
        raise ValueError("publisher preparation output must be new")
    if not _NAME.fullmatch(job_name) or not _NAME.fullmatch(registry_secret):
        raise ValueError("publisher Kubernetes resource name is invalid")
    if type(max_rootfs_bytes) is not int or max_rootfs_bytes <= 0:
        raise ValueError("publisher rootfs limit must be positive")
    bucket, key = _s3(s3_prefix.rstrip("/"))
    if packet_archive.is_symlink() or not packet_archive.is_file():
        raise ValueError("publisher handoff archive is absent or linked")
    from tempfile import TemporaryDirectory

    with TemporaryDirectory(prefix="publisher-packet-check-") as temporary:
        packet = Path(temporary) / "packet"
        unpack_handoff(packet_archive, packet)
        manifest = json.loads((packet / "manifest.json").read_text())
        if not manifest["roles"] or any(role not in {"candidate", "private_verifier"} for role in manifest["roles"]):
            raise ValueError("publisher packet roles are invalid")
        packet_manifest_sha = _sha(packet / "manifest.json")
    output.mkdir(parents=True)
    source_archive = output / "publisher-source.tar.gz"
    source_sha = _source(source_root, source_archive)
    packet_sha = _sha(packet_archive)
    packet_uri = f"s3://{bucket}/{key}/inputs/{packet_sha}/handoff.tar.gz"
    source_uri = f"s3://{bucket}/{key}/inputs/{source_sha}/publisher-source.tar.gz"
    return_prefix = f"s3://{bucket}/{key}/returns/{job_name}"
    fetch = ("set -eu; aws configure set default.s3.addressing_style virtual; "
             f"aws --endpoint-url https://cwobject.com s3 cp {packet_uri} /work/handoff.tar.gz; "
             f"aws --endpoint-url https://cwobject.com s3 cp {source_uri} /work/source.tar.gz; "
             f"printf '%s  %s\\n' {packet_sha} /work/handoff.tar.gz | sha256sum -c -; "
             f"printf '%s  %s\\n' {source_sha} /work/source.tar.gz | sha256sum -c -; "
             "tar -xzf /work/source.tar.gz -C /work/controller")
    publish = ("set -eu; umask 077; "
               "cp /run/secrets/capability-registry-publisher.json /tmp/publisher-registry.json; "
               "chmod 0400 /tmp/publisher-registry.json; "
               "trap 'rm -f /tmp/publisher-registry.json' EXIT; "
               "python3 -m pip install --disable-pip-version-check --no-cache-dir botocore==1.41.5 >/dev/null; "
               "export PYTHONPATH=/work/controller; "
               "python3 /work/controller/scripts/run_oneoff_image_publisher.py "
               f"/work/handoff.tar.gz {packet_sha} /work/source.tar.gz {source_sha} "
               f"/work/results {return_prefix} {max_rootfs_bytes} "
               "/tmp/publisher-registry.json")
    job = {
        "apiVersion": "batch/v1", "kind": "Job",
        "metadata": {"name": job_name, "namespace": "envreg", "labels": {"app.kubernetes.io/part-of": "capability-env-gen"}},
        "spec": {"backoffLimit": 0, "template": {"spec": {
            "restartPolicy": "Never", "volumes": [
                {"name": "work", "emptyDir": {"sizeLimit": "64Gi"}},
                {"name": "registry-credentials", "secret": {"secretName": registry_secret, "defaultMode": 256}},
            ],
            "initContainers": [{"name": "fetch-reviewed-input", "image": "amazon/aws-cli:2.31.26",
                                "command": ["/bin/sh", "-ec", "mkdir -p /work/controller; " + fetch],
                                "env": _s3_env(), "volumeMounts": [{"name": "work", "mountPath": "/work"}]}],
            "containers": [{"name": "publisher", "image": "python:3.12.11-slim-bookworm",
                            "command": ["/bin/sh", "-ec", publish], "env": _s3_env(),
                            "resources": {"requests": {"cpu": "2", "memory": "4Gi", "ephemeral-storage": "4Gi"},
                                          "limits": {"cpu": "4", "memory": "8Gi", "ephemeral-storage": "64Gi"}},
                            "volumeMounts": [{"name": "work", "mountPath": "/work"},
                                             {"name": "registry-credentials", "mountPath": "/run/secrets", "readOnly": True}]}],
        }}},
    }
    (output / "job.json").write_text(json.dumps(job, sort_keys=True, indent=2) + "\n")
    receipt = {"schema_version": SCHEMA, "job_name": job_name, "namespace": "envreg",
               "packet_archive_sha256": packet_sha, "packet_manifest_sha256": packet_manifest_sha,
               "source_archive_sha256": source_sha, "packet_uri": packet_uri,
               "source_uri": source_uri, "return_prefix": return_prefix,
               "roles": manifest["roles"], "max_rootfs_bytes": max_rootfs_bytes,
               "job_sha256": _sha(output / "job.json")}
    (output / "preparation.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--s3-prefix", required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--registry-secret", required=True)
    parser.add_argument("--max-rootfs-bytes", type=int, required=True)
    args = parser.parse_args()
    value = prepare(args.handoff, args.output, args.source_root, args.s3_prefix,
                    args.job_name, args.registry_secret, args.max_rootfs_bytes)
    print(json.dumps({"state": "prepared", "job": value["job_name"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # noqa: BLE001 - no provider or secret values in errors.
        print(json.dumps({"state": "failed", "error_type": type(error).__name__}))
        raise SystemExit(1) from None
