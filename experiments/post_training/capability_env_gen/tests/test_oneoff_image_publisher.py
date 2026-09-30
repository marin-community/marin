import hashlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
from test_image_publication_handoff import _fixture

from capability_pipeline.image_publication_handoff import export_handoff
from scripts.cleanup_oneoff_image_publisher import cleanup
from scripts.fetch_oneoff_image_publisher import fetch
from scripts.prepare_oneoff_image_publisher import prepare
from scripts.upload_oneoff_image_publisher import upload

ROOT = Path(__file__).resolve().parents[1]


def test_preparation_keeps_registry_secret_out_of_init_and_artifacts(tmp_path):
    status, packet, *_ = _fixture(tmp_path)
    export_handoff(status, packet)
    output = tmp_path / "prepared"
    receipt = prepare(packet, output, ROOT, "s3://bucket/oneoff/task-001",
                      "cap-image-publisher-001", "cap-image-registry-001", 20 << 30)
    job = json.loads((output / "job.json").read_text())
    pod = job["spec"]["template"]["spec"]
    init, publisher = pod["initContainers"][0], pod["containers"][0]
    assert init["volumeMounts"] == [{"name": "work", "mountPath": "/work"}]
    assert {mount["name"] for mount in publisher["volumeMounts"]} == {"work", "registry-credentials"}
    assert pod["volumes"][1]["secret"]["secretName"] == "cap-image-registry-001"
    init_command = init["command"][2]
    assert init_command.index("sha256sum -c -") < init_command.index("tar -xzf")
    assert init_command.count("sha256sum -c -") == 2
    publisher_command = publisher["command"][2]
    for command in (init_command, publisher_command):
        assert subprocess.run(["sh", "-n"], input=command, text=True,
                              capture_output=True, check=False).returncode == 0
    assert "cp /run/secrets/capability-registry-publisher.json /tmp/publisher-registry.json" in publisher_command
    assert "--credentials-file" not in publisher_command
    assert publisher_command.endswith("/tmp/publisher-registry.json")
    assert "trap 'rm -f /tmp/publisher-registry.json' EXIT" in publisher_command
    assert receipt["packet_archive_sha256"] == hashlib.sha256(packet.read_bytes()).hexdigest()
    assert "password" not in (output / "job.json").read_text().lower()
    assert "password" not in (output / "preparation.json").read_text().lower()
    assert receipt["return_prefix"].endswith("/returns/cap-image-publisher-001")
    staged_source = tmp_path / "staged-source"
    staged_source.mkdir()
    with tarfile.open(output / "publisher-source.tar.gz") as archive:
        archive.extractall(staged_source, filter="data")
    imported = subprocess.run(
        [sys.executable, "-c", "import scripts.run_oneoff_image_publisher; import scripts.publish_generic_task_image"],
        env={**os.environ, "PYTHONPATH": str(staged_source)}, cwd=tmp_path,
        capture_output=True, text=True, check=False,
    )
    assert imported.returncode == 0, imported.stderr


def test_preparation_rejects_injected_s3_prefix(tmp_path):
    status, packet, *_ = _fixture(tmp_path)
    export_handoff(status, packet)
    with pytest.raises(ValueError, match="unsafe"):
        prepare(packet, tmp_path / "prepared", ROOT, "s3://bucket/a;touch /tmp/x",
                "cap-image-publisher-001", "cap-image-registry-001", 20 << 30)


def test_return_transport_requires_exact_hashes_and_roles(tmp_path):
    status, packet, *_ = _fixture(tmp_path)
    export_handoff(status, packet)
    preparation_path = tmp_path / "prepared"
    preparation = prepare(packet, preparation_path, ROOT, "s3://bucket/oneoff/task-001",
                          "cap-image-publisher-001", "cap-image-registry-001", 20 << 30)
    data = b'{"role":"candidate","state":"published_pending_cold_pull"}\n'
    receipt_uri = preparation["return_prefix"] + "/publication-candidate.json"
    manifest = {"schema_version": "capability-oneoff-image-publisher-return-v1",
                "packet_sha256": preparation["packet_archive_sha256"],
                "source_sha256": preparation["source_archive_sha256"],
                "packet_manifest_sha256": preparation["packet_manifest_sha256"],
                "receipts": {"candidate": {"uri": receipt_uri, "sha256": hashlib.sha256(data).hexdigest(),
                                           "bytes": len(data)}}}
    objects = {preparation["return_prefix"] + "/return-manifest.json": json.dumps(manifest).encode(),
               receipt_uri: data}
    assert fetch(preparation_path / "preparation.json", tmp_path / "fetched", reader=objects.__getitem__)["state"] == "verified"
    assert (tmp_path / "fetched/publication-candidate.json").read_bytes() == data
    objects[receipt_uri] = data + b" "
    with pytest.raises(ValueError, match="bytes differ"):
        fetch(preparation_path / "preparation.json", tmp_path / "bad", reader=objects.__getitem__)
    assert not (tmp_path / "bad").exists()


def test_upload_checks_exact_prepared_inputs_and_readback(tmp_path):
    status, packet, *_ = _fixture(tmp_path)
    export_handoff(status, packet)
    preparation_path = tmp_path / "prepared"
    prepare(packet, preparation_path, ROOT, "s3://bucket/oneoff/task-001",
            "cap-image-publisher-001", "cap-image-registry-001", 20 << 30)

    class FS:
        def __init__(self):
            self.root = tmp_path / "objects"

        def exists(self, key):
            return (self.root / key).exists()

        def open(self, key, mode):
            path = self.root / key
            path.parent.mkdir(parents=True, exist_ok=True)
            return path.open(mode)

    fs = FS()
    cloud = lambda uri: (fs, uri.removeprefix("s3://"))
    assert upload(preparation_path / "preparation.json", packet, cloud=cloud)["state"] == "uploaded_verified"
    assert upload(preparation_path / "preparation.json", packet, cloud=cloud)["state"] == "uploaded_verified"
    packet.write_bytes(packet.read_bytes() + b"drift")
    with pytest.raises(ValueError, match="changed"):
        upload(preparation_path / "preparation.json", packet, cloud=cloud)


def test_cleanup_names_only_prepared_job_and_secret(tmp_path):
    status, packet, *_ = _fixture(tmp_path)
    export_handoff(status, packet)
    preparation_path = tmp_path / "prepared"
    prepare(packet, preparation_path, ROOT, "s3://bucket/oneoff/task-001",
            "cap-image-publisher-001", "cap-image-registry-001", 20 << 30)
    commands = []

    class Result:
        stdout = ""

    def run(command, **kwargs):
        commands.append(command)
        assert kwargs["check"] is True
        return Result()

    receipt = cleanup(preparation_path / "preparation.json", tmp_path / "cleanup.json", runner=run)
    assert [(row["kind"], row["name"]) for row in receipt["resources"]] == [
        ("job", "cap-image-publisher-001"), ("secret", "cap-image-registry-001")]
    assert all(command[3:5] == ["-n", "envreg"] for command in commands)
    assert len(commands) == 4
