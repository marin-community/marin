import base64
import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from capability_pipeline import publication_exchange as exchange

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("publisher_kit", ROOT / "ops/publisher/publisher_kit.py")
kit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(kit)
SHA = "d" * 64


def test_source_archive_is_deterministic_and_runs_on_a_bare_interpreter(tmp_path):
    first = kit.build_source(ROOT, tmp_path / "one.tar.gz")
    second = kit.build_source(ROOT, tmp_path / "two.tar.gz")
    assert first["sha256"] == second["sha256"]
    assert {"scripts/run_image_publisher_service.py", "capability_pipeline/publication_exchange.py",
            "ops/publisher/requirements.txt"} <= set(first["files"])
    staged = tmp_path / "controller"
    with tarfile.open(tmp_path / "one.tar.gz") as archive:
        archive.extractall(staged, filter="data")
    # -S: no site-packages at all, so the publisher imports on stdlib alone
    # (botocore is imported lazily, after pip installs it in the pod).
    env = {"PYTHONPATH": str(staged), "PATH": os.environ.get("PATH", "")}
    probe = subprocess.run([sys.executable, "-S", "-m", "scripts.run_image_publisher_service", "--help"],
                           env=env, cwd=tmp_path, capture_output=True, text=True, check=False)
    assert probe.returncode == 0, probe.stderr
    assert "--dry-run-object-root" in probe.stdout
    oneoff = subprocess.run([sys.executable, "-S", "-c", "import scripts.run_oneoff_image_publisher"],
                            env=env, cwd=tmp_path, capture_output=True, text=True, check=False)
    assert oneoff.returncode == 0, oneoff.stderr


def test_deployment_render_pins_source_and_references_secrets_only():
    text = kit.render_deployment(
        source_sha256=SHA, source_uri=f"s3://marin-us-east-02a/q/service/source/{SHA}/publisher-source.tar.gz",
        queue_uri="s3://marin-us-east-02a/q", s3_secret="capability-publisher-s3")
    assert "${" not in text and "$$" not in text
    assert f'capability-env-gen/source-sha256: "{SHA}"' in text
    assert text.index("sha256sum -c -") < text.index("differs from its pinned sha256") < text.index("extractall")
    assert 'filter="data"' in text
    assert "secretName: capability-registry-publisher" in text
    assert 'name: "capability-publisher-s3", key: accesskey' in text
    assert "type: Recreate" in text and "replicas: 1" in text
    assert "readOnlyRootFilesystem: true" in text and "automountServiceAccountToken: false" in text
    assert "--require-hashes" in text and "medium: Memory" in text
    assert kit.PYTHON_IMAGE in text and kit.AWS_CLI_IMAGE in text
    assert "@sha256:" in kit.PYTHON_IMAGE and "@sha256:" in kit.AWS_CLI_IMAGE
    assert "password" not in text.lower()
    with pytest.raises(ValueError, match="pinned sha256"):
        kit.render_deployment(source_sha256=SHA, source_uri="s3://marin-us-east-02a/q/latest.tar.gz",
                              queue_uri="s3://marin-us-east-02a/q", s3_secret="cw-s3")
    with pytest.raises(ValueError):
        kit.render_deployment(source_sha256=SHA, source_uri=f"s3://b/{SHA}/x", queue_uri="s3://b/q;rm -rf /",
                              s3_secret="cw-s3")


def test_requirements_are_fully_hash_pinned():
    lines = (ROOT / "ops/publisher/requirements.txt").read_text().splitlines()
    pins = [line for line in lines if line and not line.startswith((" ", "#"))]
    assert {line.split("==")[0] for line in pins} == {"botocore", "jmespath", "python-dateutil", "six", "urllib3"}
    assert all("==" in line and line.endswith("\\") for line in pins)


def test_registry_secret_render_validates_without_echoing(monkeypatch):
    credential = {"registry": kit.DEFAULT_REGISTRY, "user": "capability-publisher", "password": "sentinel-pw"}
    manifest = json.loads(kit.render_registry_secret(json.dumps(credential).encode(), kit.DEFAULT_REGISTRY))
    assert manifest["metadata"] == {"name": "capability-registry-publisher", "namespace": "envreg",
                                    "labels": manifest["metadata"]["labels"]}
    assert json.loads(base64.b64decode(manifest["data"]["credentials.json"])) == credential
    wrong = {**credential, "registry": "elsewhere.example"}
    with pytest.raises(ValueError) as error:
        kit.render_registry_secret(json.dumps(wrong).encode(), kit.DEFAULT_REGISTRY)
    assert "sentinel-pw" not in str(error.value)
    with pytest.raises(ValueError):
        kit.render_registry_secret(b"not json sentinel-pw", kit.DEFAULT_REGISTRY)
    s3 = json.loads(kit.render_s3_secret("capability-publisher-s3", {"CW_KEY_ID": "id", "CW_KEY_SECRET": "sec"}))
    assert base64.b64decode(s3["data"]["secretkey"]) == b"sec"
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    with pytest.raises(SystemExit, match="terminal"):
        kit._refuse_tty()


def test_upload_source_is_content_addressed_and_verified(tmp_path):
    archive = tmp_path / "source.tar.gz"
    archive.write_bytes(b"pinned source")
    digest = exchange.sha256(archive.read_bytes())
    store = {}

    class FS:
        def cat_file(self, path):
            if path not in store:
                raise FileNotFoundError(path)
            return store[path]

        def pipe_file(self, path, data):
            store[path] = data

    uri = f"s3://marin-us-east-02a/q/service/source/{digest}/publisher-source.tar.gz"
    factory = lambda value: (FS(), value.removeprefix("s3://"))
    assert kit.upload_source(archive, uri, fs_factory=factory)["state"] == "uploaded_verified"
    assert kit.upload_source(archive, uri, fs_factory=factory)["sha256"] == digest
    with pytest.raises(ValueError, match="content-addressed"):
        kit.upload_source(archive, "s3://marin-us-east-02a/q/latest.tar.gz", fs_factory=factory)


def test_health_alarms_on_absent_or_stale_heartbeat_and_counts_returns(tmp_path, capsys):
    queue = exchange.LocalQueue(tmp_path / "queue")
    code, summary = kit.health(queue, max_age=120, clock=lambda: 1000.0)
    assert code == 1 and summary["problems"] == ["heartbeat_absent"]
    queue.put(exchange.HEARTBEAT_KEY, json.dumps({"epoch": 990.0, "state": "running", "source_sha256": SHA}).encode())
    pending, published, rejected, transient = ("1" * 64, "2" * 64, "3" * 64, "4" * 64)
    for sha in (pending, published, rejected, transient):
        queue.put(exchange.request_key(sha), b"x")
    queue.put(exchange.receipt_key(published, "candidate"), b"{}")
    queue.put(exchange.result_key(published), exchange.result_document(
        packet_sha=published, attempt=1, state="published", reason="published_pending_cold_pull",
        roles=["candidate"], packet_manifest_sha256="e" * 64,
        receipts={"candidate": {"key": exchange.receipt_key(published, "candidate"), "sha256": "f" * 64, "bytes": 2}}))
    queue.put(exchange.result_key(rejected), exchange.result_document(
        packet_sha=rejected, attempt=1, state="rejected", reason="bad", rejection_class="packet_invalid"))
    queue.put(exchange.result_key(transient), exchange.result_document(
        packet_sha=transient, attempt=1, state="transient_failure", reason="registry down"))
    code, summary = kit.health(queue, max_age=120, expect_source=SHA, clock=lambda: 1000.0)
    assert code == 0, summary["problems"]
    assert summary["counts"] == {"pending": 1, "published": 1, "rejected": 1, "transient_awaiting_resubmission": 1}
    kit._print_health(code, summary)
    assert "queue depth=1" in capsys.readouterr().out
    code, summary = kit.health(queue, max_age=120, expect_source="0" * 64, clock=lambda: 5000.0)
    assert code == 1 and "heartbeat_from_other_source" in summary["problems"]
    assert any(problem.startswith("heartbeat_stale") for problem in summary["problems"])
    assert kit.requeue(queue, transient) == {"state": "requeued", "packet_sha": transient, "attempt": 2}
    assert exchange.requested_attempt(queue.keys(f"requests/{transient}"), transient) == 2
    with pytest.raises(ValueError):
        kit.requeue(queue, published)


def test_kit_main_render_deployment_writes_a_file(tmp_path, capsys):
    output = tmp_path / "deployment.yaml"
    assert kit.main(["render-deployment", "--source-sha256", SHA, "--source-uri",
                     f"s3://marin-us-east-02a/q/service/source/{SHA}/publisher-source.tar.gz",
                     "--output", str(output)]) == 0
    assert "capability-image-publisher" in output.read_text()
    source = tmp_path / "source.tar.gz"
    assert kit.main(["build-source", "--output", str(source)]) == 0
    printed = capsys.readouterr().out.strip().splitlines()[-1]
    assert printed == exchange.sha256(source.read_bytes())
    with tarfile.open(fileobj=io.BytesIO(source.read_bytes())) as archive:
        assert all(member.mtime == 0 and member.mode == 0o444 for member in archive.getmembers())
