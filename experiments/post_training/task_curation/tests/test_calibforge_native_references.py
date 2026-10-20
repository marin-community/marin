# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
import json
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from experiments.post_training.task_curation.calibforge_captured_fixtures import main as capture_main
from experiments.post_training.task_curation.calibforge_equivalence import main as equivalence_main
from experiments.post_training.task_curation.calibforge_fixtures import FIXTURE_ARCHIVE, SETUP_SCRIPT
from experiments.post_training.task_curation.calibforge_native_references import build_reference, reference_records


def test_snapshot_selection_excludes_fasttext_and_different_base_revision():
    base = "aweaiteam/calibforge:tb2-verifier-base-v1@sha256:" + "1" * 64
    records = [
        {"task_id": "ordinary", "base_image": base, "status": "compiled_pending_equivalence_and_admission"},
        {
            "task_id": "multi_solver/model-training_20260416_172928_004",
            "base_image": "aweaiteam/calibforge:fasttext-base-v1@sha256:" + "2" * 64,
            "status": "compiled_pending_equivalence_and_admission",
        },
        {
            "task_id": "changed-verifier-base",
            "base_image": "aweaiteam/calibforge:tb2-verifier-base-v1@sha256:" + "3" * 64,
            "status": "compiled_pending_equivalence_and_admission",
        },
    ]
    assert [record["task_id"] for record in reference_records(records, base)] == ["ordinary"]


def test_native_timeout_preserves_evidence_and_other_candidate_results(tmp_path, monkeypatch):
    records = []
    for identity in ("timeout", "success"):
        context = tmp_path / "source" / identity / "environment"
        context.mkdir(parents=True)
        dockerfile = b"FROM example@sha256:pinned\n"
        (context / "Dockerfile").write_bytes(dockerfile)
        records.append(
            {
                "task_id": identity,
                "source_dockerfile_sha256": hashlib.sha256(dockerfile).hexdigest(),
                "base_image": "example@sha256:pinned",
            }
        )
    output = tmp_path / "output"
    (output / "logs").mkdir(parents=True)

    def run(command, **kwargs):
        if command[-1].endswith("timeout/environment"):
            raise subprocess.TimeoutExpired(command, 300, output=b"partial build output", stderr=b"download stalled")
        return subprocess.CompletedProcess(command, 0, stdout="sha256:resolved\n", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda row: build_reference(row, tmp_path / "source", output), records))
    assert [(row["status"], row["reference_image"]) for row in results] == [
        ("build_timeout", None),
        ("built", "sha256:resolved"),
    ]
    log = (output / "logs/timeout.txt").read_bytes()
    assert b"partial build output" in log and b"download stalled" in log
    assert results[0]["build_log_sha256"] == hashlib.sha256(log).hexdigest()


@pytest.mark.parametrize("stage", ["capture", "equivalence"])
def test_failed_reference_preserves_evidence_without_blocking_built_tasks(tmp_path, monkeypatch, stage):
    failed = {"task_id": "timeout", "status": "build_timeout", "reference_image": None, "build_log_sha256": "log"}
    built = {"task_id": "success", "status": "built", "reference_image": "sha256:good", "public_fixture_roots": ["app"]}
    references = tmp_path / "references.json"
    references.write_text(json.dumps({"build_network": "none", "tasks": [failed, built]}))
    bundles = tmp_path / "bundles"
    task = bundles / "tasks/success"
    task.mkdir(parents=True)
    (task / SETUP_SCRIPT).write_text("#!/bin/sh\n")
    (bundles / "manifest.json").write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_id": "success",
                        "base_image": "base@sha256:pinned",
                        "workdir": "/app",
                        "status": "compiled_pending_equivalence_and_admission",
                        "setup_sha256": "setup",
                    }
                ]
            }
        )
    )
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as stream:
        member = tarfile.TarInfo("app/input.txt")
        member.size = 7
        stream.addfile(member, io.BytesIO(b"fixture"))
    created_images = []

    def check_output(command, **kwargs):
        if command[1] == "create":
            created_images.append(command[4])
            return "container\n"
        assert command[1] == "cp"
        return archive.getvalue()

    @contextmanager
    def popen(command, **kwargs):
        assert command[1] == "export"
        yield SimpleNamespace(stdout=io.BytesIO(archive.getvalue()), wait=lambda: 0)

    monkeypatch.setattr(subprocess, "check_output", check_output)
    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(command, 0))
    output = tmp_path / "output"
    if stage == "capture":
        argv = [
            "capture",
            "--previous",
            str(bundles),
            "--generators",
            str(bundles),
            "--references",
            str(references),
            "--output",
            str(output),
        ]
        main = capture_main
    else:
        argv = ["equivalence", "--bundles", str(bundles), "--references", str(references), "--output", str(output)]
        main = equivalence_main
    monkeypatch.setattr(sys, "argv", argv)
    main()
    report = json.loads((output / "manifest.json" if stage == "capture" else output).read_text())
    assert report["failed_native_references"] == [failed]
    assert [row["task_id"] for row in report["tasks"]] == ["success"]
    assert created_images[0] == "sha256:good"
    assert None not in created_images
    if stage == "capture":
        assert (output / "tasks/success" / FIXTURE_ARCHIVE).stat().st_size > 0
    else:
        assert report["tasks"][0]["differences"] == {}
