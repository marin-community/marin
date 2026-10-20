# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import subprocess
from concurrent.futures import ThreadPoolExecutor

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
