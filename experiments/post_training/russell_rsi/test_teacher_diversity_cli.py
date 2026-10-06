# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib
import json

from click.testing import CliRunner
from marin.execution.lazy import ArtifactStep
from marin.external_dependencies import MARIN_SKYRL

from experiments.post_training.russell_rsi import launch_interrupted_calibration_sft as foreground
from experiments.post_training.russell_rsi import launch_teacher_diversity_study as diversity


def test_collection_cli_uses_attached_foreground_runner(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
    config = {"protocol": diversity.PROTOCOL, "version": "2026.10.06.15", "runtime_commit": MARIN_SKYRL.commit}
    payload = json.dumps(config).encode()
    path = tmp_path / "config.json"
    path.write_bytes(payload)
    collected = ArtifactStep.adopt("documents/collection-only", config["version"], str(tmp_path / "completed"))
    reviews = []
    runs = []
    monkeypatch.setattr(foreground, "foreground_runner", lambda handles, count: runs.append((handles, count)))
    importlib.reload(diversity)
    monkeypatch.setattr(diversity, "require_reviewed_source", lambda uri, sha: reviews.append((uri, sha)))
    monkeypatch.setattr(diversity, "diversity_workflow", lambda value: {"collect": collected})
    result = CliRunner().invoke(
        diversity.main,
        [
            "--config-uri",
            str(path),
            "--config-sha256",
            hashlib.sha256(payload).hexdigest(),
            "--source-review-uri",
            "review.json",
            "--source-review-sha256",
            "a" * 64,
            "--stage",
            "collect",
            "--version",
            config["version"],
            "--max-concurrent",
            "1",
            "--run",
        ],
    )
    monkeypatch.undo()
    importlib.reload(diversity)
    assert result.exit_code == 0, result.output
    assert reviews == [("review.json", "a" * 64)]
    assert runs == [([collected], 1)]
