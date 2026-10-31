# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Missing environment approvals must fail before any GPU submission."""

import json

import pytest
from click.testing import CliRunner

from experiments.evaluation.mini_policy_runtime import launch


@pytest.mark.parametrize("defect", ["missing_snapshot", "wrong_dataset_scope"])
def test_incomplete_release_cannot_submit_gpu_jobs(tmp_path, monkeypatch, defect):
    manifest = {
        "required_source_keys": ["dockerfile:one", "dockerfile:two"],
        "expected_source_keys": 2,
        "environment_snapshots": {"dockerfile:one": "snapshot-one", "dockerfile:two": "snapshot-two"},
        "failures": [],
        "data_uri": "s3://example/full",
    }
    if defect == "missing_snapshot":
        del manifest["environment_snapshots"]["dockerfile:two"]
    else:
        manifest["data_uri"] = "s3://example/smoke"
    path = tmp_path / "environments.json"
    path.write_text(json.dumps(manifest))

    def forbidden_gpu_artifact(*args, **kwargs):
        pytest.fail("A GPU artifact was constructed for an unusable environment map")

    monkeypatch.setattr(launch, "skyrl_step", forbidden_gpu_artifact)
    result = CliRunner().invoke(
        launch.main,
        [
            "--version",
            "2026.10.31",
            "--run",
            "--model-uri",
            "s3://example/model",
            "--data-uri",
            "s3://example/full",
            "--environment-manifest",
            str(path),
        ],
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
