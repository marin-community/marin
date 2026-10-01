# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from click.testing import CliRunner
from rigging.provenance import Provenance

from experiments.post_training.tasktrove import pipeline


def _provenance(*, dirty: bool) -> Provenance:
    return Provenance(
        tree_hash="tree1234",
        base_commit="commit1234",
        dirty=dirty,
        branch="tasktrove",
        built_by="tester",
    )


def test_pipeline_plan_pins_the_external_verifier_commit(monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "launch_provenance", lambda: _provenance(dirty=False))
    monkeypatch.setattr(pipeline, "VERIFYIT_REVISION", "verifier5678")

    result = CliRunner().invoke(pipeline.main, ["--stage", "converted"])

    assert result.exit_code == 0
    assert '"tool_ref": "verifier5678"' in result.output


def test_pipeline_plan_rejects_a_dirty_launch(monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "launch_provenance", lambda: _provenance(dirty=True))

    result = CliRunner().invoke(pipeline.main, ["--stage", "converted"])

    assert result.exit_code == 1
    assert "uncommitted changes" in result.output
