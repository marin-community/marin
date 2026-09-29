# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from importlib.metadata import distribution

from click.testing import CliRunner
from marin.execution.lazy import materialized_config
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


def test_pipeline_plan_pins_the_verifier_commit_separately_from_marin(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(pipeline, "launch_provenance", lambda: _provenance(dirty=False))
    verifier_provenance = json.loads(distribution("verifyit").read_text("direct_url.json"))
    verifier_revision = verifier_provenance["vcs_info"]["commit_id"]

    result = CliRunner().invoke(pipeline.main, ["--stage", "converted"])

    assert result.exit_code == 0
    assert f'"tool_ref": "{verifier_revision}"' in result.output
    workflow = pipeline.build_workflow(pipeline.launch_commit(), pipeline.verifier_commit())
    routing_config = materialized_config(workflow.routing, str(tmp_path))
    assert routing_config.git_revision == "commit1234"


def test_pipeline_plan_rejects_a_dirty_launch(monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "launch_provenance", lambda: _provenance(dirty=True))

    result = CliRunner().invoke(pipeline.main, ["--stage", "converted"])

    assert result.exit_code == 1
    assert "uncommitted changes" in result.output
