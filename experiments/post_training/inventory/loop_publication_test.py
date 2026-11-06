# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise publication recovery through the real artifact cache."""

import json
from pathlib import Path

import pytest
from marin.execution.lazy import ArtifactStep, resolve
from marin.rl.skyrl import SkyRLRun

from experiments.post_training.inventory import loop_publication


def test_publication_retry_reuses_completed_training(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    training_calls = []
    publication_calls = []

    def train(config):
        output = config["output"]
        training_calls.append(output)
        export = Path(output) / "exports"
        export.mkdir(parents=True)
        (export / "loop-result.json").write_text(json.dumps({"run_id": "test/run", "attempt_id": "attempt-1"}))
        return SkyRLRun(
            path=output,
            hf_model_uri=None,
            global_step=None,
            tokenizer_uri="test/model",
            tokenizer_revision="revision",
            checkpoint_root=str(Path(output) / "checkpoints"),
            draft_checkpoint_root=None,
            terminal_manifest_uri=str(Path(output) / "terminal.json"),
            iris_job_id="test-job",
        )

    def publish(summary):
        publication_calls.append(summary)
        if len(publication_calls) == 1:
            raise ConnectionError("synthetic dashboard outage")
        return {"verified": True}

    monkeypatch.setattr(loop_publication, "publish_result", publish)
    training = ArtifactStep(
        name="test/training",
        version="2026.11.02",
        artifact_type=SkyRLRun,
        build_config=lambda ctx: {"output": ctx.output_path},
        run=train,
    )
    publication = loop_publication.publication_step(training)
    with pytest.raises(RuntimeError, match=r"step\(s\) failed"):
        resolve(publication)
    result = resolve(publication)
    assert result.attempt_id == "attempt-1"
    assert len(training_calls) == 1
    assert len(publication_calls) == 2
    assert json.loads((Path(result.path) / "publication.json").read_text()) == {"verified": True}
