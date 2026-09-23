# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for repairing pre-direct-archive Harbor evaluations."""

import json

from finestore.eval import ARCHIVE_ROLLOUTS_TABLE, EvalSample, EvaluationStore, SampleKind
from finestore.reader import ReadView

from experiments.evaluation.migrations.harbor_trajectories import backfill_harbor_trajectories


def test_backfill_uses_recorded_final_attempt_after_job_tree_relocation(tmp_path):
    root = tmp_path / "results"
    trial_id = "103__rvK8oHQ"
    trial = root / "harbor_jobs" / "job" / trial_id
    attempt = trial / "attempts" / "001" / "agent"
    attempt.mkdir(parents=True)
    raw = json.dumps({"steps": [{"step_id": 0, "source": "agent", "message": "answer"}]}).encode()
    (attempt / "trajectory.json").write_bytes(raw)
    (trial / "result.json").write_text(
        json.dumps({"task_name": "ds-1000/103", "trial_uri": f"s3://old/job/{trial_id}/attempts/001"})
    )
    with EvaluationStore.open(str(root), writer_id="harbor") as store:
        store.add_sample(EvalSample(task="ds-1000", doc_id="ds-1000/103", kind=SampleKind.AGENTIC), trial_id=trial_id)
        store.seal()

    assert backfill_harbor_trajectories(str(root)).restored == 1
    reader = ReadView(str(root))
    sample = reader.point("samples", task="ds-1000", doc_id="ds-1000/103", trial_id=trial_id, filter="")
    assert sample["trajectory_uri"] is not None
    assert reader.read_blob(f"{trial_id}/trajectory.json") == raw
    assert reader.scan(ARCHIVE_ROLLOUTS_TABLE).num_rows == 1
    assert backfill_harbor_trajectories(str(root)).restored == 0
