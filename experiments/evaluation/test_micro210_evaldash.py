# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The on-seal EvalDash publication step writes one record per sealed row."""

import json

from experiments.evaluation.micro210 import (
    Micro210Evaluation,
    Micro210Model,
    Micro210Route,
    publish_evaldash_on_seal,
)
from tests.rl.micro210_fixture import policy_fixture_reference


def _write_run_output(root, rows):
    eval_only = root / "exports/dumped_evals/attempt-a/eval_only"
    eval_only.mkdir(parents=True)
    with (eval_only / "aggregated_results.jsonl").open("w") as handle:
        handle.write("{}\n")
    with (eval_only / "aime24.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return eval_only


def _row(uid: str, score: float, *, verified: bool = True) -> dict:
    return {
        "data_source": "zwhe99_aime24",
        "uid": uid,
        "env_class": "taskcompendium",
        "verification": {"status": "verified", "score": score, "passed": score > 0} if verified else None,
    }


def _fixture_evaluation(tmp_path, monkeypatch):
    """A Micro210Evaluation bound to the fixture policy and a fake output path."""

    reference = policy_fixture_reference()
    monkeypatch.setattr("experiments.evaluation.micro210_policy.POLICY_REFERENCE", reference)
    model = Micro210Model("s3://fixture/model", "sha256:" + "a" * 64, "marin-community/marin-tokenizer-tools", "b" * 40)
    route = Micro210Route("lib/iris/config/marin.yaml", "lib/iris/config/cw-us-east-02a.yaml", "cw-us-east-02a", 4)
    return Micro210Evaluation(
        model=model, run_id="micro210-reval-fixture@2026.11.25.1", output_uri=str(tmp_path / "run"), route=route
    )


def test_publish_evaldash_on_seal_writes_and_computes(tmp_path, monkeypatch):
    records_prefix = tmp_path / "evals"
    monkeypatch.setattr("experiments.evaluation.micro210.EVALDASH_RECORDS_PREFIX", str(records_prefix))
    evaluation = _fixture_evaluation(tmp_path, monkeypatch)
    _write_run_output(
        tmp_path / "run",
        [_row("0", 1.0), _row("1", 0.0), _row("2", 0.0, verified=False)],
    )

    outcome = publish_evaldash_on_seal(evaluation=evaluation, gpu_variant="H100", num_nodes=16)

    assert outcome["published"] is True
    assert outcome["run_id"] == "micro210-reval-fixture-2026.11.25.1-fixture"
    record_path = records_prefix / outcome["run_id"] / "record.json"
    record = json.loads(record_path.read_text())
    assert record["model"]["location"] == "s3://fixture/model"
    assert record["jobs"] == {}
    # publish_evaluation computes metrics and coverage from the native rows: the two
    # verified attempts score (mean 0.5); the absent-verification attempt counts as
    # attempted but stays unscored.
    assert record["metrics"]["zwhe99_aime24"]["reward"] == 0.5
    assert record["coverage"]["zwhe99_aime24"]["n_attempted"] == 3
    assert record["coverage"]["zwhe99_aime24"]["n_scored"] == 2
    assert record["coverage"]["zwhe99_aime24"]["n_correct"] == 1
    # The FineStore store sits beside the record under the same run id.
    assert (records_prefix / outcome["run_id"] / "finestore").exists()


def test_publish_evaldash_on_seal_survives_missing_rows(tmp_path, monkeypatch):
    records_prefix = tmp_path / "evals"
    monkeypatch.setattr("experiments.evaluation.micro210.EVALDASH_RECORDS_PREFIX", str(records_prefix))
    evaluation = _fixture_evaluation(tmp_path, monkeypatch)
    (tmp_path / "run").mkdir(parents=True)

    outcome = publish_evaldash_on_seal(evaluation=evaluation, gpu_variant="H100", num_nodes=16)

    assert outcome["published"] is False
    assert "no dumped_evals" in outcome["error"]
