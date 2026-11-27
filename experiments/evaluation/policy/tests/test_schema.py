# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A policy is built from catalog models and named evaluations, and rejects what it can't run as declared."""

from dataclasses import replace

import pytest
from evalrecords.records import EvaluationOverrides

from experiments.evaluation.curated import load_evaluations
from experiments.evaluation.policy.versions import POLICIES

POLICY = POLICIES["marin-2026-11-20-trial"]

EVALUATIONS = {evaluation.name: evaluation for evaluation in POLICY.evaluations}


def test_an_evaluations_file_may_not_name_an_evaluation_twice(tmp_path):
    path = tmp_path / "evaluations.yaml"
    path.write_text("evaluations:\n  aime24:\n    benchmark: aime24\n  aime24:\n    benchmark: math500\n")
    with pytest.raises(ValueError, match="aime24"):
        load_evaluations(path)


@pytest.mark.parametrize(
    "run",
    [
        pytest.param({"name": "financebench", "update": {"judge": None}}, id="judged-without-a-judge"),
        pytest.param({"name": "aime24", "judge_from": "financebench"}, id="a-judge-where-none-grades"),
        pytest.param({"name": "aime24", "update": {"limit": 20}}, id="an-item-limit"),
    ],
)
def test_an_evaluation_the_policy_cant_run_as_declared_is_rejected(run):
    evaluation = EVALUATIONS[run["name"]]
    if "judge_from" in run:
        judged = EVALUATIONS[run["judge_from"]].run
        update = {"judge": judged.judge, "judge_endpoint": judged.judge_endpoint, "judge_api_key": judged.judge_api_key}
    else:
        update = run["update"]
    changed = evaluation.model_copy(
        update={"run": EvaluationOverrides.model_validate({**evaluation.run.model_dump(), **update})}
    )
    with pytest.raises(ValueError, match=run["name"]):
        replace(POLICY, evaluations=tuple(changed if e is evaluation else e for e in POLICY.evaluations))
