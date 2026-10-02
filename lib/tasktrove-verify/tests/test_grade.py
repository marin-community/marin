# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from tasktrove_verify import grade as grade_module
from tasktrove_verify.grade import InvalidTask, Reward, Status, run, write_reward
from tasktrove_verify.spec import McqSpec, Mode


def test_invalid_task_becomes_invalid_task_reward(tmp_path, monkeypatch):
    def broken(spec, tests_dir, workspace):
        raise InvalidTask("no reference")

    monkeypatch.setitem(grade_module.GRADERS, Mode.MCQ, broken)
    reward = grade_module.grade(McqSpec("A"), tmp_path, tmp_path)
    assert reward.status == Status.INVALID_TASK and reward.detail == {"error": "no reference"}


@pytest.mark.parametrize(
    "verdict",
    [
        Reward(float("nan"), Status.SCORED),
        Reward(float("inf"), Status.SCORED),
        Reward(1.1, Status.SCORED),
        Reward(True, Status.SCORED),
        Reward(1.0, "scored"),
        Reward(1.0, Status.INFRA_ERROR),
    ],
)
def test_invalid_direct_grader_verdict_is_unscored_and_removes_stale_rewards(tmp_path, monkeypatch, verdict):
    monkeypatch.setitem(grade_module.GRADERS, Mode.MCQ, lambda spec, tests_dir, workspace: verdict)
    spec_path = tmp_path / "verifier.toml"
    spec_path.write_text('mode="mcq"\nexpected="A"\n')
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "reward.json").write_text('{"reward":1.0}')
    result = run(spec_path, tmp_path)
    write_reward(logs, result)
    assert (result.status, result.reward) == (Status.INFRA_ERROR, 0.0)
    assert not (logs / "reward.json").exists()


@pytest.mark.parametrize(
    ("verdict", "status", "reward"),
    [
        (Reward(float("nan"), Status.SCORED), "infra_error", None),
        (Reward(1.0, "scored"), "infra_error", None),
        (Reward(1.0, Status.INVALID_TASK), "infra_error", None),
        (Reward(0.0, Status.SCORED), "scored", 0.0),
        (Reward(0.5, Status.SCORED), "scored", 0.5),
    ],
)
def test_direct_reward_persistence_rejects_invalid_verdicts_and_keeps_valid_zero(tmp_path, verdict, status, reward):
    (tmp_path / "reward.json").write_text('{"reward":1.0}')
    (tmp_path / "reward.txt").write_text("1.0")
    write_reward(tmp_path, verdict)
    persisted = json.loads((tmp_path / "verdict.json").read_text())
    assert persisted["status"] == status
    if reward is None:
        assert persisted["reward"] == 0.0
        assert not (tmp_path / "reward.json").exists()
        assert not (tmp_path / "reward.txt").exists()
    else:
        assert json.loads((tmp_path / "reward.json").read_text()) == {"reward": reward}


@pytest.mark.parametrize(
    "verdict",
    [
        Reward(10**10000, Status.SCORED),
        Reward(1.0, Status.SCORED, {"bad": object()}),
        Reward(1.0, Status.SCORED, {"bad": float("nan")}),
    ],
)
def test_invalid_persisted_verdict_overwrites_previous_positive_verdict(tmp_path, verdict):
    (tmp_path / "verdict.json").write_text('{"reward":1.0,"status":"scored"}')
    (tmp_path / "reward.json").write_text('{"reward":1.0}')
    write_reward(tmp_path, verdict)
    persisted = json.loads((tmp_path / "verdict.json").read_text())
    assert (persisted["status"], persisted["reward"]) == ("infra_error", 0.0)
    assert not (tmp_path / "reward.json").exists()


def test_deeply_nested_detail_replaces_stale_positive_verdict(tmp_path):
    detail = {}
    for _ in range(10000):
        detail = {"nested": detail}
    (tmp_path / "verdict.json").write_text('{"reward":1.0,"status":"scored"}')
    (tmp_path / "reward.json").write_text('{"reward":1.0}')
    write_reward(tmp_path, Reward(1.0, Status.SCORED, detail))
    verdict = json.loads((tmp_path / "verdict.json").read_text())
    assert (verdict["status"], verdict["reward"]) == ("infra_error", 0.0)
    assert not (tmp_path / "reward.json").exists()


@pytest.mark.parametrize(
    "policy,expected",
    [
        (grade_module.Aggregation.ALL, 0.0),
        (grade_module.Aggregation.MEAN, 0.5),
        (grade_module.Aggregation.MAX, 1.0),
        (grade_module.Aggregation.MIN, 0.0),
    ],
)
def test_aggregation_preserves_missing_component_denominator(policy, expected):
    verdict = grade_module.aggregate_rewards([grade_module.scored(1.0)], expected_total=2, policy=policy)
    assert (verdict.reward, verdict.status) == (expected, Status.SCORED)
    assert verdict.detail["missing"] == 1
    complete = grade_module.aggregate_rewards([grade_module.scored(1.0)] * 2, expected_total=2, policy=policy)
    assert complete.reward == 1.0


@pytest.mark.parametrize(
    "bad,status",
    [
        (Reward(0.0, Status.INVALID_TASK), Status.INVALID_TASK),
        (Reward(0.0, Status.INFRA_ERROR), Status.INFRA_ERROR),
        (Reward(float("nan"), Status.SCORED), Status.INFRA_ERROR),
        (Reward(True, Status.SCORED), Status.INFRA_ERROR),
        (Reward(2.0, Status.SCORED), Status.INFRA_ERROR),
    ],
)
@pytest.mark.parametrize("policy", list(grade_module.Aggregation))
def test_bad_component_discards_previous_credit_and_stale_reward(tmp_path, bad, status, policy):
    write_reward(tmp_path, grade_module.scored(1.0))
    verdict = grade_module.aggregate_rewards([grade_module.scored(1.0), bad], expected_total=2, policy=policy)
    write_reward(tmp_path, verdict)
    assert (verdict.reward, verdict.status) == (0.0, status)
    assert not (tmp_path / "reward.txt").exists()
    assert not (tmp_path / "reward.json").exists()


@pytest.mark.parametrize("total", [0, True, 1])
def test_invalid_aggregation_denominator_cannot_award_credit(total):
    verdict = grade_module.aggregate_rewards(
        [grade_module.scored(1.0)] * 2, expected_total=total, policy=grade_module.Aggregation.MEAN
    )
    assert (verdict.reward, verdict.status) == (0.0, Status.INVALID_TASK)


def test_max_aggregation_preserves_partial_credit_and_empty_failure():
    verdict = grade_module.aggregate_rewards(
        [grade_module.scored(0.25), grade_module.scored(0.75)],
        expected_total=3,
        policy=grade_module.Aggregation.MAX,
    )
    assert (verdict.reward, verdict.status) == (0.75, Status.SCORED)
    empty = grade_module.aggregate_rewards([], expected_total=3, policy=grade_module.Aggregation.MAX)
    assert (empty.reward, empty.status) == (0.0, Status.SCORED)


@pytest.mark.parametrize("numeric_gate,expected", [(0.0, 0.0), (1.0, 0.22)])
def test_fractional_gate_and_rounding_preserve_alignment_denominator(numeric_gate, expected):
    partial = grade_module.aggregate_rewards(
        [grade_module.scored(2 / 3), grade_module.scored(numeric_gate)],
        expected_total=2,
        policy=grade_module.Aggregation.MIN,
    )
    result = grade_module.aggregate_rewards(
        [partial, grade_module.scored(0), grade_module.scored(0)],
        expected_total=3,
        policy=grade_module.Aggregation.MEAN,
        round_digits=2,
    )
    assert result.reward == expected
    assert result.status == Status.SCORED
    assert result.detail["missing"] == 0


def test_missing_fractional_gate_cannot_preserve_partial_credit():
    result = grade_module.aggregate_rewards(
        [grade_module.scored(0.75)], expected_total=2, policy=grade_module.Aggregation.MIN
    )
    assert (result.reward, result.status, result.detail["missing"]) == (0, Status.SCORED, 1)


@pytest.mark.parametrize("round_digits", [True, -1, 7, float("nan")])
def test_invalid_rounding_contract_cannot_publish_positive_reward(tmp_path, round_digits):
    write_reward(tmp_path, grade_module.scored(1))
    result = grade_module.aggregate_rewards(
        [grade_module.scored(1)], expected_total=1, policy=grade_module.Aggregation.MEAN, round_digits=round_digits
    )
    write_reward(tmp_path, result)
    assert result.status == Status.INVALID_TASK
    assert not (tmp_path / "reward.json").exists()


def test_product_preserves_fractional_components_and_requires_complete_valid_set():
    result = grade_module.aggregate_rewards(
        [grade_module.scored(1.0), grade_module.scored(0.3), grade_module.scored(0.3)],
        expected_total=3,
        policy=grade_module.Aggregation.PRODUCT,
    )
    assert result.status == grade_module.Status.SCORED
    assert result.reward == pytest.approx(0.09)
    missing = grade_module.aggregate_rewards(
        [grade_module.scored(1.0), grade_module.scored(0.3)],
        expected_total=3,
        policy=grade_module.Aggregation.PRODUCT,
    )
    assert missing.reward == 0.0
    failed = grade_module.aggregate_rewards(
        [grade_module.scored(0.0), grade_module.infra_error("provider unavailable")],
        expected_total=2,
        policy=grade_module.Aggregation.PRODUCT,
    )
    assert failed.reward == 0.0
    assert failed.status == grade_module.Status.INFRA_ERROR
