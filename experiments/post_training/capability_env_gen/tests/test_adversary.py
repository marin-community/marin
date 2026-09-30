import asyncio
import copy
import json
import sys
from types import ModuleType

import pytest

from capability_pipeline.adversary import (
    STRATEGIES,
    assess_attack_results,
    run_independent_attacks,
    trial_failure,
)


def test_trial_failure_preserves_truncation_cause_without_provider_body():
    assert (
        trial_failure(
            {
                "exception_info": {
                    "exception_type": "RuntimeError",
                    "exception_message": "Incomplete GLM solver output: length",
                }
            }
        )
        == "model_output_truncated"
    )
    assert (
        trial_failure(
            {
                "exception_info": {
                    "exception_type": "SensitiveProviderError",
                    "exception_message": "provider body must not be copied",
                }
            }
        )
        == "harbor_trial_failed"
    )
    assert trial_failure({"exception_info": None}) is None
    assert (
        trial_failure(
            {
                "exception_info": {
                    "exception_type": "ExtractionError",
                }
            }
        )
        is None
    )


def test_truncated_trial_without_transcript_retains_underlying_failure(
    tmp_path, monkeypatch
):
    runner = ModuleType("taskcompendium.harbor.runner")

    async def run_trial(package, config, trials, name):
        root = trials / name
        root.mkdir(parents=True)
        (root / "result.json").write_text(
            json.dumps(
                {
                    "exception_info": {
                        "exception_type": "RuntimeError",
                        "exception_message": "Incomplete GLM solver output: length",
                    }
                }
            )
        )

    runner.run_trial = run_trial
    monkeypatch.setitem(sys.modules, "taskcompendium.harbor.runner", runner)
    package = tmp_path / "package"
    package.mkdir()
    (package / "manifest.json").write_text('{"step_names":["one"]}')
    result = asyncio.run(
        run_independent_attacks(
            package, {"agent": {"kwargs": {}}}, tmp_path / "trials", tmp_path, "none"
        )
    )
    assert result["report"]["state"] == "needs_adjudication_or_retry"
    for case in result["report"]["cases"]:
        assert case["error"] == "model_output_truncated"
        assert len(case["trial_sha256"]) == 64
        assert (tmp_path / case["trial_artifact"]).is_file()


def test_attacks_receive_explicit_extended_output_policy(tmp_path, monkeypatch):
    runner = ModuleType("taskcompendium.harbor.runner")
    seen = []

    async def run_trial(package, config, trials, name):
        seen.append(config["agent"]["kwargs"])
        root = trials / name
        root.mkdir(parents=True)
        (root / "result.json").write_text(
            json.dumps(
                {
                    "exception_info": {
                        "exception_type": "RuntimeError",
                        "exception_message": "Incomplete GLM solver output: length",
                    }
                }
            )
        )

    runner.run_trial = run_trial
    monkeypatch.setitem(sys.modules, "taskcompendium.harbor.runner", runner)
    package = tmp_path / "package"
    package.mkdir()
    (package / "manifest.json").write_text('{"step_names":["one"]}')
    asyncio.run(
        run_independent_attacks(
            package,
            {"agent": {"kwargs": {}}},
            tmp_path / "trials",
            tmp_path,
            "none",
            token_limits=(32768, 65536, 131072, None),
            request_timeout=3600,
        )
    )
    assert len(seen) == 3
    assert all(
        record["token_limits"] == [32768, 65536, 131072, None]
        and record["request_timeout"] == 3600
        for record in seen
    )


def clean_records():
    return [
        {"strategy": strategy, "result": {"status": "graded", "reward": 0.0}}
        for strategy in STRATEGIES
    ]


def test_rewarded_attack_requires_adjudication_not_automatic_exploit_claim():
    records = clean_records()
    records[0]["result"]["reward"] = 1.0
    issues = assess_attack_results(records)
    assert any("independent adjudication" in issue for issue in issues)
    assert assess_attack_results(clean_records()) == []


@pytest.mark.parametrize("reward", [float("nan"), float("inf"), True, -0.1, 1.1, None])
def test_invalid_reward_never_counts_as_rejected_attack(reward):
    records = clean_records()
    records[0]["result"]["reward"] = reward
    assert any(
        "invalid graded reward" in issue for issue in assess_attack_results(records)
    )


def test_malformed_only_suite_does_not_establish_semantic_grading():
    records = clean_records()
    for record in records:
        record["result"] = {"status": "extraction_error", "reward": None}
    assert any(
        "no well-formed attack" in issue for issue in assess_attack_results(records)
    )
    records[0]["result"] = {"status": "graded", "reward": 0}
    assert assess_attack_results(records) == []


def test_infra_failure_and_missing_strategy_cannot_be_success():
    records = clean_records()
    records[0]["error"] = "TimeoutError"
    assert any("execution failed" in issue for issue in assess_attack_results(records))
    assert assess_attack_results(clean_records()[:-1])
    assert assess_attack_results([clean_records()[0]] * 3)


def test_multistep_attack_cannot_hide_reward_in_an_earlier_step():
    records = clean_records()
    for record in records:
        result = record.pop("result")
        record["steps"] = [
            {"step_index": 0, "result": copy.deepcopy(result)},
            {"step_index": 1, "result": copy.deepcopy(result)},
        ]
    assert assess_attack_results(records) == []
    records[0]["steps"][0]["result"]["reward"] = 1
    assert any("injection:0" in issue for issue in assess_attack_results(records))
