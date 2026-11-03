"""Incomplete grading preserves evidence and cannot authorize a policy update."""

import asyncio
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from skyrl_gym.verification import VerificationResult, VerificationStatus
from skyrl_train.evaluate import evaluate, evaluate_step_wise
from skyrl_train.task_inputs import PreferenceInput, TrajectoryMetadata
from skyrl_train.trajectory_runners.trajectory_retention import make_trajectory_sink


class EvaluationRunner:
    """An in-process model/worker boundary with one missing verifier result."""

    def __init__(self, missing_grade):
        self.missing_grade = missing_grade
        self.events = []

    def set_trajectory_sink(self, sink):
        pass

    async def start_eval_session(self, **kwargs):
        self.events.append("start")

    async def stop_eval_session(self):
        self.events.append("stop")

    async def run(self, request):
        missing = self.missing_grade and request.metadata[0].uid == "second"
        return {
            "prompt_token_ids": [[1]],
            "response_ids": [[2]],
            "loss_masks": [[1]],
            "rewards": [0.0],
            "verification_results": [
                VerificationResult(
                    VerificationStatus.ERROR,
                    reason="Grader unavailable",
                    diagnostics={
                        "machine_error_causes": [{"exception_type": "TimeoutError", "origin": None}],
                        "command": "private-command-argument",
                    },
                )
                if missing
                else VerificationResult(VerificationStatus.VERIFIED, score=0.0)
            ],
            "rollout_logprobs": None,
            "stop_reasons": ["stop"],
            "trajectory_ids": list(request.trajectory_ids),
            "is_last_step": [True],
        }


class Tokenizer:
    def decode(self, tokens):
        return "answer"


@pytest.mark.parametrize("step_wise", [False, True])
@pytest.mark.parametrize("scenario", ["complete_zero_scores", "missing_grade", "loader_dropped_row"])
def test_fixed_baseline_requires_every_grade_after_dumping_answers(tmp_path, step_wise, scenario):
    config_path = Path(__file__).parents[3] / "skyrl-train/skyrl_train/config/ppo_base_config.yaml"
    cfg = OmegaConf.load(config_path)
    cfg.data.expected_eval_rows = 2
    cfg.generator.eval_n_samples_per_prompt = 2
    cfg.generator.trajectory_retention.enabled = False
    cfg.trainer.export_path = str(tmp_path)
    cfg.trainer.dump_eval_results = True
    cfg.trajectory_runner.max_concurrent_tasks = 2
    rows = [
        {"input": PreferenceInput([{"role": "user", "content": "Question"}], {}, TrajectoryMetadata(uid, "fixture", "fixture"))}
        for uid in ("first", "second")
    ]
    if scenario == "loader_dropped_row":
        rows.pop()
    runner = EvaluationRunner(scenario == "missing_grade")
    evaluator = evaluate_step_wise if step_wise else evaluate
    updates = []

    async def baseline_then_train():
        result = await evaluator([rows], runner, cfg, 0, Tokenizer(), trajectory_sink=make_trajectory_sink(cfg.generator, Tokenizer()))
        updates.append("optimizer update")
        return result

    if scenario == "complete_zero_scores":
        metrics = asyncio.run(baseline_then_train())
        assert metrics["eval/all/num_scored"] == 4
        assert updates == ["optimizer update"]
    else:
        with pytest.raises(RuntimeError, match="Fixed-cohort evaluation requires"):
            asyncio.run(baseline_then_train())
        assert updates == []
    assert runner.events == ["start", "stop"]
    directory = tmp_path / "dumped_evals/global_step_0_evals"
    saved = [json.loads(line) for line in (directory / "fixture.jsonl").read_text().splitlines()]
    assert len(saved) == (2 if scenario == "loader_dropped_row" else 4)
    assert all(row["response_token_ids"] == [2] for row in saved)
    if scenario == "missing_grade":
        failures = [row for row in saved if row["verification"]["status"] == "error"]
        assert len(failures) == 2
        assert failures[0]["verification"]["diagnostics"]["machine_error_causes"] == [
            {"exception_type": "TimeoutError", "origin": None}
        ]
        assert "private-command-argument" not in json.dumps(saved)
    metrics = json.loads((directory / "aggregated_results.jsonl").read_text())
    assert metrics["eval/all/num_scored"] == (4 if scenario == "complete_zero_scores" else 2)
