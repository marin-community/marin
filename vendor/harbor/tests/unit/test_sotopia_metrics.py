# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import math
from pathlib import Path
from uuid import uuid4

import pytest
from finestore.eval import EvalSample, EvaluationStore, Grading, SampleKind
from finestore.reader import ReadView
from upath import UPath
from harbor_config.models.job.config import SotopiaCohort
from harbor.metrics.sotopia import (
    EpisodeScore,
    correct_sotopia,
    length_controlled_scores,
)
from harbor.job import Job
from harbor.models.job.config import JobConfig
from harbor.models.job.result import JobResult
from harbor.models.trial.config import AgentConfig, TrialConfig, TaskConfig
from harbor.models.trial.result import AgentInfo, TrialResult
from harbor.models.verifier.result import VerifierResult

RANGES = {
    "believability": (0, 10),
    "relationship": (-5, 5),
    "knowledge": (0, 10),
    "secret": (-10, 0),
    "social_rules": (-10, 0),
    "financial_and_material_benefits": (-5, 5),
    "goal": (0, 10),
}


def sotopia_archive(root: Path, length_offset: int) -> float:
    scores = []
    with EvaluationStore.open(str(root), writer_id="harbor") as store:
        for index in range(100):
            length = 100 + length_offset + index * (2 if length_offset else 1)
            score = 1 / (
                1
                + math.exp(
                    -(0.5 + index % 5 * 0.03 + 0.8 * math.tanh((length - 200) / 150))
                )
            )
            scores.append(score)
            task = f"task-{index}"
            trial = f"trial-{index}"
            metrics = {"sotopia_" + name: score for name in RANGES}
            store.add_sample(
                EvalSample(
                    task="sotopia-hard",
                    doc_id=task,
                    kind=SampleKind.AGENTIC,
                    metrics=metrics,
                    grading=Grading(method="harbor", metric="reward", score=score),
                ),
                trial_id=trial,
            )
            summary = {
                "evaluated_agent_index": 0,
                "scores": [
                    {
                        name: minimum + score * (maximum - minimum)
                        for name, (minimum, maximum) in RANGES.items()
                    }
                ],
                "transcript": [
                    [
                        {"sender": "Environment", "receiver": "Alice"},
                        {"sender": "Environment", "receiver": "Bob"},
                    ],
                    [{"sender": "Alice", "receiver": "Bob", "message": "x" * length}],
                ],
            }
            path = (
                root
                / "harbor_jobs"
                / "job"
                / trial
                / "attempts"
                / "0"
                / "agent"
                / "sotopia-summary.json"
            )
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(summary))
        store.seal()
    return sum(scores) / len(scores)


def candidate_trials(root):
    trials = []
    for row in ReadView(str(root)).iter_rows("samples"):
        directory = root / "harbor_jobs" / "job" / row["trial_id"]
        config = TrialConfig(
            task=TaskConfig(path=Path("/tasks") / row["doc_id"]),
            trial_name=row["trial_id"],
            job_id=uuid4(),
        )
        trials.append(
            TrialResult(
                task_name=row["doc_id"],
                trial_name=row["trial_id"],
                trial_uri=directory.as_uri(),
                trial_relpath=row["trial_id"],
                task_id=config.task.get_task_id(),
                task_checksum="fixture",
                config=config,
                agent_info=AgentInfo(name="sotopia", version="test"),
                verifier_result=VerifierResult(rewards=dict(row["metrics"])),
            )
        )
    return trials


def test_sotopia_correction_preserves_raw_scores_and_records_reproducible_inputs(
    tmp_path,
):
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    sotopia_archive(baseline, 0)
    raw = sotopia_archive(candidate, 200)
    trials = candidate_trials(candidate)
    trials_dir = UPath(candidate / "harbor_jobs" / "job")
    before = [trial.model_dump() for trial in trials]
    cohort = {
        "version": "test-v1",
        "models": [{"model": "baseline", "results_path": str(baseline)}],
    }
    result = correct_sotopia(
        trials, model="candidate", trials_dir=trials_dir, cohort=SotopiaCohort.model_validate(cohort)
    )
    adjusted = [
        row.length_controlled_overall
        for row in result.adjusted_episodes
        if row.model == "candidate"
    ]
    assert result.canonical_metrics["length_controlled_overall"] == pytest.approx(
        sum(adjusted) / 100
    )
    assert sum(adjusted) / 100 != pytest.approx(raw, abs=0.01)
    assert [trial.model_dump() for trial in trials] == before
    assert result.cohort == cohort
    assert len(result.source_episodes) == 200
    assert correct_sotopia(
        trials, model="candidate", trials_dir=trials_dir, cohort=SotopiaCohort.model_validate(cohort)
    ) == result
    replacement = {
        **cohort,
        "models": [
            *cohort["models"],
            {"model": "candidate", "results_path": str(tmp_path / "superseded")},
        ],
    }
    replaced = correct_sotopia(
        trials, model="candidate", trials_dir=trials_dir, cohort=SotopiaCohort.model_validate(replacement)
    )
    assert replaced.source_episodes == result.source_episodes
    assert replaced.canonical_metrics == result.canonical_metrics
    relocated = tmp_path / "relocated-candidate"
    candidate.rename(relocated)
    trials_dir = UPath(relocated / "harbor_jobs" / "job")
    resumed = correct_sotopia(
        trials, model="candidate", trials_dir=trials_dir, cohort=SotopiaCohort.model_validate(cohort)
    )
    assert resumed == result
    next((baseline / "harbor_jobs").rglob("sotopia-summary.json")).unlink()
    with pytest.raises(ValueError, match="missing SOTOPIA summary"):
        correct_sotopia(
            trials, model="candidate", trials_dir=trials_dir, cohort=SotopiaCohort.model_validate(cohort)
        )


def test_length_control_removes_synthetic_verbosity_bonus() -> None:
    episodes = []
    for task_index in range(40):
        task = f"task-{task_index}"
        task_effect = (task_index % 5 - 2) * 0.08
        short_length = 80 + 3 * task_index
        long_length = 280 + 11 * task_index
        median = (short_length + long_length) / 2
        scale = 180
        for model, length in (("short", short_length), ("long", long_length)):
            length_bonus = 0.9 * math.tanh((length - median) / scale)
            score = 1 / (1 + math.exp(-(0.7 + task_effect + length_bonus)))
            episodes.append(EpisodeScore(model, task, score, length))

    fit = length_controlled_scores(episodes, regularization=(100.0,))
    by_model = {row.model: row for row in fit.scores}

    assert by_model["long"].raw_overall > by_model["short"].raw_overall + 0.15
    assert math.isclose(
        by_model["long"].length_controlled_overall,
        by_model["short"].length_controlled_overall,
        abs_tol=0.02,
    )


async def test_resumed_job_matches_original_model_to_cohort_with_new_serving_alias(tmp_path):
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    sotopia_archive(baseline, 0)
    sotopia_archive(candidate, 200)
    trials = candidate_trials(candidate)
    tasks = []
    agent = AgentConfig(
        import_path="harbor.agents.installed.sotopia:SotopiaAgent",
        model_name="hosted_vllm/org/candidate",
        model_alias="hosted_vllm/old-endpoint",
    )
    for trial in trials:
        task_dir = tmp_path / "tasks" / trial.task_name
        task_dir.mkdir(parents=True)
        (task_dir / "task.toml").write_text(f'[task]\nname = "{trial.task_name}"\n')
        (task_dir / "instruction.md").write_text("Continue the conversation.")
        (task_dir / "environment").mkdir()
        (task_dir / "tests").mkdir()
        (task_dir / "tests" / "test.sh").write_text("exit 0\n")
        task = TaskConfig(path=task_dir)
        tasks.append(task)
        trial.config = TrialConfig(task=task, agent=agent, trial_name=trial.trial_name)
        directory = candidate / "harbor_jobs" / "job" / trial.trial_name
        (directory / "config.json").write_text(trial.config.model_dump_json())
        (directory / "result.json").write_text(trial.model_dump_json())
    config = JobConfig(
        job_name="job", jobs_dir=candidate / "harbor_jobs", tasks=tasks, agents=[agent],
        sotopia_cohort=SotopiaCohort.model_validate({
            "version": "test-v1", "models": [
                {"model": "baseline", "results_path": str(baseline)},
                {"model": "org/candidate", "results_path": str(tmp_path / "superseded")},
            ],
        }),
    )
    job_dir = candidate / "harbor_jobs" / "job"
    (job_dir / "config.json").write_text(config.model_dump_json())
    resumed_config = config.model_copy(update={
        "agents": [agent.model_copy(update={"model_alias": "hosted_vllm/new-endpoint"})],
    })
    job = await Job.create(resumed_config)
    try:
        result = await job.run()
        persisted = JobResult.model_validate_json((job_dir / "result.json").read_text())
        evidence = persisted.metric_evidence["adhoc"]
        assert len(result.trial_results) == 100
        assert persisted.stats.n_completed_trials == 100
        assert {row["model"] for row in evidence["source_episodes"]} == {"baseline", "org/candidate"}
        assert len(evidence["source_episodes"]) == 200
        assert persisted.canonical_metrics["adhoc"]["length_controlled_overall"] == pytest.approx(
            next(row["length_controlled_overall"] for row in evidence["scores"] if row["model"] == "org/candidate")
        )
    finally:
        job._close_logger_handlers()
