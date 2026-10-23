from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from harbor.models.agent.context import AgentContext
from harbor.models.job.config import JobConfig
from harbor.models.job.result import JobResult, JobStats
from harbor.models.trial.config import AgentConfig, TaskConfig, TrialConfig
from harbor.models.trial.result import (
    AgentInfo,
    ExceptionInfo,
    ModelInfo,
    StepResult,
    TrialResult,
)
from harbor.models.verifier.result import VerifierResult


def _trial_result(
    name: str,
    *,
    exception_type: str | None = None,
    rewards: dict[str, float | int] | None = None,
) -> TrialResult:
    config = TrialConfig(
        task=TaskConfig(path=Path(f"/tmp/{name}")),
        trial_name=name,
        job_id=uuid4(),
    )
    exception_info = None
    if exception_type is not None:
        exception_info = ExceptionInfo(
            exception_type=exception_type,
            exception_message="failed",
            exception_traceback="traceback",
            occurred_at=datetime(2026, 4, 28, tzinfo=timezone.utc),
        )
    return TrialResult(
        task_name=name,
        trial_name=name,
        trial_uri=f"file:///tmp/{name}",
        task_id=config.task.get_task_id(),
        task_checksum="abc123",
        config=config,
        agent_info=AgentInfo(name="test-agent", version="1.0"),
        verifier_result=VerifierResult(rewards=rewards)
        if rewards is not None
        else None,
        exception_info=exception_info,
    )


def test_job_stats_groups_trials_by_model_name_across_serving_aliases() -> None:
    canonical_name = "hosted_vllm/org/canonical-model"
    trials = []
    for index, alias in enumerate(("original-endpoint", "resumed-endpoint")):
        trial = _trial_result(f"trial-{index}", rewards={"reward": index})
        trial.source = "benchmark"
        trial.config.agent = AgentConfig(
            name="test-agent",
            model_name=canonical_name,
            model_alias=f"hosted_vllm/{alias}",
        )
        trial.agent_info = AgentInfo(
            name="test-agent",
            version="1.0",
            model_info=ModelInfo(name=alias, provider="hosted_vllm"),
        )
        trials.append(trial)

    stats = JobStats.from_trial_results(trials)

    assert list(stats.evals) == ["test-agent__org/canonical-model__benchmark"]
    assert stats.evals["test-agent__org/canonical-model__benchmark"].n_trials == 2


def _heavy_rollout_detail(n_tokens: int = 4096) -> dict:
    """A rollout detail large enough to dominate a trial's memory footprint."""
    return {
        "prompt_token_ids": [list(range(n_tokens))],
        "completion_token_ids": [list(range(n_tokens))],
        "logprobs": [[-0.1] * n_tokens],
        "extra": {"observations": ["x" * 1024] * 8},
    }


def _heavy_single_step_trial() -> TrialResult:
    """Single-step trial with heavy agent and verifier payloads."""
    config = TrialConfig(
        task=TaskConfig(path=Path("/tmp/heavy")),
        trial_name="heavy",
        job_id=uuid4(),
    )
    return TrialResult(
        task_name="heavy",
        trial_name="heavy",
        trial_uri="file:///tmp/heavy",
        task_id=config.task.get_task_id(),
        task_checksum="abc123",
        config=config,
        agent_info=AgentInfo(
            name="test-agent",
            version="1.0",
            model_info=ModelInfo(name="gpt-x", provider="openai"),
        ),
        source="my-dataset",
        agent_result=AgentContext(
            n_input_tokens=100,
            n_cache_tokens=10,
            n_output_tokens=50,
            cost_usd=0.25,
            rollout_details=[_heavy_rollout_detail()],
            metadata={"trajectory": ["step" * 4096]},
        ),
        verifier_result=VerifierResult(
            rewards={"reward": 1},
            stdout="verifier stdout\n" * 4096,
            stderr="verifier stderr\n" * 4096,
        ),
    )


def _heavy_multi_step_trial() -> TrialResult:
    """Multi-step trial: token scalars live on each step's agent_result."""
    config = TrialConfig(
        task=TaskConfig(path=Path("/tmp/heavymulti")),
        trial_name="heavymulti",
        job_id=uuid4(),
    )
    steps = [
        StepResult(
            step_name=f"step-{i}",
            agent_result=AgentContext(
                n_input_tokens=10 * (i + 1),
                n_cache_tokens=i,
                n_output_tokens=5 * (i + 1),
                cost_usd=0.1 * (i + 1),
                rollout_details=[_heavy_rollout_detail()],
                metadata={"big": ["y" * 4096]},
            ),
            verifier_result=VerifierResult(rewards={"reward": i % 2}),
        )
        for i in range(3)
    ]
    return TrialResult(
        task_name="heavymulti",
        trial_name="heavymulti",
        trial_uri="file:///tmp/heavymulti",
        task_id=config.task.get_task_id(),
        task_checksum="abc123",
        config=config,
        agent_info=AgentInfo(name="test-agent", version="1.0"),
        source="my-dataset",
        verifier_result=VerifierResult(
            rewards={"reward": 1},
            stdout="verifier stdout\n" * 4096,
            stderr="verifier stderr\n" * 4096,
        ),
        step_results=steps,
    )


@pytest.mark.unit
def test_job_config_release_trial_payloads_defaults_false() -> None:
    assert JobConfig().release_trial_payloads_in_memory is False


@pytest.mark.unit
@pytest.mark.parametrize(
    "make_trial", [_heavy_single_step_trial, _heavy_multi_step_trial]
)
def test_slimmed_preserves_token_cost_totals(make_trial) -> None:
    full = make_trial()
    assert full.slimmed().compute_token_cost_totals() == (
        full.compute_token_cost_totals()
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "make_trial", [_heavy_single_step_trial, _heavy_multi_step_trial]
)
def test_slimmed_drops_heavy_fields_keeps_stats_fields(make_trial) -> None:
    full = make_trial()
    slim = full.slimmed()

    # Heavy payloads gone.
    assert slim.step_results is None
    assert slim.agent_result is not None
    assert slim.agent_result.rollout_details is None
    assert slim.agent_result.metadata is None
    assert slim.verifier_result is not None
    assert slim.verifier_result.stdout is None
    assert slim.verifier_result.stderr is None

    # Stats-relevant fields survive.
    assert slim.verifier_result.rewards == full.verifier_result.rewards
    assert slim.agent_info == full.agent_info
    assert slim.source == full.source
    assert slim.trial_name == full.trial_name


@pytest.mark.unit
def test_slimmed_keeps_exception_classification_without_diagnostics() -> None:
    full = _trial_result("failed", exception_type="RuntimeError")
    full.exception_info.exception_message = "failure details\n" * 4096
    full.exception_info.exception_traceback = "traceback\n" * 4096
    full.secondary_exception_info = [
        ExceptionInfo(
            exception_type="CleanupError",
            exception_message="cleanup details\n" * 4096,
            exception_traceback="cleanup traceback\n" * 4096,
            occurred_at=datetime(2026, 4, 28, tzinfo=timezone.utc),
        )
    ]

    slim = full.slimmed()

    assert slim.exception_info is not None
    assert slim.exception_info.exception_type == "RuntimeError"
    assert slim.exception_info.exception_message == ""
    assert slim.exception_info.exception_traceback == ""
    assert slim.exception_info.occurred_at == full.exception_info.occurred_at
    assert slim.secondary_exception_info == []
    assert JobStats.from_trial_results([slim]) == JobStats.from_trial_results([full])


@pytest.mark.unit
@pytest.mark.parametrize(
    "make_trial", [_heavy_single_step_trial, _heavy_multi_step_trial]
)
def test_increment_identical_for_full_and_slimmed(make_trial) -> None:
    full = make_trial()
    slim = full.slimmed()

    full_stats = JobStats()
    full_stats.increment(full)
    slim_stats = JobStats()
    slim_stats.increment(slim)

    assert slim_stats.model_dump() == full_stats.model_dump()


@pytest.mark.unit
@pytest.mark.parametrize(
    "make_trial", [_heavy_single_step_trial, _heavy_multi_step_trial]
)
def test_increment_then_remove_retry_roundtrip_matches(make_trial) -> None:
    # Simulate a retry: increment, then remove the previous attempt. The
    # removed copy is the slim one (what the job retains in
    # _previous_trial_results); the result must net to the same JobStats as
    # removing the full copy.
    full = make_trial()
    slim = full.slimmed()

    full_stats = JobStats()
    full_stats.increment(full)
    full_stats.remove_trial(full)

    slim_stats = JobStats()
    slim_stats.increment(full)
    slim_stats.remove_trial(slim)

    assert slim_stats.model_dump() == full_stats.model_dump()


@pytest.mark.unit
def test_job_result_derives_progress_stats_for_legacy_payload() -> None:
    started_at = datetime(2026, 4, 28, 9, 0, tzinfo=timezone.utc)
    payload = {
        "id": str(uuid4()),
        "started_at": started_at.isoformat(),
        "finished_at": None,
        "n_total_trials": 3,
        "stats": {"n_trials": 1, "n_errors": 1, "evals": {}},
    }

    result = JobResult.model_validate(payload)

    assert result.stats.n_completed_trials == 1
    assert result.stats.n_errored_trials == 1
    assert result.stats.n_running_trials == 0
    assert result.stats.n_pending_trials == 2
    assert result.stats.n_cancelled_trials == 0
    assert result.stats.n_retries == 0
    assert result.updated_at == started_at


@pytest.mark.unit
def test_job_result_derives_completed_progress_for_legacy_final_payload() -> None:
    started_at = datetime(2026, 4, 28, 9, 0, tzinfo=timezone.utc)
    finished_at = datetime(2026, 4, 28, 9, 30, tzinfo=timezone.utc)
    payload = {
        "id": str(uuid4()),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "n_total_trials": 2,
        "stats": {"n_trials": 2, "n_errors": 0, "evals": {}},
    }

    result = JobResult.model_validate(payload)

    assert result.stats.n_pending_trials == 0
    assert result.updated_at == finished_at


@pytest.mark.unit
def test_job_stats_counts_trial_results() -> None:
    trial_results = [
        _trial_result("passed"),
        _trial_result("errored", exception_type="RuntimeError"),
        _trial_result("cancelled", exception_type="CancelledError"),
    ]

    stats = JobStats.from_trial_results(
        trial_results,
        n_total_trials=5,
        n_running_trials=1,
        n_retries=2,
    )

    assert stats.n_completed_trials == 3
    assert stats.n_running_trials == 1
    assert stats.n_pending_trials == 1
    assert stats.n_errored_trials == 2
    assert stats.n_cancelled_trials == 1
    assert stats.n_retries == 2


@pytest.mark.unit
def test_job_stats_accepts_legacy_field_names() -> None:
    stats = JobStats.model_validate(
        {
            "n_trials": 2,
            "n_errors": 1,
        }
    )

    assert stats.n_completed_trials == 2
    assert stats.n_running_trials == 0
    assert stats.n_pending_trials == 0
    assert stats.n_errored_trials == 1
    assert stats.n_cancelled_trials == 0
    assert stats.n_retries == 0


@pytest.mark.unit
def test_job_result_serializes_progress_stats() -> None:
    trial_result = _trial_result("passed")
    started_at = datetime(2026, 4, 28, 9, 0, tzinfo=timezone.utc)
    finished_at = datetime(2026, 4, 28, 9, 5, tzinfo=timezone.utc)
    stats = JobStats.from_trial_results([trial_result], n_total_trials=1)

    result = JobResult(
        id=uuid4(),
        started_at=started_at,
        updated_at=finished_at,
        finished_at=finished_at,
        n_total_trials=1,
        stats=stats,
    )

    restored = JobResult.model_validate_json(result.model_dump_json())
    serialized = result.model_dump()

    assert restored.stats == stats
    assert "status" not in serialized
    assert "n_trials" not in serialized["stats"]
    assert "n_errors" not in serialized["stats"]


@pytest.mark.unit
def test_job_stats_updates_after_json_round_trip() -> None:
    started_at = datetime(2026, 4, 28, 9, 0, tzinfo=timezone.utc)
    result = JobResult(
        id=uuid4(),
        started_at=started_at,
        updated_at=started_at,
        finished_at=None,
        n_total_trials=2,
        stats=JobStats.from_trial_results(
            [_trial_result("passed", rewards={"reward": 1})],
            n_total_trials=2,
        ),
    )
    restored = JobResult.model_validate_json(result.model_dump_json())
    cancelled = _trial_result(
        "cancelled",
        exception_type="CancelledError",
        rewards={"reward": 0},
    )

    restored.stats.increment(cancelled)
    restored.stats.remove_trial(cancelled)

    eval_stats = next(iter(restored.stats.evals.values()))
    assert restored.stats.n_completed_trials == 1
    assert restored.stats.n_errored_trials == 0
    assert restored.stats.n_cancelled_trials == 0
    assert eval_stats.exception_stats["CancelledError"] == []
    assert eval_stats.reward_stats["reward"][0] == []


@pytest.mark.unit
def test_job_result_migrates_legacy_status_into_stats() -> None:
    started_at = datetime(2026, 4, 28, 9, 0, tzinfo=timezone.utc)
    payload = {
        "id": str(uuid4()),
        "started_at": started_at.isoformat(),
        "updated_at": started_at.isoformat(),
        "finished_at": None,
        "n_total_trials": 4,
        "stats": {"n_trials": 2, "n_errors": 1, "evals": {}},
        "status": {
            "n_completed_trials": 2,
            "n_running_trials": 1,
            "n_pending_trials": 1,
            "n_errored_trials": 1,
            "n_cancelled_trials": 0,
            "n_retries": 1,
        },
    }

    result = JobResult.model_validate(payload)

    assert result.stats.n_completed_trials == 2
    assert result.stats.n_errored_trials == 1
    assert result.stats.n_running_trials == 1
    assert result.stats.n_pending_trials == 1
    assert result.stats.n_cancelled_trials == 0
    assert result.stats.n_retries == 1
