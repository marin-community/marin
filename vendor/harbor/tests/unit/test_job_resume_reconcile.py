"""Tests for `Job._init_remaining_trial_configs` — the `harbor jobs resume`
reconciliation that matches freshly-PLANNED trial configs against the trial
dirs already on disk (the partial progress a resume must keep).

Regression target (2026-06-29): a partial eval resume re-runs NOTHING and
aborts with `ValueError: Existing trial config does not match planned job
config` whenever a benign runtime-serving field drifts between the original
legs and the resume, such as `agent.model_alias`, `agent.kwargs.api_base`, or a
timeout multiplier. None of those change what a trial is, so the existing
trials must be kept and the resume must proceed. A semantic mismatch, including
an immutable `agent.model_name` change, must still raise.
"""

from __future__ import annotations

import copy
import io
import logging
from pathlib import Path

import pytest

from harbor.job import Job
from harbor.models.job.reconciliation import (
    TrialPlanMismatchError,
    TrialPlanMismatchKind,
)
from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
)


def _make_trial(
    *,
    task_name: str = "harbor/hello-world",
    agent_name: str = "terminus-2",
    model_name: str = "hosted_vllm/laion/some-model",
    model_alias: str | None = None,
    api_base: str = "http://node-a:8000/v1",
    timeout_multiplier: float = 1.0,
    trials_dir: Path = Path("/jobs/run-1"),
    env_kwargs: dict | None = None,
) -> TrialConfig:
    return TrialConfig(
        task=TaskConfig(name=task_name, ref="latest"),
        trials_dir=trials_dir,
        timeout_multiplier=timeout_multiplier,
        agent=AgentConfig(
            name=agent_name,
            model_name=model_name,
            model_alias=model_alias,
            kwargs={"api_base": api_base},
        ),
        environment=EnvironmentConfig(kwargs=env_kwargs or {}),
    )


def _reconcile(planned: list[TrialConfig], existing: list[TrialConfig]) -> Job:
    """Run the reconciliation in isolation on a bare Job instance."""
    job = object.__new__(Job)
    # Deep-copy so the method's in-place reuse doesn't alias the inputs.
    job._trial_configs = copy.deepcopy(planned)
    job._existing_trial_configs = copy.deepcopy(existing)
    job._init_remaining_trial_configs()
    return job


def test_exact_match_keeps_all_trials_nothing_remaining() -> None:
    """No drift at all: every planned trial binds its exact on-disk twin."""
    existing = [_make_trial(), _make_trial(task_name="harbor/other")]
    planned = copy.deepcopy(existing)

    job = _reconcile(planned, existing)

    assert job._remaining_trial_configs == []
    assert len(job._trial_configs) == 2


def test_served_model_alias_drift_resume_proceeds() -> None:
    existing = [_make_trial(model_alias="hosted_vllm/laion/original-endpoint")]
    planned = [_make_trial(model_alias="hosted_vllm/laion/resumed-endpoint")]

    job = _reconcile(planned, existing)

    assert job._remaining_trial_configs == []
    assert job._trial_configs[0].agent.model_alias == (
        "hosted_vllm/laion/original-endpoint"
    )


def test_model_name_drift_resume_rejected() -> None:
    existing = [_make_trial(model_name="hosted_vllm/laion/model-a")]
    planned = [_make_trial(model_name="hosted_vllm/laion/model-b")]

    with pytest.raises(TrialPlanMismatchError) as exc_info:
        _reconcile(planned, existing)

    assert exc_info.value.mismatches[0].differing_fields == ("agent.model_name",)


def test_api_base_drift_resume_proceeds() -> None:
    """`api_base` rewritten to the live MASTER_ADDR is benign serving drift."""
    existing = [_make_trial(api_base="http://dead-node:8000/v1")]
    planned = [_make_trial(api_base="http://live-node:8000/v1")]

    job = _reconcile(planned, existing)

    assert job._remaining_trial_configs == []


def test_api_base_drift_log_preserves_capability_urls() -> None:
    old_endpoint = "https://old.iris.example/proxy/t/old-token/eval/v1"
    new_endpoint = "https://new.iris.example/proxy/t/new-token/eval/v1"
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    harbor_logger = logging.getLogger("harbor.utils.logger")
    harbor_logger.addHandler(handler)

    try:
        _reconcile(
            [_make_trial(api_base=new_endpoint)],
            [_make_trial(api_base=old_endpoint)],
        )
    finally:
        harbor_logger.removeHandler(handler)

    log_output = stream.getvalue()
    assert old_endpoint in log_output
    assert new_endpoint in log_output


def test_timeout_multiplier_drift_resume_proceeds() -> None:
    """A timeout-multiplier change (e.g. enforce-eager slowdown) is benign."""
    existing = [_make_trial(timeout_multiplier=1.0)]
    planned = [_make_trial(timeout_multiplier=16.0)]

    job = _reconcile(planned, existing)

    assert job._remaining_trial_configs == []


def test_serving_drift_plus_genuinely_missing_trial() -> None:
    """Mixed: one trial drifts (keep), one planned trial is new (re-run)."""
    existing = [_make_trial(api_base="http://old:8000/v1")]
    planned = [
        _make_trial(api_base="http://new:8000/v1"),  # identity-matches existing
        _make_trial(task_name="harbor/brand-new-task"),  # genuinely missing
    ]

    job = _reconcile(planned, existing)

    assert len(job._remaining_trial_configs) == 1
    assert job._remaining_trial_configs[0].task.name == "harbor/brand-new-task"


def test_task_identity_drift_still_raises() -> None:
    """SEMANTIC mismatch: the on-disk trial is for a DIFFERENT task and has no
    counterpart in the planned job — resume must still abort (never silently
    reuse trials from a different evaluation)."""
    existing = [_make_trial(task_name="harbor/task-A")]
    planned = [_make_trial(task_name="harbor/task-B")]

    with pytest.raises(ValueError, match="SEMANTIC mismatch"):
        _reconcile(planned, existing)


def test_hf_dataset_materialization_path_drift_resume_proceeds() -> None:
    """A pinned HF task retains its result after a fresh local materialization."""
    existing = [
        TrialConfig(
            task=TaskConfig(
                path=Path("/tmp/harbor-run-old/hf_dataset/astropy__astropy-13236"),
                source="hf_dataset",
            )
        )
    ]
    planned = [
        TrialConfig(
            task=TaskConfig(
                path=Path("/tmp/harbor-run-new/hf_dataset/astropy__astropy-13236"),
                source="hf_dataset",
            )
        )
    ]

    job = _reconcile(planned, existing)

    assert job._remaining_trial_configs == []


def test_extra_existing_trial_with_no_plan_counterpart_raises() -> None:
    """More on-disk trials than the plan can account for (e.g. n_attempts /
    dataset shrank) → unmatched existing trial → raise."""
    existing = [_make_trial(), _make_trial(task_name="harbor/dropped-task")]
    # Plan only contains the first (serving-drifted) trial.
    planned = [_make_trial(api_base="http://new:8000/v1")]

    with pytest.raises(ValueError, match="does not match planned job config"):
        _reconcile(planned, existing)


def test_resume_mismatch_classifies_every_incompatible_trial() -> None:
    planned = [
        _make_trial(task_name="harbor/kept"),
        _make_trial(task_name="harbor/changed", agent_name="terminus-2"),
    ]
    existing = [
        _make_trial(task_name="harbor/kept"),
        _make_trial(task_name="harbor/kept"),
        _make_trial(task_name="harbor/changed", agent_name="oracle"),
        _make_trial(task_name="harbor/removed"),
    ]

    with pytest.raises(TrialPlanMismatchError) as exc_info:
        _reconcile(planned, existing)

    error = exc_info.value
    assert error.compatible_existing_count == 1
    assert [mismatch.kind for mismatch in error.mismatches] == [
        TrialPlanMismatchKind.EXCESS_EXISTING_ATTEMPT,
        TrialPlanMismatchKind.SEMANTIC_CONFIG_MISMATCH,
        TrialPlanMismatchKind.TASK_NOT_IN_PLAN,
    ]
    assert [mismatch.task_name for mismatch in error.mismatches] == [
        "harbor/kept",
        "harbor/changed",
        "harbor/removed",
    ]
    assert error.mismatches[1].differing_fields == ("agent.name",)


def test_resume_mismatch_lists_every_incompatible_directory() -> None:
    planned = [_make_trial(task_name="harbor/kept")]
    existing = [
        _make_trial(task_name="harbor/kept"),
        _make_trial(task_name="harbor/removed-a"),
        _make_trial(task_name="harbor/removed-b"),
    ]
    existing[1].trial_name = "removed-a__trial"
    existing[2].trial_name = "removed-b__trial"

    with pytest.raises(TrialPlanMismatchError) as exc_info:
        _reconcile(planned, existing)

    assert [mismatch.trial_name for mismatch in exc_info.value.mismatches] == [
        "removed-a__trial",
        "removed-b__trial",
    ]


@pytest.mark.parametrize(
    "provisioning_kwarg",
    [
        {"auto_snapshot": True},
        {"snapshot_template_name": "harbor__{name}__snapshot"},
        {"assume_global_snapshot": True},
    ],
)
def test_environment_cache_acquisition_kwarg_drift_resume_proceeds(
    provisioning_kwarg: dict,
) -> None:
    """A Daytona cache-acquisition flag (auto_snapshot and the sibling snapshot
    knobs) does not change what a trial evaluates — only how its sandbox is
    provisioned/cached. A trial launched before auto_snapshot was enabled must
    resume (be kept, not re-run) when the flag is later turned on, instead of
    being rejected as a different evaluation because the flag leaked into the
    identity fingerprint."""
    existing = [_make_trial(env_kwargs={})]
    planned = [_make_trial(env_kwargs=provisioning_kwarg)]

    job = _reconcile(planned, existing)

    assert job._remaining_trial_configs == []


def test_environment_semantic_kwarg_drift_still_raises() -> None:
    """A genuinely environment-affecting kwarg (one that is NOT a
    cache-acquisition knob) must still count as a semantic mismatch — the fix
    must not over-broaden identity to ignore every environment kwarg. The
    existing trial has no identity counterpart in the plan, so resume aborts."""
    existing = [_make_trial(env_kwargs={"network_block_all": False})]
    planned = [_make_trial(env_kwargs={"network_block_all": True})]

    with pytest.raises(ValueError, match="SEMANTIC mismatch"):
        _reconcile(planned, existing)
