# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind policy #10197 grading and cohort correction to evaluation completion."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import yaml
from finestore.eval import sample_from_archive_row
from finestore.reader import ReadView
from marin.evaluation.evalchemy.config import EvalchemyJudgeConfig
from marin.evaluation.evalchemy.runner import EvalchemyExecutor
from marin.evaluation.hardware import AcceleratorChoice
from marin.evaluation.lm_eval_samples import task_coverage
from marin.evaluation.model_config import ModelConfig, load_model_config
from marin.evaluation.records import EvalRef, RunStatus
from marin.evaluation.runner import (
    EvalExecutor,
    EvaluationError,
    EvaluationOutcome,
    check_inference_backend,
    job_diagnostics,
)
from marin.evaluation.serving_config import inference_config_for_model
from marin.inference.iris import RemoteInferenceSession, RemoteInferenceStartupError, remote_inference
from marin.inference.model_preparation import precache_inference_model
from marin.inference.types import InferenceBackendState

POLICY = "marin-10197"
MIN_SCORED_FRACTION = 0.9
OLYMPIAD_REPETITIONS = 10


@dataclass(frozen=True)
class PolicyFinalization:
    judge: ModelConfig


def load_policy_finalization(path: Path) -> PolicyFinalization:
    """Freeze the judge model before submitting a policy launch."""
    config = yaml.safe_load(path.read_text())
    judge = load_model_config(path.parent / config["judge_model_config"])
    if (
        judge.location != "MiniMaxAI/MiniMax-M3-MXFP8"
        or judge.revision != "c5454eb03678d8710e54a4e0fc681b9f3b4a3dba"
        or judge.serve.reasoning_parser != "minimax_m3"
        or judge.serve.max_model_len != 262144
        or judge.serve.tensor_parallel_size != 8
    ):
        raise ValueError("policy #10197 requires the 262144-token TP8 answer-equivalence judge")
    return PolicyFinalization(judge)


def _complete_coverage(outcome: EvaluationOutcome) -> None:
    if not outcome.coverage:
        raise ValueError("policy finalization requires declared trial coverage")
    for task, coverage in outcome.coverage.items():
        if (
            coverage.n_attempted is None
            or coverage.n_attempted <= 0
            or coverage.n_scored < MIN_SCORED_FRACTION * coverage.n_attempted
        ):
            raise ValueError(f"{task}: fewer than 90% of declared trials have valid scores")


def trial_coverage(outcome: EvaluationOutcome, root: str, repetitions: int) -> EvaluationOutcome:
    """Count seeded repetitions separately while preserving native aggregate metrics."""
    task, original = next(iter(outcome.coverage.items()))
    if repetitions == 1:
        return outcome
    samples = [
        sample_from_archive_row(row).model_copy(update={"doc_id": json.dumps([row["doc_id"], row["trial_id"]])})
        for row in ReadView(root).iter_rows("samples")
        if row["task"] == task
    ]
    assert original.n_attempted is not None
    coverage = task_coverage(
        samples,
        n_attempted=original.n_attempted * repetitions,
        n_benchmark=original.n_benchmark * repetitions if original.n_benchmark is not None else None,
    )
    reference = outcome.tasks[0]
    assert reference.benchmark is not None
    benchmark = reference.benchmark.model_copy(
        update={
            "n_attempted": coverage.n_attempted,
            "n_benchmark": coverage.n_benchmark,
        }
    )
    return replace(outcome, coverage={task: coverage}, tasks=(reference.model_copy(update={"benchmark": benchmark}),))


@dataclass(frozen=True)
class PolicyJudge:
    """Run Evalchemy with the policy judge as its native grading endpoint."""

    source: EvalchemyExecutor
    repetitions: int
    model: ModelConfig
    accelerator: AcceleratorChoice
    priority: int

    def __call__(
        self,
        session: RemoteInferenceSession,
        output_dir: str,
        env_vars: Mapping[str, str],
        *,
        backend_state: Callable[[], InferenceBackendState],
        judge: RemoteInferenceSession | None = None,
    ) -> EvaluationOutcome:
        inference = inference_config_for_model(
            self.model,
            self.accelerator,
            env_vars=env_vars,
            priority=self.priority,
            api_model=self.model.location,
        )
        precache_inference_model(inference)
        try:
            with remote_inference(inference) as policy_judge:
                jobs = {f"policy-judge-{i}": str(job.job_id) for i, job in enumerate(policy_judge.jobs)}
                policy_judge.wait_until_ready(check_dependencies=lambda: check_inference_backend(backend_state()))
                configuration = EvalchemyJudgeConfig(
                    base_url=policy_judge.model.endpoint.base_url,
                    model=policy_judge.model.endpoint.model,
                    api_key=("env:JUDGE_API_KEY",),
                    protocol=POLICY,
                )
                executor = EvalchemyExecutor(replace(self.source.config, judge=configuration))

                def dependencies() -> InferenceBackendState:
                    check_inference_backend(policy_judge.backend_state())
                    return backend_state()

                try:
                    outcome = executor(
                        session,
                        output_dir,
                        {**env_vars, "JUDGE_API_KEY": "local-policy-judge"},
                        backend_state=dependencies,
                    )
                except EvaluationError as exc:
                    exc.jobs.update(jobs)
                    raise
                jobs.update(outcome.jobs)
                try:
                    outcome = trial_coverage(outcome, output_dir, self.repetitions)
                    _complete_coverage(outcome)
                except Exception as exc:
                    raise EvaluationError(
                        f"policy #10197 judge coverage failed: {exc}",
                        status=RunStatus.ARTIFACT_FAILED,
                        jobs=jobs,
                        coverage=outcome.coverage,
                    ) from exc
                return replace(outcome, jobs=jobs)
        except RemoteInferenceStartupError as exc:
            jobs, tails = job_diagnostics(exc.jobs, "policy-judge")
            raise EvaluationError(
                f"policy judge startup failed: {exc}",
                status=RunStatus.INFRA_FAILED,
                jobs=jobs,
                log_tails=tails,
            ) from exc


def policy_executor(
    source: EvalExecutor,
    reference: EvalRef,
    config: PolicyFinalization,
    judge_accelerator: AcceleratorChoice | None,
    priority: int,
) -> tuple[EvalExecutor, EvalRef]:
    """Bind each policy benchmark to its own grading configuration."""
    if reference.name in {"financebench", "olympiadbench"}:
        assert judge_accelerator is not None
        if not isinstance(source, EvalchemyExecutor):
            raise TypeError("policy answer-equivalence grading requires Evalchemy")
        executor = PolicyJudge(
            source,
            OLYMPIAD_REPETITIONS if reference.name == "olympiadbench" else 1,
            config.judge,
            judge_accelerator,
            priority,
        )
        configuration = {"judge": asdict(config.judge)}
    else:
        return source, reference
    return executor, reference.model_copy(update={"postprocessing": {POLICY: json.dumps(configuration, sort_keys=True)}})
