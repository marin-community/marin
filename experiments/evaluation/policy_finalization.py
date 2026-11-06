# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind policy #10197 judging to evaluation completion."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import yaml
from finestore.reader import ReadView
from marin.evaluation.curated import CuratedExecutor
from marin.evaluation.hardware import AcceleratorChoice
from marin.evaluation.model_config import ModelConfig, load_model_config
from marin.evaluation.records import EvalRef, JudgeIdentity, RunStatus
from marin.evaluation.runner import (
    EvalExecutor,
    EvaluationError,
    EvaluationOutcome,
    check_inference_backend,
    job_diagnostics,
)
from marin.evaluation.serving_config import inference_config_for_model
from marin.inference.iris import RemoteInferenceSession, RemoteInferenceStartupError, remote_inference
from marin.inference.types import InferenceBackendState
from marin.rollouts.evaluation_results import evaluation_coverage

POLICY = "marin-10197"
MIN_SCORED_FRACTION = 0.9
POLICY_JUDGED_BENCHMARKS = frozenset({"financebench", "olympiadbench"})


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


@dataclass(frozen=True)
class PolicyJudge:
    """Run curated evaluations with the policy answer-equivalence judge."""

    source: CuratedExecutor
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
        try:
            with remote_inference(inference) as policy_judge:
                jobs = {f"policy-judge-{i}": str(job.job_id) for i, job in enumerate(policy_judge.jobs)}
                policy_judge.wait_until_ready(check_dependencies=lambda: check_inference_backend(backend_state()))
                assert self.model.revision is not None
                configuration = JudgeIdentity(
                    deployment=self.model.location,
                    models={self.model.location: self.model.revision},
                    default_model=self.model.location,
                )
                executor = replace(
                    self.source, settings=self.source.settings.model_copy(update={"judge": configuration})
                )

                def dependencies() -> InferenceBackendState:
                    check_inference_backend(policy_judge.backend_state())
                    return backend_state()

                try:
                    outcome = executor(
                        session,
                        output_dir,
                        {**env_vars, "JUDGE_API_KEY": "local-policy-judge"},
                        backend_state=dependencies,
                        judge=policy_judge,
                    )
                except EvaluationError as exc:
                    exc.jobs.update(jobs)
                    raise
                jobs.update(outcome.jobs)
                coverage = evaluation_coverage(ReadView(output_dir))
                if not coverage or any(
                    not item.n_attempted or item.n_scored / item.n_attempted < MIN_SCORED_FRACTION
                    for item in coverage.values()
                ):
                    raise EvaluationError(
                        "policy judge scored fewer than 90% of selected trials",
                        status=RunStatus.ARTIFACT_FAILED,
                        jobs=jobs,
                        coverage=coverage,
                    )
                return replace(outcome, jobs=jobs, coverage=coverage)
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
    if reference.name in POLICY_JUDGED_BENCHMARKS:
        assert judge_accelerator is not None
        if not isinstance(source, CuratedExecutor):
            raise TypeError("policy answer-equivalence grading requires curated tasks")
        assert config.judge.revision is not None
        source = replace(
            source,
            settings=source.settings.model_copy(
                update={
                    "judge": JudgeIdentity(
                        deployment=config.judge.location,
                        models={config.judge.location: config.judge.revision},
                        default_model=config.judge.location,
                    )
                }
            ),
        )
        assert reference.curated is not None
        reference = reference.model_copy(
            update={"curated": reference.curated.model_copy(update={"settings": source.settings})}
        )
        executor = PolicyJudge(
            source,
            config.judge,
            judge_accelerator,
            priority,
        )
        configuration = {"judge": asdict(config.judge)}
    else:
        return source, reference
    return executor, reference.model_copy(update={"postprocessing": {POLICY: json.dumps(configuration, sort_keys=True)}})
