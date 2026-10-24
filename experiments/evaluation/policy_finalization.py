# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind policy #10197 grading and cohort correction to evaluation completion."""

from __future__ import annotations

import hashlib
import importlib
import json
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import yaml
from finestore.eval import EvaluationStore, sample_from_archive_row
from finestore.reader import ReadView
from iris.client.client import iris_ctx
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
from marin.evaluation.hardware import AcceleratorChoice
from marin.evaluation.model_config import ModelConfig, load_model_config
from marin.evaluation.records import EVALCHEMY_INFRASTRUCTURE_ERROR, EvalRef, RunStatus
from marin.evaluation.runner import (
    BACKEND_POLL_SECONDS,
    EvalExecutor,
    EvaluationError,
    EvaluationOutcome,
    InferenceBackendFinished,
    InferenceUnavailable,
    check_inference_backend,
)
from marin.evaluation.serving_config import inference_config_for_model
from marin.inference.iris import RemoteInferenceSession, remote_inference
from marin.inference.types import InferenceBackendState

from vendor.evalchemy.eval.contracts.policy import TrialVerdict

POLICY = "marin-10197"
MIN_SCORED_FRACTION = 0.9
REGRADE_ARTIFACT = "policy-10197/judge-verdicts.json"
SOTOPIA_ARTIFACT = "policy-10197/sotopia-length-control.json"
SOTOPIA_METRIC = "length_controlled_overall"


@dataclass(frozen=True)
class CohortSource:
    model: str
    results_path: str


@dataclass(frozen=True)
class PolicyFinalization:
    judge: ModelConfig
    sotopia_baselines: tuple[CohortSource, ...]

    def record_configuration(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def load_policy_finalization(path: Path) -> PolicyFinalization:
    """Freeze the judge model and baseline source runs before submitting a policy launch."""
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
    cohort = json.loads((path.parent / config["sotopia_baselines"]).read_text())
    sources = tuple(CohortSource(**row) for row in cohort["models"])
    if len(sources) != len({source.model for source in sources}):
        raise ValueError("SOTOPIA baseline cohort contains duplicate models")
    return PolicyFinalization(judge, sources)


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


def _final_metrics(outcome: EvaluationOutcome, score: float, metric: str, correct: int | None) -> EvaluationOutcome:
    if len(outcome.coverage) != 1 or not outcome.tasks or len(outcome.tasks) != 1:
        raise ValueError("policy finalization expects a single benchmark")
    task = next(iter(outcome.coverage))
    reference = outcome.tasks[0]
    if reference.benchmark is None:
        raise ValueError("policy finalization requires benchmark metric metadata")
    primary = reference.benchmark.primary_metric
    declared = next(value for value in reference.benchmark.metrics if value.name == primary)
    benchmark = reference.benchmark.model_copy(
        update={
            "primary_metric": metric,
            "metrics": (declared.model_copy(update={"name": metric, "source_name": metric}),),
            "n_benchmark": outcome.coverage[task].n_benchmark,
            "n_attempted": outcome.coverage[task].n_attempted,
        }
    )
    return replace(
        outcome,
        metrics={task: {**outcome.metrics.get(task, {}), metric: score}},
        canonical_metrics={task: {metric: score}},
        tasks=(reference.model_copy(update={"benchmark": benchmark}),),
        coverage={task: outcome.coverage[task].model_copy(update={"n_correct": correct})},
    )


@dataclass(frozen=True)
class MiniMaxRegrade:
    source: EvalExecutor
    benchmark: str
    config: PolicyFinalization
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
        outcome = self.source(session, output_dir, env_vars, backend_state=backend_state, judge=judge)
        jobs = dict(outcome.jobs)
        try:
            _complete_coverage(outcome)
            inference = inference_config_for_model(
                self.config.judge,
                self.accelerator,
                env_vars=env_vars,
                priority=self.priority,
                api_model=self.config.judge.location,
            )
            with remote_inference(inference) as regrader:
                jobs.update({f"policy-judge-{i}": str(job.job_id) for i, job in enumerate(regrader.jobs)})
                regrader.wait_until_ready(check_dependencies=lambda: check_inference_backend(backend_state()))
                payload = {
                    "results_path": output_dir,
                    "benchmark": self.benchmark,
                    "endpoint": regrader.model.endpoint.base_url,
                    "model": regrader.model.endpoint.model,
                    "judge_config": asdict(self.config.judge),
                }
                child = iris_ctx().client.submit(
                    name="policy-regrade-" + uuid.uuid4().hex[:8],
                    entrypoint=Entrypoint.from_command(
                        "bash",
                        "-c",
                        'cd "$IRIS_WORKDIR"\n'
                        'export UV_PROJECT_ENVIRONMENT="$IRIS_WORKDIR/vendor/evalchemy/.venv"\n'
                        "exec uv run --project vendor/evalchemy --frozen --no-default-groups --extra olympiadbench "
                        "python -m eval.policy_regrade",
                    ),
                    resources=ResourceSpec(cpu=8, memory="32g", disk="50g"),
                    environment=EnvironmentSpec(env_vars={**env_vars, "MARIN_POLICY_REGRADE": json.dumps(payload)}),
                    max_retries_failure=0,
                )
                jobs["policy-regrade"] = str(child.job_id)
                try:
                    while True:
                        check_inference_backend(backend_state())
                        regrader.check_alive()
                        try:
                            child.wait(timeout=BACKEND_POLL_SECONDS)
                            break
                        except TimeoutError:
                            continue
                except BaseException:
                    child.cancel()
                    child.wait(timeout=float("inf"), raise_on_failure=False)
                    raise
            payload = ReadView(output_dir).read_blob(REGRADE_ARTIFACT)
            if payload is None:
                raise ValueError("policy judge did not preserve its per-trial verdicts")
            verdicts = [TrialVerdict(**row) for row in json.loads(payload)["trials"]]
            task, original = next(iter(outcome.coverage.items()))
            repeats = 10 if self.benchmark == "olympiadbench" else 1
            assert original.n_attempted is not None
            expected = original.n_attempted * repeats
            identities = {(row.task, row.doc_id, row.trial_id) for row in verdicts}
            if len(identities) != len(verdicts) or not MIN_SCORED_FRACTION * expected <= len(verdicts) <= expected:
                raise ValueError("policy judge verdicts do not cover at least 90% of declared trials")
            correct = sum(value.score for value in verdicts)
            missing = expected - len(verdicts)
            coverage = original.model_copy(
                update={
                    "n_benchmark": original.n_benchmark * repeats if original.n_benchmark is not None else None,
                    "n_attempted": expected,
                    "n_scored": len(verdicts),
                    "errors": {EVALCHEMY_INFRASTRUCTURE_ERROR: missing} if missing else {},
                    "n_unanswered": sum(not row.extracted.strip() for row in verdicts),
                }
            )
            outcome = replace(outcome, coverage={task: coverage})
            return replace(_final_metrics(outcome, correct / len(verdicts), "accuracy", int(correct)), jobs=jobs)
        except (InferenceUnavailable, InferenceBackendFinished):
            raise
        except Exception as exc:
            raise EvaluationError(
                f"policy #10197 judge finalization failed: {exc}",
                status=RunStatus.ARTIFACT_FAILED,
                jobs=jobs,
                coverage=outcome.coverage,
            ) from exc


@dataclass(frozen=True)
class SotopiaLengthControl:
    source: EvalExecutor
    config: PolicyFinalization
    model: str

    def __call__(
        self,
        session: RemoteInferenceSession,
        output_dir: str,
        env_vars: Mapping[str, str],
        *,
        backend_state: Callable[[], InferenceBackendState],
        judge: RemoteInferenceSession | None = None,
    ) -> EvaluationOutcome:
        outcome = self.source(session, output_dir, env_vars, backend_state=backend_state, judge=judge)
        try:
            _complete_coverage(outcome)
            return finalize_sotopia(outcome, output_dir, self.model, self.config.sotopia_baselines)
        except Exception as exc:
            raise EvaluationError(
                f"policy #10197 SOTOPIA finalization failed: {exc}",
                status=RunStatus.ARTIFACT_FAILED,
                jobs=outcome.jobs,
                coverage=outcome.coverage,
            ) from exc


def finalize_sotopia(
    outcome: EvaluationOutcome, root: str, model: str, baselines: tuple[CohortSource, ...]
) -> EvaluationOutcome:
    """Fit the complete frozen comparison cohort and preserve every adjusted episode."""
    rank = importlib.import_module(
        "experiments.evaluation.campaigns.eval-campaign-09-25-unlabeled.analysis.sotopia_length_control.rank"
    )
    cells = [rank.SotopiaCell(source.model, source.results_path) for source in baselines if source.model != model]
    cells.append(rank.SotopiaCell(model, root))
    episodes = [episode for cell in cells for episode in rank.read_episodes(cell)]
    fit = rank.length_controlled_scores(episodes)
    candidate = next(score for score in fit.scores if score.model == model)
    adjustments = {row.task_id: row for row in fit.episodes if row.model == model}
    rows = list(ReadView(root).iter_rows("samples"))
    evidence = {
        "policy": POLICY,
        "cohort": [asdict(cell) for cell in cells],
        "source_episodes_sha256": (
            hashlib.sha256(json.dumps([asdict(row) for row in episodes], sort_keys=True).encode()).hexdigest()
        ),
        "source_episodes": [asdict(row) for row in episodes],
        "scores": [asdict(row) for row in fit.scores],
        "adjusted_episodes": [asdict(row) for row in fit.episodes],
        "selected_regularization": fit.selected_regularization,
        "cross_validation_losses": fit.cross_validation_losses,
    }
    final = _final_metrics(outcome, candidate.length_controlled_overall, SOTOPIA_METRIC, None)
    with EvaluationStore.open(root, writer_id="policy-sotopia-length-control") as store:
        for row in rows:
            adjusted = adjustments.get(row["doc_id"])
            if adjusted is None:
                continue
            sample = sample_from_archive_row(row)
            updated = sample.model_copy(
                update={
                    "metrics": {**sample.metrics, SOTOPIA_METRIC: adjusted.length_controlled_overall},
                    "correct": None,
                }
            )
            store.add_sample(updated, trial_id=row.get("trial_id") or "", extraction_filter=row.get("filter") or "")
        store.add_artifact(SOTOPIA_ARTIFACT, json.dumps(evidence, sort_keys=True).encode())
        store.seal()
    return final


def policy_executor(
    source: EvalExecutor,
    reference: EvalRef,
    config: PolicyFinalization,
    model: str,
    judge_accelerator: AcceleratorChoice | None,
    priority: int,
) -> tuple[EvalExecutor, EvalRef]:
    """Wrap only the policy benchmarks that require postprocessing."""
    if reference.name in {"financebench", "olympiadbench"}:
        assert judge_accelerator is not None
        executor = MiniMaxRegrade(source, reference.name, config, judge_accelerator, priority)
    elif reference.name == "sotopia-hard":
        executor = SotopiaLengthControl(source, config, model)
    else:
        return source, reference
    return executor, reference.model_copy(update={"postprocessing": {POLICY: config.record_configuration()}})
