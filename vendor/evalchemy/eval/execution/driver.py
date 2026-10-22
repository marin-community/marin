"""One evaluation driver over normalized source iterators."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Iterator
from contextlib import aclosing
from copy import copy
from dataclasses import dataclass, field, replace
from typing import Protocol

from aiohttp import ClientSession, ClientTimeout

from eval.contracts.eval_spec import EvalSpec, SourceRef
from eval.contracts.failures import TRANSPORT_FAILURE_CATEGORIES
from eval.contracts.sample_manifest import SampleCoverageError
from eval.contracts.task_outcome import (
    FailureCategory,
    FailurePhase,
    TaskOutcome,
    TaskRoute,
    TaskStatus,
    classify_task_exception,
)
from eval.execution.backend import LMBackend, prompt_encoder, prompt_token_counter
from eval.execution.engine import CollectSamples, Evaluator, ItemFailure, ItemResult
from eval.execution.grading import Grader, Graders
from eval.execution.judge import JudgeBackend
from eval.execution.metrics import SourceMetrics
from eval.execution.run_spec import RunSpec
from eval.graders.answer_equivalence import JudgeConfig
from eval.graders.specs import GRADERS
from eval.robust_api import completion_response_text_unavailable
from eval.sources.common import SourceOptions
from eval.task import BaseBenchmark
from eval.sources.lm_eval import LMEvalMetrics, LMEvalSource

logger = logging.getLogger(__name__)


class Metrics(Protocol):
    def add(self, result: ItemResult) -> None: ...
    def finish(self, name: str) -> dict: ...


@dataclass
class SourcePlan:
    name: str
    route: TaskRoute
    sources: tuple[Iterable[EvalSpec], ...]
    task_names: tuple[str, ...]
    reducer: Metrics
    graders: dict[str, Grader] = field(default_factory=dict)
    errors: list[Exception] = field(default_factory=list)
    judge_config: JudgeConfig | None = None
    expected_examples: int | None = None
    benchmark: BaseBenchmark | None = None


def _interleave(plans: list[SourcePlan]) -> Iterator[EvalSpec]:
    active = deque((plan, iter(source)) for plan in plans for source in plan.sources)
    while active:
        plan, source = active.popleft()
        try:
            spec = next(source)
        except StopIteration:
            continue
        except Exception as exc:
            logger.error("Source %s failed while producing items: %s", plan.name, exc)
            plan.errors.append(ItemFailure(FailurePhase.PREPARATION, exc))
            continue
        active.append((plan, source))
        yield spec


def _source_specs(benchmark, options: SourceOptions) -> Iterator[EvalSpec]:
    # Some sources load data before returning an iterator. Keep those failures
    # inside _interleave's per-task preparation boundary.
    yield from benchmark.iter_specs(options)


def source_plans(tasks, routes, custom, native, run: RunSpec, model) -> list[SourcePlan]:
    plans = []
    imported = set()
    for task in tasks:
        route = TaskRoute(routes[task])
        if route is TaskRoute.CUSTOM:
            benchmark = copy(custom.get_benchmark(task))
            benchmark.evaluation_limit = run.limit
            benchmark.num_samples = run.num_samples
            benchmark.pass_at_k = run.pass_at_k
            description = benchmark.describe(task)
            if description is None:
                raise ValueError(f"source {task} must declare its metrics")
            source = SourceRef(benchmark.dataset_name, benchmark.dataset_split, benchmark.dataset_revision)
            options = SourceOptions(
                task, source, run.limit, run.max_length,
                run.max_tokens if run.max_tokens is not None else benchmark._evaluation_max_tokens,
                run.generation, run.seeds[0], run.num_samples, run.system_instruction,
                prompt_encoder(model), model_name=run.model_name,
                tokenizer_encode=prompt_encoder(model, tokenizer_defaults=True),
                prompt_token_count=prompt_token_counter(model),
                chat_template=getattr(model, "_evalchemy_chat_template_kwargs", None) or {},
            )
            reducer = SourceMetrics(
                repeated=False,
                pass_at_k=(
                    run.pass_at_k if run.num_samples > 1 and description.reducers[0].trial_metric is not None else ()
                ),
                policy=description.reducers[0],
            )
            plans.append(SourcePlan(
                task, route, (_source_specs(benchmark, options),), (task,), reducer,
                judge_config=JudgeConfig.resolve(judge_model=run.judge_model or benchmark.judge_model)
                if benchmark.REQUIRES_JUDGE and not run.predict_only else None,
                expected_examples=description.n_attempted, benchmark=benchmark,
            ))
            continue
        loaded = native.load([task])
        sources = []
        graders = {}
        for name, definition in loaded["tasks"].items():
            if getattr(definition, "UNSAFE_CODE", False) and not run.confirm_run_unsafe_code:
                raise ValueError(f"task {name} requires --confirm_run_unsafe_code")
            source = LMEvalSource(
                name, definition, run.limit, run.num_fewshot, run.generation, run.seeds[3],
                run.apply_chat_template, run.system_instruction, run.fewshot_as_multiturn,
            )
            if name not in imported:
                sources.append(source)
                imported.add(name)
            graders[source.grader_name] = source.grader()
        reducer = LMEvalMetrics(loaded["tasks"], loaded["groups"])
        plans.append(SourcePlan(task, route, tuple(sources), tuple(loaded["tasks"]), reducer, graders))
    return plans


async def run_evaluations(
    model,
    plans: list[SourcePlan],
    *,
    workers: int,
    sink=None,
    checkpoints=None,
    predict_only=False,
    seeds=RunSpec.seeds,
    supports_request_seed=False,
    batch_sizes=None,
    batch_size_attribute="batch_size",
) -> dict:
    samples = CollectSamples() if sink is None else sink
    definitions = dict(GRADERS)
    for plan in plans:
        definitions.update(plan.graders)
    grading = Graders(definitions, workers)
    by_task = defaultdict(list)
    for plan in plans:
        for task in plan.task_names:
            by_task[task].append(plan)
    scored: dict[str, set[str]] = {task: set() for task in by_task}
    dropped = defaultdict(Counter)
    first_dropped = {}
    scored_samples = Counter()
    try:
        async with LMBackend(
            model,
            seeds=seeds,
            supports_request_seed=supports_request_seed,
            batch_sizes=batch_sizes or {},
            batch_size_attribute=batch_size_attribute,
        ) as backend, ClientSession(timeout=ClientTimeout(total=None)) as judge_session:
            judges = {
                task: JudgeBackend(plan.judge_config, judge_session)
                for plan in plans if plan.judge_config is not None for task in plan.task_names
            }
            engine = Evaluator(
                backend,
                grading,
                samples,
                workers=workers,
                checkpoints=checkpoints,
                judges=judges,
                predict_only=predict_only,
                extraction_filters={name: grader.extraction_filters for name, grader in definitions.items()},
            )
            async with aclosing(engine.run(_interleave(plans))) as stream:
                async for completed in stream:
                    task_plans = by_task[completed.job.task]
                    if completed.error is not None:
                        logger.error(
                            "Evaluation %s item %s failed: %s",
                            completed.job.task,
                            completed.job.source_id,
                            completed.error,
                        )
                        if (
                            isinstance(completed.error, ItemFailure)
                            and completed.error.phase is FailurePhase.GENERATION
                            and completed.error.category in TRANSPORT_FAILURE_CATEGORIES
                        ):
                            dropped[completed.job.task][completed.error.category] += 1
                            first_dropped.setdefault(completed.job.task, completed.error)
                            continue
                        for plan in task_plans:
                            plan.errors.append(completed.error)
                        continue
                    result: ItemResult = completed.result
                    if not predict_only:
                        scored[result.spec.task].add(result.record["sample_id"])
                        scored_samples[result.spec.task] += 1
                        for plan in task_plans:
                            try:
                                plan.reducer.add(result)
                            except Exception as error:
                                plan.errors.append(ItemFailure(FailurePhase.GRADING, error))
    finally:
        grading.close()
    result = {"results": {}, "task_outcomes": {}}
    for plan in plans:
        manifests = [engine.manifests[task] for task in plan.task_names if task in engine.manifests]
        expected = sum(manifest.expected_sample_count for manifest in manifests)
        generated = sum(manifest.generated_sample_count for manifest in manifests)
        score_count = sum(len(scored[task]) for task in plan.task_names)
        if plan.benchmark is not None:
            plan.expected_examples = plan.benchmark.describe(plan.name).n_attempted
        if plan.expected_examples is not None and expected != plan.expected_examples:
            plan.errors.append(
                SampleCoverageError(f"source declared {plan.expected_examples} examples but yielded {expected}")
            )
        for manifest in manifests:
            try:
                missing = manifest.expected_unit_count - manifest.generated_unit_count
                explained_drops = dropped[manifest.task_name] and missing <= sum(dropped[manifest.task_name].values())
                if not explained_drops:
                    manifest.validate_generated()
                if not predict_only:
                    manifest.mark_scored(len(scored[manifest.task_name]))
                    if not explained_drops:
                        manifest.validate_scored()
            except SampleCoverageError as exc:
                plan.errors.append(exc)
        if not expected:
            plan.errors.append(SampleCoverageError("source produced no evaluation items"))
        if not predict_only and not score_count:
            plan.errors.append(next(
                (first_dropped[task] for task in plan.task_names if task in first_dropped),
                SampleCoverageError("source produced no scored evaluation items"),
            ))
        failures = Counter()
        response_summary = Counter()
        for task in plan.task_names:
            failures.update(engine.failures[task])
            for category, count in dropped[task].items():
                failures[category] = max(failures[category], count)
            response_summary.update(engine.response_summary[task])
        if completion_response_text_unavailable(response_summary):
            plan.errors.append(
                ItemFailure(
                    FailurePhase.GENERATION,
                    ValueError("at least half of chat completions contain no scorer text"),
                    FailureCategory.MALFORMED_MODEL_RESPONSE,
                )
            )
        document = None
        if not plan.errors and not predict_only:
            try:
                document = plan.reducer.finish(plan.name)
                if plan.benchmark is not None:
                    document["results"][plan.name].update(plan.benchmark.evaluation_metadata())
            except Exception as error:
                plan.errors.append(ItemFailure(FailurePhase.GRADING, error))
        if plan.errors:
            error = plan.errors[0]
            category = (
                error.category
                if isinstance(error, ItemFailure)
                else (
                    FailureCategory.INCOMPLETE_EVALUATION
                    if isinstance(error, SampleCoverageError)
                    else classify_task_exception(FailurePhase.EVALUATION, error)
                )
            )
            outcome = TaskOutcome.failed(plan.name, plan.route, category, str(error), exception=error)
            outcome = replace(outcome, expected_count=expected, generated_count=generated, scored_count=score_count,
                              failure_counts=failures, completion_response_summary=response_summary)
        elif predict_only:
            outcome = TaskOutcome(plan.name, plan.route, TaskStatus.EXPORTED, {}, expected, generated, None,
                                  failure_counts=failures, completion_response_summary=response_summary)
        else:
            assert document is not None
            for key, value in document.items():
                result.setdefault(key, {}).update(value)
            metrics = document["results"].get(plan.name, document.get("groups", {}).get(plan.name, {}))
            outcome = TaskOutcome(
                plan.name, plan.route, TaskStatus.SUCCEEDED, metrics, expected, generated, score_count,
                failure_counts=failures, completion_response_summary=response_summary,
            )
        outcome = replace(
            outcome,
            expected_sample_count=sum(manifest.expected_unit_count for manifest in manifests),
            generated_sample_count=sum(manifest.generated_unit_count for manifest in manifests),
            scored_sample_count=None if predict_only else sum(scored_samples[task] for task in plan.task_names),
        )
        result["task_outcomes"][plan.name] = outcome.to_dict()
    if sink is None:
        result["samples"] = samples.samples
    return result
