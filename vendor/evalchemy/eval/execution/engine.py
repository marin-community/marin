"""Execute and grade source items without consulting a benchmark implementation."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping, Sequence
from contextlib import AsyncExitStack
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Protocol

from eval.completion_response import CompletionClassification, CompletionText, FailedGeneration
from eval.contracts.eval_spec import EvalSpec, Grade, JudgePlan, ModelRequest, RequestKind
from eval.contracts.failures import (
    TRANSPORT_FAILURE_CATEGORIES,
    FailureCategory,
    FailurePhase,
    MalformedJudgeResponse,
    classify_task_exception,
)
from eval.contracts.sample_manifest import SampleEntry, SampleManifest, SampleRequest
from eval.contracts.sample_results import validate_sample_metrics
from eval.runtime.checkpoints import Checkpoints, ItemJournal, ResumeStream, completed_operation
from eval.runtime.pool import Completed, ExecutionPool
from eval.runtime.resources import SerialIO
from eval.sample_logging import canonicalize_sample


class ModelBackend(Protocol):
    async def execute(self, request: ModelRequest) -> Any: ...


class ItemFailure(RuntimeError):
    """Keep lifecycle context when the pool reports a failed item."""

    def __init__(self, phase: FailurePhase, error: Exception, category: FailureCategory | None = None):
        super().__init__(str(error))
        self.phase = phase
        self.category = category or classify_task_exception(phase, error)


class SampleSink(Protocol):
    def write_sample(self, task_name: str, record: Mapping[str, Any]) -> None: ...

    def finish_task(self) -> None: ...


async def request_response(backend, request, phase, allow_failed_response):
    try:
        return await backend.execute(request)
    except Exception as error:
        category = classify_task_exception(phase, error)
        if allow_failed_response or (phase is FailurePhase.GENERATION and category in TRANSPORT_FAILURE_CATEGORIES):
            return FailedGeneration(category)
        raise ItemFailure(phase, error) from error


@dataclass
class ItemResult:
    spec: EvalSpec
    responses: tuple[Any, ...]
    grade: Grade
    record: dict[str, Any]


@dataclass(frozen=True)
class AdmittedItem:
    spec: EvalSpec
    entry: SampleEntry
    checkpoint: ItemJournal | None


@dataclass
class CollectSamples:
    samples: dict[str, list[dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))

    def write_sample(self, task_name: str, record: Mapping[str, Any]) -> None:
        self.samples[task_name].append(dict(record))

    def finish_task(self) -> None:
        pass


def scheduling_group(spec: EvalSpec) -> tuple:
    """Group compatible work within bounded lookahead without changing identity."""
    request = spec.requests[0]
    prompt = request.prompt
    text = prompt if isinstance(prompt, str) else str(prompt)
    settings = tuple(sorted((key, repr(value)) for key, value in request.generation.items()))
    return request.kind, settings, len(text).bit_length(), text[:64]


class Evaluator:
    """One request, checkpoint, grading and sample-delivery lifecycle for every source."""

    def __init__(
        self,
        model: ModelBackend,
        grade: Callable[[EvalSpec, Sequence[Any]], Awaitable[Grade | JudgePlan]],
        sink: SampleSink,
        *,
        workers: int,
        checkpoints: Mapping[str, Checkpoints] | None = None,
        judges: Mapping[str, ModelBackend] | None = None,
        predict_only: bool = False,
        extraction_filters: Mapping[str, tuple[str, ...]] | None = None,
    ):
        self.model = model
        self.grade = grade
        self.judges = judges or {}
        self.predict_only = predict_only
        self.extraction_filters = extraction_filters or {}
        self.sink = sink
        self.pool = ExecutionPool(workers)
        self.io = SerialIO()
        self.checkpoints = {task: ResumeStream(store) for task, store in (checkpoints or {}).items()}
        self.manifests: dict[str, SampleManifest] = {}
        self.failures: dict[str, Counter[str]] = defaultdict(Counter)
        self.response_summary: dict[str, Counter[str]] = defaultdict(Counter)

    async def operation(self, admitted, unit, request, backend, phase, *, allow_failed_response=False):
        spec, checkpoint = admitted.spec, admitted.checkpoint
        try:
            output = await completed_operation(
                checkpoint, unit, asdict(replace(request, metadata={})),
                lambda: request_response(backend, request, phase, allow_failed_response),
                io=self.io,
                successful=lambda value: not isinstance(value, FailedGeneration) and not (
                    isinstance(value, CompletionText) and value.response.classification is CompletionClassification.EMPTY
                    and phase is not FailurePhase.GRADING
                ),
            )
        except ItemFailure:
            raise
        except Exception as error:
            raise ItemFailure(FailurePhase.SERIALIZATION, error) from error
        if isinstance(output, CompletionText) and phase is FailurePhase.GENERATION:
            self.response_summary[spec.task][output.response.classification] += 1
        if isinstance(output, FailedGeneration):
            category = FailureCategory(output.failure_category)
            if phase is FailurePhase.GRADING:
                category = FailureCategory.GRADER_INFRASTRUCTURE
            self.failures[spec.task][category] += 1
            if allow_failed_response:
                return output
            await self.io.call(self.sink.write_sample, spec.task, canonicalize_sample(spec.task, {
                "doc_id": spec.ordinal, "doc": spec.doc, "target": spec.target,
                "arguments": [(request.prompt, dict(request.generation))],
                "resps": [[output]], "filtered_resps": [output],
                "metrics": [], "failure_category": category,
                "filter_variants": [
                    {"filter": name, "metrics": {}, "filtered_resps": [output]}
                    for name in self.extraction_filters.get(spec.grader.name, ("none",))
                ],
            }, spec.ordinal, admitted.entry))
            raise ItemFailure(phase, RuntimeError(f"{unit['phase']} request exhausted retries"), category)
        return output

    async def score(self, spec, outputs):
        try:
            return await self.grade(spec, outputs)
        except Exception as error:
            raise ItemFailure(FailurePhase.GRADING, error) from error

    async def judge(self, admitted: AdmittedItem, plan: JudgePlan) -> Grade:
        spec = admitted.spec
        attempt = 0
        while True:
            judgments = []
            for index, request in enumerate(plan.requests):
                unit = {"phase": "judge", "request_index": index}
                # Keep the first attempt's checkpoint coordinate unchanged.
                if attempt:
                    unit["parse_attempt"] = attempt
                judgment = await self.operation(
                    admitted, unit, request, self.judges[spec.task], FailurePhase.GRADING,
                    allow_failed_response=plan.on_error is not None,
                )
                judgments.append(judgment)
            try:
                grade = await self.score(replace(spec, grader=plan.grader), judgments)
            except ItemFailure as error:
                malformed = isinstance(error.__cause__, MalformedJudgeResponse) and not any(
                    isinstance(value, FailedGeneration) for value in judgments
                )
                if malformed and attempt + 1 < plan.max_attempts:
                    attempt += 1
                    continue
                fallback = plan.on_exhausted if malformed else plan.on_error
                if fallback is None:
                    raise
                if not malformed:
                    fallback = replace(fallback, reference={
                        **fallback.reference, "judge_error": {
                            "category": FailureCategory.GRADER_INFRASTRUCTURE.value,
                            "message": str(error)[:512], "exception_type": type(error.__cause__ or error).__name__,
                        },
                    })
                grade = await self.score(replace(spec, grader=fallback), judgments)
            if isinstance(grade, JudgePlan):
                raise ValueError("a judge parser must produce a grade")
            return grade

    def admit(self, specs: Iterable[EvalSpec]) -> Iterable[AdmittedItem]:
        positions = Counter()
        for spec in specs:
            if not spec.requests:
                raise ValueError("an executable eval spec must contain at least one model request")
            manifest = self.manifests.setdefault(spec.task, SampleManifest(spec.task))
            [entry] = manifest.plan_batch(
                [SampleRequest(spec.source_id, spec.ordinal, namespace=spec.namespace, repeat=spec.trial)]
            )
            checkpoint = None
            if spec.task in self.checkpoints:
                checkpoint = self.checkpoints[spec.task].admit(positions[spec.task], asdict(spec))
            positions[spec.task] += 1
            yield AdmittedItem(spec, entry, checkpoint)
        for checkpoint in self.checkpoints.values():
            checkpoint.validate_consumed()

    async def execute(self, admitted: AdmittedItem) -> ItemResult:
        spec, entry = admitted.spec, admitted.entry
        manifest = self.manifests[spec.task]
        outputs: list[Any] = []
        rendered_prompts: list[Any] = []
        arguments: list[tuple] = []
        for index, original in enumerate(spec.requests):
            request = replace(original, metadata={**original.metadata, "task": spec.task, "source_id": spec.source_id,
                                                  "ordinal": spec.ordinal, "trial": spec.trial})
            if original.previous_turn is not None:
                # A conversation is a fixed sequence of user turns. Only the
                # assistant slots depend on execution; sources never run a loop.
                previous = original.previous_turn
                if previous >= index:
                    raise ValueError("a conversation turn must depend on an earlier request")
                prefix = list(rendered_prompts[previous])
                prefix.append({"role": "assistant", "content": str(outputs[previous])})
                request = replace(request, prompt=prefix + list(original.prompt))
            rendered_prompts.append(request.prompt)
            if request.kind is RequestKind.LIKELIHOOD:
                arguments.append((request.prompt, request.continuation))
            elif request.kind is RequestKind.ROLLING_LIKELIHOOD:
                arguments.append((request.prompt,))
            else:
                arguments.append((request.prompt, dict(request.generation)))
            unit = {"phase": "generate", "request_index": index}
            output = await self.operation(admitted, unit, request, self.model, FailurePhase.GENERATION)
            outputs.append(output)
        manifest.mark_generated([entry], [outputs])
        grade = Grade({}) if self.predict_only else await self.score(spec, outputs)
        if isinstance(grade, JudgePlan):
            grade = await self.judge(admitted, grade)
        metrics = grade.metrics if grade.sample_metrics is None else grade.sample_metrics
        raw = {
            **grade.audit,
            "doc_id": spec.ordinal,
            "doc": spec.doc,
            "target": spec.target,
            "arguments": arguments,
            "resps": [[output] for output in outputs],
            "filtered_resps": list(outputs) if grade.filtered_resps is None else list(grade.filtered_resps),
            "metrics": list(metrics),
            **metrics,
        }
        try:
            record = canonicalize_sample(spec.task, raw, spec.ordinal, entry)
            if not self.predict_only:
                validate_sample_metrics(spec.task, [record])
            await self.io.call(self.sink.write_sample, spec.task, record)
        except Exception as error:
            raise ItemFailure(FailurePhase.SERIALIZATION, error) from error
        return ItemResult(spec, tuple(outputs), grade, record)

    async def run(self, items: Iterable[EvalSpec]) -> AsyncIterator[Completed[EvalSpec, ItemResult]]:
        stream = self.pool.map(self.admit(items), self.execute, group=lambda item: scheduling_group(item.spec))
        try:
            async for result in stream:
                yield Completed(result.position, result.job.spec, result.result, result.error)
        finally:
            async with AsyncExitStack() as cleanup:
                cleanup.callback(self.io.close)
                cleanup.push_async_callback(self.io.call, self.sink.finish_task)
                for store in self.checkpoints.values():
                    cleanup.push_async_callback(self.io.call, store.close)
                await stream.aclose()
