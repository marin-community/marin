"""Import lm-eval definitions as source items and ordinary grading functions."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

from lm_eval.api.instance import Instance
from lm_eval.api.task import Task
from lm_eval.evaluator_utils import _process_results, get_sample_size
from lm_eval.utils import handle_non_serializable, hash_string

from eval.contracts.eval_spec import EvalSpec, Grade, GraderSpec, ModelRequest, RequestKind, SourceRef
from eval.execution.engine import ItemResult
from eval.execution.generation import canonical_generation_kwargs, generation_parameters
from eval.execution.grading import Grader


@dataclass(frozen=True)
class ChatInput:
    messages: list[dict[str, str]]
    settings: dict[str, Any]


def chat_input(messages, **settings):
    """Import messages without rendering a tokenizer's template in the source."""
    return ChatInput(messages, settings)


def grade_lm_eval(task: Task, spec: EvalSpec, responses: Sequence[Any]) -> Grade:
    """Apply the task's filter and metric definitions to one complete document."""
    counts = spec.grader.reference["repeats"]
    originals = spec.grader.reference["indices"]
    instances = []
    offset = 0
    for index, repeats in zip(originals, counts, strict=True):
        request = spec.requests[offset]
        args = (request.prompt, dict(request.generation))
        if request.kind is RequestKind.LIKELIHOOD:
            args = (request.prompt, request.continuation)
        elif request.kind is RequestKind.ROLLING_LIKELIHOOD:
            args = (request.prompt,)
        instance = Instance(request.kind.value, spec.doc, args, index, metadata=(spec.task, spec.source_id, repeats))
        original_responses = list(responses[offset:offset + repeats])
        instance.resps.extend(str(value) if isinstance(value, str) else value for value in original_responses)
        instances.append(instance)
        offset += repeats
    # Filters consume the complete response cohort for this document; majority
    # voting must see all repeats before process_results runs.
    for pipeline in task._filters:
        pipeline.apply(instances)
    variants = []
    all_metrics = {}
    reducer_inputs = {}
    for name in instances[0].filtered_resps:
        filtered = [instance.filtered_resps[name] for instance in instances]
        raw_metrics = task.process_results(spec.doc, filtered)
        metrics = {
            metric: float(value) if isinstance(value, (float, int)) else float(task.aggregation()[metric]([value]))
            for metric, value in raw_metrics.items()
        }
        reducer_inputs.update({f"{metric},{name}": value for metric, value in raw_metrics.items()})
        all_metrics.update({f"{metric},{name}": value for metric, value in metrics.items()})
        variants.append({"filter": name, "filtered_resps": filtered, "metrics": metrics})
    primary = variants[0]
    audit = {
        "filter": primary["filter"],
        "doc_hash": hash_string(json.dumps(spec.doc, indent=2, default=handle_non_serializable, ensure_ascii=False)),
        "prompt_hash": hash_string(instances[0].arguments[0] if isinstance(instances[0].arguments[0], str)
                                   else json.dumps(instances[0].arguments[0], sort_keys=True)),
        "target_hash": hash_string(str(spec.target)),
    }
    if len(variants) > 1:
        audit["filter_variants"] = variants
    audit["reducer_inputs"] = reducer_inputs
    return Grade(all_metrics, audit, reducer_inputs=reducer_inputs,
                 sample_metrics=primary["metrics"], filtered_resps=primary["filtered_resps"])


@dataclass
class LMEvalSource:
    task_name: str
    task: Task
    limit: int | None
    num_fewshot: int | None
    generation: dict[str, Any]
    fewshot_seed: int | None
    apply_chat_template: bool = False
    system_instruction: str | None = None
    fewshot_as_multiturn: bool = False

    @property
    def grader_name(self) -> str:
        return f"lm_eval:{self.task_name}"

    def grader(self) -> Grader:
        return Grader(partial(grade_lm_eval, self.task),
                      extraction_filters=tuple(pipeline.name for pipeline in self.task._filters))

    def __iter__(self) -> Iterator[EvalSpec]:
        task = self.task
        fewshot = task.get_config("num_fewshot")
        if fewshot != 0 and self.num_fewshot is not None:
            fewshot = self.num_fewshot
        task.set_config(key="num_fewshot", value=fewshot or 0)
        task.set_fewshot_seed(self.fewshot_seed)
        overrides = canonical_generation_kwargs(self.generation)
        if task.OUTPUT_TYPE == "generate_until":
            defaults = canonical_generation_kwargs(task.get_config("generation_kwargs") or {})
            task.set_config(key="generation_kwargs", value=generation_parameters(defaults, overrides))
        limit = get_sample_size(task, self.limit)
        for ordinal, (doc_id, doc) in enumerate(task.doc_iterator(limit=limit)):
            context = task.fewshot_context(
                doc, num_fewshot=fewshot or 0, system_instruction=self.system_instruction,
                apply_chat_template=self.apply_chat_template,
                fewshot_as_multiturn=self.fewshot_as_multiturn,
                chat_template=chat_input,
                gen_prefix=task.doc_to_prefix(doc),
            )
            instances = task.construct_requests(
                doc=doc, ctx=context, metadata=(self.task_name, doc_id, task.config.repeats),
                apply_chat_template=self.apply_chat_template, chat_template=chat_input,
            )
            if isinstance(instances, Instance):
                instances = [instances]
            instances = sorted(instances, key=lambda instance: instance.idx)
            requests = []
            for instance in instances:
                kind = RequestKind(instance.request_type)
                prompt = instance.args[0]
                request = ModelRequest(
                    kind,
                    prompt.messages if isinstance(prompt, ChatInput) else prompt,
                    generation=(
                        generation_parameters(canonical_generation_kwargs(instance.args[1]), overrides)
                        if kind is RequestKind.GENERATE
                        else {}
                    ),
                    continuation=instance.args[1] if kind is RequestKind.LIKELIHOOD else None,
                    chat_template=prompt.settings if isinstance(prompt, ChatInput) else {},
                )
                requests.extend([request] * instance.repeats)
            yield EvalSpec(
                task=self.task_name,
                source=SourceRef(str(task.config.dataset_path), task.config.test_split or task.config.validation_split),
                source_id=doc_id, ordinal=ordinal,
                namespace=self.task_name, requests=tuple(requests),
                grader=GraderSpec(self.grader_name, {
                    "repeats": [instance.repeats for instance in instances],
                    "indices": [instance.idx for instance in instances],
                }),
                doc=doc, target=task.doc_to_target(doc),
            )


class LMEvalMetrics:
    """Use lm-eval's metric and group definitions without its execution flow."""

    def __init__(self, tasks: dict[str, Task], groups: dict[str, Any], bootstrap_iters: int = 100000):
        self.tasks = tasks
        self.groups = groups
        self.bootstrap_iters = bootstrap_iters
        self.values = {name: defaultdict(list) for name in tasks}

    def add(self, item: ItemResult) -> None:
        for key, value in item.grade.reducer_inputs.items():
            metric, extraction_filter = key.rsplit(",", 1)
            self.values[item.spec.task][metric, extraction_filter].append(value)

    def finish(self, _task_name: str) -> dict[str, Any]:
        accumulators = {
            name: {"task": task, "raw_metrics": self.values[name], "logged_samples": []}
            for name, task in self.tasks.items()
        }
        return _process_results(accumulators, self.groups, self.bootstrap_iters)._to_eval_results()
