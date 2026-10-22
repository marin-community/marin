"""Shared construction of ordinary text evaluation items."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sized
from dataclasses import dataclass, field, replace
from itertools import islice
from typing import Any, Literal

from eval.contracts.eval_spec import EvalSpec, GraderSpec, ModelRequest, RequestKind, SourceRef
from eval.execution.generation import generation_parameters


@dataclass(frozen=True)
class SourceOptions:
    """Resolved run settings used while a dataset produces its item stream."""

    task: str
    source: SourceRef
    limit: int | float | None = None
    context_length: int | None = None
    max_tokens: int | None = None
    generation: Mapping[str, Any] = field(default_factory=dict)
    seed: int = 0
    num_samples: int = 1
    system_instruction: str | None = None
    encode: Callable[[str], list[int]] | None = None
    model_name: str | None = None
    tokenizer_encode: Callable[[str], list[int]] | None = None


def text_specs(
    records: Iterable[Mapping[str, Any]],
    options: SourceOptions,
    *,
    prompt: Callable[[Mapping[str, Any]], Any],
    target: Callable[[Mapping[str, Any]], Any],
    grader: str,
    generation_kwargs: Mapping[str, Any] | Callable[[Mapping[str, Any]], Mapping[str, Any]],
    source_id: Callable[[Mapping[str, Any]], Any] | None = None,
    prompt_format: Literal["chat", "completion"] = "chat",
    repeats: int = 1,
    seed_repeats: bool = True,
    namespace: str = "default",
    reference: Callable[[Mapping[str, Any]], Mapping[str, Any]] = lambda _doc: {},
    tags: Callable[[Mapping[str, Any]], Mapping[str, str]] = lambda _doc: {},
) -> Iterator[EvalSpec]:
    """Apply an example cap before expanding trials.

    Sources may omit legacy repeat seeds; explicit sample counts use trial seeds.
    """
    limit = options.limit
    if limit is not None:
        if 0 < limit < 1:
            if not isinstance(records, Sized):
                records = list(records)
            limit = int(len(records) * limit)
        else:
            limit = int(limit)
    selected = islice(records, limit) if limit is not None else records
    count = options.num_samples if options.num_samples > 1 else repeats
    for ordinal, record in enumerate(selected):
        if source_id is not None:
            identity = source_id(record)
        else:
            identity = next((record[key] for key in ("id", "task_id", "source_index", "index", "key", "question_id", "instance_id", "session_id") if key in record), None)
            if identity is None:
                # Equal rows at different dataset positions are separate examples.
                identity = ordinal
        model_input = prompt(record)
        if prompt_format == "chat":
            if isinstance(model_input, str):
                model_input = [{"role": "user", "content": model_input}]
            if options.system_instruction:
                model_input = [{"role": "system", "content": options.system_instruction}, *model_input]
        for trial in range(count):
            parameters = dict(generation_kwargs(record) if callable(generation_kwargs) else generation_kwargs)
            if options.num_samples > 1:
                parameters.update(do_sample=True, temperature=0.7, top_p=1.0)
            source_do_sample = parameters.get("do_sample")
            parameters = generation_parameters(parameters, options.generation)
            if (seed_repeats or options.num_samples > 1) and (source_do_sample or parameters.get("do_sample")) and "seed" not in parameters and options.seed is not None:
                parameters["seed"] = options.seed + trial
            if options.max_tokens is not None:
                parameters["max_gen_toks"] = options.max_tokens
            yield EvalSpec(
                task=options.task, source=options.source, source_id=identity, ordinal=ordinal,
                trial=trial if count > 1 else None, namespace=namespace,
                requests=(ModelRequest(RequestKind.GENERATE, model_input, parameters, context_length=options.context_length,
                                       metadata={"source_do_sample": source_do_sample}),),
                grader=GraderSpec(grader, reference(record)),
                doc=dict(record), target=target(record), tags=tags(record),
            )


def conversation_specs(records, options, *, turns, prompt_renderer: Literal["fastchat"] | None = None, **kwargs):
    """Expand fixed conversation turns into dependent model requests."""
    for spec in text_specs(records, options, prompt=lambda row: turns(row)[0], **kwargs):
        first = replace(spec.requests[0], prompt_format=prompt_renderer)
        requests = [first]
        for index, turn in enumerate(turns(spec.doc)[1:], start=1):
            messages = [{"role": "user", "content": turn}] if isinstance(turn, str) else turn
            requests.append(replace(first, prompt=messages, previous_turn=index - 1))
        yield replace(spec, requests=tuple(requests))
