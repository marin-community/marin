# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare pinned DROP spans for Exact, Schema and shared score composition."""

import functools
import re
import string
from importlib import import_module

from tasktrove_verify.adapters.harness_validation import pinned_function_namespace
from tasktrove_verify.grade import Aggregation, InvalidTask, Status, aggregate_rewards, scored
from tasktrove_verify.modes.grade_exact import grade_collection_f1

SOURCE_HASH = "a5782ee7d2e537968a5b83976195ae2b9b6e2a565a0bfb5c24d221b4d61a04f5"
FILTER_HASH = "c5363aea4b449e515e970967dac45e67c96042cdd39e37be4383d8c0dad51509"


def drop_config_profile(config: dict) -> bool:
    return (
        config.get("task") == "drop"
        and config.get("dataset_path") == "EleutherAI/drop"
        and config.get("output_type") == "generate_until"
        and config.get("doc_to_target") == "Answer: {{ answers[0]|join(',')}}"
        and type(config.get("repeats", 1)) is int
        and config.get("repeats", 1) == 1
    )


def _validate_task(task):
    source = pinned_function_namespace(task.config.process_results, "process_results", {SOURCE_HASH})
    if (
        task.config.process_docs is not source["process_docs"]
        or task.prompt is not None
        or source.get("re") is not re
        or source.get("string") is not string
        or source.get("_ARTICLES") != re.compile(r"\b(a|an|the)\b", re.UNICODE)
    ):
        raise InvalidTask("DROP document preparation changed")
    if set(task._metric_fn_list) != {"em", "f1"} or any(task._metric_fn_kwargs.values()):
        raise InvalidTask("DROP metric configuration changed")
    mean = import_module("lm_eval.api.metrics").mean
    if any(task.aggregation().get(name) is not mean for name in ("em", "f1")):
        raise InvalidTask("DROP metric aggregation changed")
    filters = task._filters
    selection = import_module("lm_eval.filters.selection").TakeFirstFilter
    custom = import_module("lm_eval.filters.custom").CustomFilter
    ensemble = import_module("lm_eval.api.filter").FilterEnsemble
    if len(filters) != 1 or type(filters[0]) is not ensemble or filters[0].name != "extract_answer":
        raise InvalidTask("DROP requires its single source extraction filter")
    constructors = filters[0].filters
    if (
        "apply" in vars(filters[0])
        or len(constructors) != 2
        or any(not isinstance(item, functools.partial) or item.args for item in constructors)
        or constructors[0].func is not custom
        or set(constructors[0].keywords) != {"filter_fn"}
        or constructors[1].func is not selection
        or constructors[1].keywords
    ):
        raise InvalidTask("DROP requires source extraction followed by take_first")
    pinned_function_namespace(constructors[0].keywords["filter_fn"], "drop_answer_extraction_filter", {FILTER_HASH})
    return source


def drop_task_metrics(task, doc, responses) -> dict | None:
    if not drop_config_profile(task.dump_config()):
        return None
    source = _validate_task(task)
    # Schema is optional until this recognized task actually executes.
    from tasktrove_verify.modes.grade_json_schema import grade_json_schema_candidate  # noqa: PLC0415

    references = doc.get("answers")
    if (
        not isinstance(references, (list, tuple))
        or not references
        or any(
            not isinstance(answer, (list, tuple))
            or not answer
            or any(not isinstance(span, str) or not span.strip() for span in answer)
            for answer in references
        )
    ):
        raise InvalidTask("DROP requires nonempty textual reference spans")
    prepared = [[source["_normalize"](span) for span in answer] for answer in references]
    # Preflight every trusted collection before inspecting a candidate.
    for answer in prepared:
        for span in answer:
            grade_collection_f1(span.split(), [], multiplicity="set", empty_reference="zero", round_digits=None)
    if len(responses) != 1:
        raise InvalidTask("DROP requires exactly one predicted span")
    candidate = source["_normalize"](responses[0]) if isinstance(responses[0], str) else ""
    tokens = candidate.split()
    numbers = [token for token in tokens if source["_is_number"](token)]
    candidate_present = grade_json_schema_candidate({"type": "string", "minLength": 1}, candidate)
    exact, partial = [], []
    for answer in prepared:
        exact_match = grade_json_schema_candidate(
            {"const": {"spans": sorted(set(answer)), "size": len(answer)}}, {"spans": [candidate], "size": 1}
        )
        exact.append(aggregate_rewards([candidate_present, exact_match], expected_total=2, policy=Aggregation.MIN))
        pairs = []
        for span in answer:
            reference_tokens = span.split()
            reference_numbers = [token for token in reference_tokens if source["_is_number"](token)]
            numeric_schema: dict[str, object] = {"type": "array"}
            if reference_numbers:
                numeric_schema["contains"] = {"enum": reference_numbers}
            gate = grade_json_schema_candidate(numeric_schema, numbers)
            overlap = grade_collection_f1(
                reference_tokens, tokens, multiplicity="set", empty_reference="zero", round_digits=None
            )
            pairs.append(aggregate_rewards([gate, overlap], expected_total=2, policy=Aggregation.MIN))
        best = aggregate_rewards(pairs, expected_total=len(pairs), policy=Aggregation.MAX)
        partial.append(
            aggregate_rewards(
                [best, *[scored(0.0) for _ in answer[1:]]],
                expected_total=len(answer),
                policy=Aggregation.MEAN,
                round_digits=2,
            )
        )
    metrics = {
        "em": aggregate_rewards(exact, expected_total=len(exact), policy=Aggregation.MAX),
        "f1": aggregate_rewards(partial, expected_total=len(partial), policy=Aggregation.MAX),
    }
    for verdict in metrics.values():
        if verdict.status is Status.INVALID_TASK:
            raise InvalidTask(str(verdict.detail))
        if verdict.status is not Status.SCORED:
            raise RuntimeError(f"DROP grading failed: {verdict.detail}")
    return {name: verdict.reward for name, verdict in metrics.items()}
