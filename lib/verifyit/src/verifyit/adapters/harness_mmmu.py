# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2026 The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""MMMU source extraction composed with MCQ, numeric and substring exact grading."""

import ast
import functools
import inspect
import math
from importlib import import_module

from verifyit.adapters import harness_validation as validation
from verifyit.adapters.skyrl import grade_literal_candidate
from verifyit.grade import InvalidTask, Reward, scored
from verifyit.modes.grade_exact import grade_exact_candidate
from verifyit.modes.grade_math import grade_numeric_candidate
from verifyit.modes.grade_mcq import grade_mcq_candidate
from verifyit.spec import ExactSpec, McqSpec, NumericSpec

_SUFFIX = "/tasks/mmmu/utils.py"
_ORIGINAL = "65e52c4a7694c68df5fdd250be8c3998af3a835d22e669a64235197d42a83057"
_PATCHED = "5c63ed4a1fd1bc271d4903ed0b523fc57ea5c3699874664eeb48b455c91e6154"
_OPTION_LABELS = "ABCDEFGHI"


@functools.lru_cache(maxsize=4)
def _source_constants(source: bytes) -> tuple[tuple[str, str], ...]:
    names = {"START_CHR", "MULTI_CHOICE_EXAMPLE_FORMAT", "SHORT_ANS_EXAMPLE_FORMAT"}
    return tuple(
        (node.targets[0].id, ast.literal_eval(node.value))
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in names
    )


def _pinned_function(callback, name: str) -> bool:
    path = validation.utils_callback_path(callback, name)
    if path is None or not str(path).endswith(_SUFFIX):
        return False
    source = validation.pinned_source_bytes(path, {_PATCHED} if inspect.isfunction(callback) else {_ORIGINAL, _PATCHED})
    if source is None:
        return False
    if inspect.isfunction(callback):
        namespace = callback.__globals__
        if namespace.get(name) is not callback:
            return False
        if (
            namespace.get("ast") is not ast
            or namespace.get("re") is not import_module("re")
            or namespace.get("np") is not import_module("numpy")
        ):
            return False
        if any(namespace.get(name) != value for name, value in _source_constants(source)):
            return False
        if not validation.source_functions_match(namespace, validation.compiled_source_functions(source, str(path)), {}):
            return False
    return True


def mmmu_config_profile(config: dict) -> bool:
    return (
        _pinned_function(config.get("process_results"), "process_results")
        and _pinned_function(config.get("doc_to_text"), "doc_to_text")
        and _pinned_function(config.get("doc_to_image"), "doc_to_image")
        and config.get("process_docs") is None
        and config.get("output_type") == "generate_until"
        and config.get("doc_to_target") == "answer"
        and config.get("doc_to_choice") is None
        and config.get("metric_list") == [{"metric": "acc", "aggregation": "mean", "higher_is_better": True}]
        and not config.get("class")
    )


def validate_mmmu_task(task) -> bool:
    """Return False for unsupported tasks; raise InvalidTask for changed recognized contracts."""
    callback = getattr(getattr(task, "config", None), "process_results", None)
    if callback is None:
        return False
    config = {
        key: getattr(task.config, key)
        for key in (
            "process_results",
            "process_docs",
            "doc_to_choice",
            "doc_to_target",
            "doc_to_text",
            "doc_to_image",
            "metric_list",
        )
    }
    config["output_type"] = task.OUTPUT_TYPE
    if not mmmu_config_profile(config):
        if inspect.isfunction(callback) and callback.__code__.co_filename.endswith(_SUFFIX):
            raise InvalidTask("changed MMMU extraction, source helpers or task contract")
        return False
    mean = import_module("lm_eval.api.metrics").mean
    if (
        tuple(task._metric_fn_list) != ("acc",)
        or task._metric_fn_kwargs.get("acc", {})
        or task._aggregation_list.get("acc") is not mean
    ):
        raise InvalidTask("MMMU requires the registered acc/mean metric contract")
    validation.validate_default_filter(task, "MMMU")
    return True


def _mcq_references(choices, gold) -> list[str]:
    if (
        not isinstance(choices, list)
        or not 1 <= len(choices) <= len(_OPTION_LABELS)
        or any(type(choice) is not str or not choice for choice in choices)
    ):
        raise InvalidTask("MMMU requires one to nine nonempty string options")
    references = gold if isinstance(gold, list) else [gold]
    labels = set(_OPTION_LABELS[: len(choices)])
    if not references or any(type(value) is not str or value not in labels for value in references):
        raise InvalidTask("MMMU gold labels must identify available source options")
    return references


def grade_mmmu_mcq(choices, gold, extracted: object) -> Reward:
    references = _mcq_references(choices, gold)
    labels = set(_OPTION_LABELS[: len(choices)])
    if extracted is None:
        return scored(0.0, reason="missing_choice_extraction")
    if type(extracted) is not str or extracted not in labels:
        return scored(0.0, reason="unrecognized_choice_extraction")
    grades = [
        (
            grade_literal_candidate(value, extracted)
            if len(choices) == 1
            else grade_mcq_candidate(McqSpec(value, len(choices)), extracted)
        )
        for value in references
    ]
    return max(grades, key=lambda result: result.reward)


def grade_mmmu_open(references, candidates) -> Reward:
    if not isinstance(references, list) or not references:
        raise InvalidTask("MMMU requires nonempty normalized open references")
    for reference in references:
        if type(reference) is str:
            if not reference.strip():
                raise InvalidTask("MMMU normalized references must be nonempty")
        elif type(reference) is not float or not math.isfinite(reference):
            raise InvalidTask("MMMU numeric references must be finite source-normalized floats")
    if not isinstance(candidates, list) or any(type(value) not in (str, float) for value in candidates):
        raise RuntimeError("MMMU source parser returned an invalid normalized candidate vector")
    if any(type(value) is float and not math.isfinite(value) for value in candidates):
        return scored(0.0, reason="nonfinite_candidate")
    for candidate in candidates:
        for reference in references:
            if type(candidate) is str and type(reference) is str:
                result = grade_exact_candidate(
                    ExactSpec(
                        (reference,),
                        ignore_case=False,
                        ignore_whitespace=False,
                        strip_outer_whitespace=False,
                        substring=True,
                    ),
                    candidate,
                )
            elif type(candidate) is float and type(reference) is float:
                result = grade_numeric_candidate(NumericSpec(reference, tolerance_abs=0.0, tolerance_rel=0.0), candidate)
            else:
                continue
            if result.reward:
                return result
    return scored(0.0, reason="no_normalized_answer_match")


def mmmu_task_metrics(task, doc, responses) -> dict | None:
    if not validate_mmmu_task(task):
        return None
    if not isinstance(responses, list) or len(responses) != 1 or type(responses[0]) is not str:
        raise InvalidTask("MMMU requires one string generation response")
    namespace = task.config.process_results.__globals__
    gold = doc.get("answer")
    if doc.get("question_type") == "multiple-choice":
        try:
            options = ast.literal_eval(doc.get("options", ""))
        except (ValueError, SyntaxError) as error:
            raise InvalidTask("MMMU source options must be a literal option list") from error
        # Validate gold/options before executing extraction, including the missing-answer path.
        _mcq_references(options, gold)
        labels = list(_OPTION_LABELS[: len(options)])
        extracted = namespace["parse_multi_choice_response"](
            responses[0], labels, dict(zip(labels, options, strict=True))
        )
        result = grade_mmmu_mcq(options, gold, extracted)
    elif doc.get("question_type") == "open":
        references = gold if isinstance(gold, list) else [gold]
        if not references or any(type(value) is not str or not value.strip() for value in references):
            raise InvalidTask("MMMU open gold must contain nonempty strings")
        normalized = [value for reference in references for value in namespace["normalize_str"](reference)]
        result = grade_mmmu_open(normalized, namespace["parse_open_response"](responses[0]))
    else:
        raise InvalidTask("MMMU question_type must be multiple-choice or open")
    return {"acc": result.reward}
