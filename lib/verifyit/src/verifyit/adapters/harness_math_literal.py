# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Strict exact grading after trusted Hendrycks string normalization."""

import inspect
import math
from collections.abc import Callable
from importlib import import_module
from pathlib import Path

from verifyit.adapters import harness_validation as validation
from verifyit.adapters.skyrl import grade_literal_candidate
from verifyit.grade import InvalidTask, Reward, Status, invalid_task, scored

_NONFINITE = {"nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}


def grade_normalized_math(expected: object, candidate: object, normalize: Callable[[str], str]) -> Reward:
    """Normalize source-extracted strings, then use existing strict exact grading.

    Known malformed-expression errors never fall back to raw equality. Unexpected
    normalizer failures propagate to the caller's infrastructure boundary.
    """
    if type(expected) not in (str, int, float):
        return invalid_task("normalized math reference must be a string or finite nonboolean number")
    if isinstance(expected, int | float):
        try:
            finite = math.isfinite(expected)
        except OverflowError:
            finite = False
        if not finite:
            return invalid_task("normalized math reference must be finite")
    reference = str(expected)
    if not reference.strip() or reference.strip().lower() in _NONFINITE:
        return invalid_task("normalized math reference is empty or nonfinite")
    try:
        reference = normalize(reference)
    except (AssertionError, IndexError, ValueError):
        return invalid_task("source math reference normalization failed")
    if not isinstance(reference, str) or not reference.strip() or reference.strip().lower() in _NONFINITE:
        return invalid_task("source math reference normalization produced no answer")
    if not isinstance(candidate, str) or not candidate.strip() or candidate.strip().lower() in _NONFINITE:
        return scored(0, reason="missing_or_nonfinite_candidate")
    try:
        answer = normalize(candidate)
    except (AssertionError, IndexError, ValueError):
        return scored(0, reason="candidate_normalization_failed")
    if not isinstance(answer, str) or not answer.strip() or answer.strip().lower() in _NONFINITE:
        return scored(0, reason="missing_normalized_candidate")
    return grade_literal_candidate(reference, answer)


def hendrycks_config_profile(config: dict) -> bool:
    """Recognize eight pinned pure-normalization/exact task configurations."""
    callback = config.get("process_results")
    if isinstance(callback, dict) and callback.get("tag") == "function":
        name = callback.get("value")
        path = Path(str(callback.get("source_dir", ""))) / "utils.py"
        recognized = name == "utils.process_results"
    elif inspect.isfunction(callback):
        path = Path(callback.__code__.co_filename)
        recognized = callback.__name__ == "process_results"
    else:
        return False
    if not recognized or not str(path).endswith("/tasks/hendrycks_math/utils.py"):
        return False
    source = validation.pinned_source_bytes(path, {"8332e42f23c62043b23d74ee6d3934f997b209cd60a1ad45b7882d6b00929d5c"})
    if source is None:
        return False
    if inspect.isfunction(callback):
        if not validation.source_functions_match(
            callback.__globals__, validation.compiled_source_functions(source, str(path)), {"is_equiv": (False,)}
        ):
            return False
        if callback.__globals__.get("process_results") is not callback:
            return False
    docs = config.get("process_docs")
    if inspect.isfunction(callback):
        if docs is not callback.__globals__.get("process_docs"):
            return False
    elif not (
        isinstance(docs, dict)
        and docs.get("tag") == "function"
        and docs.get("value") == "utils.process_docs"
        and Path(str(docs.get("source_dir", ""))) == path.parent
    ):
        return False
    return (
        config.get("output_type") == "generate_until"
        and config.get("doc_to_target") == "{{answer}}"
        and config.get("doc_to_choice") is None
        and config.get("metric_list") == [{"metric": "exact_match", "aggregation": "mean", "higher_is_better": True}]
        and not config.get("class")
    )


def validate_hendrycks_task(task) -> bool:
    """Validate the pinned source task before filters or custom callbacks run."""
    callback = getattr(getattr(task, "config", None), "process_results", None)
    if callback is None:
        return False
    config = {
        "process_results": getattr(task.config, "process_results", None),
        "process_docs": task.config.process_docs,
        "output_type": task.OUTPUT_TYPE,
        "doc_to_target": task.config.doc_to_target,
        "doc_to_choice": task.config.doc_to_choice,
        "metric_list": task.config.metric_list,
    }
    if not hendrycks_config_profile(config):
        if inspect.isfunction(callback) and str(callback.__code__.co_filename).endswith(
            "/tasks/hendrycks_math/utils.py"
        ):
            raise InvalidTask("changed Hendrycks source scorer or metric configuration")
        return False
    mean = import_module("lm_eval.api.metrics").mean
    if tuple(task._metric_fn_list) != ("exact_match",) or task._aggregation_list.get("exact_match") is not mean:
        raise InvalidTask("Hendrycks exact requires the registered exact_match/mean contract")
    if task._metric_fn_kwargs.get("exact_match", {}):
        raise InvalidTask("Hendrycks exact does not accept additional metric options")
    validation.validate_default_filter(task, "Hendrycks exact")
    return True


def hendrycks_task_metrics(task, doc, responses) -> dict | None:
    """Preserve source dollar-span/box extraction, replacing only equality scoring."""
    if not validate_hendrycks_task(task):
        return None
    if len(responses) != 1:
        raise InvalidTask("Hendrycks exact requires one filtered response")
    callback = task.config.process_results
    namespace = callback.__globals__
    solution = doc.get("solution")
    if not isinstance(solution, str):
        raise InvalidTask("Hendrycks math solution must contain a reference box")
    boxed = namespace["last_boxed_only_string"](solution)
    if boxed is None:
        raise InvalidTask("Hendrycks math solution has no reference box")
    try:
        expected = namespace["remove_boxed"](boxed)
    except (AssertionError, IndexError, ValueError):
        raise InvalidTask("Hendrycks math reference box is malformed") from None
    candidate = responses[0]
    if isinstance(candidate, str):
        indices = [index for index, value in enumerate(candidate) if value == "$"]
        if len(indices) > 1:
            candidate = candidate[indices[0] + 1 : indices[-1]]
    verdict = grade_normalized_math(expected, candidate, namespace["strip_string"])
    if verdict.status is not Status.SCORED:
        raise InvalidTask(str(verdict.detail))
    return {"exact_match": verdict.reward}


def validate_hendrycks_normalizer(normalize: object) -> bool:
    """Validate AMC's imported source normalizer and its helper namespace."""
    if not inspect.isfunction(normalize) or normalize.__globals__.get("strip_string") is not normalize:
        return False
    namespace = normalize.__globals__
    return hendrycks_config_profile(
        {
            "process_results": namespace.get("process_results"),
            "process_docs": namespace.get("process_docs"),
            "output_type": "generate_until",
            "doc_to_target": "{{answer}}",
            "doc_to_choice": None,
            "metric_list": [{"metric": "exact_match", "aggregation": "mean", "higher_is_better": True}],
        }
    )
