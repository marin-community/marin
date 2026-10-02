# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2026 The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Pinned AGIEval multi-answer likelihood scoring through existing choice grading."""

import inspect
from importlib import import_module
from pathlib import Path

from tasktrove_verify.adapters import harness_validation as validation
from tasktrove_verify.grade import InvalidTask

_SOURCE_HASH = "d54929f3cfcae8ee7f1f0171414cb81ec19b4243ea6474d8155956beb18327d3"
_SUFFIX = "/tasks/agieval/utils.py"
_METRICS = [{"metric": metric, "aggregation": "mean", "higher_is_better": True} for metric in ("acc", "acc_norm")]


def agieval_config_profile(config: dict) -> bool:
    callback = config.get("process_results")
    if isinstance(callback, dict) and callback.get("tag") == "function":
        path = Path(str(callback.get("source_dir", ""))) / "utils.py"
        name = "process_results_mcqa" if callback.get("value") == "utils.process_results_mcqa" else ""
    elif isinstance(callback, str) and "::" in callback:
        filename, name = callback.rsplit("::", 1)
        path = Path(filename)
    elif inspect.isfunction(callback):
        path, name = Path(callback.__code__.co_filename), callback.__name__
    else:
        return False
    if name != "process_results_mcqa" or not str(path).endswith(_SUFFIX):
        return False
    source = validation.pinned_source_bytes(path, {_SOURCE_HASH})
    if source is None:
        return False
    if inspect.isfunction(callback):
        code = next(part for part in validation.compiled_source_functions(source, str(path)) if part.co_name == name)
        namespace = callback.__globals__
        if (
            not validation.source_functions_match(namespace, (code,), {})
            or namespace.get(name) is not callback
            or namespace.get("np") is not import_module("numpy")
        ):
            return False
    return (
        config.get("output_type") == "multiple_choice"
        and config.get("doc_to_choice") == "{{choices}}"
        and config.get("doc_to_target") == "{{gold}}"
        and config.get("process_docs") is None
        and config.get("metric_list") == _METRICS
        and not config.get("class")
    )


def validate_agieval_task(task) -> bool:
    callback = getattr(getattr(task, "config", None), "process_results", None)
    if callback is None:
        return False
    config = {
        key: getattr(task.config, key)
        for key in ("process_results", "process_docs", "doc_to_choice", "doc_to_target", "metric_list")
    }
    config["output_type"] = task.OUTPUT_TYPE
    if not agieval_config_profile(config):
        if inspect.isfunction(callback) and callback.__code__.co_filename.endswith(_SUFFIX):
            raise InvalidTask("changed AGIEval MCQA source scorer or task contract")
        return False
    mean = import_module("lm_eval.api.metrics").mean
    if (
        tuple(task._metric_fn_list) != ("acc", "acc_norm")
        or any(task._metric_fn_kwargs.get(metric, {}) for metric in ("acc", "acc_norm"))
        or any(task._aggregation_list.get(metric) is not mean for metric in ("acc", "acc_norm"))
    ):
        raise InvalidTask("AGIEval MCQA requires registered acc/acc_norm means")
    validation.validate_default_filter(task, "AGIEval MCQA")
    return True


def agieval_metrics(choices, gold, responses) -> dict[str, float]:
    """Preserve first-maximum ties and alternative gold indices for both source metrics."""
    likelihood_choice = import_module("tasktrove_verify.adapters.harness_native").likelihood_choice

    if not isinstance(choices, list) or not choices or any(type(choice) is not str or not choice for choice in choices):
        raise InvalidTask("AGIEval choices must be nonempty strings")
    if not isinstance(gold, list) or not gold or any(type(index) is not int for index in gold):
        raise InvalidTask("AGIEval requires nonempty integer gold indices")
    if not isinstance(responses, list) or len(responses) != len(choices):
        raise InvalidTask("AGIEval requires one likelihood response per choice")
    likelihoods = validation.log_likelihoods(responses, "AGIEval MCQA")
    return {
        "acc": likelihood_choice(choices, likelihoods, gold).reward,
        "acc_norm": likelihood_choice(choices, likelihoods, gold, "characters").reward,
    }


def agieval_task_metrics(task, doc, responses) -> dict | None:
    if not validate_agieval_task(task):
        return None
    return agieval_metrics(task.doc_to_choice(doc), doc.get("gold"), responses)
