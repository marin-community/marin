# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2026 The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Babilong's explicit substring benchmark contract with source preprocessing."""

import inspect
from importlib import import_module
from pathlib import Path

from verifyit.adapters import harness_validation as validation
from verifyit.grade import InvalidTask, Reward
from verifyit.modes.grade_exact import grade_exact_candidate
from verifyit.spec import ExactSpec

_SUFFIX = "/tasks/babilong/common_utils.py"
_HASH = "557de4855a370f2f251176982a110aaac9209fa50645bf110d7d42b1151878bc"


def _pinned_scorer(callback) -> bool:
    if isinstance(callback, dict) and callback.get("tag") == "function":
        path = Path(str(callback.get("source_dir", ""))) / "common_utils.py"
        recognized = callback.get("value") == "common_utils.process_results"
    elif inspect.isfunction(callback):
        path = Path(callback.__code__.co_filename)
        recognized = callback.__name__ == "process_results"
    else:
        return False
    if not recognized or not str(path).endswith(_SUFFIX):
        return False
    source = validation.pinned_source_bytes(path, {_HASH})
    if source is None:
        return False
    if inspect.isfunction(callback):
        namespace = callback.__globals__
        if namespace.get("process_results") is not callback or namespace.get("re") is not import_module("re"):
            return False
        if not validation.source_functions_match(
            namespace,
            tuple(
                code
                for code in validation.compiled_source_functions(source, str(path))
                if code.co_name in {"process_results", "postprocess_pred"}
            ),
            {},
        ):
            return False
    return True


def babilong_config_profile(config: dict) -> bool:
    return (
        _pinned_scorer(config.get("process_results"))
        and config.get("process_docs") is None
        and config.get("output_type") == "generate_until"
        and config.get("doc_to_target") == "{{target}}"
        and config.get("doc_to_choice") is None
        and config.get("metric_list") == [{"metric": "acc", "aggregation": "mean", "higher_is_better": True}]
        and not config.get("class")
    )


def validate_babilong_task(task) -> bool:
    """Return False for unsupported tasks; raise InvalidTask for changed recognized contracts."""
    callback = getattr(getattr(task, "config", None), "process_results", None)
    if callback is None:
        return False
    config = {
        key: getattr(task.config, key)
        for key in ("process_results", "process_docs", "doc_to_choice", "doc_to_target", "metric_list")
    }
    config["output_type"] = task.OUTPUT_TYPE
    if not babilong_config_profile(config):
        if inspect.isfunction(callback) and callback.__code__.co_filename.endswith(_SUFFIX):
            raise InvalidTask("changed Babilong scorer, preprocessing or task contract")
        return False
    mean = import_module("lm_eval.api.metrics").mean
    if (
        tuple(task._metric_fn_list) != ("acc",)
        or task._metric_fn_kwargs.get("acc", {})
        or task._aggregation_list.get("acc") is not mean
    ):
        raise InvalidTask("Babilong requires the registered acc/mean metric contract")
    validation.validate_default_filter(task, "BabiLong")
    return True


def grade_babilong_candidate(reference: object, preprocessed_candidate: object) -> Reward:
    """The caller preprocesses the response once; references only strip and lowercase."""
    if type(reference) is not str or type(preprocessed_candidate) is not str:
        raise InvalidTask("Babilong requires string references and responses")
    return grade_exact_candidate(
        ExactSpec(
            (reference.strip().lower(),),
            ignore_case=False,
            ignore_whitespace=False,
            strip_outer_whitespace=False,
            substring=True,
        ),
        preprocessed_candidate.lower(),
    )


def babilong_task_metrics(task, doc, responses) -> dict | None:
    if not validate_babilong_task(task):
        return None
    if not isinstance(responses, list) or len(responses) != 1 or type(responses[0]) is not str:
        raise InvalidTask("Babilong requires one string generation response")
    namespace = task.config.process_results.__globals__
    candidate = namespace["postprocess_pred"](responses)[0]
    return {"acc": grade_babilong_candidate(doc.get("target"), candidate).reward}
