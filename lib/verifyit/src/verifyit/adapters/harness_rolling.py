# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned rolling-likelihood observations with core-owned corpus diagnostics."""

import re
from importlib import import_module

from verifyit.adapters.harness_validation import (
    pinned_function_namespace,
    validate_default_filter,
)
from verifyit.grade import InvalidTask
from verifyit.modes.grade_mcq import summarize_log_likelihoods

CATEGORIES = frozenset(
    "wikipedia_english wikipedia_nonenglish github_python github_cpp github_javascript github_markdown github_other "
    "bbc_news arxiv_physics arxiv_computer_science arxiv_math arxiv_other biorxiv_all ao3_english ao3_nonenglish".split()
)
METRICS = frozenset({"word_perplexity", "byte_perplexity", "bits_per_byte"})


def rolling_config_profile(config: dict) -> bool:
    name = config.get("task", "")
    return (
        isinstance(name, str)
        and name.removeprefix("uncheatable_eval_") in CATEGORIES
        and name.startswith("uncheatable_eval_")
        and config.get("dataset_path") == "Jellyfish042/UncheatableEval-2026-07"
        and config.get("dataset_kwargs") == {"revision": "65889535d56aa38d448ce7e07b08e6e36c031545"}
        and config.get("output_type") == "loglikelihood_rolling"
        and config.get("doc_to_target") == "{{content}}"
        and config.get("doc_to_text") == ""
        and config.get("process_results") is None
    )


def _validate_task(task) -> None:
    validate_default_filter(task, "Uncheatable")
    if task.prompt is not None or task.config.doc_to_choice is not None or set(task._metric_fn_list) != METRICS:
        raise InvalidTask("Uncheatable requires its source target and corpus metrics")
    metrics = import_module("lm_eval.api.metrics")
    for name in METRICS:
        aggregation = "bits_per_byte" if name == "bits_per_byte" else "weighted_perplexity"
        if (
            task._metric_fn_list[name] is not getattr(metrics, name + "_fn")
            or task._metric_fn_kwargs.get(name)
            or task.aggregation().get(name) is not getattr(metrics, aggregation)
        ):
            raise InvalidTask("Uncheatable metric contract changed")
    namespace = pinned_function_namespace(
        task.config.process_docs,
        task.config.task.removeprefix("uncheatable_eval_"),
        {"1e8753e77ec64e18a4c5d35bd1d4f7c229b266d0de3587880a910993b053e48e"},
    )
    if namespace.get("Dataset") is not import_module("datasets").Dataset:
        raise InvalidTask("Uncheatable category source changed")


def rolling_task_metrics(task, doc, responses) -> dict | None:
    if not rolling_config_profile(task.dump_config()):
        return None
    _validate_task(task)
    if not isinstance(doc.get("content"), str) or not doc["content"]:
        raise InvalidTask("Uncheatable requires nonempty trusted content")
    target = task.doc_to_target(doc)
    if not isinstance(target, str) or not target:
        raise InvalidTask("Uncheatable target must render as nonempty text")
    if len(responses) != 1:
        raise InvalidTask("Uncheatable requires one model likelihood per document")
    # Match the source word boundary and UTF-8 units, including edge whitespace.
    words, byte_count = len(re.split(r"\s+", target)), len(target.encode("utf-8"))
    return {
        "word_perplexity": (responses[0], words),
        "byte_perplexity": (responses[0], byte_count),
        "bits_per_byte": (responses[0], byte_count),
    }


def _statistics(items):
    if not items or any(not isinstance(item, (tuple, list)) or len(item) != 2 for item in items):
        raise InvalidTask("corpus diagnostics require likelihood/length pairs")
    likelihoods, lengths = zip(*items, strict=True)
    return summarize_log_likelihoods(likelihoods, normalization_lengths=lengths)


def weighted_perplexity(items):
    return _statistics(items).perplexity


def bits_per_unit(items):
    return _statistics(items).bits_per_unit


def rolling_task_aggregation(task, metric):
    if not rolling_config_profile(task.dump_config()):
        return None
    _validate_task(task)
    if metric not in METRICS:
        raise InvalidTask("unknown Uncheatable corpus metric")
    return bits_per_unit if metric == "bits_per_byte" else weighted_perplexity
