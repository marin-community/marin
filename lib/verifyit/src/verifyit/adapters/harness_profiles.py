# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Scoring leaves for source-owned token/span preprocessing and corpus aggregation."""

import inspect
import re
from collections.abc import Sequence
from pathlib import Path

from verifyit.adapters.skyrl import grade_literal_candidate
from verifyit.grade import InvalidTask, Reward, scored
from verifyit.modes.grade_exact import grade_collection_f1


def _tokens(value: Sequence[str]) -> None:
    if isinstance(value, (str, bytes)) or any(not isinstance(token, str) for token in value):
        raise InvalidTask("prepared tokens must be a sequence of strings")


def token_f1(candidate: Sequence[str], reference: Sequence[str]) -> Reward:
    """Pass prepared tokens to the core multiplicity-aware overlap contract."""
    return grade_collection_f1(reference, candidate, multiplicity="multiset", empty_reference="zero", round_digits=None)


def token_accuracy(candidate: Sequence[str], reference: Sequence[str]) -> Reward:
    """Source POS accuracy truncates both sequences to the shorter length."""
    _tokens(candidate)
    _tokens(reference)
    length = min(len(candidate), len(reference))
    if not length:
        raise InvalidTask("POS accuracy has no comparable tokens")
    matches = sum(
        grade_literal_candidate(gold, prediction).reward
        for gold, prediction in zip(reference[:length], candidate[:length], strict=True)
    )
    return scored(matches / length, matched=matches, compared=length)


def span_equal(candidate: tuple[str, str], reference: tuple[str, str]) -> Reward:
    """Compare prepared (tag, entity) spans; caller consumes duplicate matches."""
    if (
        not isinstance(candidate, tuple)
        or not isinstance(reference, tuple)
        or len(candidate) != 2
        or len(reference) != 2
    ):
        raise InvalidTask("prepared spans require (tag, entity) tuples")
    _tokens(candidate)
    _tokens(reference)
    results = [grade_literal_candidate(gold, value).reward for gold, value in zip(reference, candidate, strict=True)]
    return scored(float(all(results)))


def generation_profile(config: dict) -> str | None:
    """Recognize pinned AfroBench source aggregator profiles, not arbitrary F1."""
    if config.get("output_type", "generate_until") != "generate_until":
        return None
    if config.get("process_results") or config.get("class"):
        return None
    metrics = config.get("metric_list")
    if not isinstance(metrics, list) or any(not isinstance(metric, dict) for metric in metrics):
        return None
    names = [metric.get("metric") for metric in metrics]
    if any(not isinstance(name, str) for name in names):
        return None
    family = {
        ("acc",): ("masakhapos", "acc_score", "pos_accuracy"),
        ("f1",): ("masakhaner", "span_f1_agg", "ner_span_f1"),
        ("exact_match", "f1"): ("afriqa", "f1", "afriqa_f1"),
    }.get(tuple(names))
    if family is None:
        return None
    source_family, name, profile = family
    aggregate = metrics[-1].get("aggregation")
    if isinstance(aggregate, dict) and aggregate.get("tag") == "function":
        symbol = aggregate.get("value", "").rsplit(".", 1)[-1]
        directory = str(aggregate.get("source_dir", ""))
    elif inspect.isfunction(aggregate):
        symbol = aggregate.__name__
        filename = inspect.getsourcefile(aggregate)
        directory = str(Path(filename).parent) if filename else ""
    else:
        return None
    if symbol != name or not re.search(rf"/afrobench/{source_family}/prompt_[1-5]$", directory):
        return None
    allowed = {"metric", "aggregation", "higher_is_better", "ignore_case", "ignore_punctuation", "regexes_to_ignore"}
    if any(metric.keys() - allowed for metric in metrics):
        return None
    return profile


def profile_task_metrics(task, doc, responses, exact_comparator) -> dict | None:
    """Preserve source observations while actual configured aggregators grade leaves.

    POS's ordered vector must not be mistaken for accepted alternative answers.
    This scoring-boundary fix leaves source targets and few-shot prompts intact.
    """
    if task.OUTPUT_TYPE != "generate_until" or tuple(task._metric_fn_list) not in {
        ("acc",),
        ("f1",),
        ("exact_match", "f1"),
    }:
        return None
    metrics = [
        {"metric": name, "aggregation": task._aggregation_list[name], **task._metric_fn_kwargs.get(name, {})}
        for name in task._metric_fn_list
    ]
    profile = generation_profile({"output_type": task.OUTPUT_TYPE, "metric_list": metrics})
    if profile is None:
        return None
    for name, function in task._metric_fn_list.items():
        if (
            getattr(function, "__module__", None) != "lm_eval.api.metrics"
            or getattr(function, "__name__", None)
            != {"acc": "acc_fn", "f1": "f1_fn", "exact_match": "exact_match_fn"}[name]
        ):
            return None
    if len(responses) != 1:
        raise InvalidTask("generation metric profile requires one filtered response")
    gold, prediction = task.doc_to_target(doc), responses[0]
    if profile == "pos_accuracy":
        if not isinstance(gold, list) or not isinstance(prediction, list):
            raise InvalidTask("POS requires an ordered gold vector and nested predicted vectors")
        if any(not isinstance(part, list) for part in prediction):
            raise InvalidTask("POS predictions require nested token lists")
        flattened = [token for part in prediction for token in part]
        token_accuracy(flattened, gold)
        return {"acc": [gold, prediction]}
    if task.multiple_target:
        raise InvalidTask("source aggregate profile does not define alternative-reference vectors")
    if profile == "ner_span_f1":
        if not isinstance(gold, str) or not (
            isinstance(prediction, str)
            or (isinstance(prediction, list) and all(isinstance(value, str) for value in prediction))
        ):
            raise InvalidTask("NER span profile requires string or source-filter string-list observations")
        return {"f1": [gold, prediction]}
    if not isinstance(gold, str) or not isinstance(prediction, str):
        raise InvalidTask("AfriQA profile requires single string reference and prediction")
    options = dict(task._metric_fn_kwargs["exact_match"])
    punctuation = options.get("ignore_punctuation")
    if isinstance(punctuation, str):
        if punctuation != 'true - "." - "," - "\\\\$"':
            raise InvalidTask("unknown source punctuation profile")
        options["ignore_punctuation"] = True
    return {"exact_match": exact_comparator(prediction, [gold], **options).reward, "f1": [gold, prediction]}
