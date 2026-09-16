# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Smooth multiple-choice metrics, using upstream lm-eval's task extension API.

These retain the definitions from the former Stanford harness pin, including
byte-normalized choice probabilities and first-valid-target scoring.
"""

import math

import numpy as np
from lm_eval.api.registry import register_metric
from lm_eval.api.task import ConfigurableTask

SMOOTH_METRICS = frozenset({"bpb", "logprob", "choice_logprob", "choice_prob_norm", "choice_logprob_norm"})
_SMOOTH_DEFAULTS = [
    {"metric": name, "aggregation": "mean", "higher_is_better": name != "bpb"}
    for name in ("bpb", "logprob", "choice_logprob", "choice_prob_norm", "choice_logprob_norm")
]
_ACCURACY = {"metric": "acc", "aggregation": "mean", "higher_is_better": True}
_ACCURACY_NORM = {**_ACCURACY, "metric": "acc_norm"}
_BOTH_ACCURACIES = frozenset({"arc_easy", "arc_challenge", "hellaswag", "piqa", "openbookqa", "mathqa"})
_RAW_ACCURACY = frozenset({"boolq", "commonsense_qa", "copa", "winogrande", "wsc273"})


@register_metric(metric="bpb", higher_is_better=False, aggregation="mean")
@register_metric(metric="logprob", higher_is_better=True, aggregation="mean")
@register_metric(metric="choice_logprob", higher_is_better=True, aggregation="mean")
@register_metric(metric="choice_prob_norm", higher_is_better=True, aggregation="mean")
@register_metric(metric="choice_logprob_norm", higher_is_better=False, aggregation="mean")
def passthrough_metric(value):
    return value


class SmoothMultipleChoiceTask(ConfigurableTask):
    """Add smooth scores without replacing upstream accuracy scoring or prompts."""

    def __init__(self, config: dict):
        # TaskManager passes its Python-task dispatch key to custom task classes.
        super().__init__(config={key: value for key, value in config.items() if key != "class"})

    def process_results(self, doc, results):
        scores = super().process_results(doc, results)
        if self.OUTPUT_TYPE != "multiple_choice" or callable(self.config.process_results):
            return scores

        assert self.config.metric_list is not None
        metrics = {entry["metric"] for entry in self.config.metric_list}
        choices = self.doc_to_choice(doc)
        gold = self.doc_to_text(doc) if self.multiple_input else self.doc_to_target(doc)
        if isinstance(gold, str):
            gold = choices.index(gold) if gold in choices else -100
        golds = gold if isinstance(gold, list) else [gold]
        primary = next((index for index in golds if 0 <= index < len(choices)), None)
        if primary is None:
            return scores

        lls = np.array([result[0] for result in results[: len(choices)]])
        byte_lengths = np.array([max(1, len(choice.encode("utf-8"))) for choice in choices])
        bits = (-lls / byte_lengths) * (1 / math.log(2))
        weights = np.exp(-bits)
        weights /= max(weights.sum(), 1e-8)
        values = {
            # Keep NumPy scalars: Python float summation uses a different accumulator.
            "bpb": bits[primary],
            "logprob": float(lls[primary]),
            "choice_logprob": float((lls - np.logaddexp.reduce(lls))[primary]),
            "choice_prob_norm": float(weights[primary]),
            "choice_logprob_norm": float(np.log(float(weights[primary]) + 1e-30)),
        }
        scores.update({name: value for name, value in values.items() if name in metrics})
        return scores


def task_config_with_smooth_metrics(config: dict) -> dict:
    """Preserve Levanter's core-task metrics and honor explicit metric overrides."""
    name = config["task"]
    if "metric_list" not in config:
        if name in _BOTH_ACCURACIES:
            smooth = _SMOOTH_DEFAULTS[:-1] if name == "mathqa" else _SMOOTH_DEFAULTS
            config = {**config, "metric_list": [_ACCURACY, _ACCURACY_NORM, *smooth]}
        elif name in _RAW_ACCURACY:
            config = {**config, "metric_list": [_ACCURACY, *_SMOOTH_DEFAULTS]}
        elif name == "mmlu":
            # Retain the old task metadata as well as its scores during the runtime migration.
            # Correcting these historically inconsistent directions is a separate protocol change.
            smooth = [
                {**metric, "higher_is_better": metric["metric"] == "choice_logprob_norm"}
                for metric in _SMOOTH_DEFAULTS
            ]
            config = {**config, "metric_list": [_ACCURACY, _ACCURACY_NORM, *smooth]}
        elif name == "social_iqa":
            config = {**config, "metric_list": [_ACCURACY, _SMOOTH_DEFAULTS[2]]}
    metrics = {entry["metric"] for entry in config.get("metric_list", [])}
    if name == "mmlu" and "aggregate_metric_list" not in config:
        config = {
            **config,
            "aggregate_metric_list": [
                {
                    "metric": metric["metric"],
                    "aggregation": "mean",
                    "weight_by_size": metric["metric"] in {"acc", "acc_norm"},
                }
                for metric in config["metric_list"]
            ],
        }
    if metrics & SMOOTH_METRICS:
        return {**config, "class": SmoothMultipleChoiceTask}
    return config
