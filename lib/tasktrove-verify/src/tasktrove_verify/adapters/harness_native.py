# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Native existing-mode scoring after harness-owned response filtering.

These functions do not invoke source scorers. Unknown metric options fail closed.
Task loading, extraction filters and corpus aggregations remain caller-owned.
"""

from collections.abc import Sequence
from dataclasses import asdict, replace
from typing import Any

from tasktrove_verify.adapters.harness_agieval import agieval_config_profile, agieval_task_metrics
from tasktrove_verify.adapters.harness_babilong import babilong_config_profile, babilong_task_metrics
from tasktrove_verify.adapters.harness_crows import crows_config_profile, crows_task_metrics
from tasktrove_verify.adapters.harness_drop import drop_config_profile, drop_task_metrics
from tasktrove_verify.adapters.harness_math_literal import hendrycks_config_profile, hendrycks_task_metrics
from tasktrove_verify.adapters.harness_mmmu import mmmu_config_profile, mmmu_task_metrics
from tasktrove_verify.adapters.harness_probability import truthfulqa_mc2_profile, truthfulqa_task_metrics
from tasktrove_verify.adapters.harness_profiles import generation_profile, profile_task_metrics
from tasktrove_verify.adapters.harness_rolling import rolling_config_profile, rolling_task_metrics
from tasktrove_verify.grade import Aggregation, InvalidTask, Reward, aggregate_rewards
from tasktrove_verify.modes.grade_exact import grade_exact_candidate
from tasktrove_verify.modes.grade_mcq import LikelihoodScoring, grade_mcq_likelihoods
from tasktrove_verify.preparation.text import TextNormalization, TextPolicy, normalize_text, structure_text
from tasktrove_verify.spec import EmptyOutputPolicy, ExactSpec


def exact_match(
    candidate: str,
    references: Sequence[str],
    *,
    empty_output: EmptyOutputPolicy = EmptyOutputPolicy.GRADE,
    normalization_policy: TextPolicy = TextPolicy.HARNESS_EXACT,
    **options,
) -> Reward:
    """Normalize source text and compare aliases under an explicit empty-output policy.

    ZERO rejects answers that become empty during normalization; GRADE preserves
    source literal-empty matching. Trusted aliases are validated in either case.
    """
    if not isinstance(empty_output, EmptyOutputPolicy):
        raise InvalidTask("unknown exact empty-output policy")
    allowed = {"regexes_to_ignore", "ignore_case", "ignore_punctuation", "ignore_numbers"}
    if unknown := options.keys() - allowed:
        raise InvalidTask(f"unsupported exact_match options: {sorted(unknown)}")
    prepared = normalize_text(
        structure_text(candidate, references),
        TextNormalization(
            policy=normalization_policy,
            regexes_to_ignore=tuple(options.get("regexes_to_ignore") or ()),
            ignore_case=options.get("ignore_case", False),
            ignore_punctuation=options.get("ignore_punctuation", False),
            ignore_numbers=options.get("ignore_numbers", False),
        ),
    )
    value = prepared.candidate
    normalized_references = prepared.references
    results = [
        grade_exact_candidate(
            ExactSpec(
                (reference,),
                ignore_case=False,
                ignore_whitespace=False,
                strip_outer_whitespace=False,
                empty_output=empty_output,
            ),
            value,
        )
        for reference in normalized_references
    ]
    verdict = aggregate_rewards(results, expected_total=len(normalized_references), policy=Aggregation.MAX)
    return replace(
        verdict,
        detail={**verdict.detail, "preparation": {**asdict(prepared.normalization), "empty_output": empty_output}},
    )


def likelihood_choice(
    choices: Sequence[str], likelihoods: Sequence[float], gold: Sequence[int], normalization: str = "raw"
) -> Reward:
    """Select the first maximum likelihood, then grade its MCQ option.

    Character and UTF-8 byte normalization deliberately differ. Source F1/MCC
    aggregators still require original (gold, selected index) values.
    """
    if normalization not in {"raw", "characters", "bytes"}:
        raise InvalidTask(f"unsupported likelihood normalization: {normalization}")
    if not choices or any(not isinstance(choice, str) for choice in choices):
        raise InvalidTask("likelihood choices must be nonempty strings")
    lengths = [
        1 if normalization == "raw" else len(value if normalization == "characters" else value.encode())
        for value in choices
    ]
    return grade_mcq_likelihoods(likelihoods, gold, normalization_lengths=lengths, policy=LikelihoodScoring.MOST_LIKELY)


def native_config_route(config: dict) -> str | None:
    """Recognize implemented source branches; unknown options are not native coverage."""
    if drop_config_profile(config):
        return "drop_span_metrics"
    if rolling_config_profile(config):
        return "rolling_likelihood_statistics"
    if mmmu_config_profile(config):
        return "mmmu_typed_answers"
    if babilong_config_profile(config):
        return "babilong_substring"
    if crows_config_profile(config):
        return "crows_pair_preference"
    if agieval_config_profile(config):
        return "agieval_mcqa"
    if hendrycks_config_profile(config):
        return "hendrycks_literal_exact"
    if truthfulqa_mc2_profile(config):
        return "truthfulqa_mc2"
    if config.get("process_results") or config.get("class"):
        return None
    profile = generation_profile(config)
    if profile is not None:
        return profile
    output = config.get("output_type", "generate_until")
    metrics = config.get("metric_list")
    if metrics is None:
        metrics = (
            [{"metric": "acc"}, {"metric": "acc_norm"}]
            if output == "multiple_choice"
            else [{"metric": "exact_match"}] if output == "generate_until" else []
        )
    if not isinstance(metrics, list) or not metrics:
        return None
    metadata = {"metric", "aggregation", "higher_is_better"}
    if output == "multiple_choice":
        if all(
            isinstance(metric, dict)
            and metric.get("metric") in {"acc", "acc_norm", "acc_bytes", "f1", "mcc", "exact_match", "likelihood"}
            and not metric.keys()
            - (
                metadata
                | {
                    "weight_by_size",
                    "average",
                    "hf_evaluate",
                    "ignore_case",
                    "ignore_punctuation",
                    "ignore_numbers",
                    "regexes_to_ignore",
                    "high_is_better",
                }
            )
            for metric in metrics
        ):
            return "likelihood_choice"
    if output == "generate_until" and len(metrics) == 1:
        metric = metrics[0]
        allowed = metadata | {"regexes_to_ignore", "ignore_case", "ignore_punctuation", "ignore_numbers"}
        if isinstance(metric, dict) and metric.get("metric") == "exact_match" and not metric.keys() - allowed:
            if all(
                isinstance(metric.get(flag, False), bool)
                for flag in ("ignore_case", "ignore_punctuation", "ignore_numbers")
            ):
                regexes = metric.get("regexes_to_ignore")
                if regexes is None or (
                    isinstance(regexes, list) and all(isinstance(pattern, str) for pattern in regexes)
                ):
                    return "exact_match"
    return None


def native_task_metrics(
    task, doc, responses, *, exact_empty_output: EmptyOutputPolicy = EmptyOutputPolicy.GRADE
) -> dict | None:
    """Route only the pinned ConfigurableTask implementation and recognized metrics.

    None means compatibility-only source scoring is required; it is never a reward.
    Invalid recognized task/sample contracts raise InvalidTask instead of fallback.
    """
    if not isinstance(exact_empty_output, EmptyOutputPolicy):
        raise InvalidTask("unknown exact empty-output policy")
    method = task.process_results
    if (
        getattr(method, "__module__", None) != "lm_eval.api.task"
        or getattr(method, "__qualname__", None) != "ConfigurableTask.process_results"
    ):
        return None
    drop = drop_task_metrics(task, doc, responses)
    if drop is not None:
        return drop
    rolling = rolling_task_metrics(task, doc, responses)
    if rolling is not None:
        return rolling
    mmmu = mmmu_task_metrics(task, doc, responses)
    if mmmu is not None:
        return mmmu
    babilong = babilong_task_metrics(task, doc, responses)
    if babilong is not None:
        return babilong
    crows = crows_task_metrics(task, doc, responses)
    if crows is not None:
        return crows
    agieval = agieval_task_metrics(task, doc, responses)
    if agieval is not None:
        return agieval
    math_metrics = hendrycks_task_metrics(task, doc, responses)
    if math_metrics is not None:
        return math_metrics
    probability_metrics = truthfulqa_task_metrics(task, doc, responses)
    if probability_metrics is not None:
        return probability_metrics
    if task.config.process_results is not None:
        return None
    profile_metrics = profile_task_metrics(task, doc, responses, exact_match)
    if profile_metrics is not None:
        return profile_metrics
    config = {
        "output_type": task.OUTPUT_TYPE,
        "metric_list": [{"metric": metric, **task._metric_fn_kwargs.get(metric, {})} for metric in task._metric_fn_list],
    }
    route = native_config_route(config)
    if route is None:
        return None
    if route == "exact_match":
        fn = task._metric_fn_list["exact_match"]
        if getattr(fn, "__module__", None) != "lm_eval.api.metrics" or getattr(fn, "__name__", None) != "exact_match_fn":
            return None
        if len(responses) != 1:
            raise InvalidTask("generation exact_match requires one filtered response")
        candidate = responses[0]
        gold = task.doc_to_target(doc)
        if task.config.doc_to_choice is not None:
            choices = task.doc_to_choice(doc)
            if not isinstance(gold, int) or not 0 <= gold < len(choices):
                raise InvalidTask("generation target index is outside available choices")
            gold = choices[gold]
        elif task.multiple_target:
            gold = list(gold)
        elif type(gold) is not type(candidate):
            gold = type(candidate)(gold)
        references = gold if task.multiple_target else [gold]
        return {
            "exact_match": (
                exact_match(
                    candidate, references, empty_output=exact_empty_output, **task._metric_fn_kwargs["exact_match"]
                ).reward
            )
        }
    choices = task.doc_to_choice(doc)
    likelihoods, greedy = zip(*responses, strict=True)
    gold = task.doc_to_text(doc) if task.multiple_input else task.doc_to_target(doc)
    if isinstance(gold, str):
        if gold not in choices:
            raise InvalidTask("choice target text is absent from choices")
        gold = choices.index(gold)
    targets = gold if task.multiple_target else [gold]
    metrics: dict[str, Any] = {}
    for metric in task._metric_fn_list:
        if metric in {"acc", "acc_norm", "acc_bytes"}:
            normalization = {"acc": "raw", "acc_norm": "characters", "acc_bytes": "bytes"}[metric]
            metrics[metric] = likelihood_choice(choices, likelihoods, targets, normalization).reward
        else:
            # Validate likelihoods/targets even when only a structured metric is requested.
            verdict = likelihood_choice(choices, likelihoods, targets)
            selected = verdict.detail["selected_index"]
            if metric == "exact_match":
                if any(type(flag) is not bool for flag in greedy):
                    raise InvalidTask("greedy completion flags must be booleans")
                alternatives = [
                    grade_exact_candidate(ExactSpec(("True",), ignore_case=False), str(greedy[index]))
                    for index in targets
                ]
                metrics[metric] = aggregate_rewards(
                    alternatives, expected_total=len(targets), policy=Aggregation.MAX
                ).reward
            elif metric == "likelihood":
                metrics[metric] = (gold, likelihoods)
            else:
                metrics[metric] = (gold, selected)
    return metrics
