"""Normalize Evalchemy's lm-eval sample records into the FineStore contract."""

from __future__ import annotations

import json
from collections.abc import Mapping

from finestore.eval import Choice, EvalSample, Grading, Message, SampleKind

from eval.contracts.failures import FailureCategory
from eval.contracts.sample_results import RESERVED_RECORD_FIELDS

_PRIMARY_METRIC_PRIORITY = ("exact_match", "accuracy", "acc_norm", "acc", "pass@1")
_OPERATIONAL_METRICS = frozenset({"judge_failed"})
_FILTER_PRIORITY = ("flexible-extract",)
_INFRASTRUCTURE_ERROR_MARKER = "[EVALCHEMY_INFRASTRUCTURE_ERROR]"
_GENERATION_FAILURE_CATEGORIES = frozenset(
    {
        FailureCategory.AGENT_TIMEOUT.value,
        FailureCategory.MODEL_TRANSPORT.value,
    }
)
# Everything a sample record carries that is not one of its metrics. Sample identity
# coordinates are numeric, so harvesting metrics without this would record a sample's
# ordinal and repeat index as scores.
_STRUCTURAL_KEYS = RESERVED_RECORD_FIELDS


def _base_metric(name: str) -> str:
    return name.split(",", 1)[0]


def _primary_metric(metrics: Mapping[str, float]) -> tuple[str, float] | None:
    candidates = {
        name: value
        for name, value in metrics.items()
        if not _base_metric(name).endswith("_stderr") and _base_metric(name) not in _OPERATIONAL_METRICS
    }
    if not candidates:
        return None
    for preferred in _PRIMARY_METRIC_PRIORITY:
        matches = {name: value for name, value in candidates.items() if _base_metric(name) == preferred}
        if not matches:
            continue
        for metric_filter in _FILTER_PRIORITY:
            for name, value in matches.items():
                if name.endswith(f",{metric_filter}"):
                    return name, value
        name = min(matches)
        return name, matches[name]
    name = min(candidates)
    return name, candidates[name]


def _loglikelihood_pair(entry) -> tuple[float, bool] | None:
    if isinstance(entry, list) and len(entry) == 1:
        entry = entry[0]
    if (
        isinstance(entry, list)
        and len(entry) == 2
        and isinstance(entry[0], int | float)
        and not isinstance(entry[0], bool)
        and isinstance(entry[1], bool)
    ):
        return float(entry[0]), entry[1]
    return None


def _is_multiple_choice(arguments, responses) -> bool:
    if not isinstance(arguments, list) or len(arguments) <= 1:
        return False
    if not isinstance(responses, list) or len(responses) != len(arguments):
        return False
    return all(_loglikelihood_pair(entry) is not None for entry in responses)


def _choice_labels(doc, count: int) -> list[str]:
    choices = doc.get("choices") if isinstance(doc, dict) else None
    labels = choices.get("label") if isinstance(choices, dict) else None
    if isinstance(labels, list) and len(labels) == count and all(isinstance(label, str) for label in labels):
        return labels
    return [chr(ord("A") + index) for index in range(count)]


def _resolve_target_choice(target, choices: list[Choice]) -> int | None:
    if isinstance(target, bool):
        return None
    if isinstance(target, int):
        return target if 0 <= target < len(choices) else None
    if isinstance(target, str):
        trimmed = target.strip()
        for index, choice in enumerate(choices):
            if choice.label == trimmed or choice.text.strip() == trimmed:
                return index
        if trimmed.isdigit():
            index = int(trimmed)
            return index if 0 <= index < len(choices) else None
    return None


def _parse_chat_messages(value: object) -> list[Message] | None:
    if isinstance(value, tuple) and len(value) == 1:
        value = value[0]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return None
    else:
        parsed = value
    if not isinstance(parsed, list) or not parsed:
        return None
    messages = []
    for item in parsed:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("role"), str)
            or not isinstance(item.get("content"), str)
        ):
            return None
        messages.append(Message(role=item["role"], content=item["content"]))
    return messages


def _sample_metrics(raw: Mapping[str, object]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in raw.items()
        if key not in _STRUCTURAL_KEYS and not isinstance(value, bool) and isinstance(value, int | float)
    }


def _grading(metrics: dict[str, float], extraction_filter: str | None, raw: Mapping[str, object]) -> Grading | None:
    detail = json.dumps(
        {key: raw[key] for key in (
            "failure_category", "judge_error", "judge_label", "judge_raw", "report_outcome", "report_status", "report",
        ) if key in raw},
        ensure_ascii=False,
    )
    picked = _primary_metric(metrics)
    if picked is None:
        if raw.get("failure_category") in _GENERATION_FAILURE_CATEGORIES:
            # Preserve the extraction key so a recovered score supersedes the
            # failed sample without inventing a numeric grade for the failure.
            return Grading(method="lm-eval:transport_failure", filter=extraction_filter or "none", detail=detail)
        if any(_base_metric(name) in _OPERATIONAL_METRICS for name in metrics):
            return Grading(method="lm-eval:judge_failed", filter=extraction_filter, detail=detail)
        return None
    name, value = picked
    metric_filter = extraction_filter or (name.split(",", 1)[1] if "," in name else None)
    return Grading(
        method=f"lm-eval:{_base_metric(name)}",
        metric=name,
        filter=metric_filter,
        score=value,
        passed=value >= 1.0,
        detail=detail,
    )


def _generation_fields(arguments: object, responses: object, raw: Mapping[str, object]) -> dict[str, object]:
    prompt = ""
    messages = None
    if isinstance(arguments, list) and arguments:
        first = arguments[-1]
        candidate = first[0] if isinstance(first, (list, tuple)) and first else first
        messages = _parse_chat_messages(candidate)
        if isinstance(candidate, str):
            prompt = candidate
    output = ""
    if isinstance(responses, list) and responses:
        first = responses[-1]
        if isinstance(first, list) and first and isinstance(first[0], str):
            output = first[0]
        elif isinstance(first, str):
            output = first
    filtered = raw.get("filtered_resps")
    if isinstance(filtered, list) and filtered:
        filtered = filtered[-1]
    failure_category = raw.get("failure_category")
    if isinstance(failure_category, str) and failure_category in _GENERATION_FAILURE_CATEGORIES:
        output = f"{_INFRASTRUCTURE_ERROR_MARKER} {failure_category}"
        filtered = output
    return {
        "prompt_text": None if messages else prompt,
        "prompt_messages": messages,
        "output": output,
        "extracted": filtered if isinstance(filtered, str) else json.dumps(filtered, ensure_ascii=False),
    }


def _multiple_choice_fields(arguments: list, responses: list, doc: object, target: object) -> dict[str, object]:
    labels = _choice_labels(doc, len(arguments))
    choices = []
    for index, entry in enumerate(arguments):
        text = entry[1] if isinstance(entry, (list, tuple)) and len(entry) > 1 and isinstance(entry[1], str) else ""
        pair = _loglikelihood_pair(responses[index])
        loglikelihood, is_greedy = pair if pair is not None else (None, None)
        choices.append(Choice(label=labels[index], text=text, loglikelihood=loglikelihood, is_greedy=is_greedy))
    scored = [(choice.loglikelihood, index) for index, choice in enumerate(choices) if choice.loglikelihood is not None]
    context = arguments[0][0] if isinstance(arguments[0], (list, tuple)) and isinstance(arguments[0][0], str) else ""
    return {
        "prompt_text": context,
        "choices": choices,
        "model_choice": max(scored)[1] if scored else None,
        "target_choice": _resolve_target_choice(target, choices),
    }


def sample_from_lm_eval(task: str, raw: Mapping[str, object]) -> EvalSample:
    """Normalize one lm-eval sample filter row."""
    arguments = raw.get("arguments")
    responses = raw.get("resps")
    doc = raw.get("doc")
    target = raw.get("target")
    metrics = _sample_metrics(raw)
    extraction_filter = raw.get("filter")
    grading = _grading(metrics, extraction_filter if isinstance(extraction_filter, str) else None, raw)
    common = {
        "task": task,
        "doc_id": str(raw.get("sample_id", raw.get("doc_id"))),
        "metrics": metrics,
        "correct": grading.passed if grading is not None else None,
        "grading": grading,
        "target_text": target if isinstance(target, str) else json.dumps(target, ensure_ascii=False),
        "doc": doc if isinstance(doc, str) else json.dumps(doc, ensure_ascii=False),
    }
    if isinstance(arguments, list) and isinstance(responses, list) and _is_multiple_choice(arguments, responses):
        return EvalSample(
            kind=SampleKind.MULTIPLE_CHOICE,
            **_multiple_choice_fields(arguments, responses, doc, target),
            **common,
        )
    return EvalSample(kind=SampleKind.GENERATION, **_generation_fields(arguments, responses, raw), **common)


def samples_from_lm_eval(task: str, raw: Mapping[str, object]) -> list[EvalSample]:
    """Normalize one Evalchemy record into all of its extraction-filter samples."""
    variants = raw.get("filter_variants")
    if not isinstance(variants, list) or not variants:
        return [sample_from_lm_eval(task, raw)]

    samples = []
    for variant in variants:
        if not isinstance(variant, dict):
            raise ValueError("lm-eval filter_variants entries must be mappings")
        metrics = variant.get("metrics", {})
        if not isinstance(metrics, dict):
            raise ValueError("lm-eval filter_variants metrics must be a mapping")
        samples.append(
            sample_from_lm_eval(
                task,
                {
                    **raw,
                    **metrics,
                    "filter": variant.get("filter"),
                    "filtered_resps": variant.get("filtered_resps", []),
                },
            )
        )
    return samples
