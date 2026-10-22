"""Metric reduction for task sessions."""

from typing import Any

NUPA_METRIC_KEYS = ("acc", "exact_match", "digit_match", "dlength", "format_valid", "no_answer")


def mean_metrics(metrics: list[dict[str, Any]]) -> dict[str, float]:
    """Average each numeric field across rows that contain it."""
    values: dict[str, list[float]] = {}
    for row in metrics:
        for key, value in row.items():
            if isinstance(value, (int, float)):
                values.setdefault(key, []).append(float(value))
    return {key: sum(items) / len(items) for key, items in values.items()}


def _fraction(rows: list[dict[str, Any]], key: str) -> float:
    return sum(bool(row[key]) for row in rows) / len(rows) if rows else 0.0


def aime_metrics(metrics: list[dict[str, Any]]) -> dict[str, float]:
    metrics = [row for row in metrics if "acc" in row]
    correct = [row for row in metrics if bool(row["acc"])]
    incorrect = [row for row in metrics if not bool(row["acc"])]
    return {
        **mean_metrics(metrics),
        "over_evaluation_budget_fraction": _fraction(metrics, "over_evaluation_budget"),
        "correct_over_evaluation_budget_fraction": _fraction(correct, "over_evaluation_budget"),
        "incorrect_over_evaluation_budget_fraction": _fraction(incorrect, "over_evaluation_budget"),
        "answered_within_evaluation_budget_fraction": _fraction(metrics, "answered_within_evaluation_budget"),
    }


def nupa_metrics(metrics: list[dict[str, Any]]) -> dict[str, float]:
    means = mean_metrics(metrics)
    return {f"nupa/{key}": means.get(key, 0.0) for key in NUPA_METRIC_KEYS}


def math_metrics(metrics: list[dict[str, Any]]) -> dict[str, float]:
    if not metrics:
        return {}
    return {"avg_steps": sum(float(row.get("steps", 0)) for row in metrics) / len(metrics)}


METRIC_REDUCERS = {"aime": aime_metrics, "nupa": nupa_metrics, "gsm8k_multi_turn": math_metrics}


def aggregate_for_task(name: str, metrics: list[dict[str, Any]]) -> dict[str, float]:
    return METRIC_REDUCERS.get(name, mean_metrics)(metrics)
