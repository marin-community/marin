"""Common online reductions for custom evaluation sources."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence

from eval.execution.engine import ItemResult
from eval.execution.grading import MeanMetrics, TrialMetrics
from eval.contracts.aggregation import outcome_report
from eval.contracts.eval_spec import ReducerSpec
from eval.execution.paired import PairedConditions


class SourceMetrics:
    """Reduce scored items, trial cohorts and tagged subgroups in one place."""

    def __init__(self, *, repeated: bool, pass_at_k: Sequence[int] = (), policy: ReducerSpec = ReducerSpec()):
        self.repeated = repeated
        self.policy = policy
        self.means = MeanMetrics()
        self.trials: dict[int, MeanMetrics] = defaultdict(MeanMetrics)
        self.groups: dict[tuple[str, str], MeanMetrics] = defaultdict(MeanMetrics)
        self.group_trials = defaultdict(lambda: defaultdict(MeanMetrics))
        self.examples: set[str] = set()
        self.squared_accuracy = 0.0
        if pass_at_k and policy.trial_metric is None:
            raise ValueError("pass@k requires a binary trial metric")
        self.pass_at_k = TrialMetrics(pass_at_k, policy.trial_metric) if pass_at_k else None
        self.paired = PairedConditions() if policy.name == "paired_conditions" else None
        self.annotations = [] if policy.name == "alpaca" else None
        self.report_outcomes = defaultdict(list) if policy.name == "outcome_report" else None
        self.report_population = ()
        self.report_metadata = {}

    def add(self, result: ItemResult) -> None:
        if self.policy.selected_trial is not None and (result.spec.trial or 0) != self.policy.selected_trial:
            return
        self.repeated |= result.spec.trial is not None
        if self.paired is not None:
            self.paired.add(result)
        if self.annotations is not None:
            self.annotations.append(result.grade.audit["alpaca_annotation"])
        if self.report_outcomes is not None:
            self.report_outcomes[result.spec.trial or 0].append(result.grade.audit["report_outcome"])
            reference = result.spec.grader.reference
            if "report_population" in reference:
                self.report_population = reference["report_population"]
                self.report_metadata = reference["report_metadata"]
        self.means.add(result.grade.metrics, result.grade.weights)
        self.examples.add(result.record["sample_id"])
        self.trials[result.spec.trial or 0].add(result.grade.metrics, result.grade.weights)
        accuracy = result.grade.metrics.get("accuracy")
        if accuracy is not None:
            self.squared_accuracy += float(accuracy) ** 2
        for tag, values in result.spec.tags.items():
            for value in (values,) if isinstance(values, str) else values:
                self.groups[tag, value].add(result.grade.metrics, result.grade.weights)
                self.group_trials[tag, value][result.spec.trial or 0].add(result.grade.metrics, result.grade.weights)
        if self.pass_at_k is not None:
            self.pass_at_k.add(result)

    def finish(self, task: str) -> dict:
        metrics = self.means.finish()
        count = len(self.examples)
        if self.policy.name == "rounded_split_mean":
            return {"results": {task: self._rounded_split_metrics()}}
        if self.policy.name == "last_trial" and self.pass_at_k is None:
            return {"results": {task: self._last_trial_metrics()}}
        pass_metrics = self.pass_at_k.finish() if self.pass_at_k is not None else {}
        unavailable_trials = any(value is None for value in pass_metrics.values())
        if self.pass_at_k is not None and not unavailable_trials:
            metrics = pass_metrics
        elif "accuracy" in metrics:
            accuracy = metrics.pop("accuracy")
            rename_repeated = self.repeated and self.policy.trial_metric is not None and self.pass_at_k is None
            accuracy_name = "accuracy_avg" if rename_repeated and self.policy.accuracy == "accuracy" else self.policy.accuracy
            stderr_name = "accuracy_std_err" if rename_repeated and self.policy.stderr == "accuracy_stderr" else self.policy.stderr
            metrics[accuracy_name] = accuracy
            if self.repeated:
                trial_scores = [(trial, self.trials[trial].finish().get("accuracy")) for trial in sorted(self.trials)]
                accuracies = [value for _, value in trial_scores]
                metrics.update(
                    num_total=count, num_repeat=len(accuracies), solved_avg=self.means.sums["accuracy"] / len(accuracies),
                    run_stats=[
                        {"repetition": trial + 1, "num_total": count, "num_solved": self.trials[trial].sums["accuracy"] if value is not None else None, "accuracy": value}
                        for trial, value in trial_scores
                    ],
                )
                denominator = len(accuracies) * (len(accuracies) - self.policy.repeat_ddof)
                if any(value is None for value in accuracies):
                    metrics[stderr_name] = None
                else:
                    metrics[stderr_name] = math.sqrt(sum((value - accuracy) ** 2 for value in accuracies if value is not None) / denominator) if denominator > 0 else 0.0
            else:
                observed = self.means.counts["accuracy"]
                variance = max(0, self.squared_accuracy - observed * accuracy ** 2) / (observed - 1) if observed > 1 else 0
                metrics.update(num_total=count, num_solved=accuracy * self.means.counts["accuracy"])
                metrics[stderr_name] = math.sqrt(variance / observed)
        if "judge_failed" in self.means.sums:
            observations = self.means.counts["judge_failed"]
            failed = self.means.sums["judge_failed"]
            judged = observations - failed
            metrics.update(
                num_total=count,
                num_judged=judged,
                num_judge_failed=failed,
                judge_coverage=judged / observations if observations else 0.0,
            )
            # Math sources retain zero accuracy for failed judges; other sources
            # leave the score unavailable when no judge returned a grade.
            if self.annotations is None and not self.means.counts["accuracy"] and not self.means.counts[self.policy.accuracy]:
                metrics[self.policy.accuracy] = None
        for (tag, group), accumulator in self.groups.items():
            for metric, value in accumulator.finish().items():
                if tag in self.policy.groups:
                    name = self.policy.groups[tag].format(group=group, metric=self.policy.accuracy if metric == "accuracy" else metric)
                else:
                    name = f"{metric}_{group}{self.policy.group_suffix}" if metric == "accuracy" else f"{tag}:{group}/{metric}"
                metrics[name] = value
                if metric == "accuracy" and self.repeated and self.policy.group_suffix == "_avg":
                    trials = [trial.finish()[metric] for trial in self.group_trials[tag, group].values()]
                    denominator = len(trials) * (len(trials) - self.policy.repeat_ddof)
                    metrics[f"accuracy_{group}_std_err"] = math.sqrt(sum((score - value) ** 2 for score in trials) / denominator) if denominator > 0 else 0.0
            if "accuracy" in accumulator.counts:
                metrics[f"count_{group}"] = accumulator.counts["accuracy"]
        if self.policy.name == "classification":
            correct = self.means.sums["accuracy"]
            incorrect = self.means.sums["incorrect"]
            not_attempted = self.means.sums["not_attempted"]
            judged = correct + incorrect + not_attempted
            accuracy = correct / judged if judged else None
            attempted = correct / (correct + incorrect) if correct + incorrect else 0.0
            metrics.update(
                accuracy=accuracy, correct_rate=accuracy, incorrect_rate=incorrect / judged if judged else None,
                not_attempted_rate=not_attempted / judged if judged else None,
                accuracy_given_attempted=attempted, f1=2 * accuracy * attempted / (accuracy + attempted) if accuracy is not None and accuracy + attempted else 0.0,
                num_correct=correct, num_incorrect=incorrect, num_not_attempted=not_attempted,
            )
        if self.policy.name == "instructions":
            for name in list(metrics):
                if ":" in name:
                    group, member = name.split(":", 1)
                    metrics.setdefault(group, {})[member] = metrics.pop(name)
        if self.policy.name == "weighted_macro":
            categories = {}
            for (_tag, value), accumulator in self.groups.items():
                values = accumulator.finish()
                if value in self.policy.group_weights and "adjusted_score" in values:
                    categories[value] = values["adjusted_score"]
            macro = (sum(value * self.policy.group_weights[name] for name, value in categories.items())
                     / sum(self.policy.group_weights.values())) if categories else None
            metrics.setdefault("score", None)
            metrics.setdefault("adjusted_score", None)
            metrics.update(task_macro_score=macro, adjusted_task_macro_score=macro, task_categorized_scores=categories)
        if self.paired is not None:
            metrics = self.paired.finish()
        metrics.update(pass_metrics)
        if self.annotations is not None:
            from alpaca_eval.metrics import get_length_controlled_winrate

            coverage = {name: metrics[name] for name in ("num_total", "num_judged", "num_judge_failed", "judge_coverage") if name in metrics}
            if any(row["preference"] is not None for row in self.annotations):
                metrics = get_length_controlled_winrate(self.annotations)
            else:
                metrics = {"win_rate": None, "length_controlled_winrate": None, "n_total": 0}
            metrics.update(coverage)
        if self.report_outcomes is not None:
            metrics.update(self.report_metadata)
            reports = {
                str(trial): outcome_report(rows, self.report_population)
                for trial, rows in sorted(self.report_outcomes.items())
            }
            if self.repeated:
                metrics["reports_by_trial"] = reports
            else:
                metrics.update(reports["0"])
        return {"results": {task: metrics}}

    def _last_trial_metrics(self) -> dict:
        # Some source protocols overwrite earlier trial answers before computing
        # their reported statistics. Select by trial index, independent of when
        # concurrent requests finish, while retaining every scored sample.
        last_trial = max(self.trials)
        accuracy = self.trials[last_trial].finish()["accuracy"]
        metrics = {
            self.policy.accuracy: accuracy,
            "overall_variance": accuracy * (1 - accuracy),
            "num_trials": len(self.trials),
            "num_examples": len(self.examples),
        }
        for (tag, group), trials in self.group_trials.items():
            if tag not in self.policy.groups:
                continue
            score = trials[last_trial].finish()["accuracy"]
            metrics.setdefault(self.policy.groups[tag], {})[group] = {
                "accuracy": score, "variance": score * (1 - score),
            }
        return metrics

    def _rounded_split_metrics(self) -> dict[str, float]:
        grouped: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
        for (tag, value), accumulator in self.groups.items():
            if tag == "split":
                grouped["overall"][value] = self._rounded_accuracy(accumulator)
            elif tag == "benchmark_split":
                benchmark, split = value.rsplit("|", 1)
                grouped[benchmark][split] = self._rounded_accuracy(accumulator)
        return {
            name: self._weighted_split_score(splits)
            for name, splits in grouped.items()
        }

    @staticmethod
    def _rounded_accuracy(accumulator: MeanMetrics) -> tuple[float, float]:
        return round(accumulator.finish()["accuracy"], 3), accumulator.counts["accuracy"]

    @staticmethod
    def _weighted_split_score(splits: dict[str, tuple[float, float]]) -> float:
        total = sum(count for _score, count in splits.values())
        return sum(score * count for score, count in splits.values()) / total
