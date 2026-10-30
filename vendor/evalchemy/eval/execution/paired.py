"""Paired-condition reduction over compact per-example measurements."""

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class PairedRow:
    tags: Mapping[str, str | tuple[str, ...]]
    pair_id: str
    trial: int | None
    metrics: Mapping[str, float]


class PairedConditions:
    def __init__(self):
        self.rows: list[PairedRow] = []

    def add(self, item):
        self.rows.append(PairedRow(dict(item.spec.tags), item.spec.grader.reference["pair_id"], item.spec.trial, dict(item.grade.metrics)))

    def finish(self):
        baseline = {(row.pair_id, row.trial): row.metrics["accuracy"] for row in self.rows if row.tags["condition"] == "baseline"}
        tasks = defaultdict(list)
        for row in self.rows:
            tasks[row.tags["task"]].append(row)
        result = {}
        def average(rows, metric):
            return 100 * sum(row.metrics[metric] for row in rows) / len(rows)
        for task, rows in tasks.items():
            result[task] = average(rows, "accuracy")
            result[task + "_no_answer"] = average(rows, "no_answer")
            conditions = defaultdict(list)
            for row in rows:
                if row.tags["condition"] != "baseline":
                    conditions[row.tags["condition"]].append(row)
            for condition, members in conditions.items():
                prefix = f"{task}:{condition}"
                result[prefix] = average(members, "accuracy")
                result[prefix + "_no_answer"] = average(members, "no_answer")
                paired = [row for row in members if (row.pair_id, row.trial) in baseline]
                if paired:
                    clean = 100 * sum(baseline[row.pair_id, row.trial] for row in paired) / len(paired)
                    result[prefix + "_clean_subset"] = clean
                    result[prefix + "_paired_delta"] = result[prefix] - clean
                    result[prefix + "_harmed"] = sum(baseline[row.pair_id, row.trial] and not row.metrics["accuracy"] for row in paired)
                    result[prefix + "_helped"] = sum(row.metrics["accuracy"] and not baseline[row.pair_id, row.trial] for row in paired)
                changed = [row for row in members if row.tags["changed"] == "True"]
                if changed and len(changed) != len(members):
                    result[prefix + "_changed_only"] = average(changed, "accuracy")
                number_words = [row for row in members if row.tags["number_words"] == "True"]
                if number_words:
                    result[prefix + "_number_words"] = average(number_words, "accuracy")
        return result
