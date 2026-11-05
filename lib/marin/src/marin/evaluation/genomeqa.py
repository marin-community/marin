# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GenomeQA reporting views derived from one full evaluation's saved metrics."""

from marin.evaluation.records import EvalRunRecord, MetricKind, TaskCoverage

SUBTASK_VIEWS = {
    "1_enhancer_vs_promoter": "genomeqa-enhancer-promoter",
    "2_splice_sites": "genomeqa-splice-sites",
    "3_species_cls": "genomeqa-taxonomy",
    "4_histone_marks": "genomeqa-histone-marks",
    "5_tfbs": "genomeqa-tfbs",
    "6_motif": "genomeqa-motifs",
}


def subtask_records(record: EvalRunRecord) -> tuple[EvalRunRecord, ...]:
    """Project complete subtask scores without inference or duplicating persisted run records."""
    if record.evaluation.name != "genomeqa" or len(record.evaluation.tasks) != 1:
        return ()
    task = record.evaluation.tasks[0]
    benchmark = task.benchmark
    if benchmark is None or benchmark.task != "GenomeQA" or benchmark.primary_metric != "genomeqa_macro":
        return ()
    views = []
    for task_key, canonical in record.canonical_metrics.items():
        raw = record.metrics.get(task_key, {})
        for subtask, view_name in SUBTASK_VIEWS.items():
            if subtask not in canonical:
                continue
            cells = (f"{subtask}/binary", f"{subtask}/mcq")
            expected = 200 if subtask == "6_motif" else 1000
            # A scalar alone is insufficient evidence that the complete subtask was graded.
            if any(raw.get(f"{cell}/graded") != expected // 2 for cell in cells):
                continue
            metric = next((metric for metric in benchmark.metrics if metric.name == subtask), None)
            if metric is None:
                continue
            view_benchmark = benchmark.model_copy(
                update={
                    "primary_metric": subtask,
                    "metric_kind": MetricKind.CONTINUOUS,
                    "n_benchmark": expected,
                    "n_attempted": expected,
                }
            )
            view_task = task.model_copy(update={"benchmark": view_benchmark})
            evaluation = record.evaluation.model_copy(
                update={
                    "name": view_name,
                    "family": None,
                    "tasks": (view_task,),
                }
            )
            coverage = TaskCoverage(
                n_benchmark=expected,
                n_attempted=expected,
                n_scored=expected,
                n_correct=int(sum(raw[f"{cell}/correct"] for cell in cells)),
                n_unanswered=int(sum(raw.get(f"{cell}/unanswered", 0) for cell in cells)),
            )
            views.append(
                record.model_copy(
                    update={
                        "evaluation": evaluation,
                        "coverage": {task_key: coverage},
                        "metrics": {task_key: raw},
                        "canonical_metrics": {task_key: canonical},
                    }
                )
            )
    return tuple(views)
