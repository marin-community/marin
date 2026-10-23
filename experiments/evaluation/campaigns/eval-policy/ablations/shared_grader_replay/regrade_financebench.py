#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay sealed FinanceBench answers through the pinned MiniMax judge."""

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import yaml
from eval.chat_benchmarks.FinanceBench import eval_instruct
from eval.chat_benchmarks.FinanceBench.eval_instruct import FinanceBenchBenchmark
from eval.graders import answer_equivalence
from eval.graders.answer_equivalence import EquivalenceJudgment, EquivalenceRequest, JudgeConfig, JudgeLabel
from finestore.reader import ReadView

from ..olympiadbench_judge.regrade import JUDGE_EXTRA_BODY, Source  # noqa: TID252

EXPECTED_TRIALS = 50
SAMPLE_COLUMNS = ["doc_id", "doc", "output", "metrics"]


def read_sources(path: Path) -> list[Source]:
    """Select the sealed FinanceBench cells from the frozen tracker manifest."""
    rows = yaml.safe_load(path.read_text())["models"]
    sources = [
        Source(
            row["model"],
            float(row["benchmarks"]["financebench"]["tracker_score"]),
            row["benchmarks"]["financebench"]["source"],
        )
        for row in rows
    ]
    if not sources or len({source.model for source in sources}) != len(sources):
        raise ValueError("expected nonempty, unique FinanceBench sources")
    return sources


def source_examples(source: Source) -> tuple[list[dict], list[dict]]:
    """Rebuild the original 50-item FinanceBench input from sealed sample rows."""
    table = ReadView(source.uri).scan("samples", columns=SAMPLE_COLUMNS)
    if table is None or table.num_rows != EXPECTED_TRIALS:
        raise ValueError(f"{source.model}: expected {EXPECTED_TRIALS} sealed samples")
    rows = table.to_pylist(maps_as_pydicts="strict")
    by_document = {int(row["doc_id"]): row for row in rows}
    if len(by_document) != EXPECTED_TRIALS or set(by_document) != set(range(EXPECTED_TRIALS)):
        raise ValueError(f"{source.model}: incomplete or duplicate document IDs")
    ordered_rows = [by_document[index] for index in range(EXPECTED_TRIALS)]
    examples = []
    for row in ordered_rows:
        example = json.loads(row["doc"])
        example["model_output"] = row["output"] or ""
        examples.append(example)
    scored_rows = [row for row in ordered_rows if "accuracy" in row["metrics"]]
    old_score = sum(float(row["metrics"]["accuracy"]) for row in scored_rows) / len(scored_rows)
    if abs(old_score - source.old_score) > 0.0005:
        raise ValueError(f"{source.model}: sample score {old_score:.6f} differs from tracker {source.old_score:.3f}")
    return examples, ordered_rows


def regrade(source: Source, base_url: str, judge_model: str) -> dict:
    """Judge frozen FinanceBench responses without regenerating model answers."""
    examples, rows = source_examples(source)
    judge_config = JudgeConfig(
        model=judge_model,
        base_url=base_url,
        api_key="local-judge-replay",
        extra_body=JUDGE_EXTRA_BODY,
    )

    async def complete(
        requests: Sequence[EquivalenceRequest], _config: JudgeConfig, num_workers: int = 16
    ) -> list[EquivalenceJudgment | BaseException]:
        return await answer_equivalence.judge_equivalence(requests, judge_config, num_workers=num_workers)

    benchmark = FinanceBenchBenchmark(
        annotator_model=judge_model,
        judge_base_url=base_url,
        judge_api_key="local-judge-replay",
    )
    benchmark.judge_config = replace(benchmark.judge_config, extra_body=JUDGE_EXTRA_BODY)
    with patch.object(eval_instruct, "judge_equivalence", complete):
        scored = benchmark.evaluate_responses({"examples": examples, "judge_model": judge_model})
    if scored["num_judge_failed"]:
        raise RuntimeError(f"{source.model}: {scored['num_judge_failed']} MiniMax judge requests failed")

    trials = []
    for index, (example, row) in enumerate(zip(examples, rows, strict=True)):
        label = JudgeLabel(example["judge_label"])
        trials.append(
            {
                "doc_id": index,
                "reference_answer": example["answer"],
                "candidate_answer": example["model_output"],
                "source_correct": bool(row["metrics"]["accuracy"]) if "accuracy" in row["metrics"] else None,
                "corrected_correct": label == JudgeLabel.CORRECT,
                "judge_label": label.value,
                "judge_raw": example["judge_raw"],
            }
        )
    corrected_score = sum(trial["corrected_correct"] for trial in trials) / EXPECTED_TRIALS
    if abs(corrected_score - float(scored["accuracy"])) > 1e-9:
        raise ValueError(f"{source.model}: native aggregate differs from trial verdicts")
    return {
        "model": source.model,
        "benchmark": "financebench",
        "source": source.uri,
        "tracker_score": source.old_score,
        "corrected_score": corrected_score,
        "num_trials": EXPECTED_TRIALS,
        "num_judge_failed": scored["num_judge_failed"],
        "judge_model": judge_model,
        "judge_request_extra_body": JUDGE_EXTRA_BODY,
        "changed_trials": sum(
            trial["source_correct"] is not None and trial["source_correct"] != trial["corrected_correct"]
            for trial in trials
        ),
        "trials": trials,
    }
