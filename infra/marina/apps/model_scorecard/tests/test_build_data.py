# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from urllib.parse import unquote

import pytest
from marin.evaluation.records import (
    BenchmarkMetadataRef,
    EvalRef,
    EvalRunRecord,
    EvalTaskRef,
    MetricKind,
    ModelConfigRef,
    ModelRef,
    Provenance,
    RunStatus,
)
from model_scorecard.build_data import LinkKind, build_scorecard, source_link

RUN_ID = "20261111-225936-model-math500-abc"


def record(
    eval_name: str = "math500",
    status: RunStatus = RunStatus.SUCCEEDED,
    value: float = 0.97856,
    *,
    location: str = "org/model",
    revision: str = "abc123",
    step_status: str | None = None,
) -> EvalRunRecord:
    benchmark = BenchmarkMetadataRef.model_construct(primary_metric="accuracy", metric_kind=MetricKind.BINARY)
    evaluation = EvalRef.model_construct(
        name=eval_name,
        tasks=(EvalTaskRef.model_construct(benchmark=benchmark, num_fewshot=0),),
        evalchemy=None,
        harbor=None,
    )
    return EvalRunRecord.model_construct(
        run_id=RUN_ID,
        status=status,
        step_status=step_status,
        model=ModelRef.model_construct(
            name="Model", location=location, config=ModelConfigRef.model_construct(revision=revision)
        ),
        evaluation=evaluation,
        metrics={"math500": {"accuracy": value}},
        canonical_metrics={"math500": {"accuracy": value, "accuracy_stderr": 0.006}},
        coverage={},
        repeat_aggregates={},
        created_at="2026-11-11T00:00:00Z",
        version=None,
        provenance=Provenance.model_construct(git_sha="abc123", eval_runtime=None),
    )


def manifest(cells: dict) -> dict:
    return {
        "evals": ["math500", "gsm8k-0shot"],
        "models": [{"name": "Model", "hf_repo": "org/model", "revision": "abc123", "cells": cells}],
    }


def test_run_cell_scores_on_0_100_scale_and_links_to_evaldash() -> None:
    scorecard = build_scorecard(manifest({"math500": {"run_id": RUN_ID, "label": "unofficial"}}), lambda _: record())

    model = scorecard["models"][0]
    assert model["url"] == "https://huggingface.co/org/model/tree/abc123"
    assert model["cells"]["math500"] == {
        "score": 97.86,
        "label": "unofficial",
        "link": {"kind": "evaldash", "url": f"https://marina.oa.dev/evaldash/runs/{RUN_ID}"},
        "run_id": RUN_ID,
    }
    assert model["cells"]["gsm8k-0shot"] == {"missing": ""}


def test_piqa_cell_uses_published_normalized_accuracy() -> None:
    piqa = record(eval_name="piqa").model_copy(
        update={
            "metrics": {"piqa": {"acc": 0.25, "acc_norm": 0.75}},
            "canonical_metrics": {"piqa": {"accuracy": 0.25, "normalized_accuracy": 0.75}},
        }
    )
    scorecard = build_scorecard(
        {**manifest({"piqa": {"run_id": RUN_ID, "label": "unofficial"}}), "evals": ["piqa"]}, lambda _: piqa
    )

    assert scorecard["models"][0]["cells"]["piqa"]["score"] == 75.0


def test_source_in_a_run_directory_links_to_evaldash() -> None:
    link = source_link(f"s3://marin-us-east-02a/marin/evals/{RUN_ID}/results")
    assert link.kind == LinkKind.EVALDASH
    assert link.url == f"https://marina.oa.dev/evaldash/runs/{RUN_ID}"


def test_source_outside_a_run_directory_links_to_filer() -> None:
    uri = "s3://marin-us-east-02a/marin/evals/ablations/regrade-1/results/math500/model.json"
    link = source_link(uri)
    assert link.kind == LinkKind.STORAGE
    assert unquote(link.url.removeprefix("https://filer.marina.oa.dev/#")) == uri


@pytest.mark.parametrize(
    ("fetched", "message"),
    [
        (record(status=RunStatus.FAILED), "status"),
        (record(eval_name="gsm8k-0shot"), "evaluates"),
        (record(step_status="RUNNING"), "step status"),
        (record(location="other/model"), "other/model"),
        (record(revision="other-revision"), "other-revision"),
    ],
)
def test_run_cell_rejects_failed_or_mismatched_runs(fetched: EvalRunRecord, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build_scorecard(manifest({"math500": {"run_id": RUN_ID, "label": "official"}}), lambda _: fetched)
