# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for Echo's graded search benchmark."""

import json

import pytest
import search_benchmark
from search_result import SearchResult


def result(identifier: str, domain: str, url: str) -> SearchResult:
    return SearchResult(
        id=identifier,
        domain=domain,
        title=identifier,
        subtitle="",
        url=url,
        snippet="",
        score=1.0,
        distance=0.1,
        lexical_score=None,
    )


def test_query_metrics_matches_github_comments_to_their_canonical_artifact():
    case = search_benchmark.BenchmarkCase(
        id="scheduler-pr",
        query="scheduler change",
        domains=("pr",),
        intent="recent_activity",
        split="dev",
        relevant=(
            search_benchmark.RelevanceJudgment(
                "pr",
                "https://github.com/marin-community/marin/pull/7747",
                3,
            ),
        ),
    )
    results = [
        result(
            "pr:10",
            "pr",
            "https://github.com/marin-community/marin/pull/7000",
        ),
        result(
            "pr:11",
            "pr",
            "https://github.com/marin-community/marin/pull/7747#issuecomment-123",
        ),
    ]

    metrics = search_benchmark.query_metrics(case, results, 10)

    assert metrics.reciprocal_rank == 0.5
    assert metrics.hit
    assert metrics.recalled == 1


def test_evaluation_measures_ranked_relevance_and_no_answer_suppression(tmp_path):
    benchmark_path = tmp_path / "benchmark.jsonl"
    capture_path = tmp_path / "capture.jsonl"
    cases = [
        {
            "id": "deploy-iris",
            "query": "how do i deploy iris",
            "domains": ["file"],
            "intent": "how_to",
            "split": "dev",
            "source": {"kind": "repository", "path": "lib/iris/OPS.md"},
            "relevant": [{"domain": "file", "target": "lib/iris/OPS.md", "grade": 3}],
        },
        {
            "id": "unanswerable",
            "query": "how do i deploy the nonexistent zeppelin service",
            "domains": ["file"],
            "intent": "no_answer",
            "split": "dev",
            "source": {"kind": "synthetic"},
            "relevant": [],
        },
    ]
    captures = [
        {
            "id": "deploy-iris",
            "latency_ms": 20,
            "results": [
                {
                    "id": "file:lib/iris/OPS.md",
                    "domain": "file",
                    "title": "Iris Operations",
                    "subtitle": "lib/iris/OPS.md:68",
                    "url": "https://github.com/marin-community/marin/blob/abc/lib/iris/OPS.md#L68",
                    "snippet": "Restart the controller.",
                    "score": 0.1,
                    "distance": 0.2,
                    "lexical_score": None,
                }
            ],
        },
        {"id": "unanswerable", "latency_ms": 30, "results": []},
    ]
    benchmark_path.write_text("\n".join(json.dumps(case) for case in cases))
    capture_path.write_text("\n".join(json.dumps(capture) for capture in captures))

    metrics = search_benchmark.evaluation(
        search_benchmark.load_benchmark(benchmark_path),
        search_benchmark.load_captured(capture_path),
        10,
    )

    assert metrics["mrr@10"] == 1.0
    assert metrics["ndcg@10"] == 1.0
    assert metrics["judgment_recall@10"] == 1.0
    assert metrics["no_answer_accuracy"] == 1.0
    assert metrics["latency_ms_p95"] == 30.0


def test_load_benchmark_rejects_duplicate_ids(tmp_path):
    benchmark_path = tmp_path / "benchmark.jsonl"
    case = {
        "id": "duplicate",
        "query": "query",
        "domains": ["file"],
        "intent": "how_to",
        "split": "dev",
        "relevant": [],
    }
    benchmark_path.write_text(f"{json.dumps(case)}\n{json.dumps(case)}\n")

    with pytest.raises(ValueError, match="duplicate benchmark id"):
        search_benchmark.load_benchmark(benchmark_path)
