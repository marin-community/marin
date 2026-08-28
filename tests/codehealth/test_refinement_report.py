# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import datetime as dt

import pytest

from infra.codehealth import refinement_report as report
from infra.codehealth.github_review_corpus import GitHubUsage, ReviewEventRecord
from infra.codehealth.review_corpus import BenchmarkLabel, BenchmarkSummary, CorpusManifest


def _manifest() -> CorpusManifest:
    return CorpusManifest(
        schema_version=1,
        snapshot_id="snapshot-a",
        repository="owner/repo",
        window_start="2026-07-29T00:00:00Z",
        window_end="2026-08-28T00:00:00Z",
        collection_started_at="2026-08-28T00:00:00Z",
        collection_completed_at="2026-08-28T00:10:00Z",
        exporter_sha="exporter-a",
        catalog_sha="catalog-a",
        benchmark_sha="benchmark-a",
        benchmark=BenchmarkSummary(
            cases=2,
            positive_cases=1,
            hard_negatives=1,
            covered_rules=1,
            catalog_rules=1,
        ),
        complete=True,
        candidate_pull_requests=20,
        included_pull_requests=3,
        review_events=3,
        human_events_in_window=3,
        github_usage=GitHubUsage(
            graphql_requests=4,
            graphql_points=5,
            rest_requests=6,
            projected_rest_requests=10,
        ),
        files=(),
        limitations=("Deleted comments are not observable.",),
    )


def _event(pr_number: int) -> ReviewEventRecord:
    event_id = f"owner/repo:inline_comment:{pr_number}"
    return ReviewEventRecord(
        event_id=event_id,
        kind="inline_comment",
        database_id=pr_number,
        node_id=f"node-{pr_number}",
        repository="owner/repo",
        pr_number=pr_number,
        pr_author="author",
        author="reviewer",
        author_type="User",
        author_association="MEMBER",
        body="Move consumer policy to its adapter.",
        state=None,
        created_at="2026-08-27T00:00:00Z",
        updated_at="2026-08-27T00:00:00Z",
        submitted_at=None,
        source_url=f"https://github.com/owner/repo/pull/{pr_number}#discussion_r{pr_number}",
        review_id=pr_number,
        thread_id=f"thread-{pr_number}",
        parent_comment_id=None,
        thread_is_resolved=True,
        thread_is_outdated=False,
        thread_resolved_by="author",
        path="shared.py",
        side="RIGHT",
        line=1,
        original_line=1,
        start_side=None,
        start_line=None,
        original_start_line=None,
        commit_id="head-a",
        original_commit_id="head-a",
        diff_hunk="@@ -1 +1 @@",
        is_bot=False,
        is_agent_marked=False,
        is_human=True,
        in_window=True,
    )


def _refinement_report() -> report.RefinementReport:
    proposal = report.RuleProposal(
        code="ml-consumer-policy-in-shared-layer",
        lane="interfaces",
        title="Consumer policy in a shared layer",
        condition="Flag consumer-specific policy implemented in a reusable lower layer.",
        when_allowed="The lower layer owns the lifecycle behavior.",
        precedence=("ml-reverse-layer-import",),
        evidence=tuple(
            report.EvidenceReference(event_id=_event(pr_number).event_id, relevance="Policy belongs to the adapter.")
            for pr_number in (1, 2, 3)
        ),
        counterexamples=("A provider name alone does not establish ownership.",),
    )
    analysis = report.RefinementAnalysis(
        schema_version=1,
        corpus_snapshot_id="snapshot-a",
        catalog_sha="catalog-a",
        benchmark_sha="benchmark-a",
        proposals=(proposal,),
        existing_rule_gaps=(
            report.ExistingRuleGap(
                pr_number=9,
                human_events=4,
                rules=("ml-local-import",),
                finding="The catalog covered the finding but the review missed it.",
            ),
        ),
        limitations=("Production findings are not independently adjudicated.",),
    )
    events = tuple(_event(pr_number) for pr_number in (1, 2, 3))
    return report.RefinementReport(
        manifest=_manifest(),
        analysis=analysis,
        proposals=(
            report.ProposalEvidence(
                proposal=proposal,
                events_30_days=3,
                pull_requests_30_days=3,
                events_7_days=3,
                pull_requests_7_days=3,
                events=events,
            ),
        ),
        benchmark=report.BenchmarkScore(
            prediction_sha="prediction-a",
            cases=2,
            exact_matches=2,
            true_positives=1,
            false_positives=0,
            false_negatives=0,
            hard_negatives=1,
            true_negatives=1,
        ),
        production_7_days=report.ProductionWindow(7, 12, 10, 8, 25),
        production_30_days=report.ProductionWindow(30, 40, 35, 28, 100),
        current_catalog=report.CurrentCatalogActivity(
            sha="catalog-deployed-a",
            observed_days=14.0,
            started_runs=20,
            successful_runs=18,
            distinct_heads=17,
            changed_files=200,
            changed_lines=5000,
            findings=50,
            runs_with_findings=12,
            zero_finding_rules=("ml-local-import",),
        ),
    )


def test_render_report_separates_synthetic_benchmark_from_production_activity() -> None:
    refinement = _refinement_report()

    markdown = report.render_markdown(refinement)
    slack = report.render_slack(
        refinement,
        report_url="https://loom.example/artifacts/report",
        catalog_pr_url="https://github.com/owner/repo/pull/10",
    )

    assert "3 events across 3 PRs in 30 days" in markdown
    assert "| 30 days | 40 | 35 | 28 | 100 |" in markdown
    assert "synthetic regression check" in markdown
    assert "not an estimate of production precision or recall" in markdown
    assert "14.0 days of observed history" in markdown
    assert "https://github.com/owner/repo/pull/1#discussion_r1" in markdown
    assert "This is not production recall" in slack
    assert "Catalog PR: https://github.com/owner/repo/pull/10" in slack
    assert len(slack) < 4_000


def test_score_benchmark_reports_false_positive_and_false_negative() -> None:
    labels = (
        BenchmarkLabel(
            alias="case-001",
            source_id="positive",
            description="positive",
            expected_rules=("ml-example",),
            provenance="catalog-example",
            source_url=None,
            source_pr=None,
            source_author=None,
        ),
        BenchmarkLabel(
            alias="case-002",
            source_id="negative",
            description="negative",
            expected_rules=(),
            provenance="synthetic-hard-negative",
            source_url=None,
            source_pr=None,
            source_author=None,
        ),
    )
    predictions = (
        report.BenchmarkPrediction(alias="case-001", predicted_rules=()),
        report.BenchmarkPrediction(alias="case-002", predicted_rules=("ml-example",)),
    )

    score = report.score_benchmark(
        predictions,
        labels,
        catalog_rules={"ml-example"},
        prediction_sha="prediction-a",
    )

    assert score.exact_matches == 0
    assert score.true_positives == 0
    assert score.false_positives == 1
    assert score.false_negatives == 1
    assert score.true_negatives == 0
    assert score.precision == 0.0
    assert score.recall == 0.0


def test_score_benchmark_rejects_incomplete_prediction_set() -> None:
    label = BenchmarkLabel(
        alias="case-001",
        source_id="positive",
        description="positive",
        expected_rules=("ml-example",),
        provenance="catalog-example",
        source_url=None,
        source_pr=None,
        source_author=None,
    )

    with pytest.raises(ValueError, match="cover every case"):
        report.score_benchmark((), (label,), catalog_rules={"ml-example"}, prediction_sha="prediction-a")


def test_production_window_counts_only_successful_pre_commit_reviews() -> None:
    end = dt.datetime(2026, 8, 28, tzinfo=dt.UTC)
    rows = (
        {
            "ts": "2026-08-27T00:00:00Z",
            "tool": "pre-commit-review",
            "agent_exit_code": 0,
            "timed_out": False,
            "diff_added_lines": 90,
            "diff_removed_lines": 11,
            "finding_count": 3,
        },
        {
            "ts": "2026-08-27T00:00:00Z",
            "tool": "pre-commit-review",
            "agent_exit_code": 1,
            "timed_out": False,
            "diff_added_lines": 500,
            "diff_removed_lines": 0,
            "finding_count": 0,
        },
        {
            "ts": "2026-08-27T00:00:00Z",
            "tool": "review-pr",
            "agent_exit_code": 0,
            "timed_out": False,
            "diff_added_lines": 500,
            "diff_removed_lines": 0,
            "finding_count": 8,
        },
    )

    metrics = report.production_window(rows, end=end, days=7)

    assert metrics == report.ProductionWindow(
        days=7,
        started_runs=2,
        successful_runs=1,
        meta_eligible_runs=1,
        findings=3,
    )
