# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from scripts.pm.render_agent_moe_report import (
    DEFAULT_DATA_PATH,
    DEFAULT_REPORT_PATH,
    AuditResult,
    Experiment,
    Metadata,
    Outcome,
    RemoteIssue,
    ReportData,
    audit_snapshot,
    load_report_data,
    render_report,
)


def test_agent_moe_snapshot_loads_expected_unique_tracker_issues():
    report = load_report_data(DEFAULT_DATA_PATH)

    issue_numbers = [experiment.issue for experiment in report.experiments]
    assert len(issue_numbers) == report.metadata.expected_issue_count
    assert len(set(issue_numbers)) == len(issue_numbers)


def test_agent_moe_generated_report_matches_snapshot():
    report = load_report_data(DEFAULT_DATA_PATH)

    assert DEFAULT_REPORT_PATH.read_text() == render_report(report)


def test_agent_moe_audit_reports_new_removed_and_changed_issues():
    metadata = Metadata(
        schema_version=1,
        snapshot_date="2026-07-24",
        repository="marin-community/marin",
        parent_issue=4281,
        title_prefix="Agent MoE Experiment:",
        expected_issue_count=2,
        headline_summary="Summary.",
    )
    report = ReportData(
        metadata=metadata,
        experiments=(
            Experiment(
                issue=1,
                title="Current",
                category="Modeling",
                section=None,
                outcome=Outcome.WORKED,
                model_flops_speedup="1.12x",
                wall_clock_speedup="1.10x",
                summary="Passed.",
                state="CLOSED",
                source_updated_at="2026-07-20T00:00:00Z",
            ),
            Experiment(
                issue=2,
                title="Removed",
                category="Modeling",
                section=None,
                outcome=Outcome.IN_PROGRESS,
                model_flops_speedup="—",
                wall_clock_speedup="—",
                summary="Pending.",
                state="OPEN",
                source_updated_at="2026-07-20T00:00:00Z",
            ),
        ),
        foundations=(),
    )
    remote_issues = (
        RemoteIssue(
            number=1,
            title="Agent MoE Experiment: Current",
            state="CLOSED",
            updated_at="2026-07-21T00:00:00Z",
        ),
        RemoteIssue(
            number=3,
            title="Agent MoE Experiment: New",
            state="OPEN",
            updated_at="2026-07-22T00:00:00Z",
        ),
        RemoteIssue(
            number=4,
            title="Unrelated tracker issue",
            state="OPEN",
            updated_at="2026-07-22T00:00:00Z",
        ),
    )

    result = audit_snapshot(report, remote_issues)

    assert result == AuditResult(
        new_issues=(remote_issues[1],),
        removed_issue_numbers=(2,),
        changed_issues=(remote_issues[0],),
    )
    assert result.has_drift
