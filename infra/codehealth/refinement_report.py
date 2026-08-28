# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build verified agentic-lint refinement reports from a frozen corpus."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import click
from pydantic import Field

from .github_review_corpus import CorpusModel, ReviewEventRecord
from .review_corpus import (
    CASE_ALIAS_PATTERN,
    BenchmarkLabel,
    CorpusManifest,
    Lane,
    catalog_rules,
    validate_corpus,
)
from .review_tables import parse_utc

REPORT_SCHEMA_VERSION = 1
REPORT_ARTIFACT_NAME = "codehealth-refinement-report"
ANALYSIS_ARTIFACT_NAME = "codehealth-refinement-analysis"
PRODUCTION_TOOL = "pre-commit-review"
RETIREMENT_DAYS = 30


class EvidenceReference(CorpusModel):
    event_id: str
    relevance: str


class RuleProposal(CorpusModel):
    code: str = Field(pattern=r"^ml-[a-z0-9-]+$")
    lane: Lane
    title: str
    condition: str
    when_allowed: str
    precedence: tuple[str, ...] = ()
    evidence: tuple[EvidenceReference, ...]
    counterexamples: tuple[str, ...] = ()


class ExistingRuleGap(CorpusModel):
    pr_number: int
    human_events: int = Field(ge=1)
    rules: tuple[str, ...]
    finding: str


class RefinementAnalysis(CorpusModel):
    schema_version: Literal[1]
    corpus_snapshot_id: str
    catalog_sha: str
    benchmark_sha: str
    proposals: tuple[RuleProposal, ...]
    existing_rule_gaps: tuple[ExistingRuleGap, ...] = ()
    limitations: tuple[str, ...] = ()


class BenchmarkPrediction(CorpusModel):
    alias: str = Field(pattern=CASE_ALIAS_PATTERN)
    predicted_rules: tuple[str, ...]


@dataclass(frozen=True)
class BenchmarkScore:
    prediction_sha: str
    cases: int
    exact_matches: int
    true_positives: int
    false_positives: int
    false_negatives: int
    hard_negatives: int
    true_negatives: int

    @property
    def precision(self) -> float:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        return 2 * self.precision * self.recall / denominator if denominator else 0.0


@dataclass(frozen=True)
class ProductionWindow:
    days: int
    started_runs: int
    successful_runs: int
    meta_eligible_runs: int
    findings: int


@dataclass(frozen=True)
class CurrentCatalogActivity:
    sha: str | None
    observed_days: float
    started_runs: int
    successful_runs: int
    distinct_heads: int
    changed_files: int
    changed_lines: int
    findings: int
    runs_with_findings: int
    zero_finding_rules: tuple[str, ...]


@dataclass(frozen=True)
class ProposalEvidence:
    proposal: RuleProposal
    events_30_days: int
    pull_requests_30_days: int
    events_7_days: int
    pull_requests_7_days: int
    events: tuple[ReviewEventRecord, ...]


@dataclass(frozen=True)
class RefinementReport:
    manifest: CorpusManifest
    analysis: RefinementAnalysis
    proposals: tuple[ProposalEvidence, ...]
    benchmark: BenchmarkScore
    production_7_days: ProductionWindow
    production_30_days: ProductionWindow
    current_catalog: CurrentCatalogActivity


def _jsonl(path: Path) -> tuple[dict[str, object], ...]:
    return tuple(json.loads(line) for line in path.read_text().splitlines())


def _event_time(event: ReviewEventRecord) -> dt.datetime:
    timestamps = [
        parse_utc(value) for value in (event.created_at, event.updated_at, event.submitted_at) if value is not None
    ]
    if not timestamps:
        raise ValueError(f"review event has no activity timestamp: {event.event_id}")
    return max(timestamps)


def _predictions(path: Path) -> tuple[BenchmarkPrediction, ...]:
    predictions = tuple(BenchmarkPrediction.model_validate_json(line) for line in path.read_text().splitlines())
    if not predictions:
        raise ValueError("benchmark predictions are empty")
    aliases = [prediction.alias for prediction in predictions]
    if aliases != sorted(set(aliases)):
        raise ValueError("benchmark predictions must have unique aliases in case order")
    return predictions


def score_benchmark(
    predictions: Sequence[BenchmarkPrediction],
    labels: Sequence[BenchmarkLabel],
    *,
    catalog_rules: set[str],
    prediction_sha: str,
) -> BenchmarkScore:
    """Score one committed prediction set after benchmark labels are opened."""
    expected_aliases = [label.alias for label in labels]
    if [prediction.alias for prediction in predictions] != expected_aliases:
        raise ValueError("benchmark predictions must cover every case in case order")

    exact_matches = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    hard_negatives = 0
    true_negatives = 0
    for prediction, label in zip(predictions, labels, strict=True):
        if len(prediction.predicted_rules) != len(set(prediction.predicted_rules)):
            raise ValueError(f"benchmark prediction {prediction.alias} repeats a rule")
        predicted = set(prediction.predicted_rules)
        unknown = predicted - catalog_rules
        if unknown:
            raise ValueError(f"benchmark prediction {prediction.alias} uses unknown rules: {sorted(unknown)}")
        expected = set(label.expected_rules)
        exact_matches += predicted == expected
        true_positives += len(predicted & expected)
        false_positives += len(predicted - expected)
        false_negatives += len(expected - predicted)
        if not expected:
            hard_negatives += 1
            true_negatives += not predicted
    return BenchmarkScore(
        prediction_sha=prediction_sha,
        cases=len(labels),
        exact_matches=exact_matches,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        hard_negatives=hard_negatives,
        true_negatives=true_negatives,
    )


def _in_window(row: dict[str, object], start: dt.datetime) -> bool:
    value = row.get("ts")
    return value is not None and parse_utc(str(value)) >= start


def _successful_run(row: dict[str, object]) -> bool:
    exit_code = row.get("agent_exit_code")
    return exit_code is not None and int(exit_code) == 0 and not bool(row.get("timed_out"))


def production_window(
    invocations: Sequence[dict[str, object]],
    *,
    end: dt.datetime,
    days: int,
) -> ProductionWindow:
    start = end - dt.timedelta(days=days)
    started = [row for row in invocations if row.get("tool") == PRODUCTION_TOOL and _in_window(row, start)]
    successful = [row for row in started if _successful_run(row)]
    return ProductionWindow(
        days=days,
        started_runs=len(started),
        successful_runs=len(successful),
        meta_eligible_runs=sum(
            int(row.get("diff_added_lines") or 0) + int(row.get("diff_removed_lines") or 0) > 100 for row in successful
        ),
        findings=sum(int(row.get("finding_count") or 0) for row in successful),
    )


def current_catalog_activity(
    invocations: Sequence[dict[str, object]],
    findings: Sequence[dict[str, object]],
    *,
    end: dt.datetime,
    catalog_rules: set[str],
) -> CurrentCatalogActivity:
    catalog_runs = [
        row
        for row in invocations
        if row.get("tool") == PRODUCTION_TOOL and row.get("lint_catalog_sha") and row.get("ts")
    ]
    if not catalog_runs:
        return CurrentCatalogActivity(None, 0.0, 0, 0, 0, 0, 0, 0, 0, tuple(sorted(catalog_rules)))
    latest = max(catalog_runs, key=lambda row: parse_utc(str(row["ts"])))
    sha = str(latest["lint_catalog_sha"])
    current = [row for row in catalog_runs if row.get("lint_catalog_sha") == sha]
    successful = [row for row in current if _successful_run(row)]
    successful_ids = {str(row["invocation_id"]) for row in successful}
    codes = {
        str(row["code"])
        for row in findings
        if str(row.get("invocation_id")) in successful_ids and row.get("code") in catalog_rules
    }
    first = min(parse_utc(str(row["ts"])) for row in current)
    return CurrentCatalogActivity(
        sha=sha,
        observed_days=max(0.0, (end - first).total_seconds() / 86_400),
        started_runs=len(current),
        successful_runs=len(successful),
        distinct_heads=len({row.get("head_sha") for row in successful if row.get("head_sha")}),
        changed_files=sum(int(row.get("diff_files") or 0) for row in successful),
        changed_lines=sum(
            int(row.get("diff_added_lines") or 0) + int(row.get("diff_removed_lines") or 0) for row in successful
        ),
        findings=sum(int(row.get("finding_count") or 0) for row in successful),
        runs_with_findings=sum(int(row.get("finding_count") or 0) > 0 for row in successful),
        zero_finding_rules=tuple(sorted(catalog_rules - codes)),
    )


def _proposal_evidence(
    proposals: Sequence[RuleProposal],
    events: Sequence[ReviewEventRecord],
    *,
    end: dt.datetime,
) -> tuple[ProposalEvidence, ...]:
    by_id = {event.event_id: event for event in events}
    seven_day_start = end - dt.timedelta(days=7)
    result: list[ProposalEvidence] = []
    for proposal in proposals:
        referenced: list[ReviewEventRecord] = []
        seen: set[str] = set()
        for reference in proposal.evidence:
            if reference.event_id in seen:
                raise ValueError(f"proposal {proposal.code} repeats evidence {reference.event_id}")
            seen.add(reference.event_id)
            event = by_id.get(reference.event_id)
            if event is None:
                raise ValueError(f"proposal {proposal.code} cites missing evidence {reference.event_id}")
            if not event.is_human or event.is_bot or event.is_agent_marked or not event.in_window:
                raise ValueError(f"proposal {proposal.code} cites ineligible evidence {reference.event_id}")
            if event.source_url is None:
                raise ValueError(f"proposal {proposal.code} cites evidence without a URL: {reference.event_id}")
            referenced.append(event)
        pull_requests = {event.pr_number for event in referenced}
        if len(pull_requests) < 3:
            raise ValueError(f"proposal {proposal.code} requires evidence from three distinct pull requests")
        recent = [event for event in referenced if _event_time(event) >= seven_day_start]
        result.append(
            ProposalEvidence(
                proposal=proposal,
                events_30_days=len(referenced),
                pull_requests_30_days=len(pull_requests),
                events_7_days=len(recent),
                pull_requests_7_days=len({event.pr_number for event in recent}),
                events=tuple(referenced),
            )
        )
    return tuple(result)


def load_report(
    corpus: Path,
    analysis_path: Path,
    predictions_path: Path,
) -> RefinementReport:
    """Validate and load every input used to render a refinement report."""
    manifest = validate_corpus(corpus)
    analysis = RefinementAnalysis.model_validate_json(analysis_path.read_text())
    if analysis.schema_version != REPORT_SCHEMA_VERSION:
        raise ValueError(f"unsupported refinement analysis schema: {analysis.schema_version}")
    for field, actual in (
        ("corpus_snapshot_id", manifest.snapshot_id),
        ("catalog_sha", manifest.catalog_sha),
        ("benchmark_sha", manifest.benchmark_sha),
    ):
        if getattr(analysis, field) != actual:
            raise ValueError(f"refinement analysis {field} does not match the corpus")

    rules = catalog_rules(corpus / "catalog")
    proposal_codes = [proposal.code for proposal in analysis.proposals]
    if len(proposal_codes) != len(set(proposal_codes)):
        raise ValueError("refinement analysis contains duplicate proposal codes")
    overlap = sorted(set(proposal_codes) & set(rules))
    if overlap:
        raise ValueError(f"refinement proposals duplicate current catalog rules: {overlap}")
    for proposal in analysis.proposals:
        unknown = sorted(set(proposal.precedence) - set(rules))
        if unknown:
            raise ValueError(f"proposal {proposal.code} has unknown precedence rules: {unknown}")
    for gap in analysis.existing_rule_gaps:
        unknown = sorted(set(gap.rules) - set(rules))
        if unknown:
            raise ValueError(f"existing-rule gap for PR #{gap.pr_number} has unknown rules: {unknown}")

    events = tuple(ReviewEventRecord.model_validate(row) for row in _jsonl(corpus / "review_events.jsonl"))
    labels = tuple(BenchmarkLabel.model_validate(row) for row in _jsonl(corpus / "benchmark" / "labels.jsonl"))
    prediction_bytes = predictions_path.read_bytes()
    predictions = _predictions(predictions_path)
    benchmark = score_benchmark(
        predictions,
        labels,
        catalog_rules=set(rules),
        prediction_sha=hashlib.sha256(prediction_bytes).hexdigest(),
    )
    invocations = _jsonl(corpus / "automation_invocations.jsonl")
    findings = _jsonl(corpus / "automation_findings.jsonl")
    end = parse_utc(manifest.window_end)
    return RefinementReport(
        manifest=manifest,
        analysis=analysis,
        proposals=_proposal_evidence(analysis.proposals, events, end=end),
        benchmark=benchmark,
        production_7_days=production_window(invocations, end=end, days=7),
        production_30_days=production_window(invocations, end=end, days=30),
        current_catalog=current_catalog_activity(
            invocations,
            findings,
            end=end,
            catalog_rules=set(rules),
        ),
    )


def _markdown_bullets(items: Iterable[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _inflected_count(noun: str, value: int) -> str:
    suffix = "" if value == 1 else "s"
    return f"{value} {noun}{suffix}"


def _proposal_section(evidence: ProposalEvidence) -> str:
    proposal = evidence.proposal
    reference_relevance = {reference.event_id: reference.relevance for reference in proposal.evidence}
    lines = [
        f"### `{proposal.code}` — {proposal.title}",
        "",
        f"Lane: `{proposal.lane}`.",
        "",
        proposal.condition,
        "",
        f"Allowed: {proposal.when_allowed}",
        "",
        (
            f"Verified support: {_inflected_count('event', evidence.events_30_days)} across "
            f"{_inflected_count('PR', evidence.pull_requests_30_days)} in 30 days; "
            f"{_inflected_count('event', evidence.events_7_days)} across "
            f"{_inflected_count('PR', evidence.pull_requests_7_days)} in 7 days."
        ),
        "",
        "Evidence:",
        "",
        _markdown_bullets(
            f"[PR #{event.pr_number}]({event.source_url}) — {reference_relevance[event.event_id]}"
            for event in evidence.events
        ),
    ]
    if proposal.precedence:
        lines.extend(["", f"Precedence: {', '.join(f'`{code}`' for code in proposal.precedence)}."])
    if proposal.counterexamples:
        lines.extend(["", "Counterexamples:", "", _markdown_bullets(proposal.counterexamples)])
    return "\n".join(lines)


def _production_section(report: RefinementReport) -> str:
    seven = report.production_7_days
    thirty = report.production_30_days
    return "\n".join(
        [
            "## Production activity",
            "",
            "| Window | Started runs | Successful runs | Meta-eligible runs | Findings |",
            "| --- | ---: | ---: | ---: | ---: |",
            (
                f"| 7 days | {seven.started_runs} | {seven.successful_runs} | "
                f"{seven.meta_eligible_runs} | {seven.findings} |"
            ),
            (
                f"| 30 days | {thirty.started_runs} | {thirty.successful_runs} | "
                f"{thirty.meta_eligible_runs} | {thirty.findings} |"
            ),
            "",
            (
                "These are successful local `pre-commit-review` runs and emitted finding counts. "
                "They do not measure production precision or recall."
            ),
        ]
    )


def _benchmark_section(score: BenchmarkScore) -> str:
    return "\n".join(
        [
            "## Blind benchmark",
            "",
            (
                f"The committed prediction set `{score.prediction_sha}` matched "
                f"{score.exact_matches}/{score.cases} cases. TP/FP/FN were "
                f"{score.true_positives}/{score.false_positives}/{score.false_negatives}; precision, recall, and "
                f"F1 were {score.precision:.3f}, {score.recall:.3f}, and {score.f1:.3f}. Hard-negative "
                f"specificity was {score.true_negatives}/{score.hard_negatives}."
            ),
            "",
            (
                "This catalog-derived benchmark is a synthetic regression check. It is not an estimate of "
                "production precision or recall, and it has one score for the frozen corpus instead of "
                "separate 7-day and 30-day scores."
            ),
        ]
    )


def _retirement_section(current: CurrentCatalogActivity) -> str:
    if current.sha is None:
        decision = "No catalog-bearing production runs were available, so no retirement decision is possible."
    elif current.observed_days < RETIREMENT_DAYS:
        decision = (
            f"No rule is ready for retirement. Catalog `{current.sha}` has {current.observed_days:.1f} days of "
            f"observed history. Its {len(current.zero_finding_rules)} zero-finding rules remain exposure gaps "
            f"until the catalog reaches {RETIREMENT_DAYS} days."
        )
    else:
        candidates = ", ".join(f"`{code}`" for code in current.zero_finding_rules) or "none"
        decision = (
            f"Catalog `{current.sha}` has {current.observed_days:.1f} days of history. "
            f"Zero-finding retirement candidates: {candidates}."
        )
    activity = (
        f"Current-catalog activity: {current.started_runs} started runs, {current.successful_runs} successful runs, "
        f"{current.distinct_heads} distinct heads, {current.changed_files} changed files, "
        f"{current.changed_lines} changed lines, and {current.findings} findings across "
        f"{current.runs_with_findings} successful runs."
    )
    return "\n\n".join(["## Retirement", decision, activity])


def render_markdown(report: RefinementReport) -> str:
    """Render the complete, evidence-linked Markdown report."""
    manifest = report.manifest
    proposals = (
        "\n\n".join(_proposal_section(evidence) for evidence in report.proposals)
        if report.proposals
        else "No new rules met the three-pull-request evidence threshold."
    )
    gaps = (
        _markdown_bullets(
            f"PR #{gap.pr_number}: {gap.human_events} human events map to "
            f"{', '.join(f'`{code}`' for code in gap.rules)}. {gap.finding}"
            for gap in report.analysis.existing_rule_gaps
        )
        if report.analysis.existing_rule_gaps
        else "No existing-rule application gaps were recorded."
    )
    sections = [
        "# Agentic-lint refinement report",
        (
            f"Snapshot `{manifest.snapshot_id}` covers {manifest.window_start} through {manifest.window_end}. "
            f"The collector scanned {manifest.candidate_pull_requests} pull requests and retained "
            f"{manifest.included_pull_requests} with {manifest.human_events_in_window} in-window human review events."
        ),
        "\n".join(
            [
                "## Corpus",
                "",
                (
                    f"All {len(manifest.files)} declared files passed hash, size, row-count, and corpus validation. "
                    f"Collection used {manifest.github_usage.graphql_points} GraphQL points and "
                    f"{manifest.github_usage.rest_requests} REST requests."
                ),
            ]
        ),
        f"## Proposed catalog changes\n\n{proposals}",
        f"## Existing-rule application gaps\n\n{gaps}",
        _production_section(report),
        _benchmark_section(report.benchmark),
        _retirement_section(report.current_catalog),
        f"## Limitations\n\n{_markdown_bullets((*manifest.limitations, *report.analysis.limitations))}",
    ]
    return "\n\n".join(sections) + "\n"


def render_slack(
    report: RefinementReport,
    *,
    report_url: str,
    catalog_pr_url: str | None,
) -> str:
    """Render the compact message delivered to the engineering Slack channel."""
    benchmark = report.benchmark
    current = report.current_catalog
    proposal_codes = ", ".join(f"`{item.proposal.code}`" for item in report.proposals) or "none"
    retirement = (
        f"no retirements; current catalog has {current.observed_days:.1f}/{RETIREMENT_DAYS} observed days"
        if current.observed_days < RETIREMENT_DAYS
        else f"{len(current.zero_finding_rules)} zero-finding retirement candidates"
    )
    links = f"Full report: {report_url}"
    if catalog_pr_url:
        links += f"\nCatalog PR: {catalog_pr_url}"
    return (
        f"Agentic-lint refinement found {len(report.proposals)} catalog proposals from "
        f"{report.manifest.included_pull_requests} PRs and {report.manifest.human_events_in_window} human review "
        f"events. Proposed: {proposal_codes}.\n"
        f"30-day production activity: {report.production_30_days.successful_runs}/"
        f"{report.production_30_days.started_runs} successful review runs and "
        f"{report.production_30_days.findings} findings; {retirement}.\n"
        f"Synthetic catalog regression: {benchmark.exact_matches}/{benchmark.cases} exact and "
        f"{benchmark.true_negatives}/{benchmark.hard_negatives} hard negatives. This is not production recall.\n"
        f"{links}"
    )


@click.command()
@click.option("--corpus", type=click.Path(path_type=Path, exists=True, file_okay=False), required=True)
@click.option("--analysis", "analysis_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--predictions", "predictions_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--report-out", type=click.Path(path_type=Path), required=True)
@click.option("--slack-out", type=click.Path(path_type=Path), required=True)
@click.option("--report-url", required=True)
@click.option("--catalog-pr-url")
def cli(
    corpus: Path,
    analysis_path: Path,
    predictions_path: Path,
    report_out: Path,
    slack_out: Path,
    report_url: str,
    catalog_pr_url: str | None,
) -> None:
    """Validate analysis outputs and render the full and Slack reports."""
    report = load_report(corpus, analysis_path, predictions_path)
    report_out.write_text(render_markdown(report))
    slack_out.write_text(render_slack(report, report_url=report_url, catalog_pr_url=catalog_pr_url) + "\n")
    click.echo(report_out)


if __name__ == "__main__":
    cli()
