# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export a frozen GitHub review corpus for agentic lint refinement."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import click
from pydantic import BaseModel, Field

from .github_review_corpus import (
    CorpusModel,
    GitHubUsage,
    PullRequestBundle,
    PullRequestRecord,
    ReviewEventRecord,
    collect_corpus,
)
from .review_tables import (
    DEFAULT_BOT_LOGINS,
    DEFAULT_DEPLOYMENT,
    DEFAULT_REPOSITORY,
    FINDINGS_NAMESPACE,
    HUMAN_COMMENTS_NAMESPACE,
    INVOCATIONS_NAMESPACE,
    open_tables_client,
    query_rows,
)

SCHEMA_VERSION = 1
DEFAULT_CATALOG_DIR = Path(__file__).parents[1] / "lint"
DEFAULT_BENCHMARK = DEFAULT_CATALOG_DIR / "eval" / "corpus.jsonl"
Lane = Literal["complexity", "interfaces", "robustness", "cruft", "prose", "meta"]
Provenance = Literal["catalog-example", "human-review", "synthetic-hard-negative"]
JsonRecord = BaseModel | dict[str, object]
RULE_HEADING = re.compile(r"^### `(?P<code>ml-[a-z0-9-]+)`", re.MULTILINE)


class CorpusFile(CorpusModel):
    path: str
    sha256: str
    rows: int | None
    bytes: int


class BenchmarkCase(CorpusModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]+$")
    lane: Lane
    description: str
    diff: str
    changed_lines: int = Field(default=0, ge=0)
    expected_rules: tuple[str, ...]
    provenance: Provenance
    source_url: str | None = None
    source_pr: int | None = None
    source_author: str | None = None


class BenchmarkPredictionCase(CorpusModel):
    alias: str = Field(pattern=r"^case-[0-9]{3}$")
    lane: Lane
    diff: str
    changed_lines: int = Field(ge=0)


class BenchmarkLabel(CorpusModel):
    alias: str = Field(pattern=r"^case-[0-9]{3}$")
    source_id: str
    description: str
    expected_rules: tuple[str, ...]
    provenance: Provenance
    source_url: str | None
    source_pr: int | None
    source_author: str | None


class BenchmarkSummary(CorpusModel):
    cases: int
    positive_cases: int
    hard_negatives: int
    covered_rules: int
    catalog_rules: int


class BenchmarkIdentity(CorpusModel):
    sha256: str
    summary: BenchmarkSummary


class BenchmarkSplit(CorpusModel):
    cases: tuple[BenchmarkPredictionCase, ...]
    labels: tuple[BenchmarkLabel, ...]
    identity: BenchmarkIdentity


class CorpusManifest(CorpusModel):
    schema_version: int
    snapshot_id: str
    repository: str
    window_start: str
    window_end: str
    collection_started_at: str
    collection_completed_at: str
    exporter_sha: str
    catalog_sha: str
    benchmark_sha: str
    benchmark: BenchmarkSummary
    complete: bool
    candidate_pull_requests: int
    included_pull_requests: int
    review_events: int
    human_events_in_window: int
    github_usage: GitHubUsage
    files: tuple[CorpusFile, ...]
    limitations: tuple[str, ...]


class TelemetryRows(CorpusModel):
    invocations: tuple[dict[str, object], ...]
    findings: tuple[dict[str, object], ...]
    annotations: tuple[dict[str, object], ...]


def _iso(value: dt.datetime) -> str:
    return value.astimezone(dt.UTC).isoformat().replace("+00:00", "Z")


def _write_jsonl(path: Path, records: Sequence[JsonRecord]) -> int:
    lines = []
    for record in records:
        payload = record.model_dump(mode="json") if isinstance(record, BaseModel) else record
        lines.append(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str))
    path.write_text("".join(f"{line}\n" for line in lines))
    return len(lines)


def _file_record(root: Path, path: Path, rows: int | None) -> CorpusFile:
    content = path.read_bytes()
    return CorpusFile(
        path=path.relative_to(root).as_posix(),
        sha256=hashlib.sha256(content).hexdigest(),
        rows=rows,
        bytes=len(content),
    )


def _snapshot_id(repository: str, start: str, end: str, files: Sequence[CorpusFile]) -> str:
    identity = {
        "schema_version": SCHEMA_VERSION,
        "repository": repository,
        "window_start": start,
        "window_end": end,
        "files": [item.model_dump(mode="json") for item in files],
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _catalog_sha(catalog_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(catalog_dir.glob("*.md")):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _catalog_rules(catalog_dir: Path) -> dict[str, str]:
    rules: dict[str, str] = {}
    for catalog_path in sorted(catalog_dir.glob("*.md")):
        if catalog_path.stem not in {"complexity", "interfaces", "robustness", "cruft", "prose", "meta"}:
            continue
        for match in RULE_HEADING.finditer(catalog_path.read_text()):
            code = match.group("code")
            if code in rules:
                raise ValueError(f"duplicate catalog rule: {code}")
            rules[code] = catalog_path.stem
    return rules


def _load_benchmark(path: Path) -> tuple[BenchmarkCase, ...]:
    cases: list[BenchmarkCase] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        try:
            case = BenchmarkCase.model_validate_json(line)
        except ValueError as error:
            raise ValueError(f"invalid benchmark case on line {line_number}: {error}") from error
        if case.id in seen_ids:
            raise ValueError(f"duplicate benchmark case id: {case.id}")
        seen_ids.add(case.id)
        cases.append(case)
    if not cases:
        raise ValueError("benchmark corpus is empty")
    return tuple(cases)


def _validate_benchmark_cases(cases: Sequence[BenchmarkCase], catalog_dir: Path) -> BenchmarkSummary:
    rules = _catalog_rules(catalog_dir)
    for case in cases:
        if len(case.expected_rules) != len(set(case.expected_rules)):
            raise ValueError(f"benchmark case {case.id} contains duplicate expected rules")
        unknown = sorted(set(case.expected_rules) - set(rules))
        if unknown:
            raise ValueError(f"benchmark case {case.id} references unknown rules: {unknown}")
        wrong_lane = sorted(code for code in case.expected_rules if rules[code] != case.lane)
        if wrong_lane:
            raise ValueError(f"benchmark case {case.id} references rules from another lane: {wrong_lane}")
        if case.lane == "meta" and case.changed_lines <= 100:
            raise ValueError(f"meta benchmark case {case.id} must represent a change larger than 100 lines")
        if case.provenance == "human-review" and not all((case.source_url, case.source_pr, case.source_author)):
            raise ValueError(f"human-review benchmark case {case.id} requires source provenance")

    covered = {code for case in cases for code in case.expected_rules}
    missing = sorted(set(rules) - covered)
    if missing:
        raise ValueError(f"benchmark corpus has no positive case for rules: {missing}")
    for lane in sorted(set(rules.values())):
        hard_negatives = sum(not case.expected_rules for case in cases if case.lane == lane)
        if hard_negatives < 3:
            raise ValueError(f"benchmark corpus requires three hard negatives for lane {lane}")
    return BenchmarkSummary(
        cases=len(cases),
        positive_cases=sum(bool(case.expected_rules) for case in cases),
        hard_negatives=sum(not case.expected_rules for case in cases),
        covered_rules=len(covered),
        catalog_rules=len(rules),
    )


def validate_benchmark(path: Path, catalog_dir: Path) -> BenchmarkSummary:
    """Validate fixed-case labels against the exact catalog copied into the corpus."""
    return _validate_benchmark_cases(_load_benchmark(path), catalog_dir)


def _prediction_payload(case: BenchmarkCase) -> dict[str, object]:
    return {
        "lane": case.lane,
        "diff": case.diff,
        "changed_lines": case.changed_lines,
    }


def _normalized_sha(cases: Sequence[BenchmarkPredictionCase], labels: Sequence[BenchmarkLabel]) -> str:
    identity = {
        "format": "blind-split-v1",
        "cases": [case.model_dump(mode="json") for case in cases],
        "labels": [label.model_dump(mode="json") for label in labels],
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _benchmark_split(source_cases: Sequence[BenchmarkCase], catalog_dir: Path) -> BenchmarkSplit:
    summary = _validate_benchmark_cases(source_cases, catalog_dir)
    keyed_cases = [
        (
            hashlib.sha256(
                json.dumps(_prediction_payload(case), sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            case,
        )
        for case in source_cases
    ]
    hashes = [content_hash for content_hash, _ in keyed_cases]
    if len(hashes) != len(set(hashes)):
        raise ValueError("benchmark corpus contains duplicate model-visible cases")
    keyed_cases.sort(key=lambda item: item[0])

    prediction_cases: list[BenchmarkPredictionCase] = []
    labels: list[BenchmarkLabel] = []
    for index, (_, source) in enumerate(keyed_cases, start=1):
        alias = f"case-{index:03d}"
        prediction_cases.append(BenchmarkPredictionCase(alias=alias, **_prediction_payload(source)))
        labels.append(
            BenchmarkLabel(
                alias=alias,
                source_id=source.id,
                description=source.description,
                expected_rules=tuple(sorted(source.expected_rules)),
                provenance=source.provenance,
                source_url=source.source_url,
                source_pr=source.source_pr,
                source_author=source.source_author,
            )
        )
    identity = BenchmarkIdentity(
        sha256=_normalized_sha(prediction_cases, labels),
        summary=summary,
    )
    return BenchmarkSplit(cases=tuple(prediction_cases), labels=tuple(labels), identity=identity)


def benchmark_split(path: Path, catalog_dir: Path) -> BenchmarkSplit:
    """Build the deterministic model-visible and hidden-label benchmark split."""
    return _benchmark_split(_load_benchmark(path), catalog_dir)


def validate_exported_benchmark(cases_path: Path, labels_path: Path, catalog_dir: Path) -> BenchmarkIdentity:
    """Validate an exported blind benchmark split and its normalized identity."""
    try:
        cases = tuple(BenchmarkPredictionCase.model_validate_json(line) for line in cases_path.read_text().splitlines())
        labels = tuple(BenchmarkLabel.model_validate_json(line) for line in labels_path.read_text().splitlines())
    except ValueError as error:
        raise ValueError(f"invalid exported benchmark split: {error}") from error
    if not cases or not labels:
        raise ValueError("exported benchmark split is empty")
    expected_aliases = tuple(f"case-{index:03d}" for index in range(1, len(cases) + 1))
    if tuple(case.alias for case in cases) != expected_aliases:
        raise ValueError("exported benchmark cases must use consecutive aliases in file order")
    if tuple(label.alias for label in labels) != expected_aliases:
        raise ValueError("exported benchmark labels must match case aliases in file order")

    source_cases = tuple(
        BenchmarkCase(
            id=label.source_id,
            lane=case.lane,
            description=label.description,
            diff=case.diff,
            changed_lines=case.changed_lines,
            expected_rules=label.expected_rules,
            provenance=label.provenance,
            source_url=label.source_url,
            source_pr=label.source_pr,
            source_author=label.source_author,
        )
        for case, label in zip(cases, labels, strict=True)
    )
    expected = _benchmark_split(source_cases, catalog_dir)
    if expected.cases != cases or expected.labels != labels:
        raise ValueError("exported benchmark aliases do not match content-hash ordering")
    return expected.identity


def _exporter_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def _sorted_rows(rows: Sequence[dict[str, object]]) -> tuple[dict[str, object], ...]:
    return tuple(sorted(rows, key=lambda row: json.dumps(row, sort_keys=True, default=str)))


def load_telemetry(
    deployment: str,
    start: dt.datetime,
    end: dt.datetime,
    events: Sequence[ReviewEventRecord],
) -> TelemetryRows:
    """Read bounded automation telemetry and matching prior annotations."""
    start_sql = start.astimezone(dt.UTC).strftime("%Y-%m-%d %H:%M:%S")
    end_sql = end.astimezone(dt.UTC).strftime("%Y-%m-%d %H:%M:%S")
    annotation_start_sql = (start - dt.timedelta(days=30)).astimezone(dt.UTC).strftime("%Y-%m-%d %H:%M:%S")
    event_kinds = {
        "inline_comment": "inline",
        "review": "review",
        "issue_comment": "issue",
    }
    event_keys = {(event_kinds[event.kind], event.database_id) for event in events}
    with open_tables_client(deployment) as client:
        invocations = query_rows(
            client,
            f'SELECT * FROM "{INVOCATIONS_NAMESPACE}" '
            f"WHERE ts >= TIMESTAMP '{start_sql}' AND ts < TIMESTAMP '{end_sql}'",
            INVOCATIONS_NAMESPACE,
        )
        findings = query_rows(
            client,
            f'SELECT * FROM "{FINDINGS_NAMESPACE}" '
            f"WHERE ts >= TIMESTAMP '{start_sql}' AND ts < TIMESTAMP '{end_sql}'",
            FINDINGS_NAMESPACE,
        )
        annotations = query_rows(
            client,
            f"""
            SELECT * FROM (
                SELECT *, row_number() OVER (
                    PARTITION BY pr_number, comment_type, comment_id ORDER BY seq DESC
                ) AS recency
                FROM "{HUMAN_COMMENTS_NAMESPACE}"
                WHERE ts >= TIMESTAMP '{annotation_start_sql}' AND ts < TIMESTAMP '{end_sql}'
            ) WHERE recency = 1
            """,
            HUMAN_COMMENTS_NAMESPACE,
        )
    matching_annotations = [
        row for row in annotations if (str(row.get("comment_type")), int(row.get("comment_id") or 0)) in event_keys
    ]
    return TelemetryRows(
        invocations=_sorted_rows(invocations),
        findings=_sorted_rows(findings),
        annotations=_sorted_rows(matching_annotations),
    )


def write_corpus(
    output: Path,
    bundles: list[PullRequestBundle],
    *,
    repository: str,
    start: dt.datetime,
    end: dt.datetime,
    collection_started: dt.datetime,
    candidate_count: int,
    complete: bool,
    telemetry: TelemetryRows,
    catalog_dir: Path = DEFAULT_CATALOG_DIR,
    benchmark: Path = DEFAULT_BENCHMARK,
    github_usage: GitHubUsage | None = None,
) -> CorpusManifest:
    """Write a deterministic corpus directory and publish it atomically."""
    if output.exists():
        raise FileExistsError(f"refusing to replace existing corpus: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    split = benchmark_split(benchmark, catalog_dir)
    pull_requests = [bundle.pull_request for bundle in bundles]
    events = [event for bundle in bundles for event in bundle.events]
    threads = [thread for bundle in bundles for thread in bundle.threads]
    changed_files = [item for bundle in bundles for item in bundle.files]
    commits = [item for bundle in bundles for item in bundle.commits]
    completed = dt.datetime.now(dt.UTC)

    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        root = Path(temporary) / "corpus"
        root.mkdir()
        (root / "diffs").mkdir()
        (root / "catalog").mkdir()
        (root / "benchmark").mkdir()
        files: list[CorpusFile] = []
        datasets: list[tuple[str, Sequence[JsonRecord]]] = [
            ("pull_requests.jsonl", pull_requests),
            ("review_events.jsonl", events),
            ("review_threads.jsonl", threads),
            ("changed_files.jsonl", changed_files),
            ("commits.jsonl", commits),
            ("automation_invocations.jsonl", telemetry.invocations),
            ("automation_findings.jsonl", telemetry.findings),
            ("derived_annotations.jsonl", telemetry.annotations),
        ]
        for name, records in datasets:
            path = root / name
            rows = _write_jsonl(path, records)
            files.append(_file_record(root, path, rows))
        for bundle in bundles:
            diff_path = bundle.pull_request.diff_path
            if bundle.diff is None:
                if diff_path is not None:
                    raise ValueError(f"PR #{bundle.pull_request.number} has a diff path without diff content")
                continue
            if diff_path is None:
                raise ValueError(f"PR #{bundle.pull_request.number} has diff content without a diff path")
            path = root / diff_path
            path.write_text(bundle.diff)
            files.append(_file_record(root, path, None))
        for source in sorted(catalog_dir.glob("*.md")):
            destination = root / "catalog" / source.name
            shutil.copy2(source, destination)
            files.append(_file_record(root, destination, None))
        benchmark_cases = root / "benchmark" / "cases.jsonl"
        benchmark_labels = root / "benchmark" / "labels.jsonl"
        files.append(_file_record(root, benchmark_cases, _write_jsonl(benchmark_cases, split.cases)))
        files.append(_file_record(root, benchmark_labels, _write_jsonl(benchmark_labels, split.labels)))

        files.sort(key=lambda item: item.path)
        snapshot_id = _snapshot_id(repository, _iso(start), _iso(end), files)
        limitations = (
            "GitHub does not expose deleted review events or prior versions of edited bodies.",
            "GraphQL changed-file rows omit REST-only SHA, URL, previous-filename, and per-file patch fields.",
            "Each available GitHub-served pull-request diff is the frozen patch context; binary content may be absent.",
            "GitHub returns HTTP 406 too_large above its 300-file diff render limit; those records have "
            "diff_path null and GraphQL file metadata is their only changed-file context.",
            "Derived comment annotations are a bounded 60-day lookup and are not an ingestion filter.",
        )
        manifest = CorpusManifest(
            schema_version=SCHEMA_VERSION,
            snapshot_id=snapshot_id,
            repository=repository,
            window_start=_iso(start),
            window_end=_iso(end),
            collection_started_at=_iso(collection_started),
            collection_completed_at=_iso(completed),
            exporter_sha=_exporter_sha(),
            catalog_sha=_catalog_sha(catalog_dir),
            benchmark_sha=split.identity.sha256,
            benchmark=split.identity.summary,
            complete=complete,
            candidate_pull_requests=candidate_count,
            included_pull_requests=len(pull_requests),
            review_events=len(events),
            human_events_in_window=sum(item.is_human and item.in_window for item in events),
            github_usage=github_usage
            or GitHubUsage(
                graphql_requests=0,
                graphql_points=0,
                rest_requests=0,
                projected_rest_requests=0,
            ),
            files=tuple(files),
            limitations=limitations,
        )
        (root / "manifest.json").write_text(json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n")
        os.replace(root, output)
    return manifest


def validate_corpus(path: Path, *, require_complete: bool = True) -> CorpusManifest:
    """Validate corpus identity, completeness, and every declared file hash."""
    root = path.resolve()
    manifest = CorpusManifest.model_validate_json((root / "manifest.json").read_text())
    if manifest.schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported corpus schema version: {manifest.schema_version}")
    if require_complete and not manifest.complete:
        raise ValueError("corpus manifest is incomplete")
    declared_paths = [item.path for item in manifest.files]
    if declared_paths != sorted(set(declared_paths)):
        raise ValueError("corpus manifest file paths must be unique and sorted")
    for item in manifest.files:
        candidate = (root / item.path).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError(f"corpus file escapes its root: {item.path}")
        if (root / item.path).is_symlink():
            raise ValueError(f"corpus file must not be a symlink: {item.path}")
        if not candidate.is_file():
            raise ValueError(f"corpus file is missing: {item.path}")
        content = candidate.read_bytes()
        if len(content) != item.bytes or hashlib.sha256(content).hexdigest() != item.sha256:
            raise ValueError(f"corpus file hash mismatch: {item.path}")
        if item.rows is not None and len(candidate.read_text().splitlines()) != item.rows:
            raise ValueError(f"corpus row count mismatch: {item.path}")
    actual_paths = sorted(
        candidate.relative_to(root).as_posix()
        for candidate in root.rglob("*")
        if candidate.is_file() and candidate.name != "manifest.json"
    )
    if actual_paths != declared_paths:
        raise ValueError("corpus contains undeclared files")
    expected_snapshot = _snapshot_id(
        manifest.repository,
        manifest.window_start,
        manifest.window_end,
        manifest.files,
    )
    if manifest.snapshot_id != expected_snapshot:
        raise ValueError("corpus snapshot identity mismatch")
    if manifest.catalog_sha != _catalog_sha(root / "catalog"):
        raise ValueError("corpus catalog identity mismatch")
    benchmark_identity = validate_exported_benchmark(
        root / "benchmark" / "cases.jsonl",
        root / "benchmark" / "labels.jsonl",
        root / "catalog",
    )
    if manifest.benchmark_sha != benchmark_identity.sha256:
        raise ValueError("corpus benchmark identity mismatch")
    if manifest.benchmark != benchmark_identity.summary:
        raise ValueError("corpus benchmark summary mismatch")
    pull_request_rows = [
        PullRequestRecord.model_validate_json(line) for line in (root / "pull_requests.jsonl").read_text().splitlines()
    ]
    event_rows = [
        ReviewEventRecord.model_validate_json(line) for line in (root / "review_events.jsonl").read_text().splitlines()
    ]
    if manifest.included_pull_requests != len(pull_request_rows):
        raise ValueError("corpus pull-request count mismatch")
    if manifest.candidate_pull_requests < manifest.included_pull_requests:
        raise ValueError("corpus candidate pull-request count is smaller than its included count")
    pull_request_numbers = [pull.number for pull in pull_request_rows]
    if len(pull_request_numbers) != len(set(pull_request_numbers)):
        raise ValueError("corpus contains duplicate pull requests")
    expected_diffs = sorted(pull.diff_path for pull in pull_request_rows if pull.diff_path is not None)
    declared_diffs = sorted(item.path for item in manifest.files if item.path.startswith("diffs/"))
    if declared_diffs != expected_diffs:
        raise ValueError("corpus diff artifacts do not match pull-request records")
    if manifest.review_events != len(event_rows):
        raise ValueError("corpus review-event count mismatch")
    event_ids = [event.event_id for event in event_rows]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("corpus contains duplicate review events")
    unknown_event_prs = sorted({event.pr_number for event in event_rows} - set(pull_request_numbers))
    if unknown_event_prs:
        raise ValueError(f"corpus review events reference unknown pull requests: {unknown_event_prs}")
    human_events = sum(event.is_human and event.in_window for event in event_rows)
    if manifest.human_events_in_window != human_events:
        raise ValueError("corpus human-event count mismatch")
    return manifest


@click.group()
def cli() -> None:
    """Build and validate frozen review corpora."""


@cli.command("export")
@click.option("--repo", default=DEFAULT_REPOSITORY, show_default=True)
@click.option("--days", type=click.IntRange(min=1), required=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option("--deployment", default=DEFAULT_DEPLOYMENT, show_default=True)
@click.option("--skip-telemetry", is_flag=True, help="Probe only; omits Finelog telemetry")
@click.option("--limit", type=click.IntRange(min=1), help="Probe only; marks the corpus incomplete")
def export_command(
    repo: str,
    days: int,
    output: Path,
    deployment: str,
    skip_telemetry: bool,
    limit: int | None,
) -> None:
    """Export one frozen review-activity window."""
    collection_started = dt.datetime.now(dt.UTC)
    end = collection_started
    start = end - dt.timedelta(days=days)
    github_result = collect_corpus(
        repo,
        start,
        end,
        bot_logins=set(DEFAULT_BOT_LOGINS),
        limit=limit,
    )
    bundles = list(github_result.bundles)
    candidate_count = github_result.candidate_pull_requests
    events = [event for bundle in bundles for event in bundle.events]
    telemetry = (
        TelemetryRows(invocations=(), findings=(), annotations=())
        if skip_telemetry
        else load_telemetry(deployment, start, end, events)
    )
    manifest = write_corpus(
        output,
        bundles,
        repository=repo,
        start=start,
        end=end,
        collection_started=collection_started,
        candidate_count=candidate_count,
        complete=limit is None and not skip_telemetry,
        telemetry=telemetry,
        github_usage=github_result.usage,
    )
    click.echo(json.dumps(manifest.model_dump(mode="json"), indent=2))


@cli.command("validate")
@click.argument("path", type=click.Path(path_type=Path, exists=True, file_okay=False))
def validate_command(path: Path) -> None:
    """Validate a corpus before handing it to an analysis agent."""
    manifest = validate_corpus(path)
    click.echo(manifest.snapshot_id)


if __name__ == "__main__":
    cli()
