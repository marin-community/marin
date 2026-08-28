# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime as dt
import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest

from infra.codehealth import review_corpus

REPOSITORY = "owner/repo"
PR_NUMBER = 7
START = dt.datetime(2026, 8, 21, tzinfo=dt.UTC)
END = dt.datetime(2026, 8, 28, tzinfo=dt.UTC)
ROOT_BODY = "Old root: preserve **Markdown**, Unicode λ, and trailing spaces  \n\n```py\nprint('root')\n```"
REPLY_BODY = "New reply\n\nDo not trim this body.\n"
PR_BODY = "Full PR body\n\n- first\n- second\n\n<details>exact</details>\n"
DIFF = (
    "diff --git a/example.py b/example.py\n"
    "index 111..222 100644\n"
    "--- a/example.py\n"
    "+++ b/example.py\n"
    "@@ -1 +1 @@\n"
    "-old\n"
    "+new  \n"
)


@dataclass(frozen=True)
class CorpusExport:
    output: Path
    catalog: Path
    benchmark: Path
    manifest: review_corpus.CorpusManifest


def _bundle() -> review_corpus.PullRequestBundle:
    pull_request = review_corpus.PullRequestRecord(
        repository=REPOSITORY,
        number=PR_NUMBER,
        node_id="PR_node",
        url=f"https://github.com/{REPOSITORY}/pull/{PR_NUMBER}",
        title="Preserve review evidence",
        body=PR_BODY,
        state="open",
        draft=False,
        author="author",
        author_type="User",
        author_association="MEMBER",
        created_at="2026-08-01T00:00:00Z",
        updated_at="2026-08-27T12:00:00Z",
        closed_at=None,
        merged_at=None,
        base_ref="main",
        base_sha="base-a",
        head_ref="feature",
        head_sha="head-a",
        additions=1,
        deletions=1,
        changed_files=0,
        commits=0,
        review_comments=2,
        issue_comments=0,
        commit_shas=(),
        diff_path=f"diffs/{PR_NUMBER}.diff",
    )

    def event(comment_id: int, body: str, *, in_window: bool) -> review_corpus.ReviewEventRecord:
        return review_corpus.ReviewEventRecord(
            event_id=f"inline_comment:{comment_id}",
            kind="inline_comment",
            database_id=comment_id,
            node_id=f"comment-{comment_id}",
            repository=REPOSITORY,
            pr_number=PR_NUMBER,
            pr_author="author",
            author="reviewer",
            author_type="User",
            author_association="MEMBER",
            body=body,
            state=None,
            created_at="2026-08-27T10:00:00Z" if in_window else "2026-08-01T10:00:00Z",
            updated_at="2026-08-27T10:00:00Z" if in_window else "2026-08-01T10:00:00Z",
            submitted_at=None,
            source_url=f"https://github.com/{REPOSITORY}/pull/{PR_NUMBER}#discussion_r{comment_id}",
            review_id=55,
            thread_id="thread-1",
            parent_comment_id=101 if comment_id == 102 else None,
            thread_is_resolved=False,
            thread_is_outdated=True,
            thread_resolved_by=None,
            path="example.py",
            side="RIGHT",
            line=1,
            original_line=1,
            start_side=None,
            start_line=None,
            original_start_line=None,
            commit_id="head-a",
            original_commit_id="head-a",
            diff_hunk="@@ -1 +1 @@\n-old\n+new  ",
            is_bot=False,
            is_agent_marked=False,
            is_human=True,
            in_window=in_window,
        )

    return review_corpus.PullRequestBundle(
        pull_request=pull_request,
        events=(event(101, ROOT_BODY, in_window=False), event(102, REPLY_BODY, in_window=True)),
        threads=(),
        files=(),
        commits=(),
        diff=DIFF,
    )


def _benchmark_files(root: Path) -> tuple[Path, Path]:
    catalog = root / "source-catalog"
    catalog.mkdir()
    (catalog / "complexity.md").write_text("# Complexity\n\n### `ml-example` — Example\n")
    benchmark = root / "benchmark.jsonl"
    cases = [
        {
            "id": "example-rule",
            "lane": "complexity",
            "description": "one positive",
            "diff": "+example()",
            "expected_rules": ["ml-example"],
            "provenance": "catalog-example",
        },
        *(
            {
                "id": f"near-miss-{index}",
                "lane": "complexity",
                "description": "accepted near miss",
                "diff": f"+accepted_{index}()",
                "expected_rules": [],
                "provenance": "synthetic-hard-negative",
            }
            for index in range(3)
        ),
    ]
    benchmark.write_text("".join(f"{json.dumps(case)}\n" for case in cases))
    return catalog, benchmark


@pytest.fixture
def corpus_export(tmp_path: Path) -> CorpusExport:
    catalog, benchmark = _benchmark_files(tmp_path)
    output = tmp_path / "corpus"
    manifest = review_corpus.write_corpus(
        output,
        [_bundle()],
        repository=REPOSITORY,
        start=START,
        end=END,
        collection_started=END,
        candidate_count=1,
        complete=True,
        telemetry=review_corpus.TelemetryRows(invocations=(), findings=(), annotations=()),
        catalog_dir=catalog,
        benchmark=benchmark,
    )
    return CorpusExport(output=output, catalog=catalog, benchmark=benchmark, manifest=manifest)


def test_load_telemetry_bounds_automation_and_keeps_only_matching_annotations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle()
    queries: list[tuple[str, str]] = []

    @contextmanager
    def fake_client(_deployment: str):
        yield object()

    def fake_query(_client: object, sql: str, namespace: str) -> list[dict]:
        queries.append((namespace, sql))
        if namespace == review_corpus.INVOCATIONS_NAMESPACE:
            return [{"invocation_id": "run-1"}]
        if namespace == review_corpus.FINDINGS_NAMESPACE:
            return [{"invocation_id": "run-1", "code": "ml-example"}]
        return [
            {"comment_type": "inline", "comment_id": 102, "reason": "matching"},
            {"comment_type": "issue", "comment_id": 999, "reason": "unrelated"},
        ]

    monkeypatch.setattr(review_corpus, "open_tables_client", fake_client)
    monkeypatch.setattr(review_corpus, "query_rows", fake_query)

    telemetry = review_corpus.load_telemetry("test", START, END, bundle.events)

    assert telemetry.invocations == ({"invocation_id": "run-1"},)
    assert telemetry.findings == ({"invocation_id": "run-1", "code": "ml-example"},)
    assert telemetry.annotations == ({"comment_type": "inline", "comment_id": 102, "reason": "matching"},)
    for namespace, sql in queries[:2]:
        assert namespace in {review_corpus.INVOCATIONS_NAMESPACE, review_corpus.FINDINGS_NAMESPACE}
        assert "2026-08-21 00:00:00" in sql
        assert "2026-08-28 00:00:00" in sql


def test_checked_in_benchmark_covers_catalog_with_lane_negatives() -> None:
    summary = review_corpus.validate_benchmark(
        review_corpus.DEFAULT_BENCHMARK,
        review_corpus.DEFAULT_CATALOG_DIR,
    )

    assert summary == review_corpus.BenchmarkSummary(
        cases=81,
        positive_cases=63,
        hard_negatives=18,
        covered_rules=63,
        catalog_rules=63,
    )


def test_write_corpus_has_deterministic_identity_and_order(
    tmp_path: Path,
    corpus_export: CorpusExport,
) -> None:
    reordered_benchmark = tmp_path / "benchmark-reordered.jsonl"
    reordered_benchmark.write_text("\n".join(reversed(corpus_export.benchmark.read_text().splitlines())) + "\n")
    reordered_output = tmp_path / "reordered-corpus"
    reordered_manifest = review_corpus.write_corpus(
        reordered_output,
        [_bundle()],
        repository=REPOSITORY,
        start=START,
        end=END,
        collection_started=END,
        candidate_count=1,
        complete=True,
        telemetry=review_corpus.TelemetryRows(invocations=(), findings=(), annotations=()),
        catalog_dir=corpus_export.catalog,
        benchmark=reordered_benchmark,
    )

    assert corpus_export.manifest.snapshot_id == reordered_manifest.snapshot_id
    assert corpus_export.manifest.benchmark_sha == reordered_manifest.benchmark_sha
    assert corpus_export.manifest.files == reordered_manifest.files
    assert tuple(item.path for item in corpus_export.manifest.files) == tuple(
        sorted(item.path for item in corpus_export.manifest.files)
    )
    for output, manifest in (
        (corpus_export.output, corpus_export.manifest),
        (reordered_output, reordered_manifest),
    ):
        validated = review_corpus.validate_corpus(output)
        assert validated.snapshot_id == manifest.snapshot_id
        for item in validated.files:
            assert hashlib.sha256((output / item.path).read_bytes()).hexdigest() == item.sha256


def test_write_corpus_preserves_evidence_and_blind_benchmark_split(corpus_export: CorpusExport) -> None:
    pull_row = json.loads((corpus_export.output / "pull_requests.jsonl").read_text())
    event_rows = [json.loads(line) for line in (corpus_export.output / "review_events.jsonl").read_text().splitlines()]
    assert pull_row["body"] == PR_BODY
    assert [row["body"] for row in event_rows] == [ROOT_BODY, REPLY_BODY]
    assert (corpus_export.output / "diffs" / f"{PR_NUMBER}.diff").read_text() == DIFF
    assert not (corpus_export.output / "benchmark" / "corpus.jsonl").exists()
    cases_path = corpus_export.output / "benchmark" / "cases.jsonl"
    labels_path = corpus_export.output / "benchmark" / "labels.jsonl"
    case_rows = [json.loads(line) for line in cases_path.read_text().splitlines()]
    label_rows = [json.loads(line) for line in labels_path.read_text().splitlines()]
    assert [row["alias"] for row in case_rows] == [f"case-{index:03d}" for index in range(1, 5)]
    assert all(set(row) == {"alias", "lane", "diff", "changed_lines"} for row in case_rows)
    assert [row["alias"] for row in label_rows] == [row["alias"] for row in case_rows]
    assert {row["source_id"] for row in label_rows} == {
        "example-rule",
        "near-miss-0",
        "near-miss-1",
        "near-miss-2",
    }
    identity = review_corpus.validate_exported_benchmark(
        cases_path,
        labels_path,
        corpus_export.output / "catalog",
    )
    assert identity.sha256 == corpus_export.manifest.benchmark_sha
    assert identity.summary == corpus_export.manifest.benchmark


def test_validate_corpus_rejects_tampered_evidence(corpus_export: CorpusExport) -> None:
    with (corpus_export.output / "review_events.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="corpus file hash mismatch"):
        review_corpus.validate_corpus(corpus_export.output)


def test_validate_exported_benchmark_rejects_missing_catalog_coverage(tmp_path: Path) -> None:
    catalog, benchmark = _benchmark_files(tmp_path)
    split = review_corpus.benchmark_split(benchmark, catalog)
    cases_path = tmp_path / "cases.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    cases_path.write_text("".join(f"{case.model_dump_json()}\n" for case in split.cases))
    label_rows = [label.model_dump(mode="json") for label in split.labels]
    positive = next(row for row in label_rows if row["expected_rules"])
    positive["expected_rules"] = []
    labels_path.write_text("".join(f"{json.dumps(row)}\n" for row in label_rows))

    with pytest.raises(ValueError, match="no positive case"):
        review_corpus.validate_exported_benchmark(cases_path, labels_path, catalog)


def test_validate_corpus_rejects_incomplete_snapshot_by_default(
    tmp_path: Path,
) -> None:
    bundle = _bundle()
    catalog, benchmark = _benchmark_files(tmp_path)
    output = tmp_path / "incomplete"
    review_corpus.write_corpus(
        output,
        [bundle],
        repository=REPOSITORY,
        start=START,
        end=END,
        collection_started=END,
        candidate_count=2,
        complete=False,
        telemetry=review_corpus.TelemetryRows(invocations=(), findings=(), annotations=()),
        catalog_dir=catalog,
        benchmark=benchmark,
    )

    with pytest.raises(ValueError, match="corpus manifest is incomplete"):
        review_corpus.validate_corpus(output)
    assert review_corpus.validate_corpus(output, require_complete=False).complete is False


def test_write_corpus_omits_unavailable_diff_artifact(tmp_path: Path) -> None:
    bundle = _bundle()
    bundle = bundle.model_copy(
        update={
            "pull_request": bundle.pull_request.model_copy(update={"diff_path": None}),
            "diff": None,
        }
    )
    catalog, benchmark = _benchmark_files(tmp_path)
    output = tmp_path / "corpus"

    manifest = review_corpus.write_corpus(
        output,
        [bundle],
        repository=REPOSITORY,
        start=START,
        end=END,
        collection_started=END,
        candidate_count=1,
        complete=True,
        telemetry=review_corpus.TelemetryRows(invocations=(), findings=(), annotations=()),
        catalog_dir=catalog,
        benchmark=benchmark,
    )

    pull_row = json.loads((output / "pull_requests.jsonl").read_text())
    assert pull_row["diff_path"] is None
    assert not (output / "diffs" / f"{PR_NUMBER}.diff").exists()
    assert all(item.path != f"diffs/{PR_NUMBER}.diff" for item in manifest.files)
    assert review_corpus.validate_corpus(output).snapshot_id == manifest.snapshot_id
