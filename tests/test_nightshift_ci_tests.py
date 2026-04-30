# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime as dt
import importlib.util
from pathlib import Path
import tempfile


def load_module():
    script_path = Path(__file__).resolve().parents[1] / "infra" / "scripts" / "nightshift_ci_tests.py"
    spec = importlib.util.spec_from_file_location("nightshift_ci_tests", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nightshift_ci_tests = load_module()


def test_parse_duration_line_strips_ansi():
    line = "\x1b[32m12.34s call     tests/example_test.py::test_case\x1b[0m"
    assert nightshift_ci_tests.parse_duration_line(line) == ("tests/example_test.py::test_case", 12.34)


def test_parse_failure_line_reads_pytest_failure():
    line = "FAILED lib/iris/tests/test_service.py::test_retries - AssertionError: boom"
    assert nightshift_ci_tests.parse_failure_line(line) == "lib/iris/tests/test_service.py::test_retries"


def test_collect_cooldowns_uses_open_issue_and_closed_cooldown():
    items = [
        {
            "state": "open",
            "body": "<!-- nightshift-ci-test: tests/a.py::test_one -->",
            "html_url": "https://example.com/open",
            "title": "open issue",
            "closed_at": None,
        },
        {
            "state": "closed",
            "body": (
                "<!-- nightshift-ci-test: tests/b.py::test_two -->\n" "<!-- nightshift-ci-cooldown-until: 2026-05-15 -->"
            ),
            "html_url": "https://example.com/closed",
            "title": "closed issue",
            "closed_at": "2026-04-20T00:00:00Z",
        },
    ]
    cooldowns = nightshift_ci_tests.collect_cooldowns(items)
    assert cooldowns["tests/a.py::test_one"][0] == dt.date.max
    assert cooldowns["tests/b.py::test_two"][0] == dt.date(2026, 5, 15)


def test_filter_recent_candidates_skips_cooldown():
    today = dt.date(2026, 4, 29)
    candidates = [
        {"test": "tests/a.py::test_one"},
        {"test": "tests/b.py::test_two"},
    ]
    cooldowns = {
        "tests/a.py::test_one": (dt.date(2026, 5, 1), "https://example.com/a", "artifact a"),
    }
    fresh, skipped = nightshift_ci_tests.filter_recent_candidates(candidates, cooldowns, today)
    assert fresh == [{"test": "tests/b.py::test_two"}]
    assert skipped == [
        {
            "test": "tests/a.py::test_one",
            "cooldown_until": "2026-05-01",
            "artifact_url": "https://example.com/a",
            "artifact_title": "artifact a",
        }
    ]


def test_canonicalize_test_name_normalizes_workspace_prefix():
    assert (
        nightshift_ci_tests.canonicalize_test_name("lib/zephyr/tests/test_dataset.py::test_reduce_sum")
        == "tests/test_dataset.py::test_reduce_sum"
    )


def test_select_candidates_filters_before_truncation_and_diversifies():
    ranked = [
        {"test": "tests/a.py::test_1"},
        {"test": "tests/a.py::test_2"},
        {"test": "tests/a.py::test_3"},
        {"test": "tests/b.py::test_1"},
        {"test": "tests/c.py::test_1"},
    ]
    cooldowns = {
        "tests/a.py::test_1": (dt.date(2026, 5, 10), "https://example.com/a1", "artifact a1"),
        "tests/a.py::test_2": (dt.date(2026, 5, 10), "https://example.com/a2", "artifact a2"),
    }
    fresh, skipped = nightshift_ci_tests.select_candidates(ranked, cooldowns, dt.date(2026, 4, 29))
    assert [candidate["test"] for candidate in fresh] == [
        "tests/a.py::test_3",
        "tests/b.py::test_1",
        "tests/c.py::test_1",
    ]
    assert [candidate["test"] for candidate in skipped] == [
        "tests/a.py::test_1",
        "tests/a.py::test_2",
    ]


def test_collect_evidence_dedupes_same_test_within_one_run():
    with tempfile.TemporaryDirectory() as temp_dir:
        log_dir = Path(temp_dir)
        (log_dir / "0_job.txt").write_text(
            "12.34s call     lib/zephyr/tests/test_dataset.py::test_reduce_sum\n"
            "FAILED lib/zephyr/tests/test_dataset.py::test_reduce_sum - AssertionError\n"
        )
        nested_dir = log_dir / "job"
        nested_dir.mkdir()
        (nested_dir / "7_Test with pytest.txt").write_text(
            "12.34s call     tests/test_dataset.py::test_reduce_sum\n"
            "FAILED tests/test_dataset.py::test_reduce_sum - AssertionError\n"
        )

        evidence = nightshift_ci_tests.collect_evidence(
            log_dir,
            workflow_name="Zephyr - Tests",
            run={"id": 123, "html_url": "https://example.com/run/123"},
        )

    record = evidence["tests/test_dataset.py::test_reduce_sum"]
    assert record["slow_hits"] == 1
    assert record["failure_runs"] == {123}
    assert len(record["slow_examples"]) == 1
    assert len(record["failure_examples"]) == 1
