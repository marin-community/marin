# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for summing CI JUnit reports into per-file test times (infra/ci/shard_timings.py)."""

from pathlib import Path

from infra.ci.shard_timings import merged_seconds


def test_reports_map_rootdir_paths_to_repo_paths_and_keep_untimed_files(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    for test_file in ("lib/levanter/tests/grug/test_moe.py", "tests/test_root.py", "tests/test_untimed.py"):
        (repo / test_file).parent.mkdir(parents=True, exist_ok=True)
        (repo / test_file).write_text("def test_case(): ...\n")
    reports = tmp_path / "reports"
    (reports / "unit-junit-0").mkdir(parents=True)
    (reports / "unit-junit-1").mkdir()
    # Levanter's pytest rootdir is lib/levanter, so its report paths start at tests/.
    (reports / "unit-junit-0/marin-levanter.xml").write_text(
        "<testsuites><testsuite>"
        '<testcase file="tests/grug/test_moe.py" name="test_a" time="20.5"/>'
        '<testcase file="tests/grug/test_moe.py" name="test_b" time="1.5"/>'
        "</testsuite></testsuites>"
    )
    (reports / "unit-junit-1/marin-core.xml").write_text(
        '<testsuites><testsuite><testcase file="tests/test_root.py" name="t" time="3.0"/></testsuite></testsuites>'
    )
    previous = {"tests/test_root.py": 99.0, "tests/test_untimed.py": 7.0, "tests/test_deleted.py": 5.0}

    seconds = merged_seconds(reports, previous, repo)

    assert seconds == {
        "lib/levanter/tests/grug/test_moe.py": 22.0,
        "tests/test_root.py": 3.0,
        "tests/test_untimed.py": 7.0,
    }
