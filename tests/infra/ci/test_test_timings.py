# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for summing CI JUnit reports into per-file test times (infra/ci/test_timings.py)."""

import textwrap
from pathlib import Path

from infra.ci.test_timings import merged_seconds


def write(path: Path, body: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body))


def test_reports_map_rootdir_paths_to_repo_paths_and_keep_untimed_files(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    write(repo / "lib/levanter/tests/grug/test_moe.py", "def test_moe(): ...\n")
    write(repo / "tests/test_root.py", "def test_root(): ...\n")
    write(repo / "tests/test_untimed.py", "def test_untimed(): ...\n")
    reports = tmp_path / "reports"
    # Levanter's pytest rootdir is lib/levanter, so its report paths start at tests/.
    write(
        reports / "unit-junit-0/marin-levanter.xml",
        """\
        <testsuites><testsuite>
          <testcase file="tests/grug/test_moe.py" name="test_a" time="20.5"/>
          <testcase file="tests/grug/test_moe.py" name="test_b" time="1.5"/>
        </testsuite></testsuites>
        """,
    )
    write(
        reports / "unit-junit-1/marin-core.xml",
        '<testsuites><testsuite><testcase file="tests/test_root.py" name="t" time="3.0"/></testsuite></testsuites>',
    )
    previous = {"tests/test_root.py": 99.0, "tests/test_untimed.py": 7.0, "tests/test_deleted.py": 5.0}

    seconds = merged_seconds(reports, previous, repo)

    assert seconds == {
        "lib/levanter/tests/grug/test_moe.py": 22.0,
        "tests/test_root.py": 3.0,
        "tests/test_untimed.py": 7.0,
    }
