# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``scripts/contree_pytracer/sitecustomize.py``.

The tracer module is loaded via ``importlib`` so ``_install()`` is skipped
(it's gated on ``__name__ == "sitecustomize"``). End-to-end tests run a real
pytest subprocess with PYTHONPATH pointed at the tracer dir and parse the
JSONL it writes.
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PYTRACER_DIR = Path(__file__).resolve().parents[2] / "scripts" / "contree_pytracer"


@pytest.fixture(scope="module")
def tracer():
    spec = importlib.util.spec_from_file_location("contree_pytracer_test_load", PYTRACER_DIR / "sitecustomize.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------
# Pure helpers
# ----------------------------------------------------------------------------


def test_safe_repr_primitives(tracer):
    assert tracer._safe_repr(1) == "1"
    assert tracer._safe_repr("foo") == "'foo'"
    assert tracer._safe_repr(None) == "None"
    assert tracer._safe_repr([1, 2, 3]) == "[1, 2, 3]"
    assert tracer._safe_repr({"a": 1}) == "{'a': 1}"


def test_safe_repr_collapses_long_string_to_length_tag(tracer):
    s = "x" * (tracer.MAX_REPR_LEN + 10)
    assert tracer._safe_repr(s) == f"<str len={len(s)}>"


def test_safe_repr_truncates_large_container(tracer):
    out = tracer._safe_repr(list(range(20)))
    assert out.startswith("[0, 1, 2, 3, 4, 5, 6, 7, 8, 9")
    assert "..." in out
    assert out.endswith("]")


def test_safe_repr_renders_object_via_dict(tracer):
    class Foo:
        def __init__(self):
            self.a = 1
            self.b = "x"

    out = tracer._safe_repr(Foo())
    assert out.startswith("Foo(")
    assert "'a': 1" in out
    assert "'b': 'x'" in out


def test_render_collapses_to_typename_beyond_max_depth(tracer):
    # depth > max_depth at entry → immediate <TypeName>.
    assert tracer._render({"a": 1}, 0, -1) == "<dict>"
    # Children one level past max_depth get type-collapsed.
    out = tracer._render({"a": [1, 2, 3]}, 0, 0)
    assert "<list>" in out


def test_render_truncates_dict_with_more_than_max_items(tracer):
    big = {f"k{i}": i for i in range(tracer.MAX_CONTAINER_ITEMS + 5)}
    out = tracer._render(big, 0, 5)
    assert out.startswith("{")
    assert out.endswith("}")
    assert ", ..." in out


# ----------------------------------------------------------------------------
# Source-line classifiers
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "src, expected",
    [
        ("if x:", "if"),
        ("    if x is not None:", "if"),
        ("elif y:", "elif"),
        ("while True:", "while"),
        ("for k, v in items:", "for"),
        ("def foo():", None),
        ("class Bar:", None),
        ("x = 1", None),
        ("if x and \\", None),
    ],
)
def test_block_keyword(tracer, src, expected):
    assert tracer._block_keyword(src) == expected


@pytest.mark.parametrize(
    "qualname, func_name, expected",
    [
        ("test_foo", "test_foo", True),
        ("TestX.test_bar", "test_bar", True),
        ("helper", "helper", False),
        ("SomeClass.method", "method", False),
        ("TestX.helper", "helper", False),
    ],
)
def test_is_test_function(tracer, qualname, func_name, expected):
    assert tracer._is_test_function(qualname, func_name) is expected


def test_current_pytest_nodeid_strips_phase(tracer, monkeypatch):
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/foo.py::test_bar (call)")
    assert tracer._current_pytest_nodeid() == "tests/foo.py::test_bar"


def test_current_pytest_nodeid_unset(tracer, monkeypatch):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    assert tracer._current_pytest_nodeid() is None


def test_diff_locals_only_changed_keys(tracer):
    prev = {"x": "1", "y": "2"}
    cur = {"x": "1", "y": "3", "z": "4"}
    assert tracer._diff_locals(cur, prev) == {"y": "3", "z": "4"}


@pytest.mark.parametrize(
    "src, expected",
    [
        ("    x = 1", "    "),
        ("\t\tx = 1", "\t\t"),
        ("x = 1", ""),
        ("", ""),
    ],
)
def test_indent_of(tracer, src, expected):
    assert tracer._indent_of(src) == expected


def test_should_skip_filters_known_markers(tracer):
    assert tracer._should_skip("/usr/lib/python3.11/site-packages/foo.py", None)
    assert tracer._should_skip("/foo/site-packages/bar.py", None)
    assert tracer._should_skip("<frozen runpy>", None)
    assert tracer._should_skip("", None)


def test_should_skip_excludes_paths_outside_repo_root(tracer, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    inside = repo / "x.py"
    inside.write_text("")
    outside = tmp_path / "other.py"
    outside.write_text("")
    assert not tracer._should_skip(str(inside), str(repo))
    assert tracer._should_skip(str(outside), str(repo))


# ----------------------------------------------------------------------------
# End-to-end via subprocess: run pytest on a tiny target with the tracer
# auto-installed via PYTHONPATH and parse the JSONL it writes.
# ----------------------------------------------------------------------------


@pytest.fixture
def traced_pytest(tmp_path):
    def run(test_source: str) -> list[dict]:
        project = tmp_path / "project"
        project.mkdir(exist_ok=True)
        test_file = project / "test_target.py"
        test_file.write_text(test_source)
        trace = tmp_path / "trace.jsonl"

        env = os.environ.copy()
        env["PYTHONPATH"] = str(PYTRACER_DIR)
        env["TRACER_OUTPUT"] = str(trace)
        env["TRACER_REPO_ROOT"] = str(project)

        subprocess.run(
            [sys.executable, "-m", "pytest", "-q", str(test_file)],
            cwd=project,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )

        return [json.loads(line) for line in trace.read_text().splitlines()]

    return run


def test_tracer_emits_one_jsonl_row_per_test(traced_pytest):
    records = traced_pytest(
        "def test_a():\n" "    x = 1\n" "    assert x == 1\n" "\n" "def test_b():\n" "    assert True\n"
    )
    assert [r["test_id"] for r in records] == [
        "test_target.py::test_a",
        "test_target.py::test_b",
    ]
    for r in records:
        assert r["file"] == "test_target.py"
        assert r["function"].startswith("test_")
        assert r["event_count"] > 0


def test_tracer_renders_state_calls_and_returns(traced_pytest):
    records = traced_pytest(
        "def helper(n):\n"
        "    return n + 1\n"
        "\n"
        "def test_calls_helper():\n"
        "    y = 1\n"
        "    z = helper(y)\n"
        "    assert z == 2\n"
    )
    assert len(records) == 1
    trace = records[0]["trace"]
    assert "# --- test source ---" in trace
    assert "def test_calls_helper" in trace
    # Per-line state annotation after the assignment.
    assert "y = 1" in trace
    assert "y=1" in trace
    # Helper got its own header + ENTER + RETURN.
    assert "::helper" in trace
    assert "# ENTER: n=1" in trace
    assert "# RETURN from helper: 2" in trace
    # Caller-side delta after the call ('# → z=2').
    assert "z=2" in trace


def test_tracer_records_branch_outcome(traced_pytest):
    records = traced_pytest(
        "def test_branch():\n"
        "    n = 1\n"
        "    if n == 1:\n"
        "        x = 'yes'\n"
        "    else:\n"
        "        x = 'no'\n"
        "    assert x == 'yes'\n"
    )
    assert "branch=if:True" in records[0]["trace"]


def test_tracer_does_not_capture_subprocesses(traced_pytest):
    records = traced_pytest(
        "import subprocess, sys\n"
        "\n"
        "def test_parent_a():\n"
        "    assert True\n"
        "\n"
        "def test_parent_b():\n"
        "    subprocess.run([sys.executable, '-c', 'print(42)'], check=True)\n"
    )
    assert [r["test_id"] for r in records] == [
        "test_target.py::test_parent_a",
        "test_target.py::test_parent_b",
    ]
