# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the colocated pytracer ``sitecustomize.py``.

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

PYTRACER_DIR = Path(__file__).resolve().parents[2] / "experiments" / "swe_rebench_trace" / "pytracer"


@pytest.fixture(scope="module")
def tracer():
    spec = importlib.util.spec_from_file_location("pytracer_test_load", PYTRACER_DIR / "sitecustomize.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------
# _safe_repr / _render — the bits whose subtle choices show up in every line
# of every trace.
# ----------------------------------------------------------------------------


def test_safe_repr_collapses_strings_too_long_to_shrink_via_depth(tracer):
    # Containers shrink by reducing recursion depth; primitives can't, so
    # long strings need an explicit length-tag fallback or the whole repr
    # leaks into the trace verbatim.
    s = "x" * (tracer.MAX_REPR_LEN + 10)
    assert tracer._safe_repr(s) == f"<str len={len(s)}>"


def test_safe_repr_caps_container_size_with_ellipsis_marker(tracer):
    # Without MAX_CONTAINER_ITEMS, a single huge list/dict inside a frame
    # would dominate the trace.
    out = tracer._safe_repr(list(range(tracer.MAX_CONTAINER_ITEMS + 50)))
    head = ", ".join(str(i) for i in range(tracer.MAX_CONTAINER_ITEMS))
    assert out == f"[{head}, ...]"


def test_safe_repr_only_falls_back_to_vars_when_repr_is_object_at(tracer):
    # The "<Foo object at 0x...>" heuristic discriminates classes with no
    # useful __repr__ from ones that already render meaningfully — clobbering
    # a custom __repr__ via vars() would erase intentional formatting.
    class Plain:
        def __init__(self):
            self.a = 1
            self.b = "x"

    class Stringy:
        def __repr__(self):
            return "Stringy(custom)"

    plain = tracer._safe_repr(Plain())
    assert plain.startswith("Plain(")
    assert "'a': 1" in plain
    assert "'b': 'x'" in plain

    assert tracer._safe_repr(Stringy()) == "Stringy(custom)"


def test_render_returns_typename_when_depth_exceeds_cap(tracer):
    # Base case of the recursive renderer — without it _safe_repr's depth
    # shrinking loop has nothing to fall back to. At max_depth=0 the top
    # container's own shape still renders, but every child collapses to
    # <Type>; that's what makes shrinking actually shrink.
    assert tracer._render({"a": 1}, 0, -1) == "<dict>"
    assert tracer._render({"a": [1, 2, 3]}, 0, 0) == "{<str>: <list>}"


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
        # Multi-line conditional — first line has no trailing ':', so we
        # *must* return None or branch annotations would mis-fire.
        ("if x and \\", None),
    ],
)
def test_block_keyword_only_matches_single_line_block_headers(tracer, src, expected):
    assert tracer._block_keyword(src) == expected


@pytest.mark.parametrize(
    "qualname, func_name, expected",
    [
        ("test_foo", "test_foo", True),
        ("TestX.test_bar", "test_bar", True),
        ("helper", "helper", False),
        ("SomeClass.method", "method", False),
        # A non-test method on a Test* class must NOT count, otherwise
        # helpers called by tests would each open a new trace row.
        ("TestX.helper", "helper", False),
    ],
)
def test_is_test_function(tracer, qualname, func_name, expected):
    assert tracer._is_test_function(qualname, func_name) is expected


@pytest.mark.parametrize(
    "value, expected",
    [
        # pytest's PYTEST_CURRENT_TEST is "<nodeid> (<phase>)" — must strip
        # the phase suffix or every test_id ends in '(call)'.
        ("tests/foo.py::test_bar (call)", "tests/foo.py::test_bar"),
        ("tests/foo.py::test_bar (setup)", "tests/foo.py::test_bar"),
        # Older/edge case: no space at all → rpartition returns ('', '', x);
        # the `or current` fallback keeps the nodeid intact.
        ("tests/foo.py::test_bar", "tests/foo.py::test_bar"),
        ("", None),
        (None, None),
    ],
)
def test_current_pytest_nodeid(tracer, monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    else:
        monkeypatch.setenv("PYTEST_CURRENT_TEST", value)
    assert tracer._current_pytest_nodeid() == expected


def test_diff_locals_flags_keys_added_since_prev_snapshot(tracer):
    # Implementation uses prev.get(k) != v — the subtle behavior is that
    # *new* keys (absent from prev) get None on the LHS and so appear in
    # the delta, which is what makes per-line assignment annotations work.
    prev = {"x": "1"}
    cur = {"x": "1", "y": "new"}
    assert tracer._diff_locals(cur, prev) == {"y": "new"}
    assert tracer._diff_locals(prev, prev) == {}


# ----------------------------------------------------------------------------
# _should_skip — gates which frames feed the tracer at all.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename",
    [
        "/usr/lib/python3.11/site-packages/foo.py",
        "/foo/site-packages/bar.py",
        "<frozen runpy>",
        "",
    ],
)
def test_should_skip_filters_stdlib_and_pseudo_files(tracer, filename):
    assert tracer._should_skip(filename, None)


def test_should_skip_excludes_paths_outside_repo_root(tracer, tmp_path):
    # repo_root scoping is what stops the tracer from following helpers into
    # installed dependencies — without it the trace fills with library noise.
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
# auto-installed via PYTHONPATH and parse the JSONL it writes. These exercise
# the full _Tracer state machine, which has multiple subtle invariants
# (call-site ordering, branch annotation, iter counters, PID guard).
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


def test_tracer_opens_one_row_per_outermost_test_function(traced_pytest):
    # The "outermost test_*" frame defines the row boundary. If a helper got
    # treated as a new test, we'd see >2 rows; if a real test got missed
    # (e.g. matching only `func_name` instead of qualname), we'd see <2.
    records = traced_pytest(
        "def helper():\n"
        "    return 1\n"
        "\n"
        "def test_a():\n"
        "    helper()\n"
        "    helper()\n"
        "\n"
        "def test_b():\n"
        "    helper()\n"
    )
    assert [r["test_id"] for r in records] == [
        "test_target.py::test_a",
        "test_target.py::test_b",
    ]


def test_tracer_renders_call_site_above_callee_frame(traced_pytest):
    # Off-by-one invariant: the line that *causes* a call must flush before
    # the callee's header is written, otherwise the call site renders below
    # the entire callee trace and the read order is wrong.
    records = traced_pytest("def helper():\n    return 7\n\ndef test_call_order():\n    z = helper()\n")
    trace = records[0]["trace"]
    call_site = trace.find("z = helper()")
    callee_header = trace.find("::helper")
    assert call_site != -1 and callee_header != -1
    assert call_site < callee_header


def test_tracer_propagates_callee_return_into_caller_locals(traced_pytest):
    # The 'z=2' annotation comes from a queued caller-snapshot diff that
    # fires on the next line event in the caller — the LHS assignment
    # bytecode runs *after* the return event, so a naive immediate-diff
    # would never see z bound.
    records = traced_pytest(
        "def helper(n):\n    return n + 1\n\ndef test_propagation():\n    z = helper(1)\n    assert z == 2\n"
    )
    trace = records[0]["trace"]
    assert "# ENTER: n=1" in trace
    assert "# RETURN from helper: 2" in trace
    assert "z=2" in trace


def test_tracer_records_branch_taken(traced_pytest):
    records = traced_pytest("def test_branch_true():\n    n = 1\n    if n == 1:\n        x = 'yes'\n")
    assert "branch=if:True" in records[0]["trace"]


def test_tracer_records_branch_skipped(traced_pytest):
    # next_lineno > lineno + 1 is the only signal that the body got skipped
    # — if that math is off, the False case never gets annotated.
    records = traced_pytest(
        "def test_branch_false():\n    n = 2\n    if n == 1:\n        x = 'never'\n    y = 'after'\n"
    )
    trace = records[0]["trace"]
    assert "branch=if:False" in trace
    assert "branch=if:True" not in trace


def test_tracer_marks_loop_iterations(traced_pytest):
    # Loop body lines fire repeatedly with the same lineno — the per-frame
    # iter counter is what keeps them distinguishable in the rendered trace.
    records = traced_pytest("def test_loop():\n    total = 0\n    for i in range(3):\n        total += i\n")
    trace = records[0]["trace"]
    assert "iter=2" in trace
    assert "iter=3" in trace


def test_tracer_pid_guard_prevents_subprocess_clobber(traced_pytest):
    # Without TRACER_ACTIVE_PID, a child python process inherits PYTHONPATH
    # and would re-install the tracer, racing on TRACER_OUTPUT and producing
    # truncated/mixed JSONL.
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
