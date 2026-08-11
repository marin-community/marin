# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[3]
VALIDATOR = REPO_ROOT / "lib" / "iris" / "images" / "h100_evidence_nsys_help.py"
PROFILE_HELP = """\
usage: nsys profile [<args>] [application] [<application args>]

  -c, --capture-range arg (=none)
      Start collection at the selected capture range.
  --capture-range-end arg (=stop-shutdown)
      Specify the desired behavior when a capture range ends. Possible values
      are 'none', 'stop', 'stop-shutdown', 'repeat[:N][:mode]' or
      'repeat-shutdown:N'[:mode]. If 'stop', collection stops and the target
      application continues running.
  --cuda-graph-trace arg (=graph)
      Select CUDA Graph activity granularity. Possible values are 'graph' and
      'node'.
  --cuda-backtrace arg (=false)
      Collect CUDA backtraces.
"""
FAILURE_DIAGNOSTIC_MARKER = " diagnostic="
MAX_FAILURE_MESSAGE_CHARS = 4096
MAX_POSSIBLE_VALUES_CLAUSE_BYTES = 1024
MAX_GRAPH_DIAGNOSTIC_LINE_BYTES = 512
MAX_GRAPH_DIAGNOSTIC_BYTES = 1536


def _validate(tmp_path: Path, payload: bytes) -> subprocess.CompletedProcess[str]:
    artifact = tmp_path / "profile-help.txt"
    artifact.write_bytes(payload)
    return subprocess.run(
        (sys.executable, str(VALIDATOR), str(artifact)),
        capture_output=True,
        check=False,
        text=True,
    )


def _failure_diagnostic(result: subprocess.CompletedProcess[str]) -> dict:
    assert result.returncode == 1
    assert result.stdout == ""
    assert FAILURE_DIAGNOSTIC_MARKER in result.stderr
    return json.loads(result.stderr.rstrip().split(FAILURE_DIAGNOSTIC_MARKER, maxsplit=1)[1])


def _profile_help_with_capture_clause_bytes(size: int) -> tuple[str, str]:
    clause_prefix = "Possible values are 'none', 'stop', 'stop-shutdown', and "
    padding = size - len(clause_prefix.encode()) - 1
    assert padding >= 0
    clause = f"{clause_prefix}{'x' * padding}."
    assert len(clause.encode()) == size
    capture = f"  --capture-range-end arg (=stop-shutdown)\n      {clause}"
    graph = "  --cuda-graph-trace arg (=graph)\n" "      Possible values are 'graph' and 'node'."
    return (
        f"usage: nsys profile\n--stop-on-range-end arg\n{capture}\n{graph}\n  --next-option arg\n",
        clause,
    )


def _padded_option_block(block: str, size: int, sentinel: str) -> str:
    padding = size - len(block.encode()) - len(sentinel.encode()) - 1
    assert padding >= 0
    block = f"{block}\n{sentinel}{'x' * padding}"
    assert len(block.encode()) == size
    return block


def test_nsys_profile_help_accepts_unique_stop_policy(tmp_path: Path) -> None:
    result = _validate(tmp_path, PROFILE_HELP.encode())

    assert result.returncode == 0, result.stderr
    assert "capture-range-end=none,stop,stop-shutdown,repeat[:N][:mode],repeat-shutdown:N[:mode]" in result.stdout
    assert "cuda-graph-trace=graph,node" in result.stdout


def test_nsys_profile_help_accepts_exact_remote_capture_range_end_clause(tmp_path: Path) -> None:
    exact_clause = (
        "Possible values are 'none', 'stop', 'stop-shutdown', " "'repeat[:N][:mode]' or 'repeat-shutdown:N'[:mode]."
    )
    help_text = PROFILE_HELP.replace(
        "Possible values\n"
        "      are 'none', 'stop', 'stop-shutdown', 'repeat[:N][:mode]' or\n"
        "      'repeat-shutdown:N'[:mode].",
        exact_clause,
    )

    result = _validate(tmp_path, help_text.encode())

    assert result.returncode == 0, result.stderr
    assert "capture-range-end=none,stop,stop-shutdown,repeat[:N][:mode],repeat-shutdown:N[:mode]" in result.stdout


@pytest.mark.parametrize(
    "mutation",
    (
        "'repeat[:N][:mode]' and\n      'repeat-shutdown:N'[:mode]",
        "'repeat[:N][:mode]', or\n      'repeat-shutdown:N'[:mode]",
        "'repeat[:N][:mode]' or\n      'repeat-shutdown:N[:mode]'",
        "'repeat[:N][:mode]' or\n      'repeat-shutdown:N'",
        "'repeat-shutdown:N'[:mode] or\n      'repeat[:N][:mode]'",
        '"repeat[:N][:mode]" or\n      "repeat-shutdown:N"[:mode]',
    ),
    ids=("and", "comma-or", "quoted-suffix", "missing-suffix", "reordered", "double-quoted"),
)
def test_nsys_profile_help_rejects_noncanonical_capture_range_end_grammar(tmp_path: Path, mutation: str) -> None:
    help_text = PROFILE_HELP.replace(
        "'repeat[:N][:mode]' or\n      'repeat-shutdown:N'[:mode]",
        mutation,
    )

    result = _validate(tmp_path, help_text.encode())

    assert result.returncode == 1
    assert result.stderr.startswith("nsys profile help validation failed: --capture-range-end")


@pytest.mark.parametrize(
    "help_text",
    (
        PROFILE_HELP.replace("  --capture-range-end arg (=stop-shutdown)\n", ""),
        PROFILE_HELP.replace(
            "  --cuda-backtrace arg (=false)\n",
            "  --capture-range-end arg (=stop-shutdown)\n"
            "      Possible values are 'none', 'stop', and 'stop-shutdown'.\n"
            "  --cuda-backtrace arg (=false)\n",
        ),
        PROFILE_HELP.replace("'none', 'stop', 'stop-shutdown'", "'none', 'stop-shutdown'"),
        PROFILE_HELP.replace("'none', 'stop', 'stop-shutdown'", "'none', 'stop', 'stop', 'stop-shutdown'"),
        PROFILE_HELP.replace(
            "      application continues running.\n",
            "      application continues running. Possible values are 'none', 'stop', and 'stop-shutdown'.\n",
        ),
        PROFILE_HELP.replace(
            "'repeat-shutdown:N'[:mode]",
            "'repeat-shutdown:N'[:mode] and future",
        ),
        PROFILE_HELP.replace(
            "'repeat-shutdown:N'[:mode]",
            "'repeat-shutdown:N'[:mode] and 'future'",
        ),
        PROFILE_HELP + "  --stop-on-range-end arg (=true)\n",
    ),
    ids=(
        "missing",
        "duplicate-option",
        "missing-stop",
        "duplicate-stop",
        "duplicate-value-list",
        "unquoted-future-value",
        "quoted-future-value",
        "obsolete-option",
    ),
)
def test_nsys_profile_help_rejects_missing_or_ambiguous_policy(tmp_path: Path, help_text: str) -> None:
    result = _validate(tmp_path, help_text.encode())

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.startswith("nsys profile help validation failed:")


@pytest.mark.parametrize(
    "help_text",
    (
        PROFILE_HELP.replace(
            "  --cuda-graph-trace arg (=graph)\n"
            "      Select CUDA Graph activity granularity. Possible values are 'graph' and\n"
            "      'node'.\n",
            "",
        ),
        PROFILE_HELP.replace(
            "  --cuda-backtrace arg (=false)\n",
            "  --cuda-graph-trace arg (=graph)\n"
            "      Possible values are 'graph' and 'node'.\n"
            "  --cuda-backtrace arg (=false)\n",
        ),
        PROFILE_HELP.replace(
            "Possible values are 'graph' and\n      'node'.",
            "Possible values are 'graph'.",
        ),
        PROFILE_HELP.replace(
            "Possible values are 'graph' and\n      'node'.",
            "Possible values are 'graph', 'node', and 'node'.",
        ),
        PROFILE_HELP.replace(
            "Possible values are 'graph' and\n      'node'.",
            "Possible values are 'graph', 'node', and 'future'.",
        ),
        PROFILE_HELP.replace(
            "Possible values are 'graph' and\n      'node'.",
            "Possible values are 'graph', 'node', and future.",
        ),
    ),
    ids=(
        "missing",
        "duplicate-option",
        "missing-node",
        "duplicate-node",
        "unknown-value",
        "unquoted-future",
    ),
)
def test_nsys_profile_help_rejects_missing_or_ambiguous_cuda_graph_policy(tmp_path: Path, help_text: str) -> None:
    result = _validate(tmp_path, help_text.encode())

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.startswith("nsys profile help validation failed:")


@pytest.mark.parametrize("payload", (b"", b"usage\x00help", b"\xff\xfe"), ids=("empty", "nul", "non-utf8"))
def test_nsys_profile_help_rejects_malformed_artifacts(tmp_path: Path, payload: bytes) -> None:
    result = _validate(tmp_path, payload)

    assert result.returncode == 1
    assert result.stdout == ""


def test_nsys_profile_help_rejects_oversized_artifact_before_reading_it(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "profile-help.txt"
    with artifact.open("wb") as stream:
        stream.seek((1 << 20) + 1)
        stream.write(b"x")

    result = subprocess.run(
        (sys.executable, str(VALIDATOR), str(artifact)),
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 1
    assert "exceeds 1048576 bytes" in result.stderr


def test_nsys_profile_help_failure_emits_only_exact_possible_values_clauses(tmp_path: Path) -> None:
    help_text = PROFILE_HELP.replace(
        "usage: nsys profile [<args>] [application] [<application args>]",
        "usage: nsys profile PRIVATE_OUTSIDE_CLAUSES",
    ).replace("'graph' and\n      'node'", "'graph', 'node', and 'future'")

    result = _validate(tmp_path, help_text.encode())

    diagnostic = _failure_diagnostic(result)
    assert result.stderr.startswith(
        "nsys profile help validation failed: --cuda-graph-trace does not expose exactly graph and node"
    )
    assert diagnostic["schema"] == "iris.h100_evidence_nsys_help_failure.v3"
    capture = diagnostic["clauses"]["capture-range-end"]
    graph = diagnostic["clauses"]["cuda-graph-trace"]
    assert capture["available"] is True
    assert graph["available"] is True
    assert capture["bytes"] == len(capture["text"].encode())
    assert capture["sha256"] == hashlib.sha256(capture["text"].encode()).hexdigest()
    assert graph["bytes"] == len(graph["text"].encode())
    assert graph["sha256"] == hashlib.sha256(graph["text"].encode()).hexdigest()
    assert capture["text"].startswith("Possible values")
    assert graph["text"] == "Possible values are 'graph', 'node', and 'future'."
    assert "--capture-range-end arg" not in result.stderr
    assert "PRIVATE_OUTSIDE_CLAUSES" not in result.stderr
    assert "Collect CUDA backtraces" not in result.stderr
    assert len(result.stderr) <= MAX_FAILURE_MESSAGE_CHARS

    context = diagnostic["cuda_graph_context"]
    assert context["available"] is True
    assert context["graph_occurrences"] == 1
    assert context["node_occurrences"] == 1
    assert [line["text"] for line in context["token_lines"]] == [
        "      Select CUDA Graph activity granularity. Possible values are 'graph', 'node', and 'future'."
    ]
    assert (
        context["sha256"]
        == hashlib.sha256(
            b"  --cuda-graph-trace arg (=graph)\n"
            b"      Select CUDA Graph activity granularity. Possible values are 'graph', 'node', and 'future'."
        ).hexdigest()
    )


def test_nsys_profile_help_failure_extracts_clauses_from_real_size_option_blocks(tmp_path: Path) -> None:
    capture_clause = "Possible values are 'none', 'stop', and future."
    graph_clause = "Possible values are 'graph', 'node', and 'future'."
    capture = _padded_option_block(
        "  --capture-range-end arg (=stop-shutdown)\n"
        f"      description before clause. {capture_clause}\n"
        "      description after clause.",
        1704,
        "PRIVATE_CAPTURE_PADDING_",
    )
    graph = _padded_option_block(
        "  --cuda-graph-trace arg (=graph)\n"
        f"      description before clause. {graph_clause}\n"
        "      description after clause.",
        1538,
        "PRIVATE_GRAPH_PADDING_",
    )
    help_text = f"usage: nsys profile PRIVATE_OUTSIDE_CLAUSES\n{capture}\n{graph}\n  --next-option arg\n"

    diagnostic = _failure_diagnostic(_validate(tmp_path, help_text.encode()))

    capture_record = diagnostic["clauses"]["capture-range-end"]
    graph_record = diagnostic["clauses"]["cuda-graph-trace"]
    assert capture_record["text"] == capture_clause
    assert graph_record["text"] == graph_clause
    assert capture_record["bytes"] == len(capture_clause.encode())
    assert graph_record["bytes"] == len(graph_clause.encode())
    serialized = json.dumps(diagnostic)
    assert "PRIVATE_CAPTURE_PADDING" not in serialized
    assert "PRIVATE_GRAPH_PADDING" not in serialized
    assert "PRIVATE_OUTSIDE_CLAUSES" not in serialized

    context = diagnostic["cuda_graph_context"]
    assert context["available"] is True
    assert context["token_line_count"] == 1
    assert [line["text"] for line in context["token_lines"]] == [f"      description before clause. {graph_clause}"]
    assert "PRIVATE_GRAPH_PADDING" not in json.dumps(context)


def test_nsys_profile_help_failure_preserves_multiline_clause_exactly(tmp_path: Path) -> None:
    help_text = PROFILE_HELP.replace("'graph' and\n      'node'", "'graph', 'node', and\n      'future'")

    diagnostic = _failure_diagnostic(_validate(tmp_path, help_text.encode()))

    clause = diagnostic["clauses"]["cuda-graph-trace"]
    expected = "Possible values are 'graph', 'node', and\n      'future'."
    assert clause["text"] == expected
    assert clause["bytes"] == len(expected.encode())
    assert clause["sha256"] == hashlib.sha256(expected.encode()).hexdigest()


@pytest.mark.parametrize("mutation", ("missing", "duplicate"))
def test_nsys_profile_help_failure_marks_nonunique_option_anchor_unavailable(tmp_path: Path, mutation: str) -> None:
    graph_block = (
        "  --cuda-graph-trace arg (=graph)\n"
        "      Select CUDA Graph activity granularity. Possible values are 'graph' and\n"
        "      'node'.\n"
    )
    replacement = "" if mutation == "missing" else f"{graph_block}{graph_block}"
    help_text = PROFILE_HELP.replace(graph_block, replacement)

    diagnostic = _failure_diagnostic(_validate(tmp_path, help_text.encode()))

    graph = diagnostic["clauses"]["cuda-graph-trace"]
    assert graph == {
        "available": False,
        "bytes": 0,
        "reason": "exact_option_anchor_unavailable",
        "sha256": None,
    }
    context = diagnostic["cuda_graph_context"]
    assert context == {
        "available": False,
        "bytes": 0,
        "declaration": None,
        "graph_occurrences": 0,
        "node_occurrences": 0,
        "reason": "exact_option_anchor_unavailable",
        "sha256": None,
        "token_line_count": 0,
    }


@pytest.mark.parametrize("mutation", ("missing", "duplicate"))
def test_nsys_profile_help_failure_marks_nonunique_clause_unavailable(tmp_path: Path, mutation: str) -> None:
    if mutation == "missing":
        help_text = PROFILE_HELP.replace("Possible values are 'graph' and\n      'node'.", "No values are documented.")
    else:
        help_text = PROFILE_HELP.replace(
            "      'node'.\n",
            "      'node'. Possible values are 'graph' and 'node'.\n",
        )
    help_text = help_text.replace("usage: nsys profile", "usage: --stop-on-range-end nsys profile")

    diagnostic = _failure_diagnostic(_validate(tmp_path, help_text.encode()))

    graph = diagnostic["clauses"]["cuda-graph-trace"]
    assert graph == {
        "available": False,
        "bytes": 0,
        "reason": "unique_possible_values_clause_unavailable",
        "sha256": None,
    }


@pytest.mark.parametrize(
    ("size", "available"),
    ((MAX_POSSIBLE_VALUES_CLAUSE_BYTES, True), (MAX_POSSIBLE_VALUES_CLAUSE_BYTES + 1, False)),
)
def test_nsys_profile_help_failure_clause_boundary_is_fail_closed(tmp_path: Path, size: int, available: bool) -> None:
    help_text, capture_text = _profile_help_with_capture_clause_bytes(size)

    diagnostic = _failure_diagnostic(_validate(tmp_path, help_text.encode()))

    capture = diagnostic["clauses"]["capture-range-end"]
    assert capture["available"] is available
    assert capture["bytes"] == size
    assert capture["sha256"] == hashlib.sha256(capture_text.encode()).hexdigest()
    if available:
        assert capture["text"] == capture_text
    else:
        assert capture["reason"] == "exceeds_1024_byte_bound"
        assert "text" not in capture


def test_nsys_profile_help_failure_escape_expansion_remains_within_total_bound(
    tmp_path: Path,
) -> None:
    control_text = "\x01" * 700
    capture_clause = (
        "Possible values\n"
        "      are 'none', 'stop', 'stop-shutdown', 'repeat[:N][:mode]' or\n"
        f"      'repeat-shutdown:N'[:mode] and {control_text}."
    )
    graph_clause = f"Possible values are 'graph' and\n      'node', and {control_text}."
    help_text = PROFILE_HELP.replace(
        "'repeat-shutdown:N'[:mode].",
        f"'repeat-shutdown:N'[:mode] and {control_text}.",
    ).replace(
        "      'node'.",
        f"      'node', and {control_text}.",
    )

    result = _validate(tmp_path, help_text.encode())

    diagnostic = _failure_diagnostic(result)
    assert diagnostic["clauses"]["capture-range-end"]["available"] is False
    assert diagnostic["clauses"]["cuda-graph-trace"]["available"] is False
    assert diagnostic["clauses"]["capture-range-end"]["reason"] == "omitted_to_fit_4096_character_bound"
    assert diagnostic["clauses"]["cuda-graph-trace"]["reason"] == "omitted_to_fit_4096_character_bound"
    assert diagnostic["clauses"]["capture-range-end"]["bytes"] == len(capture_clause.encode())
    assert diagnostic["clauses"]["capture-range-end"]["sha256"] == hashlib.sha256(capture_clause.encode()).hexdigest()
    assert diagnostic["clauses"]["cuda-graph-trace"]["bytes"] == len(graph_clause.encode())
    assert diagnostic["clauses"]["cuda-graph-trace"]["sha256"] == hashlib.sha256(graph_clause.encode()).hexdigest()
    assert "text" not in diagnostic["clauses"]["capture-range-end"]
    assert "text" not in diagnostic["clauses"]["cuda-graph-trace"]
    assert "token_lines" not in diagnostic["cuda_graph_context"]
    assert "text" not in diagnostic["cuda_graph_context"]["declaration"]
    assert len(result.stderr) <= MAX_FAILURE_MESSAGE_CHARS
    assert "\\u0001" not in result.stderr


def test_nsys_profile_help_failure_does_not_emit_unrelated_future_value_clause(tmp_path: Path) -> None:
    help_text = PROFILE_HELP.replace(
        "  --cuda-backtrace arg (=false)\n      Collect CUDA backtraces.\n",
        "  --cuda-backtrace arg (=false)\n"
        "      PRIVATE_UNRELATED Possible values are 'private-future'.\n"
        "  --stop-on-range-end arg (=true)\n",
    )

    result = _validate(tmp_path, help_text.encode())

    diagnostic = _failure_diagnostic(result)
    assert diagnostic["clauses"]["capture-range-end"]["available"] is True
    assert diagnostic["clauses"]["cuda-graph-trace"]["available"] is True
    assert "private-future" not in result.stderr
    assert "PRIVATE_UNRELATED" not in result.stderr


def test_nsys_profile_help_failure_emits_only_scoped_graph_token_lines(tmp_path: Path) -> None:
    graph_block = (
        "  --cuda-graph-trace arg (=graph)\n"
        "      PRIVATE_GRAPH_DESCRIPTION_WITHOUT_LOWERCASE_TOKENS\n"
        "      graph selects whole-graph activity.\n"
        "      node selects activity.\n"
        "      future selects PRIVATE_FUTURE_VALUE activity.\n"
    )
    help_text = PROFILE_HELP.replace(
        "  --cuda-graph-trace arg (=graph)\n"
        "      Select CUDA Graph activity granularity. Possible values are 'graph' and\n"
        "      'node'.\n",
        graph_block,
    ).replace("usage: nsys profile", "usage: --stop-on-range-end nsys profile")

    result = _validate(tmp_path, help_text.encode())

    diagnostic = _failure_diagnostic(result)
    context = diagnostic["cuda_graph_context"]
    assert context["available"] is True
    assert context["bytes"] <= MAX_GRAPH_DIAGNOSTIC_BYTES
    assert context["declaration"]["text"] == "  --cuda-graph-trace arg (=graph)"
    assert [line["text"] for line in context["token_lines"]] == [
        "      graph selects whole-graph activity.",
        "      node selects activity.",
    ]
    assert all(line["bytes"] <= MAX_GRAPH_DIAGNOSTIC_LINE_BYTES for line in context["token_lines"])
    assert all(line["sha256"] == hashlib.sha256(line["text"].encode()).hexdigest() for line in context["token_lines"])
    serialized = json.dumps(diagnostic)
    assert "PRIVATE_GRAPH_DESCRIPTION" not in serialized
    assert "PRIVATE_FUTURE_VALUE" not in serialized
    assert "future selects" not in serialized


@pytest.mark.parametrize(
    ("replacement", "graph_occurrences", "node_occurrences"),
    (
        ("      graph then graph then node.\n", 2, 1),
        ("      graph then node then graph.\n", 2, 1),
        ("      graph then node then node.\n", 1, 2),
        ("      node then graph.\n", 1, 1),
        ("      node only.\n", 0, 1),
        ("      graph only.\n", 1, 0),
    ),
    ids=("duplicate-graph", "trailing-graph", "duplicate-node", "reordered", "missing-graph", "missing-node"),
)
def test_nsys_profile_help_failure_marks_ambiguous_graph_tokens_unavailable(
    tmp_path: Path,
    replacement: str,
    graph_occurrences: int,
    node_occurrences: int,
) -> None:
    help_text = PROFILE_HELP.replace(
        "      Select CUDA Graph activity granularity. Possible values are 'graph' and\n" "      'node'.\n",
        replacement,
    ).replace("usage: nsys profile", "usage: --stop-on-range-end nsys profile")

    context = _failure_diagnostic(_validate(tmp_path, help_text.encode()))["cuda_graph_context"]

    assert context["available"] is False
    assert context["reason"] == "exact_graph_node_token_sequence_unavailable"
    assert context["graph_occurrences"] == graph_occurrences
    assert context["node_occurrences"] == node_occurrences
    assert "token_lines" not in context


def test_nsys_profile_help_failure_graph_context_line_bound_is_fail_closed(tmp_path: Path) -> None:
    oversized_line = "      graph node " + "x" * MAX_GRAPH_DIAGNOSTIC_LINE_BYTES
    help_text = PROFILE_HELP.replace(
        "      Select CUDA Graph activity granularity. Possible values are 'graph' and\n" "      'node'.",
        oversized_line,
    ).replace("usage: nsys profile", "usage: --stop-on-range-end nsys profile")

    context = _failure_diagnostic(_validate(tmp_path, help_text.encode()))["cuda_graph_context"]

    assert context["available"] is False
    assert context["reason"] == "token_line_exceeds_512_byte_bound"
    assert context["bytes"] > MAX_GRAPH_DIAGNOSTIC_LINE_BYTES
    assert "token_lines" not in context


def test_nsys_profile_help_failure_graph_declaration_bound_is_fail_closed(tmp_path: Path) -> None:
    declaration = "  --cuda-graph-trace " + "d" * (MAX_GRAPH_DIAGNOSTIC_LINE_BYTES - 20)
    assert len(declaration.encode()) == MAX_GRAPH_DIAGNOSTIC_LINE_BYTES + 1
    help_text = PROFILE_HELP.replace(
        "  --cuda-graph-trace arg (=graph)",
        declaration,
    ).replace("usage: nsys profile", "usage: --stop-on-range-end nsys profile")

    context = _failure_diagnostic(_validate(tmp_path, help_text.encode()))["cuda_graph_context"]

    assert context["available"] is False
    assert context["reason"] == "declaration_line_exceeds_512_byte_bound"
    assert context["declaration"]["available"] is False
    assert "text" not in context["declaration"]
    assert "token_lines" not in context


@pytest.mark.parametrize(
    ("graph_line_bytes", "available"),
    ((MAX_GRAPH_DIAGNOSTIC_LINE_BYTES - 2, True), (MAX_GRAPH_DIAGNOSTIC_LINE_BYTES - 1, False)),
    ids=("exact-total-bound", "one-past-total-bound"),
)
def test_nsys_profile_help_failure_graph_context_total_bound_is_fail_closed(
    tmp_path: Path,
    graph_line_bytes: int,
    available: bool,
) -> None:
    declaration = "  --cuda-graph-trace " + "d" * (MAX_GRAPH_DIAGNOSTIC_LINE_BYTES - 21)
    graph_line = "      graph " + "g" * (graph_line_bytes - 12)
    node_line = "      node " + "n" * (MAX_GRAPH_DIAGNOSTIC_LINE_BYTES - 11)
    assert len(declaration.encode()) == MAX_GRAPH_DIAGNOSTIC_LINE_BYTES
    assert len(graph_line.encode()) == graph_line_bytes
    assert len(node_line.encode()) == MAX_GRAPH_DIAGNOSTIC_LINE_BYTES
    help_text = PROFILE_HELP.replace(
        "  --cuda-graph-trace arg (=graph)\n"
        "      Select CUDA Graph activity granularity. Possible values are 'graph' and\n"
        "      'node'.\n",
        f"{declaration}\n{graph_line}\n{node_line}\n",
    ).replace("usage: nsys profile", "usage: --stop-on-range-end nsys profile")

    context = _failure_diagnostic(_validate(tmp_path, help_text.encode()))["cuda_graph_context"]

    assert context["available"] is available
    assert context["bytes"] == MAX_GRAPH_DIAGNOSTIC_BYTES + (0 if available else 1)
    if available:
        assert [line["text"] for line in context["token_lines"]] == [graph_line, node_line]
    else:
        assert context["reason"] == "context_exceeds_1536_byte_bound"
        assert "token_lines" not in context


def test_nsys_profile_help_failure_graph_escape_fallback_omits_available_lines(tmp_path: Path) -> None:
    graph_line = "      graph " + "\x01" * 500
    node_line = "      node " + "\x01" * 501
    assert len(graph_line.encode()) == MAX_GRAPH_DIAGNOSTIC_LINE_BYTES
    assert len(node_line.encode()) == MAX_GRAPH_DIAGNOSTIC_LINE_BYTES
    declaration = "  --cuda-graph-trace arg (=graph)"
    help_text = PROFILE_HELP.replace(
        "  --cuda-graph-trace arg (=graph)\n"
        "      Select CUDA Graph activity granularity. Possible values are 'graph' and\n"
        "      'node'.\n",
        f"{declaration}\n{graph_line}\n{node_line}\n",
    ).replace("usage: nsys profile", "usage: --stop-on-range-end nsys profile")

    result = _validate(tmp_path, help_text.encode())

    diagnostic = _failure_diagnostic(result)
    context = diagnostic["cuda_graph_context"]
    expected_context = f"{declaration}\n{graph_line}\n{node_line}"
    assert context["available"] is False
    assert context["reason"] == "omitted_to_fit_4096_character_bound"
    assert context["bytes"] == len(expected_context.encode())
    assert context["sha256"] == hashlib.sha256(expected_context.encode()).hexdigest()
    assert context["graph_occurrences"] == 1
    assert context["node_occurrences"] == 1
    assert "text" not in context["declaration"]
    assert "token_lines" not in context
    assert len(result.stderr) <= MAX_FAILURE_MESSAGE_CHARS
    assert "\\u0001" not in result.stderr
