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
GRAPH_DECLARATION = "--cuda-graph-trace=<granularity>[:<launch origin>]"
PROFILE_HELP = f"""\
usage: nsys profile [<args>] [application] [<application args>]

  -c, --capture-range arg (=none)
      Start collection at the selected capture range.
  --capture-range-end arg (=stop-shutdown)
      Specify the desired behavior when a capture range ends. Possible values
      are 'none', 'stop', 'stop-shutdown', 'repeat[:N][:mode]' or
      'repeat-shutdown:N'[:mode]. If 'stop', collection stops and the target
      application continues running.
\t{GRAPH_DECLARATION}
      PRIVATE_GRAPH_PROSE graph node graph node graph node.
      More PRIVATE_GRAPH_PROSE node graph node graph node node.
  --cuda-backtrace arg (=false)
      Collect CUDA backtraces.
"""
FAILURE_DIAGNOSTIC_MARKER = " diagnostic="
MAX_FAILURE_MESSAGE_CHARS = 4096
MAX_POSSIBLE_VALUES_CLAUSE_BYTES = 1024


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
    return (
        f"usage: nsys profile\n--stop-on-range-end arg\n{capture}\n  {GRAPH_DECLARATION}\n  --next-option arg\n",
        clause,
    )


def test_nsys_profile_help_accepts_capture_policy_and_exact_graph_declaration(tmp_path: Path) -> None:
    result = _validate(tmp_path, PROFILE_HELP.encode())

    assert result.returncode == 0, result.stderr
    assert "capture-range-end=none,stop,stop-shutdown,repeat[:N][:mode],repeat-shutdown:N[:mode]" in result.stdout
    assert f"cuda-graph-trace-declaration={GRAPH_DECLARATION}" in result.stdout


@pytest.mark.parametrize("indentation", ("", "  ", "\t", "\t  "), ids=("none", "spaces", "tab", "mixed"))
def test_nsys_profile_help_normalizes_graph_declaration_outer_whitespace(
    tmp_path: Path,
    indentation: str,
) -> None:
    help_text = PROFILE_HELP.replace(f"\t{GRAPH_DECLARATION}", f"{indentation}{GRAPH_DECLARATION}   ")

    result = _validate(tmp_path, help_text.encode())

    assert result.returncode == 0, result.stderr
    assert f"cuda-graph-trace-declaration={GRAPH_DECLARATION}" in result.stdout


@pytest.mark.parametrize(
    "declaration",
    (
        "--cuda-graph-trace arg (=graph)",
        "--cuda-graph-trace=<granularity>",
        "--cuda-graph-trace=<granularity>[:<launch-origin>]",
        "--cuda-graph-trace =<granularity>[:<launch origin>]",
        "--cuda-graph-trace=<granularity>[:<launch origin>] extra",
        "--CUDA-GRAPH-TRACE=<granularity>[:<launch origin>]",
    ),
    ids=("old-arg-form", "missing-origin", "hyphenated-origin", "space-before-equals", "trailing-text", "uppercase"),
)
def test_nsys_profile_help_rejects_graph_declaration_variants(tmp_path: Path, declaration: str) -> None:
    help_text = PROFILE_HELP.replace(GRAPH_DECLARATION, declaration)

    result = _validate(tmp_path, help_text.encode())

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.startswith("nsys profile help validation failed:")
    assert "--cuda-graph-trace" in result.stderr


@pytest.mark.parametrize("mutation", ("missing", "duplicate"))
def test_nsys_profile_help_rejects_nonunique_graph_declaration(tmp_path: Path, mutation: str) -> None:
    declaration = f"\t{GRAPH_DECLARATION}\n"
    replacement = "" if mutation == "missing" else declaration * 2
    help_text = PROFILE_HELP.replace(declaration, replacement)

    result = _validate(tmp_path, help_text.encode())

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.startswith(
        "nsys profile help validation failed: help must contain one exact --cuda-graph-trace option declaration"
    )


def test_nsys_profile_help_ignores_unproven_graph_prose(tmp_path: Path) -> None:
    help_text = PROFILE_HELP.replace(
        "      PRIVATE_GRAPH_PROSE graph node graph node graph node.\n"
        "      More PRIVATE_GRAPH_PROSE node graph node graph node node.\n",
        "      PRIVATE_GRAPH_PROSE future syntax with no value list.\n",
    )

    result = _validate(tmp_path, help_text.encode())

    assert result.returncode == 0, result.stderr
    assert f"cuda-graph-trace-declaration={GRAPH_DECLARATION}" in result.stdout


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
def test_nsys_profile_help_rejects_missing_or_ambiguous_capture_policy(tmp_path: Path, help_text: str) -> None:
    result = _validate(tmp_path, help_text.encode())

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.startswith("nsys profile help validation failed:")


@pytest.mark.parametrize("payload", (b"", b"usage\x00help", b"\xff\xfe"), ids=("empty", "nul", "non-utf8"))
def test_nsys_profile_help_rejects_malformed_artifacts(tmp_path: Path, payload: bytes) -> None:
    result = _validate(tmp_path, payload)

    assert result.returncode == 1
    assert result.stdout == ""


def test_nsys_profile_help_rejects_oversized_artifact_before_reading_it(tmp_path: Path) -> None:
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


def test_nsys_profile_help_failure_emits_only_capture_clause(tmp_path: Path) -> None:
    help_text = PROFILE_HELP.replace(
        "usage: nsys profile [<args>] [application] [<application args>]",
        "usage: --stop-on-range-end nsys profile PRIVATE_OUTSIDE_CLAUSES",
    )

    result = _validate(tmp_path, help_text.encode())

    diagnostic = _failure_diagnostic(result)
    assert diagnostic["schema"] == "iris.h100_evidence_nsys_help_failure.v4"
    assert set(diagnostic["clauses"]) == {"capture-range-end"}
    capture = diagnostic["clauses"]["capture-range-end"]
    assert capture["available"] is True
    assert capture["bytes"] == len(capture["text"].encode())
    assert capture["sha256"] == hashlib.sha256(capture["text"].encode()).hexdigest()
    assert "cuda-graph-trace" not in json.dumps(diagnostic)
    assert "PRIVATE_GRAPH_PROSE" not in result.stderr
    assert "PRIVATE_OUTSIDE_CLAUSES" not in result.stderr
    assert len(result.stderr) <= MAX_FAILURE_MESSAGE_CHARS


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


def test_nsys_profile_help_failure_escape_expansion_remains_within_total_bound(tmp_path: Path) -> None:
    control_text = "\x01" * 700
    capture_clause = (
        "Possible values\n"
        "      are 'none', 'stop', 'stop-shutdown', 'repeat[:N][:mode]' or\n"
        f"      'repeat-shutdown:N'[:mode] and {control_text}."
    )
    help_text = PROFILE_HELP.replace(
        "'repeat-shutdown:N'[:mode].",
        f"'repeat-shutdown:N'[:mode] and {control_text}.",
    )

    result = _validate(tmp_path, help_text.encode())

    diagnostic = _failure_diagnostic(result)
    capture = diagnostic["clauses"]["capture-range-end"]
    assert capture["available"] is False
    assert capture["reason"] == "omitted_to_fit_4096_character_bound"
    assert capture["bytes"] == len(capture_clause.encode())
    assert capture["sha256"] == hashlib.sha256(capture_clause.encode()).hexdigest()
    assert "text" not in capture
    assert len(result.stderr) <= MAX_FAILURE_MESSAGE_CHARS
    assert "\\u0001" not in result.stderr
