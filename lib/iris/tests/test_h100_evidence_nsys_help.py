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
      are 'none', 'stop', 'stop-shutdown', 'repeat[:N][:mode]', and
      'repeat-shutdown:N[:mode]'. If 'stop', collection stops and the target
      application continues running.
  --cuda-graph-trace arg (=graph)
      Select CUDA Graph activity granularity. Possible values are 'graph' and
      'node'.
  --cuda-backtrace arg (=false)
      Collect CUDA backtraces.
"""
FAILURE_DIAGNOSTIC_MARKER = " diagnostic="
MAX_FAILURE_MESSAGE_CHARS = 4096
MAX_OPTION_BLOCK_BYTES = 1024


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


def _profile_help_with_capture_block_bytes(size: int) -> tuple[str, str]:
    capture = (
        "  --capture-range-end arg (=stop-shutdown)\n"
        "      Possible values are 'none', 'stop', 'stop-shutdown', 'repeat[:N][:mode]', and\n"
        "      'repeat-shutdown:N[:mode]'."
    )
    padding = size - len(capture.encode()) - 1
    assert padding >= 0
    capture = f"{capture}\n{'x' * padding}"
    assert len(capture.encode()) == size
    graph = "  --cuda-graph-trace arg (=graph)\n" "      Possible values are 'graph' and 'node'."
    return (
        f"usage: nsys profile\n--stop-on-range-end arg\n{capture}\n{graph}\n  --next-option arg\n",
        capture,
    )


def test_nsys_profile_help_accepts_unique_stop_policy(tmp_path: Path) -> None:
    result = _validate(tmp_path, PROFILE_HELP.encode())

    assert result.returncode == 0, result.stderr
    assert "capture-range-end=none,stop,stop-shutdown,repeat[:N][:mode],repeat-shutdown:N[:mode]" in result.stdout
    assert "cuda-graph-trace=graph,node" in result.stdout


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
            "'repeat-shutdown:N[:mode]'",
            "'repeat-shutdown:N[:mode]', and future",
        ),
        PROFILE_HELP.replace(
            "'repeat-shutdown:N[:mode]'",
            "'repeat-shutdown:N[:mode]', and 'future'",
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


def test_nsys_profile_help_failure_emits_only_exact_bounded_option_blocks(
    tmp_path: Path,
) -> None:
    help_text = PROFILE_HELP.replace(
        "usage: nsys profile [<args>] [application] [<application args>]",
        "usage: nsys profile PRIVATE_OUTSIDE_BLOCK",
    ).replace("'graph' and\n      'node'", "'graph', 'node', and 'future'")

    result = _validate(tmp_path, help_text.encode())

    diagnostic = _failure_diagnostic(result)
    assert result.stderr.startswith(
        "nsys profile help validation failed: --cuda-graph-trace does not expose exactly graph and node"
    )
    assert diagnostic["schema"] == "iris.h100_evidence_nsys_help_failure.v1"
    capture = diagnostic["blocks"]["capture-range-end"]
    graph = diagnostic["blocks"]["cuda-graph-trace"]
    assert capture["available"] is True
    assert graph["available"] is True
    assert capture["bytes"] == len(capture["text"].encode())
    assert capture["sha256"] == hashlib.sha256(capture["text"].encode()).hexdigest()
    assert graph["bytes"] == len(graph["text"].encode())
    assert graph["sha256"] == hashlib.sha256(graph["text"].encode()).hexdigest()
    assert "--capture-range-end" in capture["text"]
    assert "--cuda-graph-trace" in graph["text"]
    assert "PRIVATE_OUTSIDE_BLOCK" not in result.stderr
    assert "Collect CUDA backtraces" not in result.stderr
    assert len(result.stderr) <= MAX_FAILURE_MESSAGE_CHARS


def test_nsys_profile_help_failure_marks_missing_option_block_unavailable(tmp_path: Path) -> None:
    help_text = PROFILE_HELP.replace(
        "  --cuda-graph-trace arg (=graph)\n"
        "      Select CUDA Graph activity granularity. Possible values are 'graph' and\n"
        "      'node'.\n",
        "",
    )

    diagnostic = _failure_diagnostic(_validate(tmp_path, help_text.encode()))

    graph = diagnostic["blocks"]["cuda-graph-trace"]
    assert graph == {
        "available": False,
        "bytes": 0,
        "reason": "exact_option_block_unavailable",
        "sha256": None,
    }


@pytest.mark.parametrize(
    ("size", "available"),
    ((MAX_OPTION_BLOCK_BYTES, True), (MAX_OPTION_BLOCK_BYTES + 1, False)),
)
def test_nsys_profile_help_failure_option_block_boundary_is_fail_closed(
    tmp_path: Path, size: int, available: bool
) -> None:
    help_text, capture_text = _profile_help_with_capture_block_bytes(size)

    diagnostic = _failure_diagnostic(_validate(tmp_path, help_text.encode()))

    capture = diagnostic["blocks"]["capture-range-end"]
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
    help_text = PROFILE_HELP.replace(
        "      application continues running.\n",
        f"      application continues running. {control_text}\n",
    ).replace(
        "      'node'.\n",
        f"      'node'. {control_text}\n",
    )
    help_text = help_text.replace("usage: nsys profile", "usage: --stop-on-range-end nsys profile")

    result = _validate(tmp_path, help_text.encode())

    diagnostic = _failure_diagnostic(result)
    assert diagnostic["blocks"]["capture-range-end"]["available"] is False
    assert diagnostic["blocks"]["cuda-graph-trace"]["available"] is False
    assert diagnostic["blocks"]["capture-range-end"]["reason"] == "omitted_to_fit_4096_character_bound"
    assert diagnostic["blocks"]["cuda-graph-trace"]["reason"] == "omitted_to_fit_4096_character_bound"
    assert len(result.stderr) <= MAX_FAILURE_MESSAGE_CHARS
    assert "\\u0001" not in result.stderr
