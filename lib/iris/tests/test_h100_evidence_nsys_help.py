# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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


def _validate(tmp_path: Path, payload: bytes) -> subprocess.CompletedProcess[str]:
    artifact = tmp_path / "profile-help.txt"
    artifact.write_bytes(payload)
    return subprocess.run(
        (sys.executable, str(VALIDATOR), str(artifact)),
        capture_output=True,
        check=False,
        text=True,
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
        PROFILE_HELP + "  --stop-on-range-end arg (=true)\n",
    ),
    ids=("missing", "duplicate-option", "missing-stop", "duplicate-stop", "duplicate-value-list", "obsolete-option"),
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
        PROFILE_HELP.replace("Possible values are 'graph' and\n      'node'.", "Possible values are 'graph'."),
        PROFILE_HELP.replace(
            "Possible values are 'graph' and\n      'node'.", "Possible values are 'graph', 'node', and 'node'."
        ),
        PROFILE_HELP.replace(
            "Possible values are 'graph' and\n      'node'.", "Possible values are 'graph', 'node', and 'future'."
        ),
    ),
    ids=("missing", "duplicate-option", "missing-node", "duplicate-node", "unknown-value"),
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
