"""scripts/submit.sh forwards the synthesis worker's knobs (integration review 2026-09-29).

The forwarding block is executed on its own, exactly as submit.sh runs it, with a
stub ``die`` and an empty ``IRIS_CMD``.
"""

import os
import re
import subprocess
from pathlib import Path

import pytest

from capability_pipeline.conveyor import DEFAULT_WAIT_BUDGETS

ROOT = Path(__file__).resolve().parents[1]
SUBMIT = ROOT / "scripts/submit.sh"


def _block() -> str:
    text = SUBMIT.read_text()
    start = text.index("  # BEGIN worker knob forwarding")
    end = text.index("  # END worker knob forwarding")
    return text[start:end]


def _forward(env: dict[str, str], phase: str = "synthesize") -> subprocess.CompletedProcess:
    script = (
        "set -euo pipefail\n"
        "die() { printf 'error: %s\\n' \"$*\" >&2; exit 2; }\n"
        f"PHASE={phase}\nIRIS_CMD=()\n"
        + _block()
        + '\nif [ "${#IRIS_CMD[@]}" -gt 0 ]; then printf "%s\\n" "${IRIS_CMD[@]}"; fi\n'
    )
    clean = {key: value for key, value in os.environ.items() if not key.startswith("CAPABILITY_")}
    # /bin/bash is the macOS system bash 3.2; the block must not need bash 4.
    return subprocess.run(["/bin/bash", "-c", script], env={**clean, **env}, capture_output=True, text=True,
                          check=False)


def _pairs(stdout: str) -> dict[str, str]:
    lines = stdout.splitlines()
    assert len(lines) % 3 == 0 and all(flag == "-e" for flag in lines[0::3]), lines
    return dict(zip(lines[1::3], lines[2::3], strict=True))


def test_submit_script_parses():
    subprocess.run(["bash", "-n", str(SUBMIT)], check=True)
    subprocess.run(["/bin/bash", "-n", str(SUBMIT)], check=True)


def test_worker_knobs_are_forwarded_with_the_types_their_consumers_accept():
    env = {
        "CAPABILITY_PUBLICATION_QUEUE": "s3://marin-us-east-02a/users/x/publication",
        "CAPABILITY_IMAGE_CAPTURE_MAX_ATTEMPTS": "8",
        "CAPABILITY_IMAGE_COMMAND_TIMEOUT_SECONDS": "7200.5",
        "CAPABILITY_VERIFIER_PROVISIONING_SECONDS": "1200",
        "CAPABILITY_INFRA_PROBE_TTL_SECONDS": "300.25",
        "CAPABILITY_RUNTIME_INFRA_MAX_RETRIES": "4",
        "CAPABILITY_WAIT_RUNTIME_INFRASTRUCTURE_SECONDS": "43200.5",
        "CAPABILITY_WAIT_RUNTIME_INFRASTRUCTURE_ATTEMPTS": "16",
        "CAPABILITY_WAIT_RUNTIME_INFRASTRUCTURE_BACKOFF_SECONDS": ".5",
        "CAPABILITY_WAIT_IMAGE_REVIEW_TRANSPORT_ATTEMPTS": "12",
        "CAPABILITY_WAIT_IMAGE_REVIEW_BACKOFF_SECONDS": "300",
        "CAPABILITY_CONVEYOR_HEARTBEAT_SECONDS": "0.5",
        "CAPABILITY_CONVEYOR_MAX_CALLS_PER_ITEM": "400",
    }
    result = _forward(env)
    assert result.returncode == 0, result.stderr
    assert _pairs(result.stdout) == env
    assert result.stderr == ""


def test_unknown_conveyor_and_wait_variables_warn_and_are_skipped():
    result = _forward({
        "CAPABILITY_CONVEYOR_TYPO_SECONDS": "5",
        "CAPABILITY_WAIT_NOT_A_KIND_ATTEMPTS": "3",
        "CAPABILITY_WAIT_BUILDER_PROCESS_ATTEMPTS": "12",
    })
    assert result.returncode == 0, result.stderr
    assert _pairs(result.stdout) == {"CAPABILITY_WAIT_BUILDER_PROCESS_ATTEMPTS": "12"}
    assert "CAPABILITY_CONVEYOR_TYPO_SECONDS is not a knob" in result.stderr
    assert "CAPABILITY_WAIT_NOT_A_KIND_ATTEMPTS is not a knob" in result.stderr


@pytest.mark.parametrize("name,value", [
    ("CAPABILITY_WAIT_IMAGE_CAPTURE_ATTEMPTS", "2.5"),
    ("CAPABILITY_RUNTIME_INFRA_MAX_RETRIES", "four"),
    ("CAPABILITY_IMAGE_COMMAND_TIMEOUT_SECONDS", "-1"),
    ("CAPABILITY_CONVEYOR_HEARTBEAT_SECONDS", "1e3"),
])
def test_malformed_values_stop_the_submission(name, value):
    result = _forward({name: value})
    assert result.returncode == 2
    assert name in result.stderr


def test_other_phases_forward_nothing():
    result = _forward({"CAPABILITY_IMAGE_CAPTURE_MAX_ATTEMPTS": "8"}, phase="evaluate")
    assert result.returncode == 0 and result.stdout == ""


def test_wait_kind_list_matches_the_conveyor():
    kinds = re.search(r'WAIT_KINDS=" ([A-Z_ ]+) "', _block()).group(1).split()
    assert kinds == [kind.upper() for kind in DEFAULT_WAIT_BUDGETS]
