# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regressions for grading-time host delivery of hidden root-actor verifiers."""

import base64
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import taskcompendium.harbor.root_delivery as root_delivery
from taskcompendium.harbor.root_delivery import (
    MAX_DELIVERY_BYTES,
    QUIESCE_PROGRAM,
    DeliveredVerifier,
    delivery_commands,
    quiesce_command,
)
from taskcompendium.models import VerifierCommand

VERIFIER_TEST = b"#!/bin/bash\npython3 -m pytest /tests/test_state.py -rA\necho 1 > /logs/verifier/reward.txt\n"
VERIFIER_STATE = b"def test_state():\n    assert True\n"


def delivered() -> DeliveredVerifier:
    return DeliveredVerifier(files={"test.sh": VERIFIER_TEST, "test_state.py": VERIFIER_STATE})


def test_quiesce_command_runs_isolated_python() -> None:
    command = quiesce_command()
    assert isinstance(command, VerifierCommand)
    assert command.argv[:3] == ("python3", "-I", "-c")
    assert command.cwd == "/"


def test_delivery_commands_embed_and_execute_the_original_script() -> None:
    (delivery,) = delivery_commands(delivered(), cwd="/app")
    script = delivery.argv[-1]
    assert base64.b64encode(VERIFIER_TEST).decode() in script
    assert base64.b64encode(VERIFIER_STATE).decode() in script
    assert "chmod 755 /tests/test.sh" in script
    assert "chmod 644 /tests/test_state.py" in script
    assert "bash /tests/test.sh" in script
    assert delivery.cwd == "/app"


def test_delivery_requires_the_entry_script() -> None:
    with pytest.raises(ValueError, match="entry script"):
        delivery_commands(DeliveredVerifier(files={"other.sh": b"true\n"}))
    with pytest.raises(ValueError, match="executable"):
        delivery_commands(delivered(), entrypoint="test_state.py")


def test_delivery_rejects_oversized_or_escaping_payloads() -> None:
    with pytest.raises(ValueError, match="argv delivery bound"):
        DeliveredVerifier(files={"test.sh": b"x" * (MAX_DELIVERY_BYTES + 1)})
    with pytest.raises(ValueError, match="escapes"):
        DeliveredVerifier(files={"test.sh": b"true\n", "../escape": b"y"})
    with pytest.raises(ValueError, match="empty"):
        DeliveredVerifier(files={})


@pytest.mark.manual
@pytest.mark.skipif(sys.platform != "linux" or os.geteuid() != 0, reason="process quiesce requires Linux root")
@pytest.mark.skipif(shutil.which("unshare") is None, reason="PID namespace isolation requires unshare")
def test_quiesce_kills_strays_inside_its_own_pid_namespace(tmp_path):
    # The quiesce kills every process outside its ancestry, so the probe runs
    # inside a fresh PID namespace: the stray dies, the namespace's own tree and
    # its init survive, and nothing outside the namespace is touched.
    driver = (
        "import pathlib, subprocess, sys, time\n"
        "marker = pathlib.Path(sys.argv[1])\n"
        "stray = subprocess.Popen([sys.executable, '-c',\n"
        "    \"import pathlib, time, sys\\npathlib.Path(sys.argv[1]).write_text('alive')\\n"
        'while pathlib.Path(sys.argv[1]).exists(): time.sleep(0.05)",\n'
        "    str(marker)])\n"
        "deadline = time.monotonic() + 10\n"
        "while not marker.exists() and time.monotonic() < deadline:\n"
        "    time.sleep(0.05)\n"
        "assert marker.exists(), 'stray never started'\n"
        "result = subprocess.run([sys.executable, '-I', '-c', sys.argv[2]],\n"
        "    capture_output=True, text=True, timeout=60)\n"
        "assert result.returncode == 0, result.stderr[-2000:]\n"
        "int(result.stdout.strip())\n"
        "marker.unlink()\n"
        "assert stray.wait(timeout=30) != 0, 'the stray survived quiesce'\n"
        "print('quiesce-ok')\n"
    )
    result = subprocess.run(
        [
            "unshare",
            "--pid",
            "--fork",
            "--mount-proc",
            sys.executable,
            "-I",
            "-c",
            driver,
            str(tmp_path / "stray.txt"),
            QUIESCE_PROGRAM,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert "quiesce-ok" in result.stdout, result.stderr[-2000:]


@pytest.mark.manual
@pytest.mark.skipif(sys.platform != "linux" or os.geteuid() != 0, reason="delivery requires Linux root")
def test_delivery_commands_recreate_and_run_the_original_verifier(tmp_path):
    # The delivery writes into the production /tests path; the fixture entry
    # script exports the delivered bytes and its own reward for assertion.
    root = Path("/tmp/root-delivery-fixture")
    if root.exists():
        shutil.rmtree(root)
    (root / "tests").mkdir(parents=True)
    fixture = DeliveredVerifier(
        files={
            "test.sh": (
                b"#!/bin/bash\n"
                b"cp /tests/test_state.py /tmp/root-delivery-fixture/tests/test_state.py\n"
                b"stat -c '%a' /tests/test.sh > /tmp/root-delivery-fixture/mode.txt\n"
                b"echo 1 > /tmp/root-delivery-fixture/reward.txt\n"
            ),
            "test_state.py": VERIFIER_STATE,
        }
    )
    (delivery,) = delivery_commands(fixture)
    result = subprocess.run(delivery.argv, cwd="/", capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-2000:]
    assert (root / "tests/test_state.py").read_bytes() == VERIFIER_STATE
    assert (root / "mode.txt").read_text().strip() == "755"
    assert (root / "reward.txt").read_text().strip() == "1"
    shutil.rmtree(root)


def test_delivery_output_path_redirects_the_entry_script_streams() -> None:
    (delivery,) = delivery_commands(delivered(), output_path="/logs/verifier/test-output.txt", cwd="/app")
    script = delivery.argv[-1]
    assert "bash /tests/test.sh > /logs/verifier/test-output.txt 2>&1" in script
    assert "mkdir -p /logs/verifier" in script
    assert delivery.cwd == "/app"


def test_delivery_output_path_must_be_absolute() -> None:
    with pytest.raises(ValueError, match="absolute"):
        delivery_commands(delivered(), output_path="relative/output.txt")


def test_native_reward_mode_persists_exit_code_and_lets_collect_finish(tmp_path, monkeypatch) -> None:
    """A failing source run's trap-written reward still reaches the reward parser."""
    fixture = Path("/tmp/root-delivery-native")
    if fixture.exists():
        shutil.rmtree(fixture)
    (fixture / "logs" / "verifier").mkdir(parents=True)
    # This host mounts /tests read-only; point the delivery root at the fixture.
    monkeypatch.setattr(root_delivery, "DELIVERY_ROOT", "/tmp/root-delivery-native/tests")
    monkeypatch.setattr(root_delivery, "REWARD_PATH", "/tmp/root-delivery-native/logs/verifier/reward.txt")
    # The source script fails (pytest failures) but its trap writes reward 0.
    source = DeliveredVerifier(
        files={
            "test.sh": b"#!/bin/bash\n" b"echo 0 > /tmp/root-delivery-native/logs/verifier/reward.txt\n" b"exit 3\n",
            "test_state.py": VERIFIER_STATE,
        }
    )
    (delivery,) = delivery_commands(
        source,
        output_path="/tmp/root-delivery-native/logs/verifier/test-output.txt",
        exit_code_path="/tmp/root-delivery-native/logs/verifier/exit-code.txt",
    )
    result = subprocess.run(delivery.argv, cwd="/", capture_output=True, text=True, timeout=120)
    # The collect command finishes 0 even though the source run exited 3.
    assert result.returncode == 0, result.stderr[-2000:]
    assert (fixture / "logs/verifier/exit-code.txt").read_text().strip() == "3"
    assert (fixture / "logs/verifier/reward.txt").read_text().strip() == "0"
    assert (fixture / "logs/verifier/test-output.txt").exists()

    # A missing reward still surfaces through the caller's reward parsing; the
    # helper itself only guarantees the exit code and collect completion.
    shutil.rmtree(fixture)


def test_native_reward_mode_paths_must_be_absolute() -> None:
    with pytest.raises(ValueError, match="exit-code path"):
        delivery_commands(delivered(), exit_code_path="relative/exit.txt")
