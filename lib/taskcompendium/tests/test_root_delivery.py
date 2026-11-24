# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regressions for grading-time host delivery of hidden root-actor verifiers."""

import io
import os
import shutil
import stat
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

import taskcompendium.harbor.root_delivery as root_delivery
from taskcompendium.harbor.root_delivery import (
    MAX_DELIVERY_BYTES,
    QUIESCE_PROGRAM,
    DeliveredVerifier,
    delivery_commands,
    home_restore_commands,
    multiswe_run_command,
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


def test_delivery_commands_embed_and_execute_the_original_script(tmp_path, monkeypatch):
    """The delivery command writes the files and runs the entry script for real."""
    delivery_root = tmp_path / "tests"
    reward_path = tmp_path / "logs" / "verifier" / "reward.txt"
    monkeypatch.setattr(root_delivery, "DELIVERY_ROOT", str(delivery_root))
    monkeypatch.setattr(root_delivery, "REWARD_PATH", str(reward_path))
    verifier = DeliveredVerifier(
        files={
            "test.sh": (
                f"#!/bin/bash\npython3 -m pytest {delivery_root}/test_state.py -rA\necho 1 > {reward_path}\n".encode()
            ),
            "test_state.py": VERIFIER_STATE,
        }
    )
    (delivery,) = delivery_commands(verifier, cwd="/app")
    result = subprocess.run(delivery.argv, check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (delivery_root / "test.sh").read_bytes().startswith(b"#!/bin/bash")
    assert (delivery_root / "test_state.py").read_bytes() == VERIFIER_STATE
    assert stat.S_IMODE((delivery_root / "test.sh").stat().st_mode) & 0o111
    assert not stat.S_IMODE((delivery_root / "test_state.py").stat().st_mode) & 0o111
    # the entry script ran verbatim and wrote its own reward
    assert reward_path.read_text().strip() == "1"
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
@pytest.mark.skipif(sys.platform != "linux" or os.geteuid() != 0, reason="process quiesce requires Linux root")
@pytest.mark.skipif(shutil.which("unshare") is None, reason="PID namespace isolation requires unshare")
def test_quiesce_spares_the_machine_keeper_sleeper(tmp_path):
    # The local sandbox's PID 1 is bwrap and its keeper payload `sleep
    # 2147483647` is PID 2: a quiesce that killed it closed the machine
    # mid-grading (facet_v2's unscored attempts). Both quiesce variants must
    # kill strays while sparing that exact sleeper argv.
    script_path = tmp_path / "shell-quiesce.sh"
    script_path.write_text(root_delivery.SHELL_QUIESCE_SCRIPT)
    marker = tmp_path / "stray.txt"
    driver = (
        "import pathlib, subprocess, sys, time\n"
        "marker = pathlib.Path(sys.argv[1])\n"
        "keeper = subprocess.Popen(['sleep', '2147483647'])\n"
        "stray = subprocess.Popen([sys.executable, '-c',\n"
        "    \"import pathlib, time, sys\\npathlib.Path(sys.argv[1]).write_text('alive')\\n"
        'while pathlib.Path(sys.argv[1]).exists(): time.sleep(0.05)",\n'
        "    str(marker)])\n"
        "deadline = time.monotonic() + 10\n"
        "while not marker.exists() and time.monotonic() < deadline:\n"
        "    time.sleep(0.05)\n"
        "assert marker.exists(), 'stray never started'\n"
        "shell = subprocess.run(['bash', '-c', pathlib.Path(sys.argv[3]).read_text()],\n"
        "    capture_output=True, text=True, timeout=60)\n"
        "assert shell.returncode == 0, shell.stderr[-2000:]\n"
        "assert keeper.poll() is None, 'the shell quiesce killed the keeper sleeper'\n"
        "python_quiesce = subprocess.run([sys.executable, '-I', '-c', sys.argv[2]],\n"
        "    capture_output=True, text=True, timeout=60)\n"
        "assert python_quiesce.returncode == 0, python_quiesce.stderr[-2000:]\n"
        "assert keeper.poll() is None, 'the python quiesce killed the keeper sleeper'\n"
        "marker.unlink()\n"
        "assert stray.wait(timeout=30) != 0, 'the stray survived quiesce'\n"
        "print('keeper-spared')\n"
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
            str(marker),
            QUIESCE_PROGRAM,
            str(script_path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert "keeper-spared" in result.stdout, result.stderr[-2000:]


def test_quiesce_scripts_embed_the_keeper_sleeper_sentinel() -> None:
    """Both quiesce variants carry the exact keeper argv they must spare."""
    args = root_delivery.SANDBOX_KEEPER_SLEEP_ARGS
    assert repr(args) in root_delivery.QUIESCE_PROGRAM
    assert " ".join(args) in root_delivery.SHELL_QUIESCE_SCRIPT


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


def test_home_restore_commands_chunk_round_trip_exact_modes_and_mtime(tmp_path):
    """The archive restores verbatim under the home root; tar carries modes+mtimes."""
    home = tmp_path / "home"
    home.mkdir(parents=True)
    # Real capture archives carry flat /home members (fix-run.sh, test.patch, ...).
    member = home / "staged-fix-run.sh"
    member.write_text("#!/bin/bash\necho ran\n")
    member.chmod(0o755)
    old = 1_600_000_000
    os.utime(member, (old, old))
    staged_patch = home / "staged-test.patch"
    staged_patch.write_bytes(b"diff\n")
    staged_patch.chmod(0o644)
    os.utime(staged_patch, (old, old))
    archive_bytes = io.BytesIO()
    with tarfile.open(fileobj=archive_bytes, mode="w:gz") as bundle:
        for path, arcname in ((member, "fix-run.sh"), (staged_patch, "test.patch")):
            bundle.add(path, arcname=arcname, recursive=False)
    payload = archive_bytes.getvalue()
    assert len(payload) > 64  # force multiple chunks

    commands = home_restore_commands(payload, restore_root=str(home), chunk_bytes=64)
    assert len(commands) >= 3
    # A leftover staging file from an interrupted run must be truncated fresh.
    (home / ".capture.tar.gz").write_bytes(b"leftover")
    for command in commands:
        result = subprocess.run(command.argv, cwd="/", capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stderr[-2000:]
    restored = home / "fix-run.sh"
    assert restored.read_text() == "#!/bin/bash\necho ran\n"
    assert stat.S_IMODE(restored.stat().st_mode) == 0o755
    assert int(restored.stat().st_mtime) == old
    assert (home / "test.patch").read_bytes() == b"diff\n"
    assert not (home / ".capture.tar.gz").exists()


def test_multiswe_run_command_captures_streams_and_status_independently(tmp_path):
    """Prep nonzero fails; pipeline/cat streams and statuses persist separately."""
    home = tmp_path / "home"
    repo = home / "myrepo"
    repo.mkdir(parents=True)
    (home / "extract_multiswe_fix_patch.sh").write_text("#!/bin/bash\nprintf prep > prep.out\n")
    (home / "fix-run.sh").write_text("#!/bin/bash\necho pipeline-stdout-line\necho pipeline-stderr-line >&2\n")
    (home / "test_output.txt").write_text("combined tee output\n")
    command = multiswe_run_command(
        repo="myrepo",
        base="0" * 40,
        restore_root=str(home),
        test_output_path=str(home / "test_output.txt"),
        output_path=str(tmp_path / "cat-stdout.txt"),
        read_exit_code_path=str(tmp_path / "read-exit-code.txt"),
        pipeline_stdout_path=str(tmp_path / "pipeline-stdout.txt"),
        pipeline_stderr_path=str(tmp_path / "pipeline-stderr.txt"),
        pipeline_rc_path=str(tmp_path / "pipeline-rc.txt"),
        cat_rc_path=str(tmp_path / "cat-rc.txt"),
    )
    result = subprocess.run(command.argv, cwd="/", capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-2000:]
    # Upstream merges fix-run streams through 2>&1 | tee: the outer -c stdout is
    # the tee passthrough (merged content) and its stderr is the shell's own.
    merged = "pipeline-stdout-line\npipeline-stderr-line\n"
    assert (tmp_path / "pipeline-stdout.txt").read_text() == merged
    assert (tmp_path / "pipeline-stderr.txt").read_text() == ""
    assert (home / "test_output.txt").read_text() == merged
    assert (tmp_path / "cat-stdout.txt").read_text() == merged
    assert (tmp_path / "cat-rc.txt").read_text().strip() == "0"
    assert (tmp_path / "read-exit-code.txt").read_text().strip() == "0"
    # The pipeline exit is evidence only: nonzero must not fail the collection.
    (home / "fix-run.sh").write_text("#!/bin/bash\nexit 9\n")
    result = subprocess.run(command.argv, cwd="/", capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-2000:]
    assert (tmp_path / "pipeline-rc.txt").read_text().strip() == "9"
    # The failing fix-run produced no output: tee's test_output is empty and the
    # exact cat stdout is empty too — host-side selection owns the fallback.
    assert (tmp_path / "cat-stdout.txt").read_text() == ""
    assert (tmp_path / "cat-rc.txt").read_text().strip() == "0"


def test_multiswe_prep_nonzero_fails_the_collection(tmp_path):
    home = tmp_path / "home"
    repo = home / "myrepo"
    repo.mkdir(parents=True)
    (home / "extract_multiswe_fix_patch.sh").write_text("#!/bin/bash\nexit 4\n")
    command = multiswe_run_command(
        repo="myrepo",
        base="0" * 40,
        restore_root=str(home),
        test_output_path=str(home / "test_output.txt"),
        output_path=str(tmp_path / "cat-stdout.txt"),
        read_exit_code_path=str(tmp_path / "read-exit-code.txt"),
        pipeline_stdout_path=str(tmp_path / "pipeline-stdout.txt"),
        pipeline_stderr_path=str(tmp_path / "pipeline-stderr.txt"),
        pipeline_rc_path=str(tmp_path / "pipeline-rc.txt"),
        cat_rc_path=str(tmp_path / "cat-rc.txt"),
    )
    result = subprocess.run(command.argv, cwd="/", capture_output=True, text=True, timeout=120)
    assert result.returncode != 0, "prep nonzero must fail the collection"
    assert not (tmp_path / "cat-stdout.txt").exists()


def test_multiswe_run_command_guards(tmp_path):
    with pytest.raises(ValueError, match="repo and base"):
        multiswe_run_command(
            repo="",
            base="",
            output_path="/o",
            read_exit_code_path="/e",
            pipeline_stdout_path="/p",
            pipeline_stderr_path="/s",
            pipeline_rc_path="/r",
            cat_rc_path="/c",
        )
    with pytest.raises(ValueError, match="absolute"):
        multiswe_run_command(
            repo="myrepo",
            base="0" * 40,
            output_path="relative",
            read_exit_code_path="/e",
            pipeline_stdout_path="/p",
            pipeline_stderr_path="/s",
            pipeline_rc_path="/r",
            cat_rc_path="/c",
        )


def test_home_restore_replaces_destinations_that_became_directories(tmp_path):
    """The source removes each member destination (rm -rf) before extracting."""
    home = tmp_path / "home"
    member = home / "fix-run.sh"
    home.mkdir(parents=True)
    # The actor replaced the fix-run.sh destination with a directory during work.
    (member / "junk").mkdir(parents=True)
    (member / "junk" / "nested.txt").write_text("actor junk\n")
    archive_bytes = io.BytesIO()
    with tarfile.open(fileobj=archive_bytes, mode="w:gz") as bundle:
        info = tarfile.TarInfo("fix-run.sh")
        content = b"#!/bin/bash\necho ran\n"
        info.size = len(content)
        info.mode = 0o755
        bundle.addfile(info, io.BytesIO(content))
    commands = home_restore_commands(archive_bytes.getvalue(), restore_root=str(home))
    for command in commands:
        result = subprocess.run(command.argv, cwd="/", capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stderr[-2000:]
    restored = home / "fix-run.sh"
    assert restored.is_file()
    assert restored.read_bytes() == content
    assert not (home / "fix-run.sh" / "junk").exists()


def test_home_restore_rejects_unexpected_member_paths(tmp_path):
    """A member with a subdirectory path fails the restore (source semantics)."""
    home = tmp_path / "home"
    home.mkdir(parents=True)
    archive_bytes = io.BytesIO()
    with tarfile.open(fileobj=archive_bytes, mode="w:gz") as bundle:
        info = tarfile.TarInfo("sub/escaped.sh")
        content = b"escaped\n"
        info.size = len(content)
        bundle.addfile(info, io.BytesIO(content))
    commands = home_restore_commands(archive_bytes.getvalue(), restore_root=str(home))
    for command in commands[:-1]:
        result = subprocess.run(command.argv, cwd="/", capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stderr[-2000:]
    result = subprocess.run(commands[-1].argv, cwd="/", capture_output=True, text=True, timeout=120)
    assert result.returncode != 0, "unexpected member paths must fail the restore"


@pytest.mark.manual
@pytest.mark.skipif(sys.platform != "linux" or os.geteuid() != 0, reason="process quiesce requires Linux root")
@pytest.mark.skipif(shutil.which("unshare") is None, reason="PID namespace isolation requires unshare")
def test_shell_quiesce_kills_strays_inside_its_own_pid_namespace(tmp_path):
    driver = (
        "import pathlib, subprocess, sys, time\n"
        "marker = pathlib.Path(sys.argv[1])\n"
        "stray = subprocess.Popen([sys.executable, '-c',\n"
        "    'import pathlib, time, sys\\n"
        'pathlib.Path(sys.argv[1]).write_text("alive")\\n'
        "while pathlib.Path(sys.argv[1]).exists(): time.sleep(0.05)',\n"
        "    str(marker)])\n"
        "deadline = time.monotonic() + 10\n"
        "while not marker.exists() and time.monotonic() < deadline:\n"
        "    time.sleep(0.05)\n"
        "assert marker.exists(), 'stray never started'\n"
        "result = subprocess.run(['bash', '-c', sys.argv[2]], capture_output=True, text=True, timeout=60)\n"
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
            root_delivery.SHELL_QUIESCE_SCRIPT,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert "quiesce-ok" in result.stdout, result.stderr[-2000:]


def test_shell_quiesce_treats_vanished_victims_as_benign():
    """ESRCH between the /proc scan and the kill is a benign race, not a failure."""
    # Drive the script's own kill branch with a victim that vanished: the
    # elif existence re-check must continue (benign) instead of exiting 1.
    branch = (
        "set -eu\n"
        "killed=0\n"
        "for victim in 999999 999998; do\n"
        '  if kill -9 "$victim" 2>/dev/null; then\n'
        "    killed=$((killed + 1))\n"
        '  elif [ ! -d "/proc/$victim" ]; then\n'
        "    continue\n"
        "  else\n"
        '    echo "quiesce kill failed for $victim" >&2\n'
        "    exit 1\n"
        "  fi\n"
        "done\n"
        'echo "done $killed"\n'
    )
    result = subprocess.run(["bash", "-c", branch], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr[-500:]
    assert result.stdout.strip() == "done 0"
