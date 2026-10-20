"""Canonical code graders must not leave a candidate's descendants running."""

import json
import os
from pathlib import Path
import selectors
import shlex
import signal
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[2]
MODULES = (
    "eval.chat_benchmarks.HumanEval.human_eval.execution",
    "eval.chat_benchmarks.HumanEvalPlus.human_eval_plus.execution",
    "eval.chat_benchmarks.MBPPPlus.mbpp_plus.execution",
    "eval.chat_benchmarks.MultiPLE.multiple.execution",
)
DRIVER = """
import importlib
import json
import sys

module = importlib.import_module(sys.argv[1])
check = module.check_code_correctness if 'MultiPLE' in sys.argv[1] else module.check_correctness
try:
    result = check('cleanup/child', {'test_code': sys.argv[2]}, 'sh', float(sys.argv[3]), sys.argv[4], 0)
except KeyboardInterrupt:
    print(json.dumps({'cancelled': True}), flush=True)
else:
    print(json.dumps(result), flush=True)
"""
NESTED_DRIVER = """
import importlib
import json
import multiprocessing
import sys

from eval.runtime.program_process import run_program_process, start_program_session

def execute():
    start_program_session()
    if sys.argv[1] == 'safe_subprocess':
        from eval.chat_benchmarks.MultiPLE.multiple.safe_subprocess import run
        run(['/bin/bash', '-c', sys.argv[2]], timeout_seconds=60)
    else:
        from eval.chat_benchmarks.MultiPLE.multiple.libeval import run_without_exn
        run_without_exn(['/bin/bash', '-c', sys.argv[2]])

process = multiprocessing.get_context('fork').Process(target=execute)
run_program_process(process, timeout=1)
print(json.dumps({'finished': True, 'exitcode': process.exitcode}), flush=True)
"""

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Uses Linux process state to distinguish exited zombies")


def process_running(pid):
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[0]
    except FileNotFoundError:
        return False
    return state != "Z"


@pytest.mark.parametrize("module", MODULES)
@pytest.mark.parametrize("finish", ("return", "fail", "timeout", "cancel"))
def test_canonical_grader_reaps_candidate_descendants(tmp_path, module, finish):
    pid_file = tmp_path / "child.pid"
    ready = tmp_path / "ready"
    os.mkfifo(ready)
    candidate = f"sleep 60 >/dev/null 2>&1 &\necho $! > {shlex.quote(str(pid_file))}\n"
    if finish == "cancel":
        candidate += f"echo ready > {shlex.quote(str(ready))}\n"
    if finish == "fail":
        candidate += "exit 1\n"
    elif finish != "return":
        candidate += "wait\n"
    timeout = 30 if finish == "cancel" else 0.25
    process = subprocess.Popen(
        [sys.executable, "-c", DRIVER, module, candidate, str(timeout), str(tmp_path)],
        cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True,
    )
    try:
        if finish == "cancel":
            with open(ready, "rb", buffering=0, opener=lambda path, flags: os.open(path, flags | os.O_NONBLOCK)) as stream:
                with selectors.DefaultSelector() as selector:
                    selector.register(stream, selectors.EVENT_READ)
                    assert selector.select(timeout=10), "Candidate did not start its child"
                    assert stream.read() == b"ready\n"
            process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, stderr
        result = json.loads(stdout.splitlines()[-1])
        if finish == "cancel":
            assert result == {"cancelled": True}
        else:
            assert result["passed"] is (finish == "return")
            if finish == "fail":
                assert result["result"].startswith("failed:")
            else:
                assert result["result"] == ("passed" if finish == "return" else "timed out")
        assert not process_running(int(pid_file.read_text())), "Candidate child outlived the grader"
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        # Reap only this fixture's session and explicitly recorded child, even
        # when demonstrating the pre-fix leak. Never target the pytest group.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize("helper", ("safe_subprocess", "libeval"))
def test_outer_deadline_reaps_nested_helper_session(tmp_path, helper):
    pid_file = tmp_path / "child.pid"
    candidate = f"sleep 60 >/dev/null 2>&1 &\necho $! > {shlex.quote(str(pid_file))}\nwait\n"
    process = subprocess.Popen(
        [sys.executable, "-c", NESTED_DRIVER, helper, candidate],
        cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, stderr
        assert json.loads(stdout.splitlines()[-1]) == {"finished": True, "exitcode": -signal.SIGKILL}
        assert not process_running(int(pid_file.read_text())), "Nested command escaped the native owner's cleanup"
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
