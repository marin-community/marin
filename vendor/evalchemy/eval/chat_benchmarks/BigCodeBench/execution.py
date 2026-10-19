# Copyright (c) OpenAI (https://openai.com)

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.


# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# ============================================================================
import contextlib
import faulthandler
import io
import multiprocessing
from eval.runtime.program_process import run_program_process, start_program_session
import os
import platform
import signal
import random
import subprocess
import tempfile
import gzip
import json
from typing import Dict, Optional
import traceback
import types
import unittest
import sys
import time
import resource
import shutil
import builtins
from contextlib import contextmanager


def check_correctness(
    task_id: str,
    sample: dict,
    language_type: str,
    timeout: float = 3.0,
    tmp_dir: str = None,
    completion_id: Optional[int] = None,
    safe_mode: bool = False,
) -> Dict:
    """
    Evaluates the functional correctness of a completion by running the test
    suite provided in the problem.
    """

    def unsafe_execute(tmp_dir):
        start_program_session()
        random_id = random.randint(1, 100000)
        if "python" in language_type.lower():
            with create_tempdir():
                # These system calls are needed when cleaning up tempdir.
                import os
                import shutil

                rmtree = shutil.rmtree
                rmdir = os.rmdir
                chdir = os.chdir

                module_name = "__test__"
                new_module = types.ModuleType(module_name)
                # Set necessary attributes for the module
                new_module.__dict__.update(
                    {
                        "__builtins__": builtins,
                        "__file__": f"{module_name}.py",
                        "__package__": None,
                        "__doc__": None,
                        "sys": sys,
                        "os": os,
                        "environ": os.environ,
                    }
                )

                # Disable functionalities that can make destructive changes to the test.
                if safe_mode:
                    reliability_guard()

                try:
                    exec_globals = {}
                    full_code = sample["test_code"]
                    with swallow_io():
                        exec(compile(full_code, f"{module_name}.py", "exec"), new_module.__dict__)
                        sys.modules[module_name] = new_module
                        TestCases = getattr(new_module, "TestCases")
                        loader = unittest.TestLoader()
                        suite = loader.loadTestsFromTestCase(TestCases)
                        test_result = unittest.TestResult()
                        start_time = time.time()
                        with time_limit(timeout):
                            suite.run(test_result)
                        issues = test_result.failures + test_result.errors
                        if issues:
                            #  for test, trace in issues:
                            #     result.append(f"failed: {trace}")
                            result.append(f"failed: {issues[0][1]}")
                        else:
                            result.append("passed")
                except TimeoutException:
                    result.append("timed out")
                except AssertionError as e:
                    result.append(f"failed: AssertionError")
                except BaseException as e:
                    result.append(f"failed: {e}")
                # print("==================================== result ====================================")
                # print(result)
                # Needed for cleaning up.
                shutil.rmtree = rmtree
                os.rmdir = rmdir
                os.chdir = chdir

    manager = multiprocessing.Manager()
    result = manager.list()

    p = multiprocessing.Process(target=unsafe_execute, args=(tmp_dir,))
    run_program_process(p, timeout=timeout + 1)

    if not result:
        result.append("timed out")

    return {
        "task_id": task_id,
        "completion_id": completion_id,
        "result": result[0],
        "passed": result[0] == "passed",
        "finish": -1 if "finish" not in sample else sample["finish"],
        "code": sample["test_code"],
    }


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)


@contextlib.contextmanager
def safe_environment():
    # Save original functions
    original_kill = os.kill
    original_killpg = os.killpg
    original_system = os.system
    original_subprocess_call = subprocess.call
    original_subprocess_check_output = subprocess.check_output
    original_subprocess_run = subprocess.run
    original_subprocess_popen = subprocess.Popen
    original_os_popen = os.popen
    original_os_execv = os.execv
    original_os_execvp = os.execvp
    original_os_execvpe = os.execvpe

    current_pid = os.getpid()
    current_pgid = os.getpgid(current_pid)
    manager = multiprocessing.Manager()
    child_pids = manager.list()

    def safe_kill(pid, sig):
        try:
            pgid = os.getpgid(pid)
            if pid == current_pid or pid in child_pids:
                original_kill(pid, sig)
            else:
                print(f"Prevented attempt to kill PID {pid} with signal {sig}")
        except ProcessLookupError:
            pass

    def safe_killpg(pgid, sig):
        if pgid == current_pgid or pgid in {os.getpgid(pid) for pid in child_pids}:
            original_killpg(pgid, sig)
        else:
            print(f"Prevented attempt to kill PGID {pgid} with signal {sig}")

    def safe_system(command):
        print(f"Intercepted system command: {command}")
        if "kill" in command or "killall" in command:
            return 0  # Simulate successful execution without doing anything
        return original_system(command)

    def safe_subprocess_call(command, *args, **kwargs):
        print(f"Intercepted subprocess call: {command}")
        if "kill" in command or "killall" in command:
            return 0  # Simulate successful execution without doing anything
        return original_subprocess_call(command, *args, **kwargs)

    def safe_subprocess_check_output(command, *args, **kwargs):
        print(f"Intercepted command: {command}")
        if "ps" in command:
            return b""  # Simulate no processes found
        return original_subprocess_check_output(command, *args, **kwargs)

    def safe_subprocess_run(*args, **kwargs):
        print(f"Intercepted subprocess run command: {args}")
        if "kill" in args[0] or "killall" in args[0]:
            return subprocess.CompletedProcess(args, 0, b"", b"")  # Simulate successful execution
        return original_subprocess_run(*args, **kwargs)

    class SafePopen(subprocess.Popen):
        def __init__(self, *args, **kwargs):
            print(f"Intercepted Popen command: {args}")
            kwargs["preexec_fn"] = os.setsid  # Start the process in a new session
            super().__init__(*args, **kwargs)
            child_pids.append(self.pid)

        def communicate(self, *args, **kwargs):
            try:
                return super().communicate(*args, **kwargs)
            except subprocess.TimeoutExpired:
                print("Timeout expired, intercepted and returning None")
                return None, None

        def kill(self):
            print(f"Intercepted kill call for PID {self.pid}")
            safe_kill(self.pid, signal.SIGTERM)

        def terminate(self):
            print(f"Intercepted terminate call for PID {self.pid}")
            safe_kill(self.pid, signal.SIGTERM)

    def safe_os_popen(command):
        print(f"Intercepted os.popen command: {command}")
        if "kill" in command or "killall" in command:
            return os.popen("echo Intercepted")
        return original_os_popen(command)

    def safe_exec(*args, **kwargs):
        print(f"Intercepted exec command: {args}")

    # Override the risky functions with the safe versions
    os.kill = safe_kill
    os.killpg = safe_killpg
    os.system = safe_system
    subprocess.call = safe_subprocess_call
    subprocess.check_output = safe_subprocess_check_output
    subprocess.run = safe_subprocess_run
    subprocess.Popen = SafePopen
    os.popen = safe_os_popen
    os.execv = safe_exec
    os.execvp = safe_exec
    os.execvpe = safe_exec

    try:
        yield
    finally:
        for pid in child_pids:
            try:
                os.kill(pid, signal.SIGTERM)
                for _ in range(10):
                    time.sleep(0.1)
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        break
                else:
                    os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except Exception as e:
                print(f"Error handling process {pid}: {e}")

        os.kill = original_kill
        os.killpg = original_killpg
        os.system = original_system
        subprocess.call = original_subprocess_call
        subprocess.check_output = original_subprocess_check_output
        subprocess.run = original_subprocess_run
        subprocess.Popen = original_subprocess_popen
        os.popen = original_os_popen
        os.execv = original_os_execv
        os.execvp = original_os_execvp
        os.execvpe = original_os_execvpe


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from"""

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        """Returns True if the IO object can be read."""
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = "stdin"


@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)


def reliability_guard(max_as_limit: int = 30 * 1024, max_data_limit: int = 30 * 1024, max_stack_limit: int = 10):
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)

    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    import os
    import time
    from datetime import datetime

    os.environ["TZ"] = "UTC"
    time.tzset()

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    if max_as_limit and max_data_limit and max_stack_limit:
        import resource

        max_as_limit = max_as_limit * 1024 * 1024
        max_data_limit = max_data_limit * 1024 * 1024
        max_stack_limit = max_stack_limit * 1024 * 1024

        resource.setrlimit(resource.RLIMIT_AS, (max_as_limit, max_as_limit))
        resource.setrlimit(resource.RLIMIT_DATA, (max_data_limit, max_data_limit))
        if not platform.uname().system == "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (max_stack_limit, max_stack_limit))

    faulthandler.disable()

    import builtins

    builtins.exit = None
    builtins.quit = None

    try:
        import matplotlib.pyplot as plt

        plt.close("all")
    except:
        pass
