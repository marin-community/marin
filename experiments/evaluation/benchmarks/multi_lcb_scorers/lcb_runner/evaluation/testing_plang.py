import fcntl
import os
import platform
import re
import resource
import signal
import subprocess
import tempfile
import time

from decimal import Decimal, InvalidOperation
from enum import Enum
from math import isclose
from pathlib import Path

# from time import time, sleep
from typing import List, Literal, Dict, Tuple, Callable, Optional
from dataclasses import dataclass
from itertools import zip_longest

IO_BUF_BLOCK_SZ = 4096
DEFAULT_IO_BUF_SZ = 256 * 1024 * 1024
MAX_PRC_VIRT_MEM = 16 * 1024 * 1024 * 1024
MAX_PRC_STACK_MEM = 8 * 1024 * 1024 * 1024
TIK = 0.1


def limit_virtual_memory():

    # TODO: For linux system use 'ulimit' to limit subprocess memory and cpu.

    # The tuple below is of the form (soft limit, hard limit).
    # When the limit cannot be changed, setrlimit() raises ValueError.
    # soft, hard = resource.getrlimit(resource.RLIMIT_AS)

    # resource.setrlimit(resource.RLIMIT_AS, (MAX_VIRTUAL_MEMORY, resource.RLIM_INFINITY))
    resource.setrlimit(resource.RLIMIT_AS, (MAX_PRC_VIRT_MEM, MAX_PRC_VIRT_MEM))
    resource.setrlimit(resource.RLIMIT_DATA, (MAX_PRC_VIRT_MEM, MAX_PRC_VIRT_MEM))
    if not platform.uname().system == "Darwin":
        resource.setrlimit(
            resource.RLIMIT_STACK, (MAX_PRC_STACK_MEM, MAX_PRC_STACK_MEM)
        )


@dataclass
class SubprocessConfig:
    """TODO: maybe link with yaml files."""

    plang: str  # plang or process name
    limit_memory: bool = True
    bufsize: Optional[int] = None
    env: dict = (
        None  # env variables, will overwrite current ENV variables for the process
    )
    build_timeout: int = 60
    run_timeout: int = 15  # time out for single test
    project_dir: Optional[str] = (
        None  # Sets the current directory before the subprocess is executed.
    )

    def __post_init__(self):
        plang = self.plang

        if self.bufsize is None:
            self.bufsize = DEFAULT_IO_BUF_SZ

        if self.env and "PATH" not in self.env:
            # all necessary variables must be copied from parent env.
            self.env["PATH"] = os.environ["PATH"]

        if plang in ("js", "ts", "javascript", "typescript", "rust"):
            self.limit_memory = False

    def set_cwd(self, cwd: str | Path):
        self.project_dir = Path(cwd)


class Result:
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    plang: str = "python"

    def __init__(self, exit_code: int, stdout: str, stderr: str, plang: str):
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.plang = plang

        # error might be in the output
        if self.stderr == "" and (self.exit_code is None or self.exit_code != 0):
            self.stderr = stdout

    def __str__(self):
        data = {
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }
        return str(data)


class Status(Enum):
    # pylint: disable=C0103
    UNK = 0
    Done = 1
    BuildDone = 2
    BuildFailed = 3
    BuildTimeOut = 4
    SyntaxError = 5
    AbnormalTermination = 6
    Exception = 7
    ValueError = 8
    TimeoutExpired = 9
    OutOfMemory = 10
    WrongAnswer = 11
    EmptyCode = 12
    NPMFailed = 13

    def is_failure(self):
        if self.name not in ("UNK", "Done", "BuildDone"):
            return True
        return False

    def __str__(self):
        return self.name


def get_build_status(result: Result) -> Status:

    # rust specific
    if "what():  Resource temporarily unavailable" in result.stderr:
        return Status.OutOfMemory

    if "TimeoutExpired" in result.stderr:
        return Status.BuildTimeOut

    if result.exit_code is None:
        return Status.BuildFailed

    if result.exit_code != 0:
        return Status.SyntaxError

    return Status.BuildDone


def get_run_status(result: Result) -> Status:
    status = Status.Done

    if "SyntaxError" in result.stderr:
        status = Status.SyntaxError
        return status

    if "ValueError" in result.stderr:
        status = Status.ValueError
        return status

    if "TimeoutExpired" in result.stderr:
        status = Status.TimeoutExpired
        return status

    if (
        "OutOfMemoryException" in result.stderr
        or "OutOfMemoryError" in result.stderr
        or "out of memory" in result.stderr
        or "FatalProcessOutOfMemory" in result.stderr
        or re.search("memory allocation .*? failed", result.stderr)
    ):
        status = Status.OutOfMemory
        return status

    if result.exit_code is None:
        status = Status.AbnormalTermination
        return status

    if result.exit_code != 0:
        status = Status.Exception
        return status

    return status


def set_nonblocking(reader):
    fd = reader.fileno()
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)


def kill_process(process):
    if process is None:
        return

    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        process.stdin.close()
    except Exception:
        pass

    try:
        os.kill(process.pid, signal.SIGKILL)
    except Exception:
        pass


def run(
    args: List[str],
    timeout_seconds: int = 8,
    input_data: str = None,
    sconf: SubprocessConfig = None,
) -> Result:
    """
    Runs the given program with arguments. After the timeout elapses, kills the process
    and all other processes in the process group. Captures at most max_output_size bytes
    of stdout and stderr each, and discards any output beyond that.
    """

    if not sconf:
        raise NotImplementedError("Subprocess config is expected")

    plang = sconf.plang
    env = sconf.env
    limit_memory = sconf.limit_memory
    bufsize = sconf.bufsize
    cwd = sconf.project_dir

    p = None  # global var sthat stores currenlty running subprocess

    # convert input data to stdin
    if input_data is not None:
        if input_data[-1:] != "\n":
            input_data += "\n"

        input_data = input_data.encode("utf-8")

    def _start_proc() -> Result:
        nonlocal p

        stdout, stderr = b"", b""
        exit_code = None

        p = subprocess.Popen(
            args,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            # increase bufsize to fit estimated output size
            bufsize=4 * bufsize,
            cwd=cwd,
            preexec_fn=limit_virtual_memory if limit_memory else None,
        )

        # set_nonblocking(p.stdin)
        set_nonblocking(p.stdout)
        set_nonblocking(p.stderr)

        time.sleep(TIK)

        try:
            stdout, stderr = p.communicate(input=input_data, timeout=timeout_seconds)
            p.send_signal(signal.SIGINT)
            exit_code = p.returncode
        except subprocess.TimeoutExpired as ex:
            stderr = f"[{type(ex)}][{ex}]".encode("utf8")
        except Exception as ex:
            stderr = f"[{type(ex)}][{ex}]".encode("utf8")

        try:
            stdout = stdout.decode("utf-8")
            stderr = stderr.decode("utf-8")
        except Exception as ex:
            stdout = ""
            stderr = f"[{type(ex)}][{ex}]"
            # args = [str(s) for s in args]
            # print(f"ERROR! Can't decode stdout/stderr {stderr} on run('" + "', '".join(args) + "')")

        if len(stdout) > 64 * bufsize:
            print(
                f"the output size ({len(stdout)}) exceeded the expected size ({bufsize // 4}) by 64 times, so we cut it"
            )
            stdout = stdout[: 64 * bufsize]

        return Result(plang=plang, exit_code=exit_code, stdout=stdout, stderr=stderr)

    try:
        result = _start_proc()
    finally:
        # Cleanup remaining zombie process
        kill_process(p)

    return result


# pylint: disable=W0613
def eval_script_java(
    path: Path, input_data: List[str], sconf: SubprocessConfig, **kwargs
):

    program = open(str(path), "r", encoding="utf8").read()
    project_dir = path.parent

    fbasename = "Main"
    i = program.find("\npublic class ")
    if i >= 0:
        i += 14
        while program[i] in [" ", "\n", "\t"]:
            i += 1
        j = program.find("{", i)
        k = program.find(" ", i)
        l = program.find("\n", i)
        j = min(j, k, l)
        if j > i:
            fbasename = program[i:j].strip()
            # print(f'Java file name [{fbasename}]')
            path = os.path.join(project_dir, fbasename + ".java")
            with open(path, "w", encoding="utf8") as f:
                f.write(program)
                f.flush()

    outputs, errors, status = [], [], None

    with tempfile.TemporaryDirectory() as outdir:
        # Each Java file contains the class with same name `JAVA_CLASS_NAME`
        # Hence, javac will same JAVA_CLASS_NAME.class file for each problem
        # Write class for each problem to a different temp dir
        # Use UTF8 encoding with javac
        result = run(
            ["javac", "-encoding", "UTF8", "-d", outdir, path],
            timeout_seconds=sconf.build_timeout,
            sconf=sconf,
        )

        if result.exit_code is None:
            status = Status.BuildFailed
            outputs.append(result.stdout)
            errors.append(result.stderr)
        elif result.exit_code != 0:
            # Well, it's a compile error. Maybe a type error or
            # something. But, why break the set convention
            status = Status.SyntaxError
            outputs.append(result.stdout)
            errors.append(result.stderr)
        else:
            for input_str in input_data:
                result = run(
                    ["java", "-ea", "-cp", f"{outdir}", fbasename],
                    input_data=input_str,
                    timeout_seconds=sconf.run_timeout,
                    sconf=sconf,
                )
                outputs.append(result.stdout)
                errors.append(result.stderr)
                status = get_run_status(result)
                if status.is_failure():
                    break

    if status is None and result.exit_code == 0:
        status = Status.Done

    return {
        "status": status,
        "exit_code": result.exit_code,
        "stdout": outputs,
        "stderr": errors,
    }


# pylint: disable=W0613
def eval_script_cs(
    path: Path, input_data: List[str], sconf: SubprocessConfig, **kwargs
):

    outputs, errors, status = [], [], None
    exec_name = "Main.exe"

    # 'msc' is part of 'mono' library
    result = run(
        ["mcs", f"-out:{exec_name}", path],
        timeout_seconds=sconf.build_timeout,
        sconf=sconf,
    )

    status = get_build_status(result)

    if status != Status.BuildDone:
        outputs.append(result.stdout)
        errors.append(result.stderr)
    else:
        for input_str in input_data:
            result = run(
                ["mono", exec_name],
                input_data=input_str,
                timeout_seconds=sconf.run_timeout,
                sconf=sconf,
            )
            outputs.append(result.stdout)
            errors.append(result.stderr)
            status = get_run_status(result)
            if status.is_failure():
                break

    if status is None and result.exit_code == 0:
        status = Status.Done

    return {
        "status": status,
        "exit_code": result.exit_code,
        "stdout": outputs,
        "stderr": errors,
    }


# pylint: disable=W0613
def eval_script_cpp(
    path: Path, input_data: List[str], sconf: SubprocessConfig, **kwargs
):

    outputs, errors, status = [], [], None
    exec_name = path.with_suffix("")

    result = run(
        ["g++", path, "-o", exec_name, "-std=c++17", "-mcmodel=medium"],
        timeout_seconds=sconf.build_timeout,
        sconf=sconf,
    )

    status = get_build_status(result)

    if status != Status.BuildDone:
        outputs.append(result.stdout)
        errors.append(result.stderr)
    else:
        for input_str in input_data:
            result = run(
                [exec_name],
                input_data=input_str,
                timeout_seconds=sconf.run_timeout,
                sconf=sconf,
            )

            if (
                "In file included from /shared/centos7/gcc/9.2.0-skylake/"
                in result.stderr
            ):
                raise SyntaxError("Skylake bug encountered")
            if "/4.8.2" in result.stderr:
                raise SyntaxError("Ancient compiler encountered")

            outputs.append(result.stdout)
            errors.append(result.stderr)

            status = get_run_status(result)
            if status.is_failure():
                break

    if status is None and result.exit_code == 0:
        status = Status.Done

    return {
        "status": status,
        "exit_code": result.exit_code,
        "stdout": outputs,
        "stderr": errors,
    }


# pylint: disable=W0613
def eval_script_py(
    path: Path, input_data: List[str], sconf: SubprocessConfig, **kwargs
):

    outputs, errors, status = [], [], None

    path = str(path)
    if path.endswith(".py"):
        interpreter = "python"
    elif path.endswith(".py2"):
        interpreter = "python2"
    elif path.endswith(".py3"):
        interpreter = "python3"
    else:
        raise RuntimeError(f"Invalid python file extention [{path}]")

    if not input_data:
        input_data = [None]

    for input_str in input_data:
        result = run(
            [interpreter, str(path)],
            input_data=input_str,
            timeout_seconds=sconf.run_timeout,
            sconf=sconf,
        )
        outputs.append(result.stdout)
        errors.append(result.stderr)

        status = get_run_status(result)
        if status.is_failure():
            break

    if status is None and result.exit_code == 0:
        status = Status.Done

    return {
        "status": status,
        "exit_code": result.exit_code,
        "stdout": outputs,
        "stderr": errors,
    }


# pylint: disable=W0613
def eval_script_rust(
    path: Path, input_data: List[str], sconf: SubprocessConfig, **kwargs
):
    outputs, errors, status = [], [], None
    exec_name = path.with_suffix(".c")

    # rustc --help -v
    result = run(
        ["rustc", str(path), "-C", "debuginfo=2", "--verbose", "-o", exec_name],
        timeout_seconds=sconf.build_timeout,
        sconf=sconf,
    )

    status = get_build_status(result)

    if status != Status.BuildDone:
        outputs.append(result.stdout)
        errors.append(result.stderr)
    else:
        for input_str in input_data:
            result = run(
                [exec_name],
                input_data=input_str,
                timeout_seconds=sconf.run_timeout,
                sconf=sconf,
            )
            outputs.append(result.stdout)
            errors.append(result.stderr)

            status = get_run_status(result)
            if status.is_failure():
                break

    if status is None and result.exit_code == 0:
        status = Status.Done

    return {
        "status": status,
        "exit_code": result.exit_code,
        "stdout": outputs,
        "stderr": errors,
    }


def check_js_runtime(program) -> Literal["deno", "node"]:
    """Check if script must be run using 'deno' runtime environment.

    Examples:
    Deno.stdin
    Deno.readTextFromStdin
    Deno.readTextFileSync
    input = Deno.readAllSync(Deno.stdin)
    import { readline } from "https://deno.land/std@0.129.0/testing/readline.ts";

    """

    if re.search(
        "Deno[.]stdin|Deno[.]readTextFromStdin|Deno[.]readTextFileSync| Deno[.]|https://deno[.]land",
        program,
    ):
        return "deno"
    return "node"


def eval_script_js(
    js_path: Path, input_data: List[str], sconf: SubprocessConfig, **kwargs
):

    outputs, errors, status = [], [], None

    if not input_data:
        input_data = [None]

    runtime_name = check_js_runtime(kwargs["code"])

    for input_str in input_data:

        if runtime_name == "node":
            run_args = ["node", str(js_path)]
        elif runtime_name == "deno":
            run_args = ["deno", "run", str(js_path)]
        else:
            raise RuntimeError("Wrong JavaScript type")

        result = run(
            run_args,
            input_data=input_str,
            timeout_seconds=sconf.run_timeout,
            sconf=sconf,
        )

        outputs.append(result.stdout)
        errors.append(result.stderr)

        status = get_run_status(result)
        if status.is_failure():
            break

    if status is None and result.exit_code == 0:
        status = Status.Done

    return {
        "status": status,
        "exit_code": result.exit_code,
        "stdout": outputs,
        "stderr": errors,
    }


def install_npm_packages(sconf: SubprocessConfig) -> Result:
    """Install standard npm packages."""

    # must be run without limits on memomry
    assert not sconf.limit_memory
    return run(
        [
            "npm",
            "i",
            "-D",
            "@types/node",
            "@types/readline-sync",
            "readline-sync",
            "yargs",
            "js-combinatorics",
        ],
        timeout_seconds=sconf.build_timeout,
        sconf=sconf,
    )


def eval_script_ts(ts_path: Path, input_data, sconf: SubprocessConfig, **kwargs):

    outputs, errors, status = [], [], None
    js_path = ts_path.with_suffix(".js")

    runtime_name = check_js_runtime(kwargs["code"])

    # install npm packages
    result = install_npm_packages(sconf)

    if result.exit_code is None or result.exit_code != 0:
        return {
            "status": Status.NPMFailed,
            "exit_code": result.exit_code,
            "stdout": [result.stdout],
            "stderr": [result.stderr],
        }

    if runtime_name == "node":
        # compile typescript to javascript
        # https://www.typescriptlang.org/docs/handbook/compiler-options.html
        result = run(
            [
                "tsc",
                "--target",
                "esnext",
                "--module",
                "nodenext",
                "--moduleResolution",
                "nodenext",
                str(ts_path),
            ],
            timeout_seconds=sconf.build_timeout,
            sconf=sconf,
        )
    elif runtime_name == "deno":
        # don't need to compile anything
        pass

    status = get_build_status(result)

    if status != Status.BuildDone:
        outputs.append(result.stdout)
        errors.append(result.stderr)
    else:
        for input_str in input_data:
            run_args = (
                ["deno", "run", ts_path]
                if runtime_name == "deno"
                else ["node", js_path]
            )

            result = run(
                run_args,
                input_data=input_str,
                timeout_seconds=sconf.run_timeout,
                sconf=sconf,
            )

            outputs.append(result.stdout)
            errors.append(result.stderr)

            status = get_run_status(result)
            if status.is_failure():
                break

    if status is None and result.exit_code == 0:
        status = Status.Done

    return {
        "status": status,
        "exit_code": result.exit_code,
        "stdout": outputs,
        "stderr": errors,
    }


# pylint: disable=W0613
def eval_script_ruby(
    path: Path, input_data: List[str], sconf: SubprocessConfig, **kwargs
):
    outputs, errors, status = [], [], None

    if not input_data:
        input_data = [None]

    for input_str in input_data:
        result = run(
            ["ruby", str(path)],
            input_data=input_str,
            timeout_seconds=sconf.run_timeout,
            sconf=sconf,
        )
        outputs.append(result.stdout)
        errors.append(result.stderr)

        status = get_run_status(result)
        if status.is_failure():
            break

    if status is None and result.exit_code == 0:
        status = Status.Done
    return {
        "status": status,
        "exit_code": result.exit_code,
        "stdout": outputs,
        "stderr": errors,
    }


# pylint: disable=W0613
def eval_script_go(
    path: Path, input_data: List[str], sconf: SubprocessConfig, **kwargs
):

    outputs, errors, status = [], [], None
    exec_name = ".".join(str(path).split(".")[:-1])

    result = run(
        ["go", "build", "-o", exec_name, str(path)],
        timeout_seconds=sconf.build_timeout,
        sconf=sconf,
    )
    status = get_build_status(result)

    if status != Status.BuildDone:
        outputs.append(result.stdout)
        errors.append(result.stderr)
    else:
        for input_str in input_data:
            result = run(
                [exec_name],
                input_data=input_str,
                timeout_seconds=sconf.run_timeout,
                sconf=sconf,
            )
            outputs.append(result.stdout)
            errors.append(result.stderr)

            status = get_run_status(result)
            if status.is_failure():
                break

    if status is None and result.exit_code == 0:
        status = Status.Done

    return {
        "status": status,
        "exit_code": result.exit_code,
        "stdout": outputs,
        "stderr": errors,
    }


# pylint: disable=W0613
def eval_script_php(
    path: Path,
    input_data: List[str],
    sconf: SubprocessConfig,
    **kwargs,
) -> dict:
    """
    Evaluates a PHP script.

    :param path: Path to the PHP source file.
    :type path: Path
    :param input_data: List of input strings for each test case.
    :type input_data: List[str]
    :param timeout_seconds: Timeout for execution in seconds.
    :type timeout_seconds: int
    :param bufsizes: List of buffer sizes for each test case.
    :type bufsizes: List[int | None]
    :param kwargs: Additional keyword arguments, including 'output_data'.
    :type kwargs: dict
    :return: A dictionary containing the status, exit code, stdout, and stderr.
    :rtype: dict
    """
    outputs, errors, status = [], [], None

    if not input_data:
        input_data = [None]

    # Original loop for multiple test cases with input_data
    for input_str in input_data:
        result = run(
            ["php", str(path)],
            input_data=input_str,
            timeout_seconds=sconf.run_timeout,
            sconf=sconf,
        )
        outputs.append(result.stdout)
        errors.append(result.stderr)
        status = get_run_status(result)

        if status.is_failure():
            break

    if status is None and result.exit_code == 0:
        status = Status.Done

    return {
        "status": status,
        "exit_code": result.exit_code,
        "stdout": outputs,
        "stderr": errors,
    }


def eval_script_scala(
    path: Path,
    input_data: List[str],
    sconf: SubprocessConfig,
    **kwargs,
) -> dict:
    """
    Evaluates a Scala script by compiling directly with scalac and running with scala.
    """

    outputs, errors, status = [], [], None
    project_dir = sconf.project_dir

    # Extract class name from file
    class_name = _extract_scala_object_name(kwargs["code"])

    # Compile with scalac
    result = run(
        ["scalac", "-d", project_dir, str(path)],
        timeout_seconds=sconf.build_timeout,
        sconf=sconf,
    )

    status = get_build_status(result)

    if status != Status.BuildDone:
        outputs.append(result.stdout)
        errors.append(result.stderr)
    else:
        for input_str in input_data:
            # Run with scala
            result = run(
                ["scala", "-cp", project_dir, class_name],
                input_data=input_str,
                timeout_seconds=sconf.run_timeout,
                sconf=sconf,
            )

            outputs.append(result.stdout)
            errors.append(result.stderr)

            status = get_run_status(result)
            if status.is_failure():
                break

    # Clean up compiled class files
    for class_file in project_dir.glob("*.class"):
        class_file.unlink()

    if status is None and result.exit_code == 0:
        status = Status.Done

    return {
        "status": status,
        "exit_code": result.exit_code,
        "stdout": outputs,
        "stderr": errors,
    }


# pylint: disable=W0613
def eval_script_kotlin(
    path: Path,
    input_data: List[str],
    sconf: SubprocessConfig,
    **kwargs,
) -> dict:
    """
    Evaluates a Kotlin script.

    :param path: Path to the Kotlin source file.
    :type path: Path
    :param input_data: List of input strings for each test case.
    :type input_data: List[str]
    :param timeout_seconds: Timeout for execution in seconds.
    :type timeout_seconds: int
    :param bufsizes: List of buffer sizes for each test case.
    :type bufsizes: List[int | None]
    :param kwargs: Additional keyword arguments, including 'program' and 'output_data'.
    :type kwargs: dict
    :raises RuntimeError: If there is an issue during Kotlin compilation or execution.
    :return: A dictionary containing the status, exit code, stdout, and stderr.
    :rtype: dict
    """
    outputs, errors, status = [], [], None
    exec_name = path.with_suffix(".jar")

    result = run(
        ["kotlinc", str(path), "-include-runtime", "-d", str(exec_name)],
        timeout_seconds=sconf.build_timeout,
        sconf=sconf,
    )

    if result.exit_code is None:
        status = Status.BuildFailed
        outputs.append(result.stdout)
        errors.append(result.stderr)
    elif result.exit_code != 0:
        status = Status.SyntaxError
        outputs.append(result.stdout)
        errors.append(result.stderr)
    else:
        for input_str in input_data:
            result = run(
                ["java", "-jar", str(exec_name)],
                input_data=input_str,
                sconf=sconf,
                timeout_seconds=sconf.run_timeout,
            )

            outputs.append(result.stdout)
            errors.append(result.stderr)

            status = get_run_status(result)
            if status.is_failure():
                break

    if status is None and result.exit_code == 0:
        status = Status.Done

    return {
        "status": status,
        "exit_code": result.exit_code,
        "stdout": outputs,
        "stderr": errors,
    }


eval_scripts: Dict[str, Tuple[Callable, str]] = {
    "python": (eval_script_py, ".py"),
    "python2": (eval_script_py, ".py2"),
    "python3": (eval_script_py, ".py3"),
    "java": (eval_script_java, ".java"),
    "c++": (eval_script_cpp, ".cpp"),
    "rust": (eval_script_rust, ".rs"),
    "javascript": (eval_script_js, ".js"),
    "typescript": (eval_script_ts, ".ts"),
    "ruby": (eval_script_ruby, ".rb"),
    "go": (eval_script_go, ".go"),
    "c#": (eval_script_cs, ".cs"),
    "kotlin": (eval_script_kotlin, ".kt"),
    "php": (eval_script_php, ".php"),
    "scala": (eval_script_scala, ".scala"),
}


def compile_and_run(code: str, input_data: List[str], sconf: SubprocessConfig):

    plang = sconf.plang

    eval_script, file_ext = eval_scripts[plang]

    with tempfile.TemporaryDirectory() as tmpdirname:
        fname = Path(tmpdirname, "Main" + file_ext)

        sconf.set_cwd(tmpdirname)

        if plang == "scala":
            scala_object_name = _extract_scala_object_name(code)
            fname = Path(tmpdirname, scala_object_name + file_ext)

        with open(fname, "w", encoding="utf8") as f:
            f.write(code)
            f.flush()

        result = eval_script(fname, input_data, sconf=sconf, code=code)

        assert isinstance(result["stdout"], list)
        assert isinstance(result["stderr"], list)
        assert isinstance(result["status"], Status)

        subprocess.run(["pkill", "-f", str(fname)], check=False)

    return {
        "stdout": [x.strip() for x in result["stdout"]],
        "stderr": result["stderr"][-1],
        "exit_code": result["exit_code"],
        "status": result["status"],
    }


def get_stripped_lines(val: str) -> List[str]:
    ## you don't want empty lines to add empty list after splitlines!
    val = val.strip()

    return [val_line.strip() for val_line in val.split("\n")]


def truncatefn(s, length=300) -> str:
    if isinstance(s, str):
        pass
    else:
        s = str(s)

    if len(s) > length:
        return s[: length - 3] + "..."
    else:
        return s


def convert_line_to_decimals(line: str) -> tuple[bool, list[Decimal]]:

    try:
        decimal_line = [Decimal(elem) for elem in line.split()]
    except (TypeError, InvalidOperation):
        return False, []

    if any([x.is_nan() for x in decimal_line]):
        return False, []

    return True, decimal_line


def patch_prog(program: str, plang: str) -> str:
    """Minor compilation/run errors can be fixed by patching the code.

    Args:
        program (str): code of a program
        plang (str): name of a programming language

    Returns:
        (str): patched program code
    """
    plang = plang.lower()
    if plang == "python":
        # patch blas: fix numpy import error when working in single thread
        patch = "import os\nos.environ['OPENBLAS_NUM_THREADS'] = '1'\n"
        program = patch + program
    elif plang == "c++":
        pass
    elif plang == "go":
        pass
    elif plang == "javascript":
        pass

    return program


@dataclass(kw_only=True)
class NoCodeMeta:
    error: Status = Status.EmptyCode
    error_code: int = -5
    error_message: str = "Empty string instead of a program"
    success: bool = False


@dataclass(kw_only=True)
class ExecutionErrorMeta:
    error: Status
    error_code: int
    error_message: str
    success: bool = False


@dataclass(kw_only=True)
class WrongAnswerMeta:
    # mismatch in one of the tests
    error: Status = Status.WrongAnswer
    output: str = ""
    expected: str = ""
    inputs: str = ""
    error_code: int = -2
    error_message: str = ""
    success: bool = False


@dataclass(kw_only=True)
class SuccessRunMeta:
    execution_time: float
    success: bool = True


class TestScore(Enum):
    # positive if test passed, or negative number otherwise
    __test__ = False  # not a pytest
    EXECFAIL = -5  # Code execution Failed
    FAILED = -2  # Wrong Answer
    PASSED = 1  # Test passed

    def __str__(self):
        return self.value

    def __eq__(self, other):
        return self.value == other


EvalScores = List[TestScore]
ResultMeta = NoCodeMeta | ExecutionErrorMeta | SuccessRunMeta | WrongAnswerMeta


def eval_plang_code(
    program: str,
    input_data: List[str],
    output_data: List[str],
    plang: str,
    timeout: int,
) -> Tuple[EvalScores, ResultMeta]:
    """Main entry point.

    Args:
        program (str): program code.
        input_data (List[str]): tests input data in stdin format. Each string is one test.
        output_data (List[str]): tests output data in stdin format. This list is matched correspondingly to input data.
        plang (str): name of the programming language ["c++","c#", ...]
        timeout (int): test timeout

    Returns:
        EvalScores: list with scores for each test
        ResultMeta: information on error or other problems during execution
    """

    if program is None or program == "":
        return [TestScore.EXECFAIL], NoCodeMeta()

    start = time.time()

    program = patch_prog(program, plang)

    # Compile and run the program
    sconf = SubprocessConfig(plang=plang, run_timeout=timeout)
    res = compile_and_run(program, input_data, sconf=sconf)

    total_exec_time = time.time() - start

    if res["status"] != Status.Done:
        result = [TestScore.EXECFAIL]
        metadata = ExecutionErrorMeta(
            error=res["status"],
            error_code=res["exit_code"],
            error_message=res["stderr"],
        )
        return result, metadata

    all_results, metadata = match_tests_groud_truth(
        res["stdout"], input_data, output_data
    )

    if not metadata.success:
        return all_results, metadata

    # update output with real execution time
    return all_results, SuccessRunMeta(execution_time=total_exec_time)


def match_tests_groud_truth(
    code_outputs: List[str], input_data: List[str], output_data: List[str]
) -> Tuple[EvalScores, WrongAnswerMeta | SuccessRunMeta]:
    """Compare code outputs with ground truth.

    Args:
        code_outputs (List[str]): results of code execution, stdout format, one variable for each test
        input_data (List[str]): tests input data, stdin format, one variable for each test
        output_data (List[str]): expected tests outputs, stdout format, one variable for each test

    Returns:
        Tuple[EvalScores, WrongAnswerMeta | SuccessRunMeta]: _description_
    """
    epsilon = 1e-5  # LeetCode format for floats precision

    ## Compare code output with expected output
    all_results = []
    for prediction, gt_inp, gt_out in zip_longest(
        code_outputs, input_data, output_data, fillvalue="None"
    ):

        stripped_prediction_lines = get_stripped_lines(prediction)
        stripped_gt_out_lines = get_stripped_lines(gt_out)

        ## WA happens in multiple circumstances
        ## so cache the return to make it clean!
        wa_meta = WrongAnswerMeta(
            output=truncatefn(prediction),
            inputs=truncatefn(gt_inp),
            expected=truncatefn(gt_out),
            error_message="",  # will be added later
        )

        if len(stripped_prediction_lines) != len(stripped_gt_out_lines):
            all_results.append(TestScore.FAILED)
            wa_meta.error_message = "Wrong answer: mismatched output length"
            return all_results, wa_meta

        for output_line_idx, (
            stripped_prediction_line,
            stripped_gt_out_line,
        ) in enumerate(zip(stripped_prediction_lines, stripped_gt_out_lines)):

            # prepare output message in case of WA
            wa_meta.error_message = f"Wrong answer at {output_line_idx=}: {truncatefn(stripped_prediction_line)} != {truncatefn(stripped_gt_out_line)}"

            ## CASE 1: exact match
            if stripped_prediction_line == stripped_gt_out_line:
                continue

            ## CASE 2: bool match
            if stripped_prediction_line in [
                "True",
                "true",
            ] and stripped_gt_out_line in ["True", "true"]:
                continue

            if stripped_prediction_line in [
                "False",
                "false",
            ] and stripped_gt_out_line in ["False", "false"]:
                continue

            ## CASE 3: element-wise comparision
            ## if there are floating elements
            ## note that we should always be able to convert to decimals

            success, decimal_prediction_line = convert_line_to_decimals(
                stripped_prediction_line
            )
            if not success:
                all_results.append(TestScore.FAILED)
                return all_results, wa_meta

            success, decimal_gtout_line = convert_line_to_decimals(stripped_gt_out_line)

            if not success:
                all_results.append(TestScore.FAILED)
                return all_results, wa_meta

            if len(decimal_prediction_line) == len(decimal_gtout_line):
                # check all Decimals are close

                all_good = all(
                    [
                        isclose(a, b, abs_tol=epsilon, rel_tol=0)
                        for a, b in zip(decimal_prediction_line, decimal_gtout_line)
                    ]
                )

                if all_good:
                    continue

            all_results.append(TestScore.FAILED)
            return all_results, wa_meta

        all_results.append(TestScore.PASSED)

    return all_results, SuccessRunMeta(execution_time=-1)


def _extract_scala_object_name(program: str) -> str:
    """
    Extracts the Scala object name from the program string.

    Args:

    program =  '''
    object Main {
            def main(args: Array[String]): Unit = {
                ...
            }
        }
    '''

    """
    match = re.search(r"object\s+Main", program)
    if match:
        return "Main"

    match = re.search(r"object\s+(\w+)", program)
    if match:
        return match.group(1)
    return "Main"  # Default if not found


def prepare_plang_env(plang: str, timeout: int = None) -> None:
    """Some languages might fail to compile and run if your system is not prepared properly.

    Args:
        plang (str): name of a programming language.
        timeout (int, optional): value of the timeout that will be passed to process. Defaults to None.

    Returns:
        None
    """

    if plang == "go":
        # if go cache is full all go tasks may fail with Timeout
        # 5 min timeout will give enough cache space. Otherwise cache clean might take up to an hour.
        if not timeout:
            timeout = 60 * 5

        sconf = SubprocessConfig(plang="go", limit_memory=False, build_timeout=timeout)

        print("Cleaning go cache... ", end="")
        _ = run(
            ["go", "clean", "-cache"], timeout_seconds=sconf.build_timeout, sconf=sconf
        )
        print("Done")

    return None
