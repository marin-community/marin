"""Task-local IPython process. This script runs only inside the task machine."""

import io
import json
import os
import resource
import signal
import socket
import subprocess
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

STARTUP_TIMEOUT = 30.0
OUTPUT_LIMIT_BYTES = 65536
FRAME_LIMIT_BYTES = 12 * OUTPUT_LIMIT_BYTES + 1024
KERNEL_TIMEOUT_EXIT_CODE = 124
KERNEL_TERMINATED_EXIT_CODE = 125


class BoundedOutput(io.TextIOBase):
    def __init__(self, limit: int):
        self.limit = limit
        self.data = bytearray()
        self.truncated = False

    def write(self, text: str) -> int:
        value = text.encode("utf-8", errors="replace")
        remaining = self.limit - len(self.data)
        self.data.extend(value[:remaining])
        self.truncated |= len(value) > remaining
        return len(text)

    def getvalue(self) -> str:
        return self.data.decode("utf-8", errors="replace")


def request(directory: Path, value: dict, *, socket_timeout: float | None = None) -> dict:
    with socket.socket(socket.AF_UNIX) as connection:
        connection.settimeout(
            socket_timeout if socket_timeout is not None else value.get("timeout", STARTUP_TIMEOUT) + 5.0
        )
        connection.connect(str(directory / "socket"))
        connection.sendall(json.dumps(value).encode() + b"\n")
        with connection.makefile("rb") as stream:
            line = stream.readline(FRAME_LIMIT_BYTES)
        if not line:
            raise RuntimeError("The Python kernel stopped without an execution result")
        return json.loads(line)


def start(directory: Path, memory_bytes: int | None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "server.log").open("wb") as output:
        process = subprocess.Popen(
            [sys.executable, __file__, "serve", str(directory), str(memory_bytes or 0)],
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=output,
            start_new_session=True,
        )
    (directory / "pid").write_text(str(process.pid))
    deadline = time.monotonic() + STARTUP_TIMEOUT
    try:
        while True:
            if process.poll() is not None:
                raise RuntimeError((directory / "server.log").read_text())
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("The Python kernel did not start")
            try:
                request(directory, {"ping": True}, socket_timeout=remaining)
                break
            except (FileNotFoundError, ConnectionRefusedError):
                # bind creates the path before listen makes the socket ready.
                pass
            time.sleep(0.01)
    except BaseException:
        close(directory)
        process.wait()
        raise


def serve(directory: Path, memory_bytes: int | None) -> None:
    # IPython is an optional dependency supplied by the task image.
    from IPython.core.interactiveshell import InteractiveShell

    shell = InteractiveShell.instance()
    shell.colors = "NoColor"
    shell.displayhook.write_output_prompt = lambda: None
    if memory_bytes is not None:
        for kind in (resource.RLIMIT_AS, resource.RLIMIT_DATA):
            soft, hard = resource.getrlimit(kind)
            bound = min(value for value in (memory_bytes, soft, hard) if value != resource.RLIM_INFINITY)
            resource.setrlimit(kind, (bound, bound))
    timed_out = False

    def execution_timeout(signum, frame):
        nonlocal timed_out
        timed_out = True
        raise TimeoutError("Python execution timed out")

    signal.signal(signal.SIGALRM, execution_timeout)
    with socket.socket(socket.AF_UNIX) as server:
        server.bind(str(directory / "socket"))
        server.listen(1)
        while True:
            with server.accept()[0] as connection, connection.makefile("rb") as stream:
                value = json.loads(stream.readline())
                if value.get("ping"):
                    connection.sendall(b'{"ready":true}\n')
                    continue
                limit = min(value.get("output_limit_bytes", OUTPUT_LIMIT_BYTES), OUTPUT_LIMIT_BYTES)
                stdout = BoundedOutput(limit)
                stderr = BoundedOutput(limit)
                timed_out = False
                signal.setitimer(signal.ITIMER_REAL, value["timeout"])
                try:
                    with redirect_stdout(stdout), redirect_stderr(stderr):
                        result = shell.run_cell(value["code"], store_history=False)
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                connection.sendall(
                    json.dumps(
                        {
                            "exit_code": None if timed_out else int(not result.success),
                            "stdout": stdout.getvalue(),
                            "stderr": stderr.getvalue(),
                            "stdout_truncated": stdout.truncated,
                            "stderr_truncated": stderr.truncated,
                            "reason": "timed_out" if timed_out else "exited",
                        },
                        ensure_ascii=False,
                    ).encode()
                    + b"\n"
                )


def close(directory: Path) -> None:
    pid = int((directory / "pid").read_text())
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    (directory / "socket").unlink(missing_ok=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    directory = Path(sys.argv[2])
    if mode == "start":
        start(directory, int(sys.argv[3]) or None)
    elif mode == "serve":
        serve(directory, int(sys.argv[3]) or None)
    elif mode == "call":
        try:
            result = request(directory, json.load(sys.stdin))
        except TimeoutError:
            print("The candidate did not return before its execution deadline", file=sys.stderr)
            raise SystemExit(KERNEL_TIMEOUT_EXIT_CODE)
        except (ConnectionError, FileNotFoundError, RuntimeError) as error:
            print(f"The candidate kernel terminated: {error}", file=sys.stderr)
            raise SystemExit(KERNEL_TERMINATED_EXIT_CODE)
        print(json.dumps(result, ensure_ascii=False))
    elif mode == "close":
        close(directory)
    else:
        raise ValueError(f"Unknown kernel operation: {mode}")
