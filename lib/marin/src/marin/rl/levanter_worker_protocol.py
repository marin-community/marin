# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Local process transport between the SkyRL host worker and its JAX learner.

Only trusted processes in the same worker exchange these frames. Keeping the wire
payload to Python containers and NumPy arrays lets the two processes use separate
Torch and JAX environments without importing each other's framework.
"""

import concurrent.futures
import os
import pickle
import struct
import subprocess
from collections.abc import Mapping
from typing import BinaryIO

_HEADER = struct.Struct("!Q")
_SHUTDOWN_GRACE_SECONDS = 5


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            raise EOFError("Levanter learner process closed its response pipe")
        chunks.extend(chunk)
    return bytes(chunks)


def read_frame(stream: BinaryIO):
    size = _HEADER.unpack(_read_exact(stream, _HEADER.size))[0]
    return pickle.loads(_read_exact(stream, size))


def write_frame(stream: BinaryIO, payload) -> None:
    data = pickle.dumps(payload, protocol=5)
    stream.write(_HEADER.pack(len(data)))
    stream.write(data)
    stream.flush()


class LearnerProcess:
    """Own a learner subprocess and bound the complete request/response exchange."""

    def __init__(
        self,
        python_executable: str,
        *,
        environment: Mapping[str, str],
        timeout_seconds: float,
        runtime_module: str = "marin.rl.levanter_worker_runtime",
    ):
        self.timeout_seconds = timeout_seconds
        response_read, response_write = os.pipe()
        try:
            self._responses = os.fdopen(response_read, "rb")
        except BaseException:
            os.close(response_read)
            os.close(response_write)
            raise
        try:
            self.process = subprocess.Popen(
                [
                    python_executable,
                    "-u",
                    "-m",
                    runtime_module,
                    "--response-fd",
                    str(response_write),
                ],
                stdin=subprocess.PIPE,
                stdout=None,
                stderr=None,
                pass_fds=(response_write,),
                env=dict(environment),
            )
        except BaseException:
            self._responses.close()
            raise
        finally:
            os.close(response_write)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="levanter-child")

    def request(self, method: str, payload: dict, *, timeout_seconds: float | None = None):
        timeout = self.timeout_seconds if timeout_seconds is None else timeout_seconds

        def exchange():
            assert self.process.stdin is not None
            write_frame(self.process.stdin, {"method": method, "payload": payload})
            return read_frame(self._responses)

        future = self._executor.submit(exchange)
        try:
            response = future.result(timeout=timeout)
        except concurrent.futures.TimeoutError as error:
            self.close()
            raise TimeoutError(f"Levanter learner {method} exceeded {timeout} seconds") from error
        except (EOFError, BrokenPipeError) as error:
            status = self.process.poll()
            self.close()
            raise RuntimeError(
                f"Levanter learner exited during {method}, exit_code={status}; see worker stderr"
            ) from error
        if response["error"] is not None:
            raise RuntimeError(f"Levanter learner {method} failed:\n{response['error']}")
        if method == "shutdown":
            try:
                status = self.process.wait(timeout=_SHUTDOWN_GRACE_SECONDS)
            except subprocess.TimeoutExpired as error:
                self.close()
                raise TimeoutError(
                    "Levanter learner acknowledged shutdown but did not exit within its grace period"
                ) from error
            if status != 0:
                self.close()
                raise RuntimeError(f"Levanter learner acknowledged shutdown but exited with status {status}")
        return response["result"]

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()
        if self.process.stdin is not None:
            self.process.stdin.close()
        self._responses.close()
        self._executor.shutdown(wait=False, cancel_futures=True)
