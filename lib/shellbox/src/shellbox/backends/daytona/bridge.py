# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Encode binary process pipes across Daytona's UTF-8 session transport."""

import base64
import contextlib
import json
import os
import signal
import subprocess
import sys
import threading

CHUNK_BYTES = 65536
FRAME_BYTES = 4 * ((CHUNK_BYTES + 2) // 3) + 32


def main() -> None:
    config = json.loads(base64.b64decode(sys.argv[1]))
    child = subprocess.Popen(
        config["argv"],
        cwd=config["cwd"],
        env={**os.environ, **config["environment"]},
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    assert child.stdin is not None and child.stdout is not None and child.stderr is not None

    def input_stream() -> None:
        pending = bytearray()
        try:
            while data := os.read(0, FRAME_BYTES):
                pending.extend(data)
                while (end := pending.find(b"\n")) >= 0:
                    if end > FRAME_BYTES:
                        raise ValueError("Daytona process input frame exceeds its limit")
                    decoded = base64.b64decode(pending[:end], validate=True)
                    del pending[: end + 1]
                    if len(decoded) > CHUNK_BYTES:
                        raise ValueError("Daytona process input exceeds its limit")
                    child.stdin.write(decoded)
                    child.stdin.flush()
                if len(pending) > FRAME_BYTES:
                    raise ValueError("Daytona process input frame exceeds its limit")
        except BrokenPipeError:
            return
        finally:
            with contextlib.suppress(BrokenPipeError):
                child.stdin.close()

    def output_stream(source, target, prefix: bytes) -> None:
        while data := os.read(source.fileno(), CHUNK_BYTES):
            target.write(prefix + base64.b64encode(data) + b"\n")
            target.flush()

    threading.Thread(target=input_stream, daemon=True).start()
    stdout = threading.Thread(target=output_stream, args=(child.stdout, sys.stdout.buffer, b"D"))
    stderr = threading.Thread(target=output_stream, args=(child.stderr, sys.stderr.buffer, b"E"))
    stdout.start()
    stderr.start()
    # Keep the leader unreaped until its descendants are stopped, preventing
    # its process-group ID from being reused between exit and cleanup.
    os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOWAIT)
    os.killpg(child.pid, signal.SIGKILL)
    code = child.wait()
    stdout.join()
    stderr.join()
    sys.stdout.buffer.write(f"X{code}\n".encode())
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
