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

from shellbox.process_framing import (
    CHUNK_BYTES,
    EXIT,
    FRAME_BYTES,
    PAD,
    STDERR,
    STDIN,
    STDOUT,
    FrameDecoder,
    decode_frame,
    encode_frame,
)

# Daytona corrupts a command's log stream when stdout and stderr interleave, and
# its SDK withholds a websocket message shorter than three bytes until more
# output arrives. Every frame goes to stdout followed by padding, so the stream
# never interleaves and only padding can be withheld.
TRAILER = encode_frame(PAD, b"")


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
        decoder = FrameDecoder()
        try:
            while data := os.read(0, FRAME_BYTES):
                for frame in decoder.feed(data):
                    kind, decoded = decode_frame(frame)
                    if kind != STDIN:
                        raise ValueError("Unexpected process input frame")
                    child.stdin.write(decoded)
                    child.stdin.flush()
            decoder.finish()
        except BrokenPipeError:
            return
        finally:
            with contextlib.suppress(BrokenPipeError):
                child.stdin.close()

    output_lock = threading.Lock()

    def write_frame(kind: bytes, data: bytes) -> None:
        with output_lock:
            sys.stdout.buffer.write(encode_frame(kind, data) + TRAILER)
            sys.stdout.buffer.flush()

    def output_stream(source, kind: bytes) -> None:
        while data := os.read(source.fileno(), CHUNK_BYTES):
            write_frame(kind, data)

    threading.Thread(target=input_stream, daemon=True).start()
    stdout = threading.Thread(target=output_stream, args=(child.stdout, STDOUT))
    stderr = threading.Thread(target=output_stream, args=(child.stderr, STDERR))
    stdout.start()
    stderr.start()
    # Keep the leader unreaped until its descendants are stopped, preventing
    # its process-group ID from being reused between exit and cleanup.
    os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOWAIT)
    os.killpg(child.pid, signal.SIGKILL)
    code = child.wait()
    stdout.join()
    stderr.join()
    write_frame(EXIT, str(code).encode())


if __name__ == "__main__":
    main()
