"""Hold a real foreground-group descendant unreaped during interruption."""

import os
import signal
import sys
from pathlib import Path


def terminate(signum, frame):
    raise SystemExit


def hold_descendant():
    os.mkfifo("child-joined")
    os.mkfifo("join-group")
    with open("join-group") as channel:
        group = int(channel.read())
    ready, joined = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(ready)
        os.setpgid(0, group)
        os.write(joined, b"1")
        os.close(joined)
        signal.pause()
        os._exit(0)
    os.close(joined)
    try:
        assert os.read(ready, 1) == b"1"
    finally:
        os.close(ready)
    Path("descendant.pid").write_text(str(child))
    signal.signal(signal.SIGTERM, terminate)
    try:
        with open("child-joined", "w") as channel:
            channel.write("ready")
        signal.pause()
    finally:
        try:
            os.kill(child, signal.SIGKILL)
        except ProcessLookupError:
            pass
        os.waitpid(child, 0)
        Path("descendant-reaped").write_text("done")


def foreground():
    with open("join-group", "w") as channel:
        channel.write(str(os.getpgrp()))
    with open("child-joined") as channel:
        assert channel.read() == "ready"
    Path("foreground.pid").write_text(str(os.getpid()))
    signal.pause()


if __name__ == "__main__":
    if sys.argv[1] == "holder":
        hold_descendant()
    else:
        foreground()
